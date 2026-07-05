// TODO: Add --srt / --vtt format selection (currently only SRT is supported)
// NOTE: captions are verbatim by design (fillers, stutters, and all), and
// that's intentional and permanent: transcript cleanup changes what was said
// and will not be added here. The README points users at post-processing
// with an LLM instead.

use anyhow::Context;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tokio::io::AsyncWriteExt;
use transcribe_rs::accel::{set_ort_accelerator, OrtAccelerator};
use transcribe_rs::onnx::parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity};
use transcribe_rs::onnx::Quantization;
use walkdir::WalkDir;

// The ONNX export of Parakeet we use bakes a relative-position table sized for
// 2504 encoder frames into the model, and one frame covers 80ms of audio, so a
// single inference call can accept at most ~200 seconds of audio; the encoder
// errors out beyond that ("Attempting to broadcast an axis by a dimension other
// than 1. 2504 by N"). We therefore split long videos into segments and stitch
// the transcripts back together at natural sentence boundaries (see the
// split-point logic in `main`). transcribe-rs also prepends 250ms of silence to
// every call, so the default stays a few seconds under the true ceiling.
const DEFAULT_MAX_SEGMENT_LENGTH: f64 = 195.0; // 3m15s

// Rough CPU inference throughput, used only for --dry-run time estimates.
// Measured end-to-end (extraction + inference) at ~5.9x realtime on a 32-core
// Zen 3; transcribe-rs quotes 20-30x on other hardware, so this is conservative.
const ESTIMATED_REALTIME_FACTOR: f64 = 6.0;

// ============================================================================
// Cue shaping constants (see the cue-building section below for the
// algorithm that uses them)
// ============================================================================

/// The standard subtitle envelope per the BBC/Netflix guidelines: at most two
/// lines of at most 42 characters each, and at most ~7 seconds on screen
/// before a static cue reads as "stuck". Note there is deliberately no
/// "max chars per cue" constant: the real limit is that the cue's words can
/// be laid out as two fitting lines, which depends on where its spaces fall,
/// not just on the total (see `split_sentence_into_cues`).
const MAX_LINE_CHARS: usize = 42;
const MAX_CUE_SECS: f64 = 7.0;

/// Below roughly a second, a cue "flashes" and can't be read. Cues this short
/// are penalized (not forbidden: a lone "Cool." sentence has nowhere to go,
/// since we never merge cues across sentence boundaries), so the cue builder
/// avoids shaving tiny fragments off a sentence when a more balanced split is
/// available.
const MIN_CUE_SECS: f64 = 1.0;
const SHORT_CUE_PENALTY_PER_SEC: f64 = 25.0;

/// Ditto for characters: a cue with just a word or two of text reads as a
/// crumb. Without this, equally-priced mid-clause breaks tie, and the tie
/// resolves to whatever the DP scans first — observed as a long sentence
/// shedding a lone "doing." cue at its end.
const MIN_CUE_CHARS: usize = 20;
const SHORT_CUE_PENALTY_PER_CHAR: f64 = 1.0;

/// Costs for ending a cue after a given word, from best to worst: `:` and `;`
/// mark strong clause boundaries; an inferred pause (see
/// `WORD_SPEECH_PER_CHAR` below) is just as good, since silence is a stronger
/// cue than punctuation; `,` and `...` are acceptable; and a mid-clause break
/// is a last resort, but must stay finitely priced because long unpunctuated
/// stretches have to break *somewhere*. Sentence-final punctuation has no
/// cost constant: sentence ends are always cue boundaries.
const BOUNDARY_COLON: f64 = 4.0;
const BOUNDARY_PAUSE: f64 = 4.0;
const BOUNDARY_CLAUSE: f64 = 8.0;
const BOUNDARY_NONE: f64 = 30.0;

/// Breaking right after a function word severs it from the phrase it
/// introduces ("...being a tool we | can make use of..." splits a subject
/// from its verb). Added on top of `BOUNDARY_NONE` so the DP prefers
/// sliding a forced mid-clause break a word or two to a phrase edge.
/// Punctuated boundaries are exempt: a break after "so," is fine even
/// though "so" is a conjunction.
const FUNCTION_WORD_BREAK_PENALTY: f64 = 10.0;

/// A deliberately generous (slow) estimate of how long a word takes to say:
/// ~11 chars/s plus fixed per-word overhead. Parakeet's word timestamps are
/// contiguous — each word ends exactly where the next begins — so silence is
/// absorbed into the preceding word's raw duration. A word whose raw duration
/// exceeds this estimate by `PAUSE_MIN_SECS` therefore marks a real pause.
/// Over-estimating errs toward keeping cues on screen slightly too long
/// rather than cutting them off mid-word.
const WORD_SPEECH_FLOOR: f64 = 0.25;
const WORD_SPEECH_PER_CHAR: f64 = 0.09;
const PAUSE_MIN_SECS: f64 = 0.8;

/// How long a cue lingers after its last word has (by estimate) been spoken,
/// so trimmed cues don't vanish the instant the voice stops.
const CUE_LINGER_SECS: f64 = 0.5;

/// The HuggingFace repository holding the ONNX export of Parakeet that we use.
const MODEL_REPO: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";

/// The files `ParakeetModel::load` opens from the model directory, using the
/// int8 quantization (fast on CPU, and a ~670MB download instead of ~2.5GB).
/// These names must match exactly: transcribe-rs silently falls back to the
/// fp32 `encoder-model.onnx` if the int8 file is missing.
const MODEL_FILES: &[&str] = &[
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "nemo128.onnx",
    "vocab.txt",
];

/// Command-line arguments.
struct Args {
    files: Vec<PathBuf>,
    /// None means "not explicitly set on command line"
    model: Option<PathBuf>,
    segment_length: Option<f64>,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
    one_file_system: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut files = Vec::new();
    let mut model = None;
    let mut segment_length = None;
    let mut dry_run = false;
    let mut quiet = false;
    let mut verbose = false;
    let mut one_file_system = false;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Short('h') | Long("help") => {
                print_help();
                std::process::exit(0);
            }
            Short('V') | Long("version") => {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            Long("model") => {
                model = Some(PathBuf::from(parser.value()?));
            }
            Long("segment-length") => {
                segment_length = Some(parser.value()?.parse()?);
            }
            Long("dry-run") => {
                dry_run = true;
            }
            Short('q') | Long("quiet") => {
                quiet = true;
            }
            Short('v') | Long("verbose") => {
                verbose = true;
            }
            Short('x') | Long("one-file-system") => {
                one_file_system = true;
            }
            Value(val) => {
                files.push(val.into());
            }
            _ => return Err(arg.unexpected()),
        }
    }

    Ok(Args {
        files,
        model,
        segment_length,
        dry_run,
        quiet,
        verbose,
        one_file_system,
    })
}

fn print_help() {
    println!(
        "\
{name} {version}
Generate SRT subtitle files from video using local speech-to-text.

Transcription runs fully locally using NVIDIA's Parakeet model via ONNX
Runtime; no audio ever leaves your machine. On first run, the model files
(~670 MB) are downloaded from https://huggingface.co/{model_repo}.

USAGE:
    {name} [OPTIONS] <PATH>...

ARGS:
    <PATH>...    Video files or directories to transcribe

OPTIONS:
    -h, --help                  Print help information
    -V, --version               Print version information
        --model <DIR>           Directory holding the Parakeet model files
                                (default: auto-download to the user data dir)
        --segment-length <SEC>  Max segment length in seconds (default: {segment_length})
                                The Parakeet model accepts at most ~200 seconds of
                                audio per inference call, so this cannot be raised
                                meaningfully; lowering it mainly reduces memory use.
        --dry-run               List what would be transcribed without transcribing
    -q, --quiet                 Minimal output (errors only)
    -v, --verbose               Print split point details when segmenting long videos
    -x, --one-file-system       Don't cross filesystem boundaries when recursing directories

CONFIGURATION:
    Optional config file at $XDG_CONFIG_HOME/sirtea/config.toml (usually
    ~/.config/sirtea/config.toml); see config.example.toml for the available
    settings (model_path, segment_length).

REQUIREMENTS:
    - ffmpeg and ffprobe in PATH (https://ffmpeg.org/download.html)
",
        name = env!("CARGO_PKG_NAME"),
        version = env!("CARGO_PKG_VERSION"),
        model_repo = MODEL_REPO,
        segment_length = DEFAULT_MAX_SEGMENT_LENGTH,
    );
}

// deny_unknown_fields is deliberate: it turns config keys from the Gladia-based
// versions of this tool (gladia_api_key, max_cost, parallel) into a clear parse
// error naming the config file, instead of silently ignoring settings the user
// believes are in effect.
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {
    // Optional settings that can be overridden by CLI flags
    model_path: Option<PathBuf>,
    segment_length: Option<f64>,
}

fn config_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "sirtea").map(|dirs| dirs.config_dir().to_path_buf())
}

/// Where auto-downloaded model files live (under the user's data dir, since
/// model weights are data we fetch, not configuration the user edits).
fn model_data_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "sirtea")
        .map(|dirs| dirs.data_dir().join("models").join(MODEL_REPO_DIRNAME))
}

/// Directory name for the model under `model_data_dir`. Kept in sync with
/// MODEL_REPO so a future model upgrade lands in a fresh directory rather than
/// mixing files from two exports.
const MODEL_REPO_DIRNAME: &str = "parakeet-tdt-0.6b-v3-int8";

/// Check whether all required model files are present in `dir`.
fn model_files_present(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|f| dir.join(f).exists())
}

/// Load configuration from the appropriate location.
fn load_config() -> anyhow::Result<Config> {
    let mut config = Config::default();

    if let Some(config_path) = config_dir().map(|d| d.join("config.toml")) {
        if config_path.exists() {
            let contents = std::fs::read_to_string(&config_path)
                .with_context(|| format!("read config from '{}'", config_path.display()))?;
            config = toml::from_str(&contents)
                .with_context(|| format!("parse config from '{}'", config_path.display()))?;
        }
    }

    Ok(config)
}

/// Check that a required external tool is available in PATH.
async fn check_external_tool(name: &str) -> anyhow::Result<()> {
    match tokio::process::Command::new(name)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
    {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!(
                "{name} not found in PATH. Please install ffmpeg: https://ffmpeg.org/download.html"
            )
        }
        Err(e) => Err(e).with_context(|| format!("check for {name}")),
    }
}

/// Figure out where the model files live, downloading them if necessary.
///
/// An explicitly configured directory (--model or model_path in the config) is
/// trusted but verified: we never download into it, and error with pointers if
/// files are missing. Otherwise we use the per-user data dir and download the
/// model on first use.
async fn resolve_model_dir(explicit: Option<PathBuf>, quiet: bool) -> anyhow::Result<PathBuf> {
    if let Some(dir) = explicit {
        for file in MODEL_FILES {
            anyhow::ensure!(
                dir.join(file).exists(),
                "model file '{}' not found in '{}'; download the files {} \
                 from https://huggingface.co/{}",
                file,
                dir.display(),
                MODEL_FILES.join(", "),
                MODEL_REPO,
            );
        }
        return Ok(dir);
    }

    let dir = model_data_dir().context("determine per-user model data directory")?;
    if model_files_present(&dir) {
        return Ok(dir);
    }

    let config_path = config_dir()
        .map(|d| d.join("config.toml").display().to_string())
        .unwrap_or_else(|| "~/.config/sirtea/config.toml".to_string());
    download_model(&dir, quiet).await.with_context(|| {
        format!(
            "download the Parakeet model from https://huggingface.co/{MODEL_REPO}; \
             if you are offline, download the files {} manually into a directory \
             and set model_path in {config_path}",
            MODEL_FILES.join(", "),
        )
    })?;
    Ok(dir)
}

/// Download the model files from HuggingFace into `target`.
///
/// Files are downloaded into a sibling `.tmp` directory that is renamed into
/// place only once every file completed, so an interrupted download can never
/// leave a directory that passes the `model_files_present` check with
/// truncated weights in it.
async fn download_model(target: &Path, quiet: bool) -> anyhow::Result<()> {
    // TODO: pin sha256 digests for the model files instead of trusting
    // HuggingFace + TLS alone.
    let tmp_name = target
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| format!("{n}.tmp"))
        .expect("model dir name is a fixed utf-8 constant");
    let tmp = target.with_file_name(tmp_name);
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp).context("remove stale partial model download")?;
    }
    std::fs::create_dir_all(&tmp).context("create model download directory")?;

    if !quiet {
        eprintln!(
            "downloading Parakeet model from https://huggingface.co/{MODEL_REPO} \
             to {} (first run only)…",
            target.display()
        );
    }

    let client = reqwest::Client::new();
    for file in MODEL_FILES {
        let url = format!("https://huggingface.co/{MODEL_REPO}/resolve/main/{file}");
        let mut resp = client
            .get(&url)
            .send()
            .await
            .and_then(|resp| resp.error_for_status())
            .with_context(|| format!("fetch '{url}'"))?;

        let progress_bar = if quiet {
            None
        } else {
            let pb = match resp.content_length() {
                Some(len) => ProgressBar::new(len).with_style(
                    ProgressStyle::with_template(
                        "{prefix} {bar:30} {bytes}/{total_bytes} ({bytes_per_sec})",
                    )
                    .expect("valid template"),
                ),
                None => ProgressBar::new_spinner(),
            };
            pb.set_prefix(*file);
            Some(pb)
        };

        let out_path = tmp.join(file);
        let mut out = tokio::fs::File::create(&out_path)
            .await
            .with_context(|| format!("create '{}'", out_path.display()))?;
        while let Some(chunk) = resp
            .chunk()
            .await
            .with_context(|| format!("download '{url}'"))?
        {
            out.write_all(&chunk)
                .await
                .with_context(|| format!("write to '{}'", out_path.display()))?;
            if let Some(ref pb) = progress_bar {
                pb.inc(chunk.len() as u64);
            }
        }
        out.flush()
            .await
            .with_context(|| format!("flush '{}'", out_path.display()))?;
        if let Some(pb) = progress_bar {
            pb.finish();
        }
    }

    std::fs::rename(&tmp, target).context("move completed model download into place")?;
    Ok(())
}

/// How the previous segment ended, for progress status messages.
enum PrevSegment {
    /// Ended at a natural boundary after this phrase.
    Phrase(String),
    /// Transcribed to nothing at all.
    Silence,
}

/// A transcribed span of speech with timestamps. Used both for the raw
/// word-level output of Parakeet (one word per `Utterance`) and for the
/// re-grouped cues that become SRT entries (see `build_cues`).
#[derive(Debug)]
struct Utterance {
    /// Start time in seconds
    start: f64,
    /// End time in seconds
    end: f64,
    /// The transcribed text
    text: String,
}

// ============================================================================
// Cue building: re-grouping word timestamps into subtitle-sized cues
// ============================================================================
//
// Parakeet's own `Segment` timestamp granularity splits only at
// sentence-final punctuation, and real (lecture) speech is full of run-on
// sentences, so its cues routinely blow past what a viewer can read
// (measured on a 73-minute lecture: 47% of cues over 84 chars, the worst at
// 456 chars / 29 seconds). We instead transcribe at `Word` granularity and
// re-group words into cues ourselves, aiming for the standard subtitle
// envelope (two lines of `MAX_LINE_CHARS`, at most `MAX_CUE_SECS` each).
//
// One property of Parakeet's output does a lot of work here: token
// timestamps are contiguous — each word ends exactly where the next one
// begins — so silence is absorbed into the *end* of the preceding word. That
// means (a) there are no inter-word gaps to split at, but (b) a word whose
// raw duration far exceeds the time needed to actually say it marks a real
// pause. We use (b) both to prefer split points at pauses and to trim cue
// end times so captions don't linger on screen through silence.

/// Estimated time to speak `text`; a deliberate over-estimate (see the
/// comment on `WORD_SPEECH_PER_CHAR`).
fn est_spoken_secs(text: &str) -> f64 {
    WORD_SPEECH_FLOOR + WORD_SPEECH_PER_CHAR * text.chars().count() as f64
}

/// Does `word` end a sentence, given the word that follows it (if any)?
///
/// Sentence-final punctuation alone is not enough: abbreviations like "e.g."
/// or "Dr." also end with '.'. Requiring the next word to look like a
/// sentence start (capital letter or digit) filters most of those out.
fn ends_sentence(word: &str, next: Option<&str>) -> bool {
    let word = word.trim_end_matches(['"', '\'', ')', '”', '’']);
    if !word.ends_with(['.', '?', '!']) {
        return false;
    }
    next.is_none_or(|next| {
        next.chars()
            .find(|c| c.is_alphanumeric())
            .is_some_and(|c| c.is_uppercase() || c.is_numeric())
    })
}

/// Words that introduce the phrase that follows them: articles,
/// prepositions, conjunctions, pronouns, auxiliaries, and common
/// intensifiers. A line or cue should not end right after one of these
/// (see `FUNCTION_WORD_BREAK_PENALTY`).
fn is_function_word(word: &str) -> bool {
    let word = word.trim_end_matches(|c: char| !c.is_alphanumeric());
    let lower = word.to_lowercase();
    matches!(
        lower.as_str(),
        "the" | "a" | "an" // articles
        | "and" | "or" | "but" | "nor" | "so" | "yet" // conjunctions
        | "to" | "of" | "in" | "on" | "at" | "by" | "for" | "with" | "from"
        | "into" | "onto" | "about" | "over" | "under" | "between" // prepositions
        | "as" | "if" | "that" | "which" | "who" | "whose" | "whom"
        | "what" | "when" | "where" | "how" | "why" | "because" // subordinators
        | "is" | "are" | "was" | "were" | "be" | "been" | "being" | "am"
        | "do" | "does" | "did" | "will" | "would" | "can" | "could"
        | "should" | "shall" | "may" | "might" | "must" // auxiliaries
        | "i" | "we" | "you" | "he" | "she" | "it" | "they"
        | "my" | "our" | "your" | "his" | "her" | "its" | "their" // pronouns
        | "this" | "these" | "those"
        | "not" | "no" | "very" | "really" | "just" | "quite"
        | "some" | "any" | "each" | "every" // determiners/intensifiers
    )
}

/// The cost of ending a cue after `word`, mid-sentence. Sentence-final
/// boundaries never reach this function: `build_cues` splits at sentence
/// ends before the per-sentence segmentation runs.
fn boundary_cost(word: &Utterance) -> f64 {
    // Check `...` before `,`/`:`/`;`: it must not fall through to the
    // single-character cases, and certainly not to BOUNDARY_NONE.
    let punctuation = if word.text.ends_with("...") || word.text.ends_with(',') {
        BOUNDARY_CLAUSE
    } else if word.text.ends_with([':', ';']) {
        BOUNDARY_COLON
    } else if is_function_word(&word.text) {
        BOUNDARY_NONE + FUNCTION_WORD_BREAK_PENALTY
    } else {
        BOUNDARY_NONE
    };
    // Contiguous timestamps absorb silence into the preceding word, so an
    // inflated raw duration reveals a pause (see WORD_SPEECH_PER_CHAR).
    let trailing_silence = (word.end - word.start) - est_spoken_secs(&word.text);
    if trailing_silence >= PAUSE_MIN_SECS {
        punctuation.min(BOUNDARY_PAUSE)
    } else {
        punctuation
    }
}

/// Re-group word-level utterances into subtitle-sized cues.
///
/// Words are first split into sentences (a cue never spans a sentence
/// boundary), and each sentence is then segmented into cues that fit the
/// subtitle envelope by `split_sentence_into_cues`.
fn build_cues(words: &[Utterance]) -> Vec<Utterance> {
    let mut cues = Vec::new();
    let mut sentence_start = 0;
    for i in 0..words.len() {
        let next = words.get(i + 1).map(|w| w.text.as_str());
        // The last word always terminates the final sentence, punctuated or
        // not: hard cuts at segment ends can leave unpunctuated tails.
        if next.is_none() || ends_sentence(&words[i].text, next) {
            split_sentence_into_cues(&words[sentence_start..=i], &mut cues);
            sentence_start = i + 1;
        }
    }
    cues
}

/// Split one sentence's words into cues, appending them to `cues`.
///
/// This is a Knuth-Plass-style minimum-cost segmentation over word
/// boundaries: `best[i]` is the cheapest way to emit `sentence[..i]` as
/// whole cues, built up by considering every feasible last cue
/// `sentence[j..i]`. Compared to greedily bisecting over-long sentences,
/// the global optimum avoids shaving off awkward single-word tail cues, and
/// all tuning lives in the cost constants at the top of the file. Cost is
/// negligible: the inner loop is bounded by how many words fit in a cue.
fn split_sentence_into_cues(sentence: &[Utterance], cues: &mut Vec<Utterance>) {
    let n = sentence.len();
    if n == 0 {
        return;
    }

    // Prefix sums of word lengths so any candidate cue's character count
    // (words plus joining spaces) is O(1).
    let mut chars_before = Vec::with_capacity(n + 1);
    chars_before.push(0usize);
    for w in sentence {
        chars_before.push(chars_before.last().expect("vec starts non-empty") + w.text.chars().count());
    }
    let cue_chars = |j: usize, i: usize| chars_before[i] - chars_before[j] + (i - j - 1);

    // Can sentence[j..i] be laid out as at most two lines of at most
    // MAX_LINE_CHARS? This — not a total-character cap — is the real size
    // limit: an 84-char cue only wraps into two fitting lines if a space
    // falls exactly in the middle, so a cue must be rejected unless *some*
    // word boundary splits it into two fitting lines. (Shrinking a
    // wrappable span keeps it wrappable, so the early-break in the DP loop
    // below remains valid.)
    let fits_two_lines = |j: usize, i: usize| {
        cue_chars(j, i) <= MAX_LINE_CHARS
            || (j + 1..i)
                .any(|k| cue_chars(j, k) <= MAX_LINE_CHARS && cue_chars(k, i) <= MAX_LINE_CHARS)
    };

    let mut best = vec![f64::INFINITY; n + 1];
    let mut back = vec![0usize; n + 1];
    best[0] = 0.0;
    for i in 1..=n {
        // Ending a cue at the sentence end is free — it's a mandatory
        // break — while anywhere else costs by how natural the boundary is.
        let break_cost = if i == n {
            0.0
        } else {
            boundary_cost(&sentence[i - 1])
        };
        // Measure candidate cues as they will be displayed: the final
        // word's trailing absorbed silence gets trimmed at emission below,
        // so it must not count against the duration limit — otherwise a
        // long pause after a short sentence fragment makes every multi-word
        // candidate infeasible and forces a lone-word crumb cue before the
        // pause. Pauses *inside* a candidate still count in full, since an
        // earlier word's inflated end delays every word after it — so a
        // long internal pause still blows the limit and forces a split at
        // the pause, exactly where one belongs.
        let last = &sentence[i - 1];
        let display_end = last
            .end
            .min(last.start + est_spoken_secs(&last.text) + CUE_LINGER_SECS);
        for j in (0..i).rev() {
            let duration = display_end - sentence[j].start;
            if !fits_two_lines(j, i) || duration > MAX_CUE_SECS {
                // A single word (j == i - 1) is exempt from the limits: it
                // cannot be split any further, and letting it through as an
                // over-long cue beats dropping it. This also guarantees
                // best[i] is always reachable.
                if j != i - 1 {
                    // Both metrics only grow as j decreases; we're done.
                    break;
                }
            }
            let short_penalty = SHORT_CUE_PENALTY_PER_SEC * (MIN_CUE_SECS - duration).max(0.0)
                + SHORT_CUE_PENALTY_PER_CHAR
                    * MIN_CUE_CHARS.saturating_sub(cue_chars(j, i)) as f64;
            let cost = best[j] + short_penalty + break_cost;
            if cost < best[i] {
                best[i] = cost;
                back[i] = j;
            }
        }
    }

    // Walk the backpointers to recover the boundaries, then emit in order.
    let mut boundaries = vec![n];
    while *boundaries.last().expect("vec starts non-empty") > 0 {
        boundaries.push(back[*boundaries.last().expect("vec starts non-empty")]);
    }
    boundaries.reverse();
    for pair in boundaries.windows(2) {
        let words = &sentence[pair[0]..pair[1]];
        let text = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let start = words[0].start;
        let last = words.last().expect("cue spans at least one word");
        // Trim the display time so a cue doesn't linger through the silence
        // absorbed into its final word's raw end, but never below the
        // readability floor and never past the raw end (which would overlap
        // the next cue).
        // TODO: measuring RMS energy on the raw samples would trim (and
        // place pause boundaries) exactly instead of by estimate; that
        // belongs together with the split-preference TODO in `main`.
        let end = last
            .end
            .min(last.start + est_spoken_secs(&last.text) + CUE_LINGER_SECS)
            .max(start + MIN_CUE_SECS)
            .min(last.end);
        cues.push(Utterance { start, end, text });
    }
}

/// Wrap cue text into at most two lines of `MAX_LINE_CHARS`, breaking at the
/// space that best balances the lines, with a preference for breaking just
/// after clause punctuation near the middle. Text that fits on one line is
/// returned untouched. We do this ourselves rather than leaving it to the
/// player because SRT has no wrap hinting and players wrap unpredictably
/// (at arbitrary widths, at arbitrary words, or not at all).
fn balance_lines(text: &str) -> String {
    let total = text.chars().count();
    if total <= MAX_LINE_CHARS {
        return text.to_string();
    }
    // Breaking at the space at character position p yields a first line of p
    // chars and a second of total - p - 1. Scan every space and keep the
    // cheapest break; `<` (not `<=`) on the comparison makes ties resolve to
    // the earlier space, i.e. the bottom-heavy split, which is the
    // conventional subtitle shape.
    let mut best: Option<(usize, f64)> = None; // (byte index, cost)
    let mut prev: Option<char> = None;
    let mut word_start = 0; // byte index where the word before the space began
    for (char_pos, (byte_idx, c)) in text.char_indices().enumerate() {
        if c == ' ' {
            let line1 = char_pos;
            let line2 = total - char_pos - 1;
            let mut cost = (line1 as f64 - line2 as f64).abs();
            // A clause boundary beats pure balance when it's within ~10
            // chars of the middle (the discount outweighs up to 20 units of
            // imbalance); conversely, ending a line on a function word
            // severs it from its phrase, so nudge the break elsewhere.
            // TODO: the nudge still loses to balance on ~8% of two-line cues
            // (measured on a 73-min lecture eval), which end line 1 on a
            // function word anyway; raising it is the knob to try, at the
            // cost of more lopsided lines.
            if prev.is_some_and(|p| matches!(p, ',' | ';' | ':' | '.' | '?' | '!')) {
                cost -= 20.0;
            } else if is_function_word(&text[word_start..byte_idx]) {
                cost += 15.0;
            }
            // An over-long line is far worse than any imbalance, but keep it
            // finite: a cue containing a >42-char token should still break
            // at its least-bad space rather than not at all.
            if line1 > MAX_LINE_CHARS || line2 > MAX_LINE_CHARS {
                cost += 1000.0;
            }
            if best.is_none_or(|(_, c)| cost < c) {
                best = Some((byte_idx, cost));
            }
            word_start = byte_idx + 1;
        }
        prev = Some(c);
    }
    match best {
        // No spaces at all: one giant token; nothing sensible to do.
        None => text.to_string(),
        Some((byte_idx, _)) => format!("{}\n{}", &text[..byte_idx], &text[byte_idx + 1..]),
    }
}

/// Pick the cue to end a transcription segment after, from the trailing
/// window of `utterances`. Returns an index into `utterances`.
///
/// We prefer breaking at natural sentence boundaries, as that reduces the
/// likelihood that the next transcription will start at a weird point in a
/// sentence. `...` and `,` endings are also acceptable, but slightly less
/// "good" since we may end up with a capital letter starting the next
/// caption.
fn pick_split_index(utterances: &[Utterance]) -> usize {
    let window = utterances.len().min(20);
    let mut best: Option<(usize, f64)> = None;
    // offset 0 (the very last caption) is deliberately excluded: the
    // model punctuates speech truncated by the hard cut as if it were a
    // complete sentence, so the final caption's "sentence end" is often
    // fabricated, and splitting there is equivalent to not splitting at
    // a boundary at all. (The pre-Parakeet gap-based scoring excluded it
    // implicitly: the last caption's gap was always zero.) The truncated
    // caption gets dropped by the caller and re-transcribed in the next
    // segment.
    for offset in 1..window {
        let i = utterances.len() - offset - 1;
        let text = &utterances[i].text;
        let punctuation_score = if text.ends_with("...") || text.ends_with(',') {
            1.3
        } else if text.ends_with(['.', '?', '!', ':']) {
            2.0
        } else {
            1.0
        };
        // among equally-good boundaries, prefer later ones so that we keep as
        // much as possible of the audio we have already transcribed. the
        // fall-off is gentle enough that a sentence end deep in the window
        // still beats an unpunctuated caption at the very end.
        let recency_score = 1.0 - 0.5 * (offset as f64 / window as f64);
        let score = punctuation_score * recency_score;
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((i, score));
        }
    }
    // A segment with a single caption leaves no choice but the hard cut.
    best.map(|(i, _)| i).unwrap_or(utterances.len() - 1)
}

/// Reinterpret the raw little-endian f32 PCM bytes emitted by ffmpeg's
/// `-f f32le` as audio samples.
fn pcm_f32le_to_samples(bytes: &[u8]) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        bytes.len().is_multiple_of(4),
        "PCM byte stream length {} is not a whole number of f32 samples",
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| {
            f32::from_le_bytes(chunk.try_into().expect("chunks_exact yields 4-byte chunks"))
        })
        .collect())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LocalVideo {
    // NOTE: the order of the fields matter for Ord here
    // We order by length first so shorter videos are processed first
    length: Duration,
    path: PathBuf,
    delay: Duration,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = parse_args().context("parse arguments")?;

    if args.files.is_empty() {
        anyhow::bail!("no video files specified. Run with --help for usage.");
    }

    // Check for required external tools
    check_external_tool("ffmpeg").await?;
    check_external_tool("ffprobe").await?;

    let config = load_config()?;

    // Merge config with CLI args (CLI takes precedence, then config, then defaults)
    let segment_length = args
        .segment_length
        .or(config.segment_length)
        .unwrap_or(DEFAULT_MAX_SEGMENT_LENGTH);
    let model_dir_override = args.model.or(config.model_path);

    // Collect all candidate file paths, expanding directories with WalkDir.
    // Track whether each path was explicitly specified (should error on failure)
    // or discovered via directory walk (should silently skip non-media files).
    let mut candidate_paths: Vec<(PathBuf, bool)> = Vec::new();
    for arg in &args.files {
        anyhow::ensure!(arg.exists(), "path '{}' does not exist", arg.display());

        if arg.is_dir() {
            let walker = WalkDir::new(arg).same_file_system(args.one_file_system);
            for entry in walker {
                let entry = entry.with_context(|| format!("walk directory '{}'", arg.display()))?;
                if entry.file_type().is_file() {
                    candidate_paths.push((entry.into_path(), false));
                }
            }
        } else {
            candidate_paths.push((arg.clone(), true));
        }
    }

    let mut videos = BTreeSet::new();
    for (path, explicit) in candidate_paths {
        // Get extension for format detection
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let src = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if explicit => {
                anyhow::bail!("failed to open '{}': {}", path.display(), e);
            }
            Err(e) => {
                if !args.quiet {
                    eprintln!("warning: skipping '{}': {}", path.display(), e);
                }
                continue;
            }
        };
        let mss = MediaSourceStream::new(Box::new(src), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(ext);
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();
        let probed = match symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)
        {
            Ok(p) => p,
            Err(e) if explicit => {
                anyhow::bail!("unsupported format for '{}': {}", path.display(), e);
            }
            Err(_) => {
                // Not a recognized media format - silently skip when walking directories
                continue;
            }
        };
        let Some(track) = probed
            .format
            .tracks()
            .iter()
            .find(|t| t.codec_params.sample_rate.is_some())
        else {
            if !args.quiet {
                eprintln!(
                    "warning: skipping '{}': no audio track found",
                    path.display()
                );
            }
            continue;
        };
        let (Some(time_base), Some(n_frames)) =
            (track.codec_params.time_base, track.codec_params.n_frames)
        else {
            if !args.quiet {
                eprintln!(
                    "warning: skipping '{}': unable to determine audio duration",
                    path.display()
                );
            }
            continue;
        };
        let length = time_base.calc_time(n_frames);
        let length = Duration::from_secs(length.seconds) + Duration::from_secs_f64(length.frac);
        // TODO: for whatever reason, track.codec_params.start_ts is always 0, so use ffprobe
        let delay = tokio::process::Command::new("ffprobe")
            .arg("-i")
            .arg(&path)
            .arg("-show_entries")
            .arg("stream=start_time")
            .arg("-select_streams")
            .arg("a")
            .arg("-hide_banner")
            .arg("-of")
            .arg("default=noprint_wrappers=1:nokey=1")
            .output()
            .await
            .with_context(|| format!("ffprobe '{}'", path.display()))?;
        let delay = std::str::from_utf8(&delay.stdout)
            .with_context(|| format!("non-utf8 in ffprobe '{}'", path.display()))?;
        let delay: f64 = delay
            .trim()
            .parse()
            .with_context(|| format!("bad delay float in ffprobe '{}': {delay}", path.display()))?;
        let delay = if delay.is_sign_negative() {
            anyhow::ensure!(
                delay.abs() < 0.05,
                "very negative audio delay, {}, in '{}'",
                delay,
                path.display()
            );
            Duration::default()
        } else {
            Duration::from_secs_f64(delay)
        };
        videos.insert(LocalVideo {
            length,
            path,
            delay,
        });
    }

    // Handle dry-run mode: list what would be transcribed and exit
    if args.dry_run {
        for video in &videos {
            println!(
                "{} ({:.1} minutes)",
                video.path.display(),
                video.length.as_secs_f64() / 60.0
            );
        }
        let total_seconds: f64 = videos.iter().map(|v| v.length.as_secs_f64()).sum();
        println!(
            "Dry run: {} video(s), {:.1} minutes total",
            videos.len(),
            total_seconds / 60.0
        );
        println!(
            "Estimated processing time: {:.1} minutes (at ~{ESTIMATED_REALTIME_FACTOR:.0}x realtime)",
            total_seconds / ESTIMATED_REALTIME_FACTOR / 60.0
        );
        let model_present = match &model_dir_override {
            Some(dir) => model_files_present(dir),
            None => model_data_dir()
                .map(|d| model_files_present(&d))
                .unwrap_or(false),
        };
        if !model_present {
            println!(
                "Note: the first real run will download the Parakeet model (~670 MB) \
                 from https://huggingface.co/{MODEL_REPO}"
            );
        }
        return Ok(());
    }

    // Get the model ready before starting on any video so that configuration
    // problems (or a failed download) surface immediately rather than after
    // minutes of audio extraction.
    let model_dir = resolve_model_dir(model_dir_override, args.quiet).await?;

    // Run inference on the GPU via ONNX Runtime's WebGPU execution provider
    // (works on AMD and NVIDIA alike; compiled in via the default `webgpu`
    // cargo feature — prebuilt release binaries disable it and stay on CPU).
    // WebGPU must be selected explicitly, and before the sessions are
    // created below: transcribe-rs's default `Auto` mode never picks it,
    // because WebGPU forces sequential session execution, which upstream
    // won't impose on other backends. This is safe to set unconditionally:
    // builds without the feature, and machines without a working GPU/Vulkan
    // stack, fall back to CPU at session creation.
    set_ort_accelerator(OrtAccelerator::WebGpu);
    let webgpu_compiled_in = OrtAccelerator::available().contains(&OrtAccelerator::WebGpu);

    if !args.quiet {
        eprintln!(
            "loading Parakeet model…{}",
            if webgpu_compiled_in { " (WebGPU)" } else { "" }
        );
    }
    let mut model = ParakeetModel::load(&model_dir, &Quantization::Int8)
        .with_context(|| format!("load Parakeet model from '{}'", model_dir.display()))?;

    // Set up progress tracking
    let multi_progress = if args.quiet {
        None
    } else {
        Some(MultiProgress::new())
    };
    let waiting_style =
        ProgressStyle::with_template("{prefix:.dim} {wide_msg:.dim}").expect("valid template");
    let skipped_style =
        ProgressStyle::with_template("{prefix:.yellow} {wide_msg:.dim}").expect("valid template");
    let active_style =
        ProgressStyle::with_template("{prefix:.bold.green} {spinner:.green} {wide_msg}")
            .expect("valid template")
            .tick_chars("╶─╺━╺─");

    // Create all progress bars up front so every queued video is visible as
    // "waiting" while earlier ones are processed.
    let videos: Vec<(LocalVideo, String, Option<ProgressBar>)> = videos
        .into_iter()
        .map(|video| {
            let video_name = video
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("output")
                .to_string();
            let progress_bar = multi_progress.as_ref().map(|mp| {
                let pb = mp.add(ProgressBar::new_spinner());
                pb.set_style(waiting_style.clone());
                pb.set_prefix(format!("{} ⏸", video_name));
                pb.set_message("waiting…");
                pb
            });
            (video, video_name, progress_bar)
        })
        .collect();

    // Process videos one at a time; sequential keeps the code (and the progress
    // story) simple. Measured on a 16-core machine, inference averages only ~5
    // busy cores: Parakeet TDT's decode loop is inherently serial (one tiny
    // decoder_joint inference per 80ms audio frame, thousands per segment) and
    // bounds the critical path, while the encoder's parallel sections are too
    // short to saturate ONNX Runtime's thread pools. Transcribing videos
    // concurrently would therefore raise throughput, but at ~5GB RAM per
    // in-flight inference; revisit if sequential ever feels too slow.
    // TODO: overlap the *extraction* of the next segment with inference of the
    // current one if profiling ever shows extraction to be a meaningful share.
    // NOTE: transcribe-rs ships its own chunked-transcription wrappers
    // (`transcribe_rs::transcriber::{EnergyAdaptiveChunked, VadChunked}`)
    // that split long audio internally (energy/VAD-based split-point search)
    // and merge the results — conceptually overlapping the segment/split
    // machinery below. Ours picks sentence boundaries from the transcript
    // itself rather than energy dips, which gives arguably better splits, but
    // if this file ever needs to shrink, replacing the segment loop with
    // `EnergyAdaptiveChunked` is worth evaluating.
    for (video, video_name, progress_bar) in videos {
        let verbose = args.verbose;
        let result: anyhow::Result<()> = async {
            let srt = video.path.with_extension("srt");
            if tokio::fs::try_exists(&srt)
                .await
                .context("check for existence")?
            {
                if let Some(ref pb) = progress_bar {
                    pb.set_style(skipped_style.clone());
                    pb.set_prefix(video_name.clone());
                    pb.finish_with_message("skipped (exists)");
                }
                return Ok(());
            }

            // We're not skipping, so switch to active style
            if let Some(ref pb) = progress_bar {
                pb.set_style(active_style.clone());
                pb.enable_steady_tick(Duration::from_millis(120));
            }

            // Split long videos into segments to bound Parakeet's memory use
            // (see the comment on DEFAULT_MAX_SEGMENT_LENGTH). Note that this
            // must be an open-ended loop rather than iterating over a segment
            // count computed up front: every split point slides `start`
            // backwards to a sentence boundary, and over many segments that
            // slippage adds up to extra segments at the end.
            let mut start = Duration::default();
            let mut captions = Vec::new();
            // How the previous segment ended, for status messages
            let mut prev_split_info: Option<PrevSegment> = None;
            let mut segment_index: u64 = 0;
            loop {
                let remaining = video.length.saturating_sub(start);
                let is_last = remaining.as_secs_f64() <= segment_length;
                let duration_limit = if is_last {
                    // Last segment: extract to end of file
                    None
                } else if segment_index == 0 {
                    // The adelay filter below pads the segment with silence at
                    // the front, so consume correspondingly less input to keep
                    // the output segment at the intended length.
                    Some(segment_length - video.delay.as_secs_f64())
                } else {
                    Some(segment_length)
                };

                // Update progress bar with current segment and action. The
                // total is an estimate (except once we're on the last segment)
                // since split-point slippage can add segments.
                if let Some(ref pb) = progress_bar {
                    let total = if is_last {
                        format!("{}", segment_index + 1)
                    } else {
                        let est = segment_index
                            + (remaining.as_secs_f64() / segment_length).ceil().max(1.0) as u64;
                        format!("~{est}")
                    };
                    pb.set_prefix(format!("{} [{}/{}]", video_name, segment_index + 1, total));
                    let msg = if segment_index == 0 && is_last {
                        "extracting audio".to_string()
                    } else {
                        match prev_split_info {
                            Some(PrevSegment::Phrase(ref phrase)) => {
                                format!("extracting audio segment after '{phrase}'")
                            }
                            Some(PrevSegment::Silence) => {
                                "extracting audio segment after a silent stretch".to_string()
                            }
                            None => "extracting initial audio segment".to_string(),
                        }
                    };
                    pb.set_message(msg);
                }

                let mut ffmpeg = tokio::process::Command::new("ffmpeg");

                // NOTE: we don't use -acodec copy because that would be limited to extracting time
                // segments at block boundaries for the audio codec (e.g., blocks in AAC).
                // instead, we need to reencode, which allows extracting exact times since the
                // input is muxed. the model wants 16kHz mono f32 samples, so we have ffmpeg do
                // the resampling and downmixing and emit raw samples; no container needed since
                // we know the exact sample format on both sides of the pipe.
                let start_f64 = start.as_secs_f64();
                let ss = start_f64.to_string();
                ffmpeg
                    .arg("-ss")
                    .arg(ss)
                    .arg("-i")
                    .arg(&video.path)
                    .arg("-vn")
                    .arg("-ac")
                    .arg("1")
                    .arg("-ar")
                    .arg("16000")
                    .arg("-c:a")
                    .arg("pcm_f32le")
                    .arg("-f")
                    .arg("f32le")
                    .arg("-nostdin")
                    .arg("-hide_banner")
                    .arg("-loglevel")
                    .arg("error");

                if !video.delay.is_zero() && segment_index == 0 {
                    // all=1 delays every input channel; without it adelay only
                    // delays the first channel, which the -ac 1 downmix would
                    // then blend with the undelayed remainder.
                    ffmpeg
                        .arg("-af")
                        .arg(format!("adelay={}:all=1", video.delay.as_millis()));
                }

                if let Some(limit) = duration_limit {
                    ffmpeg.arg("-t").arg(limit.to_string());
                }

                let ffmpeg = ffmpeg
                    .arg("-")
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .context("ffmpeg split")?;

                // A segment is at most tens of MB of PCM (16kHz mono f32 is
                // ~3.8MB/min), so just buffer it all up; wait_with_output
                // drains stdout and stderr concurrently, so this can't deadlock
                // on a full pipe.
                let ffmpeg = ffmpeg.wait_with_output().await.context("extract audio")?;
                if !ffmpeg.status.success() {
                    let ffmpeg_err = String::from_utf8_lossy(&ffmpeg.stderr);
                    return Err(anyhow::anyhow!("{ffmpeg_err}")).context("extract audio segment");
                }
                let samples =
                    pcm_f32le_to_samples(&ffmpeg.stdout).context("interpret extracted PCM")?;

                if let Some(ref pb) = progress_bar {
                    let msg = if segment_index == 0 && is_last {
                        "transcribing".to_string()
                    } else {
                        match prev_split_info {
                            Some(PrevSegment::Phrase(ref phrase)) => {
                                format!("transcribing segment after '{phrase}'")
                            }
                            Some(PrevSegment::Silence) => {
                                "transcribing segment after a silent stretch".to_string()
                            }
                            None => "transcribing initial segment".to_string(),
                        }
                    };
                    pb.set_message(msg);
                }

                // Run inference synchronously right here, blocking the async
                // runtime. That's fine — and not worth "fixing": nothing else
                // needs to make progress during inference (the progress bars
                // tick on their own thread).
                let transcription = model
                    .transcribe_with(
                        &samples,
                        &ParakeetParams {
                            language: None,
                            // Word granularity gives per-word timestamps,
                            // which build_cues re-groups into subtitle-sized
                            // cues. (Parakeet's own Segment granularity
                            // splits only at sentence-final punctuation,
                            // which produces cues far too long to read; see
                            // the cue-building section.)
                            timestamp_granularity: Some(TimestampGranularity::Word),
                        },
                    )
                    .context("transcribe audio segment")?;
                let words: Vec<Utterance> = transcription
                    .segments
                    .unwrap_or_default()
                    .into_iter()
                    .map(|s| Utterance {
                        start: f64::from(s.start),
                        end: f64::from(s.end),
                        text: s.text,
                    })
                    .collect();
                let mut utterances = build_cues(&words);

                // A fully-silent stretch (a break in a lecture, credits, etc.) transcribes
                // to zero captions. That's not an error for a single segment — skip past
                // the extracted window and keep going. A video that yields no captions at
                // all is still reported as an error after the loop, since that more likely
                // indicates broken audio than a genuinely silent recording.
                if utterances.is_empty() {
                    if is_last {
                        break;
                    }
                    // With no transcript there is no sentence boundary to split at either,
                    // so resume from the hard cut at the end of the extracted window.
                    let consumed = duration_limit.expect("non-last segments always have a limit");
                    if verbose {
                        eprintln!(
                            "{}: silent segment at {}, skipping to {}",
                            video_name,
                            format_srt_timestamp(start.as_secs_f64()),
                            format_srt_timestamp(start.as_secs_f64() + consumed),
                        );
                    }
                    start += Duration::from_secs_f64(consumed);
                    prev_split_info = Some(PrevSegment::Silence);
                    segment_index += 1;
                    continue;
                }

                // if we're not at the last segment, we need to find a good place to split
                if !is_last {
                    if let Some(ref pb) = progress_bar {
                        pb.set_message("finding split point");
                    }
                    // Here's the trick: we grabbed captions for [start..start + segment_length]
                    // but the end point is usually in the middle of a sentence! So we pick a
                    // recent caption that ends at a natural sentence boundary, drop all captions
                    // following it, and resume transcription from that point rather than from
                    // the hard cut.
                    //
                    // Note that we cannot look for silence gaps between captions here (as this
                    // code did in its cloud-API days): Parakeet's token timestamps are contiguous
                    // by construction, so consecutive *words* never have a gap between them
                    // (build_cues instead infers pauses from inflated word durations).
                    // TODO: we hold the raw samples right here; measuring RMS energy around
                    // candidate boundaries would let us prefer genuinely quiet ones again.
                    let split_idx = pick_split_index(&utterances);
                    // Resume from the *raw* end of the chosen cue's last word, which by
                    // timestamp contiguity is the next cue's start. The chosen cue's own
                    // `end` is display-trimmed (trailing silence removed), so resuming
                    // there would re-transcribe audio we already emitted captions for.
                    // With a single cue there is no next cue; that's the hard-cut
                    // fallback, where the trimmed end only re-covers inferred silence.
                    let slice_at_secs = utterances
                        .get(split_idx + 1)
                        .map(|u| u.start)
                        .unwrap_or(utterances[split_idx].end);
                    let slice_at = start + Duration::from_secs_f64(slice_at_secs);

                    if verbose {
                        eprintln!(
                            "{}: split at {} (after '{}')",
                            video_name,
                            format_srt_timestamp(slice_at.as_secs_f64()),
                            utterances[split_idx].text,
                        );
                    }

                    // Store split info for the next segment's status messages
                    prev_split_info = Some(PrevSegment::Phrase(utterances[split_idx].text.clone()));

                    utterances.truncate(split_idx + 1);
                    start = slice_at;
                }

                // whatever captions are left, adjust their start times for the start offset
                captions.extend(utterances.into_iter().map(|mut u| {
                    // note: this is specifically start_f64, which is not affected by us updating
                    // start at the end of the gap slicing above.
                    u.start += start_f64;
                    u.end += start_f64;
                    u
                }));

                segment_index += 1;
                if is_last {
                    break;
                }
            }

            // Every silent segment was skipped above, but a video with no captions at
            // all is worth a hard error: broken/missing audio is far more likely than
            // someone captioning a genuinely speech-free recording, and writing an
            // empty .srt would permanently mark the file as done.
            anyhow::ensure!(
                !captions.is_empty(),
                "transcription produced no captions for the entire video; is the audio silent?"
            );

            // Write the SRT file. Cue text is kept single-line internally
            // (progress messages quote it); the two-line wrapping happens
            // only here, at serialization, where a '\n' inside the text
            // block simply becomes the cue's second line.
            let mut outfile = tokio::fs::File::create(&srt).await.context("create srt")?;
            for (i, utterance) in captions.into_iter().enumerate() {
                let line = format!(
                    "{}{}\n{} --> {}\n{}\n",
                    if i != 0 { "\n" } else { "" },
                    i + 1,
                    format_srt_timestamp(utterance.start),
                    format_srt_timestamp(utterance.end),
                    balance_lines(&utterance.text),
                );
                outfile
                    .write_all(line.as_bytes())
                    .await
                    .context("write out srt line")?;
            }
            outfile.flush().await.context("flush srt")?;

            // Mark as complete
            if let Some(ref pb) = progress_bar {
                pb.finish_with_message("done");
            }
            Ok(())
        }
        .await;
        result.with_context(|| format!("while transcribing {}", video_name))?;
    }

    // Skip all teardown. ONNX Runtime's WebGPU provider is buggy on shutdown:
    // destroying the Dawn device segfaults when a GPU was in use, and spins
    // forever when device enumeration failed (observed with ONNX Runtime
    // 1.24.2 prebuilts). The spin lives in an atexit-registered C++ static
    // destructor, so `std::process::exit` (which runs atexit handlers) is not
    // enough — terminate with the raw `_exit` syscall instead. Every output
    // is already written and flushed by this point, so skipping destructors
    // and atexit handlers is safe; this is the standard workaround for this
    // class of GPU-runtime teardown bug.
    //
    // Upstream status (as of 2026-07): the exit crash is fixed by
    // https://github.com/microsoft/onnxruntime/pull/27569 (merged 2026-03,
    // after the 1.24.x branch). When transcribe-rs/ort move to an ONNX
    // Runtime that contains that fix, remove this workaround (after
    // confirming the no-Vulkan atexit spin is gone too — that one appears
    // unreported upstream as of 2026-07).
    //
    // The error path (an `Err` return from `main`) still runs normal teardown
    // and may crash or hang *after* the error message has been printed.
    // That's cosmetic — the user has their error by then — and accepted to
    // keep error propagation ordinary.
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    unsafe { libc::_exit(0) }
}

fn format_srt_timestamp(total_seconds: f64) -> String {
    // Negative timestamps should never occur - they would indicate a bug
    // in our timestamp calculation logic
    let mut remaining_secs = total_seconds as i64;
    debug_assert!(remaining_secs >= 0, "negative timestamp: {total_seconds}");
    remaining_secs = remaining_secs.max(0); // Saturate to 0 in release builds rather than panic
    let h = remaining_secs / 3600;
    remaining_secs -= h * 3600;
    let m = remaining_secs / 60;
    remaining_secs -= m * 60;
    let s = remaining_secs;
    let millis_str = total_seconds.fract();
    let millis_str = format!("{:.3}", millis_str);
    let millis_str = if let Some(millis) = millis_str.strip_prefix("0.") {
        format!(",{millis}")
    } else if millis_str == "1.000" {
        // 0.9995 would be truncated to 1.000 at {:.3}
        String::from(",999")
    } else if millis_str == "0" {
        // integral number of seconds
        String::from(",000")
    } else {
        unreachable!(
            "bad fractional second: {} -> {millis_str}",
            total_seconds.fract()
        )
    };
    format!("{h:02}:{m:02}:{s:02}{millis_str}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ones() {
        assert!(dbg!(format_srt_timestamp(3661.3)).starts_with("01:01:01,3"));
    }

    #[test]
    fn zero_fract() {
        assert_eq!(format_srt_timestamp(3661.0), "01:01:01,000");
    }

    #[test]
    fn pcm_roundtrip() {
        let samples = [0.0f32, 1.0, -1.0, 0.5, f32::MIN_POSITIVE];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        assert_eq!(pcm_f32le_to_samples(&bytes).unwrap(), samples);
    }

    #[test]
    fn pcm_empty() {
        assert_eq!(pcm_f32le_to_samples(&[]).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn pcm_truncated() {
        let err = pcm_f32le_to_samples(&[0, 0, 0]).unwrap_err();
        assert!(err.to_string().contains("not a whole number"));
    }

    // ------------------------------------------------------------------
    // Cue building
    // ------------------------------------------------------------------

    /// Construct a word-level `Utterance`.
    fn w(start: f64, end: f64, text: &str) -> Utterance {
        Utterance {
            start,
            end,
            text: text.to_string(),
        }
    }

    /// Build contiguous word timestamps from text, one word every `pace`
    /// seconds (mirroring Parakeet's contiguous token timestamps).
    fn words_at_pace(text: &str, start: f64, pace: f64) -> Vec<Utterance> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, word)| w(start + i as f64 * pace, start + (i + 1) as f64 * pace, word))
            .collect()
    }

    /// Every cue must wrap into at most two lines of at most
    /// `MAX_LINE_CHARS` (the escape hatch for a lone over-long word is
    /// exercised by a dedicated test, not by these fixtures).
    fn assert_cue_fits_envelope(cue: &Utterance) {
        let wrapped = balance_lines(&cue.text);
        let lines: Vec<&str> = wrapped.split('\n').collect();
        assert!(lines.len() <= 2, "too many lines: {wrapped:?}");
        for line in lines {
            assert!(
                line.chars().count() <= MAX_LINE_CHARS,
                "line too long: {line:?} in {wrapped:?}"
            );
        }
        assert!(cue.end - cue.start <= MAX_CUE_SECS, "too slow: {cue:?}");
    }

    /// Invariants that must hold for any build_cues output: no dropped,
    /// duplicated, or reordered words; sane, non-overlapping timestamps.
    fn assert_cue_invariants(words: &[Utterance], cues: &[Utterance]) {
        let original = words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let rebuilt = cues
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(original, rebuilt, "cue text must round-trip the words");
        for pair in cues.windows(2) {
            assert!(
                pair[0].start < pair[1].start,
                "cue starts must strictly increase: {pair:?}"
            );
            assert!(
                pair[0].end <= pair[1].start,
                "cues must not overlap: {pair:?}"
            );
        }
        for cue in cues {
            assert!(cue.start < cue.end, "cue must have positive duration: {cue:?}");
        }
        if let (Some(first_word), Some(first_cue)) = (words.first(), cues.first()) {
            assert_eq!(first_word.start, first_cue.start);
        }
    }

    #[test]
    fn cues_empty_input() {
        assert!(build_cues(&[]).is_empty());
    }

    #[test]
    fn cues_short_sentence_roundtrips() {
        let words = words_at_pace("I think we might as well get started.", 0.5, 0.3);
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, "I think we might as well get started.");
        assert_eq!(cues[0].start, 0.5);
    }

    #[test]
    fn cues_run_on_sentence_splits_at_commas() {
        // ~200 chars of comma-separated clauses; every cue must fit the
        // limits and every internal break must land on a comma.
        let text = "as great as computers are in terms of being a tool, \
                    they also have a tendency to do only exactly what we told them, \
                    which is not necessarily what we intended for them to do, \
                    and hence the need for debugging and profiling today.";
        let words = words_at_pace(text, 0.0, 0.28);
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert!(cues.len() > 1, "a 200+ char sentence must split");
        for cue in &cues {
            assert_cue_fits_envelope(cue);
        }
        for cue in &cues[..cues.len() - 1] {
            assert!(cue.text.ends_with(','), "break not at a comma: {cue:?}");
        }
    }

    #[test]
    fn cues_unpunctuated_stream_forced_breaks() {
        let text = vec!["word"; 100].join(" ");
        let words = words_at_pace(&text, 0.0, 0.3);
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        for cue in &cues {
            assert!(!cue.text.is_empty());
            assert_cue_fits_envelope(cue);
        }
    }

    #[test]
    fn cues_never_merge_sentences() {
        // Both sentences would fit in one cue, but sentences stay separate.
        let words = words_at_pace("Cool. So today we will talk about debugging.", 0.0, 0.3);
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "Cool.");
        assert_eq!(cues[1].text, "So today we will talk about debugging.");
    }

    #[test]
    fn cues_no_orphan_tail() {
        // 18 unpunctuated 4-char words: too much for one cue, and every
        // possible split costs the same BOUNDARY_NONE — without a
        // char-based short-cue penalty, the tie used to resolve to a
        // maximal first cue plus a lone-word crumb ("doing."-style tail).
        let text = vec!["word"; 18].join(" ");
        let words = words_at_pace(&text, 0.0, 0.3);
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert!(cues.len() > 1);
        for cue in &cues {
            assert!(
                cue.text.chars().count() >= MIN_CUE_CHARS,
                "crumb cue: {cue:?}"
            );
        }
    }

    #[test]
    fn cues_trailing_pause_does_not_force_crumb() {
        // A sentence long enough to need one split, whose final word
        // absorbs a long silence (inflated raw end). The duration limit
        // must be judged on the *displayed* (trimmed) cue, not the raw
        // end — otherwise every multi-word tail candidate looks over-long
        // and the sentence sheds a lone-word "doing."-style crumb cue
        // before the pause.
        let text = "stick a bunch of print statements that print out information \
                    that helps you think through what the program is doing.";
        let mut words = words_at_pace(text, 0.0, 0.3);
        words.last_mut().expect("test text is non-empty").end += 6.0;
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert!(cues.len() > 1, "sentence over two lines' worth must split");
        for cue in &cues {
            assert!(
                cue.text.chars().count() >= MIN_CUE_CHARS,
                "crumb cue: {cue:?}"
            );
        }
    }

    #[test]
    fn cues_split_when_no_balanced_break_exists() {
        // Three 26-char words: 80 chars total — under two lines' worth of
        // characters, but no word boundary splits it into two lines of ≤42
        // (any split leaves one side at 53). The cue builder must therefore
        // split it into two cues rather than emit an unwrappable one.
        let long = "a".repeat(26);
        let text = format!("{long} {long} {long}");
        let words = words_at_pace(&text, 0.0, 0.8);
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert_eq!(cues.len(), 2);
        for cue in &cues {
            assert_cue_fits_envelope(cue);
        }
    }

    #[test]
    fn cues_lone_overlong_word_passes_through() {
        let word = "a".repeat(100);
        let words = vec![w(0.0, 2.0, &word)];
        let cues = build_cues(&words);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text, word);
    }

    #[test]
    fn cues_split_at_inferred_pause_over_comma() {
        // A sentence too long for one cue, containing both a comma boundary
        // and (later) a word with heavily inflated duration (an absorbed
        // pause). The pause must win as the split point, and the cue ending
        // there must have its display time trimmed rather than lingering
        // through the silence.
        let mut words = words_at_pace(
            "so this is where we get things like logging instead, \
             and logging is really just a more principled use of print statements",
            0.0,
            0.28,
        );
        // Inflate "instead," (word index 9): 4s of absorbed silence. All
        // later words shift by 4s to stay contiguous.
        for word in &mut words[10..] {
            word.start += 4.0;
            word.end += 4.0;
        }
        words[9].end += 4.0;
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert!(
            cues[0].text.ends_with("instead,"),
            "should split at the pause: {cues:?}"
        );
        // Display end trimmed: near the estimated end of speech, far from
        // the raw end (which extends 4s into the silence).
        let last_word = &words[9];
        let est_end = last_word.start + est_spoken_secs(&last_word.text) + CUE_LINGER_SECS;
        assert!(
            (cues[0].end - est_end).abs() < 1e-9,
            "cue end {} should be trimmed to ~{est_end}",
            cues[0].end
        );
        assert!(cues[0].end < last_word.end - 2.0);
    }

    #[test]
    fn cues_avoid_breaking_after_function_word() {
        // 18 unpunctuated words force one mid-clause split, and the
        // no-penalty tie region is seeded with "the": the break must slide
        // off the function word onto a content word.
        let mut word_list = vec!["word"; 18];
        word_list[12] = "them";
        word_list[11] = "the";
        let text = word_list.join(" ");
        let words = words_at_pace(&text, 0.0, 0.3);
        let cues = build_cues(&words);
        assert_cue_invariants(&words, &cues);
        assert!(cues.len() > 1);
        for cue in &cues[..cues.len() - 1] {
            let last_word = cue.text.split(' ').next_back().expect("cue not empty");
            assert_ne!(last_word, "the", "cue ends on a function word: {cue:?}");
        }
    }

    #[test]
    fn function_word_detection() {
        assert!(is_function_word("the"));
        assert!(is_function_word("The"));
        assert!(is_function_word("we"));
        // Trailing punctuation is stripped before matching (though callers
        // treat punctuated boundaries as clause breaks first).
        assert!(is_function_word("to"));
        assert!(!is_function_word("computer"));
        assert!(!is_function_word("debugging"));
    }

    #[test]
    fn balance_avoids_breaking_after_function_word() {
        // The most balanced split lands right after "the"; the break must
        // move to a neighboring space instead.
        let text = "they also have a tendency to do only the exact thing we told";
        let wrapped = balance_lines(text);
        let first_line = wrapped.split('\n').next().expect("has a line");
        assert!(
            !first_line.ends_with(" the") && !first_line.ends_with(" to"),
            "line ends on a function word: {wrapped:?}"
        );
    }

    #[test]
    fn sentence_end_detection() {
        // Ordinary sentence ends, with and without a following word.
        assert!(ends_sentence("started.", Some("If")));
        assert!(ends_sentence("started.", None));
        assert!(ends_sentence("right?", Some("Yes")));
        assert!(ends_sentence("statements.\"", Some("And")));
        // Abbreviations followed by a lowercase word are not sentence ends.
        assert!(!ends_sentence("e.g.", Some("apples")));
        assert!(!ends_sentence("Dr.", Some("who")));
        // No sentence-final punctuation at all.
        assert!(!ends_sentence("started", Some("If")));
        assert!(!ends_sentence("started,", Some("If")));
    }

    // ------------------------------------------------------------------
    // Line balancing
    // ------------------------------------------------------------------

    #[test]
    fn balance_short_text_untouched() {
        assert_eq!(balance_lines("I think we might get started."), "I think we might get started.");
    }

    #[test]
    fn balance_long_text_two_lines() {
        let text = "the biggest problem with print debugging is you start from scratch each time";
        let wrapped = balance_lines(text);
        let lines: Vec<&str> = wrapped.split('\n').collect();
        assert_eq!(lines.len(), 2);
        for line in &lines {
            assert!(line.chars().count() <= MAX_LINE_CHARS, "line too long: {line:?}");
            assert!(!line.starts_with(' ') && !line.ends_with(' '));
        }
        assert_eq!(wrapped.replace('\n', " "), text, "wrapping must not alter words");
    }

    #[test]
    fn balance_prefers_clause_punctuation_near_center() {
        // The comma sits a little off-center; a pure balance split would
        // break elsewhere, but the clause boundary should win.
        let text = "they also have a tendency to do, only exactly what we told them to";
        let wrapped = balance_lines(text);
        assert_eq!(
            wrapped,
            "they also have a tendency to do,\nonly exactly what we told them to"
        );
    }

    // ------------------------------------------------------------------
    // Segment split-point selection
    //
    // Parakeet fabricates sentence-final punctuation at hard cuts, so the
    // very last caption must never be picked as the split point. That
    // failure mode is pinned here.
    // ------------------------------------------------------------------

    #[test]
    fn split_never_picks_last_caption() {
        // The last caption ends with '.', but only because the hard cut made
        // the model fabricate it; the comma-ended caption before it must win.
        let utterances = vec![
            w(0.0, 5.0, "we talked about logging levels,"),
            w(5.0, 10.0, "and then the hard cut fabricated this."),
        ];
        assert_eq!(pick_split_index(&utterances), 0);
    }

    #[test]
    fn split_prefers_sentence_end_over_recent_comma() {
        let utterances = vec![
            w(0.0, 5.0, "filler so the window has depth,"),
            w(5.0, 10.0, "and this sentence ends properly."),
            w(10.0, 15.0, "while this one trails off with a comma,"),
            w(15.0, 20.0, "hard-cut tail caption."),
        ];
        assert_eq!(pick_split_index(&utterances), 1);
    }

    #[test]
    fn split_prefers_recent_among_equal_punctuation() {
        let utterances = vec![
            w(0.0, 5.0, "an early complete sentence."),
            w(5.0, 10.0, "a later complete sentence."),
            w(10.0, 15.0, "hard-cut tail caption."),
        ];
        assert_eq!(pick_split_index(&utterances), 1);
    }

    #[test]
    fn split_single_caption_hard_cut_fallback() {
        let utterances = vec![w(0.0, 5.0, "only one caption here")];
        assert_eq!(pick_split_index(&utterances), 0);
    }
}
