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

/// A transcribed utterance: the unit that becomes one SRT cue.
#[derive(Debug)]
struct Utterance {
    /// Start time in seconds
    start: f64,
    /// End time in seconds
    end: f64,
    /// The transcribed text
    text: String,
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
                            // Segment granularity groups words into
                            // sentence-ish chunks at punctuation, which is
                            // exactly the size we want an SRT cue to be.
                            timestamp_granularity: Some(TimestampGranularity::Segment),
                        },
                    )
                    .context("transcribe audio segment")?;
                let mut utterances: Vec<Utterance> = transcription
                    .segments
                    .unwrap_or_default()
                    .into_iter()
                    .map(|s| Utterance {
                        start: f64::from(s.start),
                        end: f64::from(s.end),
                        text: s.text,
                    })
                    .collect();

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
                    // by construction — each token ends exactly where the next one begins — so
                    // the gap between consecutive captions is always zero.
                    // TODO: we hold the raw samples right here; measuring RMS energy around
                    // candidate boundaries would let us prefer genuinely quiet ones again.
                    let window = utterances.len().min(20);
                    let mut best: Option<(usize, f64)> = None;
                    // offset 0 (the very last caption) is deliberately excluded: the
                    // model punctuates speech truncated by the hard cut as if it were a
                    // complete sentence, so the final caption's "sentence end" is often
                    // fabricated, and splitting there is equivalent to not splitting at
                    // a boundary at all. (The pre-Parakeet gap-based scoring excluded it
                    // implicitly: the last caption's gap was always zero.) The truncated
                    // caption gets dropped below and re-transcribed in the next segment.
                    for offset in 1..window {
                        let i = utterances.len() - offset - 1;
                        let text = &utterances[i].text;
                        // prefer breaking at natural sentence boundaries as it reduces the
                        // likelihood that the next transcription will start at a weird point in a
                        // sentence. the ... and , endings are also acceptable, but slightly less
                        // "good" since we may end up with a captial letter starting the next
                        // caption.
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
                    let (split_idx, _) = best.unwrap_or((utterances.len() - 1, 0.0));
                    let slice_at = start + Duration::from_secs_f64(utterances[split_idx].end);

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

            // Write the SRT file
            let mut outfile = tokio::fs::File::create(&srt).await.context("create srt")?;
            for (i, utterance) in captions.into_iter().enumerate() {
                let line = format!(
                    "{}{}\n{} --> {}\n{}\n",
                    if i != 0 { "\n" } else { "" },
                    i + 1,
                    format_srt_timestamp(utterance.start),
                    format_srt_timestamp(utterance.end),
                    utterance.text,
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
}
