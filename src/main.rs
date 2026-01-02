// TODO: Add --srt / --vtt format selection (currently only SRT is supported)
// TODO: Support alternative transcription backends (currently only Gladia)

use anyhow::Context;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tokio::io::AsyncWriteExt;
use tokio_util::codec::{BytesCodec, FramedRead};
use walkdir::WalkDir;

const DEFAULT_MAX_SEGMENT_LENGTH: f64 = 5400.0; // 1h30m
const DEFAULT_CONCURRENT_TRANSCRIBES: usize = 3;
const DEFAULT_MAX_COST: f64 = 10.0;
const PRICE_PER_SECOND: f64 = 0.0001694;

/// Command-line arguments.
struct Args {
    files: Vec<PathBuf>,
    /// None means "not explicitly set on command line"
    max_cost: Option<f64>,
    parallel: Option<usize>,
    segment_length: Option<f64>,
    dry_run: bool,
    quiet: bool,
    one_file_system: bool,
}

fn parse_args() -> Result<Args, lexopt::Error> {
    use lexopt::prelude::*;

    let mut files = Vec::new();
    let mut max_cost = None;
    let mut parallel = None;
    let mut segment_length = None;
    let mut dry_run = false;
    let mut quiet = false;
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
            Long("max-cost") => {
                max_cost = Some(parser.value()?.parse()?);
            }
            Long("parallel") => {
                parallel = Some(parser.value()?.parse()?);
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
        max_cost,
        parallel,
        segment_length,
        dry_run,
        quiet,
        one_file_system,
    })
}

fn print_help() {
    println!(
        "\
{name} {version}
Generate SRT subtitle files from video using the Gladia speech-to-text API.

USAGE:
    {name} [OPTIONS] <PATH>...

ARGS:
    <PATH>...    Video files or directories to transcribe

OPTIONS:
    -h, --help                  Print help information
    -V, --version               Print version information
        --max-cost <DOLLARS>    Maximum cost before stopping (default: {max_cost})
        --parallel <N>          Concurrent transcriptions (default: {parallel})
                                See: https://docs.gladia.io/chapters/limits-and-specifications/concurrency
        --segment-length <SEC>  Max segment length in seconds (default: {segment_length})
                                See: https://docs.gladia.io/chapters/limits-and-specifications/supported-formats#gladia-api-current-limitations
        --dry-run               Estimate cost without transcribing
    -q, --quiet                 Minimal output (errors only)
    -x, --one-file-system       Don't cross filesystem boundaries when recursing directories

CONFIGURATION:
    Set GLADIA_API_KEY environment variable or create a config file at
    $XDG_CONFIG_HOME/sirtea/config.toml (usually ~/.config/sirtea/config.toml).

REQUIREMENTS:
    - ffmpeg and ffprobe in PATH (https://ffmpeg.org/download.html)
    - Gladia API key (https://docs.gladia.io/chapters/introduction/getting-started)
",
        name = env!("CARGO_PKG_NAME"),
        version = env!("CARGO_PKG_VERSION"),
        max_cost = DEFAULT_MAX_COST,
        parallel = DEFAULT_CONCURRENT_TRANSCRIBES,
        segment_length = DEFAULT_MAX_SEGMENT_LENGTH,
    );
}

#[derive(Deserialize, Default)]
struct Config {
    gladia_api_key: Option<String>,
    // Optional settings that can be overridden by CLI flags
    max_cost: Option<f64>,
    parallel: Option<usize>,
    segment_length: Option<f64>,
}

fn config_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "sirtea").map(|dirs| dirs.config_dir().to_path_buf())
}

/// Load configuration from the appropriate location.
/// Priority: env var (for API key) > config file
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

    // Environment variable takes precedence for API key
    if let Ok(key) = std::env::var("GLADIA_API_KEY") {
        config.gladia_api_key = Some(key);
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

/// Get the API key, returning a helpful error if not configured.
fn get_api_key(config: &Config) -> anyhow::Result<String> {
    config.gladia_api_key.clone().ok_or_else(|| {
        let config_path = config_dir()
            .map(|d| d.join("config.toml").display().to_string())
            .unwrap_or_else(|| "~/.config/sirtea/config.toml".to_string());

        anyhow::anyhow!(
            "No Gladia API key found. Set GLADIA_API_KEY environment variable \
             or add gladia_api_key to {}", config_path
        )
    })
}

// Gladia API v2 response types
// See: https://docs.gladia.io/api-reference/v2/

/// Response from POST /v2/upload
#[derive(Deserialize, Debug)]
struct UploadResponse {
    audio_url: String,
}

/// Response from POST /v2/pre-recorded
#[derive(Deserialize, Debug)]
struct TranscriptionInitResponse {
    id: String,
    // result_url is also returned but we construct it ourselves
}

/// Response from GET /v2/pre-recorded/{id}
#[derive(Deserialize, Debug)]
struct TranscriptionStatusResponse {
    status: String,
    #[serde(default)]
    result: Option<TranscriptionResult>,
    #[serde(default)]
    error: Option<TranscriptionError>,
}

#[derive(Deserialize, Debug)]
struct TranscriptionError {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize, Debug)]
struct TranscriptionResult {
    transcription: Transcription,
}

#[derive(Deserialize, Debug)]
struct Transcription {
    utterances: Vec<Utterance>,
}

#[derive(Deserialize, Debug)]
struct Utterance {
    /// Start time in seconds (v2 API returns seconds, not milliseconds)
    start: f64,
    /// End time in seconds
    end: f64,
    /// The transcribed text
    text: String,
    // NOTE: The API also returns `language`, `channel`, `confidence`, and per-word timing in
    // `words`, but we use only utterance-level timing for SRT output.
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
    let gladia_api_key = get_api_key(&config)?;

    // Merge config with CLI args (CLI takes precedence, then config, then defaults)
    let max_cost = args
        .max_cost
        .or(config.max_cost)
        .unwrap_or(DEFAULT_MAX_COST);
    let parallel = args.parallel
        .or(config.parallel)
        .unwrap_or(DEFAULT_CONCURRENT_TRANSCRIBES);
    let segment_length = args.segment_length
        .or(config.segment_length)
        .unwrap_or(DEFAULT_MAX_SEGMENT_LENGTH);

    let client = reqwest::Client::new();

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

    // Handle dry-run mode: estimate cost and exit
    if args.dry_run {
        let total_seconds: f64 = videos.iter().map(|v| v.length.as_secs_f64()).sum();
        let estimated_cost = total_seconds * PRICE_PER_SECOND;
        println!(
            "Dry run: {} video(s), {:.1} minutes total",
            videos.len(),
            total_seconds / 60.0
        );
        println!("Estimated cost: ${:.2}", estimated_cost);
        return Ok(());
    }

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

    let semaphore = Arc::new(tokio::sync::Semaphore::new(parallel));
    let cost_in_cents = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut tasks = tokio::task::JoinSet::new();
    for video in videos {
        let semaphore = Arc::clone(&semaphore);
        let video_name = video
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
            .to_string();
        let cost_in_cents = Arc::clone(&cost_in_cents);
        let client = client.clone();
        let gladia_api_key = gladia_api_key.clone();

        // Create progress bar for this video (initially paused until semaphore acquired)
        let progress_bar = multi_progress.as_ref().map(|mp| {
            let pb = mp.add(ProgressBar::new_spinner());
            pb.set_style(waiting_style.clone());
            pb.set_prefix(format!("{} ⏸", video_name));
            pb.set_message("waiting…");
            pb
        });

        // Clone for use inside the async block (video_name is also used in error context)
        let video_name_inner = video_name.clone();
        let skipped_style = skipped_style.clone();
        let active_style = active_style.clone();
        let fut = async move {
            let _permit = semaphore.acquire().await.expect("semaphore is not closed");

            let srt = video.path.with_extension("srt");
            if tokio::fs::try_exists(&srt)
                .await
                .context("check for existence")?
            {
                if let Some(ref pb) = progress_bar {
                    pb.set_style(skipped_style.clone());
                    pb.set_prefix(video_name_inner.clone());
                    pb.finish_with_message("skipped (exists)");
                }
                return Ok(());
            }

            let video_cost_in_cents =
                (100.0 * PRICE_PER_SECOND * video.length.as_secs_f64()).round() as u64;
            let accumulated_cost =
                cost_in_cents.fetch_add(video_cost_in_cents, Ordering::AcqRel) as f64 / 100.0;
            if accumulated_cost > max_cost {
                // decrement again in case a shorter video comes along
                cost_in_cents.fetch_sub(video_cost_in_cents, Ordering::AcqRel);
                if let Some(ref pb) = progress_bar {
                    pb.set_style(skipped_style);
                    pb.set_prefix(video_name_inner.clone());
                    pb.finish_with_message("skipped (cost limit)");
                }
                return Ok(());
            }

            // We're not skipping, so switch to active style
            if let Some(ref pb) = progress_bar {
                pb.set_style(active_style);
                pb.enable_steady_tick(Duration::from_millis(120));
            }

            // Split long videos into segments to stay under Gladia's duration limit.
            // See: https://docs.gladia.io/chapters/limits-and-specifications/supported-formats#gladia-api-current-limitations
            let segment_count = (video.length.as_secs_f64() / segment_length).ceil() as u64;

            let mut start = Duration::default();
            let segment_duration = video.length.as_secs() / segment_count;
            let mut captions = Vec::new();
            // Track info about the previous split point for status messages
            let mut prev_split_info: Option<(f64, String)> = None;
            for split in 0..segment_count {
                let duration_limit = if split < segment_count - 1 {
                    Some(if split == 0 {
                        segment_duration - video.delay.as_secs()
                    } else {
                        segment_duration
                    })
                } else {
                    // Last segment: extract to end of file
                    None
                };

                // Update progress bar with current segment and action
                if let Some(ref pb) = progress_bar {
                    pb.set_prefix(format!(
                        "{} [{}/{}]",
                        video_name_inner,
                        split + 1,
                        segment_count
                    ));
                    let msg = if segment_count == 1 {
                        "uploading audio to Gladia".to_string()
                    } else if let Some((gap_secs, ref phrase)) = prev_split_info {
                        format!(
                            "uploading audio segment following {gap_secs:.1}s gap after '{phrase}'"
                        )
                    } else {
                        "uploading initial audio segment".to_string()
                    };
                    pb.set_message(msg);
                }

                let mut ffmpeg = tokio::process::Command::new("ffmpeg");

                // NOTE: we don't use -acodec copy because that would be limited to extracting time
                // segments at block boundaries for the audio codec (e.g., blocks in AAC).
                // instead, we need to reencode, which allows extracting exact times since the
                // input is muxed. it's tempting to reencode to flac, which is lossless, but then
                // we quickly run into the 1000MB file size limit
                // (https://docs.gladia.io/chapters/limits-and-specifications/supported-formats#gladia-api-current-limitations).
                // so, we go with opus, which is modern, compact, and high-quality. we avoid aac
                // because some aac encoders are bad.
                let start_f64 = start.as_secs_f64();
                let ss = start_f64.to_string();
                ffmpeg
                    .arg("-ss")
                    .arg(ss)
                    .arg("-i")
                    .arg(&video.path)
                    .arg("-vn")
                    .arg("-c:a")
                    .arg("libopus")
                    .arg("-f")
                    .arg("ogg")
                    .arg("-b:a")
                    .arg("192k")
                    .arg("-nostdin")
                    .arg("-hide_banner")
                    .arg("-loglevel")
                    .arg("error");

                if !video.delay.is_zero() && split == 0 {
                    ffmpeg
                        .arg("-af")
                        .arg(format!("adelay={}", video.delay.as_millis()));
                }

                if let Some(limit) = duration_limit {
                    ffmpeg.arg("-t").arg(limit.to_string());
                }

                let mut ffmpeg = ffmpeg
                    .arg("-")
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .context("ffmpeg split")?;

                // Step 1: Upload audio to Gladia
                // See: https://docs.gladia.io/api-reference/v2/upload/audio-file
                let upload_req = client
                    .post("https://api.gladia.io/v2/upload")
                    .header("x-gladia-key", &gladia_api_key)
                    .multipart(
                        Form::new().part(
                            "audio",
                            Part::stream(reqwest::Body::wrap_stream(FramedRead::new(
                                ffmpeg.stdout.take().expect("set to piped"),
                                BytesCodec::new(),
                            )))
                            .mime_str("audio/opus")
                            .context("valid mime string")?
                            .file_name("audio.ogg"),
                        ),
                    )
                    .send();

                let (upload_res, ffmpeg) = tokio::join!(upload_req, ffmpeg.wait_with_output());
                let ffmpeg = ffmpeg.context("extract audio")?;
                let upload_res = match (upload_res, ffmpeg.status.success()) {
                    (Ok(res), true) if res.status().is_success() => res,
                    (Ok(res), _) => {
                        let ffmpeg_err = String::from_utf8_lossy(&ffmpeg.stderr);
                        let code = res.status();
                        let gladia = res
                            .text()
                            .await
                            .unwrap_or_else(|_| String::from("<failed to read>"));
                        return Err(anyhow::anyhow!(gladia))
                            .with_context(|| format!("HTTP status: {code}"))
                            .with_context(|| format!("ffmpeg output:\n{ffmpeg_err}"))
                            .context("upload audio to Gladia");
                    }
                    (Err(e), _) => {
                        let ffmpeg_err = String::from_utf8_lossy(&ffmpeg.stderr);
                        Err(e)
                            .with_context(|| format!("ffmpeg output:\n{ffmpeg_err}"))
                            .context("upload audio to Gladia")?
                    }
                };

                let upload: UploadResponse = upload_res
                    .json()
                    .await
                    .context("parse upload response")?;

                // Step 2: Initiate transcription
                // See: https://docs.gladia.io/api-reference/v2/pre-recorded/init
                let init_res = client
                    .post("https://api.gladia.io/v2/pre-recorded")
                    .header("x-gladia-key", &gladia_api_key)
                    .header("Content-Type", "application/json")
                    .json(&serde_json::json!({
                        "audio_url": upload.audio_url,
                        "diarization": false
                    }))
                    .send()
                    .await
                    .context("initiate transcription")?;

                if !init_res.status().is_success() {
                    let code = init_res.status();
                    let body = init_res
                        .text()
                        .await
                        .unwrap_or_else(|_| String::from("<failed to read>"));
                    return Err(anyhow::anyhow!(body))
                        .with_context(|| format!("HTTP status: {code}"))
                        .context("initiate transcription");
                }

                let init: TranscriptionInitResponse = init_res
                    .json()
                    .await
                    .context("parse transcription init response")?;

                if let Some(ref pb) = progress_bar {
                    let msg = if segment_count == 1 {
                        "transcribing".to_string()
                    } else if let Some((gap_secs, ref phrase)) = prev_split_info {
                        format!(
                            "transcribing segment following {gap_secs:.1}s gap after '{phrase}'"
                        )
                    } else {
                        "transcribing initial segment".to_string()
                    };
                    pb.set_message(msg);
                }

                // Step 3: Poll for results
                // See: https://docs.gladia.io/api-reference/v2/pre-recorded/get
                let result_url = format!("https://api.gladia.io/v2/pre-recorded/{}", init.id);
                let mut utterances = loop {
                    let status_res = client
                        .get(&result_url)
                        .header("x-gladia-key", &gladia_api_key)
                        .send()
                        .await
                        .context("poll transcription status")?;

                    if !status_res.status().is_success() {
                        let code = status_res.status();
                        let body = status_res
                            .text()
                            .await
                            .unwrap_or_else(|_| String::from("<failed to read>"));
                        return Err(anyhow::anyhow!(body))
                            .with_context(|| format!("HTTP status: {code}"))
                            .context("poll transcription status");
                    }

                    let status: TranscriptionStatusResponse = status_res
                        .json()
                        .await
                        .context("parse transcription status")?;

                    match status.status.as_str() {
                        "done" => {
                            let result = status
                                .result
                                .context("transcription done but no result")?;
                            break result.transcription.utterances;
                        }
                        "error" => {
                            let err = status.error.unwrap_or(TranscriptionError {
                                code: None,
                                message: None,
                            });
                            let msg = err.message.unwrap_or_else(|| "unknown error".to_string());
                            let code = err.code.unwrap_or_else(|| "UNKNOWN".to_string());
                            anyhow::bail!("Gladia transcription failed: {} ({})", msg, code);
                        }
                        "queued" | "processing" => {
                            // Wait before polling again
                            tokio::time::sleep(Duration::from_millis(500)).await;
                        }
                        other => {
                            anyhow::bail!("unexpected transcription status: {}", other);
                        }
                    }
                };

                // Gladia should always return at least one caption for non-silent audio
                if utterances.is_empty() {
                    anyhow::bail!(
                        "Gladia returned no captions for segment starting at {}. \
                         This may indicate silent audio or an API issue.",
                        format_srt_timestamp(start.as_secs_f64())
                    );
                }

                // if we're not at the last segment, we need to find a good place to split
                if split < segment_count - 1 {
                    if let Some(ref pb) = progress_bar {
                        pb.set_message("finding split point");
                    }
                    // Here's the trick: we grabbed captions for [start..start + segment_duration]
                    // but the end point may be in the middle of a caption! So we find the time of
                    // last "gap", drop all captions following that, and resume captioning from
                    // that point rather than from start. This finds natural sentence boundaries.
                    let mut next_starts = utterances
                        .last()
                        .expect("checked non-empty above")
                        .end;
                    let mut best_gap: Option<(usize, f64, f64)> = None;
                    for i in 0..utterances.len().min(20) {
                        let i = utterances.len() - i - 1;
                        let gap = next_starts - utterances[i].end;
                        // prefer breaking at natural sentence boundaries as it reduces the
                        // likelihood that the next transcription will start at a weird point in a
                        // sentence. the ... and , endings are also acceptable, but slightly less
                        // "good" since we may end up with a captial letter starting the next
                        // caption.
                        let score = if utterances[i].text.ends_with("...")
                            || utterances[i].text.ends_with(',')
                        {
                            1.3 * gap
                        } else if utterances[i].text.ends_with(['.', '?', '!', ':']) {
                            2.0 * gap
                        } else {
                            gap
                        };
                        best_gap = Some(if let Some(prev) = best_gap.take() {
                            if score > prev.2 {
                                (i, gap, score)
                            } else {
                                prev
                            }
                        } else {
                            (i, gap, score)
                        });
                        next_starts = utterances[i].start;
                    }
                    let gap = best_gap.expect("always a gap");
                    let slice_at =
                        start + Duration::from_secs_f64(utterances[gap.0].end + gap.1 / 2.0);

                    // Store split info for the next segment's status messages
                    prev_split_info = Some((gap.1, utterances[gap.0].text.clone()));

                    utterances.truncate(gap.0 + 1);
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
            }

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
        };
        tasks.spawn(async move {
            fut.await
                .with_context(|| format!("while transcribing {}", video_name))
        });
    }

    while !tasks.is_empty() {
        let v = tasks.join_next().await.expect("!is_empty");
        v.context("join failed")?.context("job failed")?;
    }

    Ok(())
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
}
