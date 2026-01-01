use anyhow::Context;
use chrono::{DateTime, Utc};
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

const MAX_SEGMENT_LENGTH: f64 = 3300.0;
const CONCURRENT_TRANSCRIBES: usize = 3;
const MAX_PRICE: f64 = 10.0;
const PRICE_PER_SECOND: f64 = 0.000193;

#[derive(Deserialize)]
struct Config {
    gladia_api_key: String,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
struct GladiaTranscribeResponse {
    prediction: Vec<Prediction>,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
struct Prediction {
    // confidence: f64,
    // language: String,
    time_begin: f64,
    time_end: f64,
    transcription: String,
    // NOTE: ignoring words
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LocalVideo {
    // NOTE: the order of the fields matter for Ord here
    length: Duration,
    recorded: DateTime<Utc>,
    path: PathBuf,
    delay: Duration,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = tokio::fs::read_to_string("config.toml")
        .await
        .context("read config")?;
    let config: Config = toml::from_str(&config).context("parse config")?;

    let client = reqwest::Client::new();

    println!("==> finding video files locally");
    let mut local_videos = BTreeSet::new();
    for arg in std::env::args().skip(1) {
        let path = std::path::PathBuf::from(arg);
        anyhow::ensure!(path.exists(), "file '{}' does not exist", path.display());
        let Some(name) = path.file_name() else {
            continue;
        };
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some((name, ext)) = name.split_once('.') else {
            continue;
        };
        let Ok(dt) = name.parse::<DateTime<Utc>>() else {
            continue;
        };
        let src = std::fs::File::open(&path).context("failed to open media")?;
        let mss = MediaSourceStream::new(Box::new(src), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(ext);
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .context("unsupported format")?;
        let Some(track) = probed
            .format
            .tracks()
            .iter()
            .find(|t| t.codec_params.channel_layout.is_some())
        else {
            continue;
        };
        let (Some(time_base), Some(n_frames)) =
            (track.codec_params.time_base, track.codec_params.n_frames)
        else {
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
        local_videos.insert(LocalVideo {
            recorded: dt,
            length,
            path,
            delay,
        });
    }
    println!(" -> found {} videos", local_videos.len());

    println!("==> transcribing videos with Gladia");
    let s = Arc::new(tokio::sync::Semaphore::new(CONCURRENT_TRANSCRIBES));
    let cost_in_cents = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut tasks = tokio::task::JoinSet::new();
    for video in local_videos {
        let s = Arc::clone(&s);
        let date = video.recorded;
        let cost_in_cents = Arc::clone(&cost_in_cents);
        let client = client.clone();
        let gladia_api_key = config.gladia_api_key.clone();
        let fut = async move {
            let _permit = s.acquire().await;

            let srt = format!("{}.srt", video.recorded.date_naive());
            if tokio::fs::try_exists(&srt)
                .await
                .context("check for existence")?
            {
                println!(
                    " -> not transcribing '{}' (.srt already exists)",
                    video.path.display()
                );
                return Ok(());
            }

            let this_in_cents =
                (100.0 * PRICE_PER_SECOND * video.length.as_secs_f64()).round() as u64;
            let total_cost =
                cost_in_cents.fetch_add(this_in_cents, Ordering::AcqRel) as f64 / 100.0;
            if total_cost > MAX_PRICE {
                // decrement again in case a shorter video comes along
                cost_in_cents.fetch_sub(this_in_cents, Ordering::AcqRel);
                println!(
                    " -> not transcribing '{}' (would exceed cost limit)",
                    video.path.display()
                );
                return Ok(());
            }

            println!(
                " -> transcribing {} ({})",
                video.recorded.date_naive(),
                video.path.display(),
            );

            // https://gladia-stt.nolt.io/23
            // https://gladia-stt.nolt.io/24
            let nsplits = (video.length.as_secs_f64() / MAX_SEGMENT_LENGTH).ceil() as u64;
            let mut start = Duration::default();
            let dur = video.length.as_secs() / nsplits;
            let mut captions = Vec::new();
            for split in 0..nsplits {
                let t = if split < nsplits - 1 {
                    Some(if split == 0 {
                        dur - video.delay.as_secs()
                    } else {
                        dur
                    })
                } else {
                    None
                };

                let mut ffmpeg = tokio::process::Command::new("ffmpeg");

                // NOTE: we don't use -acodec copy because that would be limited to extracting time
                // segments at block boundaries for the audio codec (e.g., blocks in AAC).
                // instead, we need to reencode, which allows extracting exact times since the
                // input is muxed. it's tempting to reencode to flac, which is lossless, but then
                // we quickly run into the 500MB file size limit. so, we go with opus, which is
                // modern, compact, and high-quality. we avoid aac because some aac encoders are
                // bad.
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

                if let Some(t) = t {
                    ffmpeg.arg("-t").arg(t.to_string());
                }

                // eprintln!("{ffmpeg:?}");

                // if split >= 3 {
                //     println!(" .. {} | exit early", video.youtube.id);
                //     break;
                // }

                println!(
                    " .. {} | {} -> {}",
                    video.recorded.date_naive(),
                    seconds_to_timestamp(start.as_secs_f64()),
                    seconds_to_timestamp(
                        t.map(|t| start + Duration::from_secs(t))
                            .unwrap_or(video.length)
                            .as_secs_f64()
                    )
                );

                let mut ffmpeg = ffmpeg
                    .arg("-")
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .context("ffmpeg split")?;

                let req = client
                    .post("https://api.gladia.io/audio/text/audio-transcription/")
                    .header("x-gladia-key", &gladia_api_key)
                    .header("Accept", "application/json")
                    .multipart(
                        Form::new()
                            .part(
                                "audio",
                                Part::stream(reqwest::Body::wrap_stream(FramedRead::new(
                                    ffmpeg.stdout.take().expect("set to piped"),
                                    BytesCodec::new(),
                                )))
                                .mime_str("audio/opous")
                                .context("valid mime string")?,
                            )
                            .text("toggle_diarization", "false"),
                    )
                    .send();

                let (res, ffmpeg) = tokio::join!(req, ffmpeg.wait_with_output());
                let ffmpeg = ffmpeg.context("extract audio")?;
                let res = match (res, ffmpeg.status.success()) {
                    (Ok(res), true) if res.status().is_success() => res,
                    (Ok(res), _) => {
                        // we got an error response, so print all we can
                        // if res is a 2XX but ffmpeg failed, we'll also land here
                        // but that's probably appropriate
                        let ffmpeg =
                            std::str::from_utf8(&ffmpeg.stderr).expect("ffmpeg stderr is utf-8");
                        let code = res.status();
                        let gladia = res
                            .text()
                            .await
                            .unwrap_or_else(|_| String::from("<failed to read>"));
                        return Err(anyhow::anyhow!(gladia))
                            .with_context(|| format!("HTTP status: {code}"))
                            .with_context(|| format!("ffmpeg output:\n{ffmpeg}"))
                            .context("run transcription");
                    }
                    (Err(e), _) => {
                        // the request couldn't even be issued. probably an I/O error.
                        let ffmpeg =
                            std::str::from_utf8(&ffmpeg.stderr).expect("ffmpeg stderr is utf-8");
                        Err(e)
                            .with_context(|| format!("ffmpeg output:\n{ffmpeg}"))
                            .context("issue transcribe request")?
                    }
                };

                let res: serde_json::Value = res.json().await.context("parse json")?;
                let mut res: GladiaTranscribeResponse = serde_json::from_value(res).unwrap();

                // if we're not at the last segment, we need to find a good place to split
                if split < nsplits - 1 {
                    // here comes the trick
                    // we grabbed captions for [start..start + dur]
                    // _but_ the end point may be in the middle of a caption!
                    // so, we find the time of last "gap"
                    // drop all the captions following that
                    // and resume captioning from that point rather than start
                    let mut next_starts = res
                        .prediction
                        .last()
                        .expect("always at least one caption")
                        .time_end;
                    let mut best_gap: Option<(usize, f64, f64)> = None;
                    for i in 0..res.prediction.len().min(20) {
                        let i = res.prediction.len() - i - 1;
                        let gap = next_starts - res.prediction[i].time_end;
                        // prefer breaking at natural sentence boundaries as it reduces the
                        // likelihood that the next transcription will start at a weird point in a
                        // sentence. the ... and , endings are also acceptable, but slightly less
                        // "good" since we may end up with a captial letter starting the next
                        // caption.
                        let score = if res.prediction[i].transcription.ends_with("...")
                            || res.prediction[i].transcription.ends_with(',')
                        {
                            1.3 * gap
                        } else if res.prediction[i]
                            .transcription
                            .ends_with(|c| c == '.' || c == '?' || c == '!' || c == ':')
                        {
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
                        next_starts = res.prediction[i].time_begin;
                    }
                    let gap = best_gap.expect("always a gap");
                    let slice_at = start
                        + Duration::from_secs_f64(res.prediction[gap.0].time_end + gap.1 / 2.0);
                    println!(
                        " .. {} | slicing at {} in {:?} gap after: {}",
                        video.recorded.date_naive(),
                        seconds_to_timestamp(slice_at.as_secs_f64()),
                        Duration::from_secs_f64(gap.1),
                        res.prediction[gap.0].transcription
                    );
                    res.prediction.truncate(gap.0 + 1);
                    start = slice_at;
                }

                // whatever captions are left, adjust their start times for the start offset
                captions.extend(res.prediction.into_iter().map(|mut p| {
                    // note: this is specifically start_f64, which is not affected by us updating
                    // start at the end of the gap slicing above.
                    p.time_begin += start_f64;
                    p.time_end += start_f64;
                    p
                }));
            }

            println!(" .. {} | writing .srt", video.recorded.date_naive());
            let mut outfile = tokio::fs::File::create(&srt).await.context("create srt")?;
            for (i, segment) in captions.into_iter().enumerate() {
                let line = format!(
                    "{}{}\n{} --> {}\n{}\n",
                    if i != 0 { "\n" } else { "" },
                    i + 1,
                    seconds_to_timestamp(segment.time_begin),
                    seconds_to_timestamp(segment.time_end),
                    segment.transcription,
                );
                outfile
                    .write_all(line.as_bytes())
                    .await
                    .context("write out srt line")?;
            }
            outfile.flush().await.context("flush srt")?;

            println!(" .. {} | done", video.recorded.date_naive());
            Ok(())
        };
        tasks.spawn(async move {
            fut.await
                .with_context(|| format!("while transcribing {}", date.date_naive()))
        });
    }

    while !tasks.is_empty() {
        let v = tasks.join_next().await.expect("!is_empty");
        let _ = v.context("join failed")?.context("job failed")?;
    }
    eprintln!("==> all transcription completed");

    Ok(())
}

fn seconds_to_timestamp(fracs: f64) -> String {
    let mut is = fracs as i64;
    assert!(is >= 0);
    let h = is / 3600;
    is -= h * 3600;
    let m = is / 60;
    is -= m * 60;
    let s = is;
    let frac = fracs.fract();
    let frac = format!("{:.3}", frac);
    let frac = if let Some(frac) = frac.strip_prefix("0.") {
        format!(",{frac}")
    } else if frac == "1.000" {
        // 0.9995 would be truncated to 1.000 at {:.3}
        String::from(",999")
    } else if frac == "0" {
        // integral number of seconds
        String::from(",000")
    } else {
        unreachable!("bad fractional second: {} -> {frac}", fracs.fract())
    };
    format!("{h:02}:{m:02}:{s:02}{frac}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ones() {
        assert!(dbg!(seconds_to_timestamp(3661.3)).starts_with("01:01:01,3"));
    }

    #[test]
    fn zero_fract() {
        assert_eq!(seconds_to_timestamp(3661.0), "01:01:01");
    }
}
