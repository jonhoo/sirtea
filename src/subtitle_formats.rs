use crate::{balance_lines, Utterance};
use anyhow::{anyhow, Context};

/// Subtitle format enum to specify the desired output format for subtitles.
#[derive(PartialEq, Eq)]
pub enum SubtitleFormat {
    /// SRT (SubRip Subtitle) format
    Srt,
    /// VTT (Web Video Text Tracks) format
    Vtt,
}

impl std::str::FromStr for SubtitleFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            _ => Err(anyhow!("unsupported subtitle format: {s}")),
        }
    }
}

impl SubtitleFormat {
    /// Get the file extension associated with the subtitle format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }

    /// Format a timestamp in seconds into the SRT format string (HH:MM:SS,mmm).
    pub fn format_timestamp(&self, total_seconds: f64) -> String {
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
        let millis_sep = match self {
            Self::Srt => ',',
            Self::Vtt => '.',
        };
        let millis_str = total_seconds.fract();
        let millis_str = format!("{:.3}", millis_str);
        let millis_str = if let Some(millis) = millis_str.strip_prefix("0.") {
            format!("{millis_sep}{millis}")
        } else if millis_str == "1.000" {
            // 0.9995 would be truncated to 1.000 at {:.3}
            format!("{millis_sep}999")
        } else if millis_str == "0" {
            // integral number of seconds
            format!("{millis_sep}000")
        } else {
            unreachable!(
                "bad fractional second: {} -> {millis_str}",
                total_seconds.fract()
            )
        };
        format!("{h:02}:{m:02}:{s:02}{millis_str}")
    }

    /// Format a single caption block for the given utterance and cue number.
    ///
    /// This function is used internally by `to_file` to generate the string representation
    /// of a caption block. It takes an [`Utterance`] and a cue, which is anything that implements
    /// [`std::fmt::Display`], and returns a formatted String according to the subtitle format.
    ///
    /// Cue text is kept single-line internally; the two-line wrapping happens
    /// only here, at serialization, where a '\n' inside the text block simply becomes
    /// the cue's second line.
    fn format_caption_block<C>(&self, utterance: &Utterance, cue: C) -> String
    where
        C: std::fmt::Display,
    {
        match self {
            Self::Srt => format!(
                "{cue}\n{} --> {}\n{}\n\n",
                self.format_timestamp(utterance.start),
                self.format_timestamp(utterance.end),
                balance_lines(&utterance.text),
            ),
            Self::Vtt => format!(
                // Technically VTT cues are strings to allow extra settings, but we don't support that yet.
                // Fortunately, they can be omitted entirely, so that's what we do here.
                "{} --> {}\n{}\n\n",
                self.format_timestamp(utterance.start),
                self.format_timestamp(utterance.end),
                balance_lines(&utterance.text),
            ),
        }
    }

    /// Write the file format
    ///
    /// An [`Utterance`] is a single caption block, and the cue number is automatically generated
    /// based on the order of the utterances in the slice.
    ///
    /// The `writer` parameter is any type that implements [`std::io::Write`],
    /// in case we'll want to write to something other than a file in the future.
    pub fn write_out(
        &self,
        captions: &[Utterance],
        mut writer: impl std::io::Write,
    ) -> anyhow::Result<()> {
        match self {
            SubtitleFormat::Srt => {}
            SubtitleFormat::Vtt => writer
                .write_all(b"WEBVTT\n\n")
                .context("write out vtt header")?,
        }
        for (i, utterance) in captions.iter().enumerate() {
            let line = self.format_caption_block(utterance, i + 1);
            writer
                .write_all(line.as_bytes())
                .context("write out caption block")?;
        }
        writer.flush().context("flush subtitle file")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ones() {
        assert!(dbg!(SubtitleFormat::Srt.format_timestamp(3661.3)).starts_with("01:01:01,3"));
        assert!(dbg!(SubtitleFormat::Vtt.format_timestamp(3661.3)).starts_with("01:01:01.3"));
    }

    #[test]
    fn zero_fract() {
        assert_eq!(SubtitleFormat::Srt.format_timestamp(3661.0), "01:01:01,000");
        assert_eq!(SubtitleFormat::Vtt.format_timestamp(3661.0), "01:01:01.000");
    }

    fn make_utterances() -> Vec<Utterance> {
        vec![
            Utterance {
                start: 1.0,
                end: 2.0,
                text: "Hello, world!".to_string(),
            },
            Utterance {
                start: 3.0,
                end: 4.0,
                text: "hello, again...".to_string(),
            },
        ]
    }

    #[test]
    fn srt() {
        let mut out = Vec::new();
        SubtitleFormat::Srt
            .write_out(&make_utterances(), &mut out)
            .unwrap();
        let out = String::from_utf8(out).unwrap();
        let expected = "1
00:00:01,000 --> 00:00:02,000
Hello, world!

2
00:00:03,000 --> 00:00:04,000
hello, again...

";
        assert_eq!(out, expected);
    }

    #[test]
    fn vtt() {
        let mut out = Vec::new();
        SubtitleFormat::Vtt
            .write_out(&make_utterances(), &mut out)
            .unwrap();
        let out = String::from_utf8(out).unwrap();
        let expected = "WEBVTT

00:00:01.000 --> 00:00:02.000
Hello, world!

00:00:03.000 --> 00:00:04.000
hello, again...

";
        assert_eq!(out, expected);
    }
}
