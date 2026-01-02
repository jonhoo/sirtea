# caption

Generate SRT subtitle files from video using the [Gladia](https://www.gladia.io/) speech-to-text API.

**This tool uses a paid service.** Gladia charges ~$0.50-0.61/hour depending on your plan, with a [free tier](https://gladia.io/pricing) offering 10 hours/month. Use `--dry-run` to estimate costs before transcribing.

## Why this tool?

Gladia (like most transcription APIs) has a [135-minute limit](https://docs.gladia.io/chapters/limits-and-specifications/supported-formats) per request. This tool handles arbitrarily long videos by:

1. Splitting audio into segments (~90 minutes by default)
2. Finding natural sentence boundaries for clean splits
3. Stitching captions back together with correct timestamps

It also allows transcribing multiple videos concurrently, while ensuring that your total Gladia use cost does not exceed a set threshold.

For the full story, see [the blog post](https://thesquareplanet.com/blog/ai-captioning/).

## Installation

### Pre-built binaries

Download from the [releases page](https://github.com/jonhoo/sirtea/releases).

### From source

```bash
cargo install sirtea
```

### Requirements

- [ffmpeg and ffprobe](https://ffmpeg.org/download.html) in your PATH
- A [Gladia API key](https://app.gladia.io/)

## Usage

```bash
sirtea video.mkv                   # Transcribe a single video
sirtea lecture1.mp4 lecture2.mp4   # Transcribe multiple videos
sirtea --dry-run *.mkv             # Estimate cost without transcribing
sirtea -q video.mp4                # Quiet mode (errors only)
```

Output SRT files are created alongside the input videos (e.g., `video.mkv` → `video.srt`). Videos with existing `.srt` files are skipped.

## Configuration

### API Key

Set your [Gladia API key](https://docs.gladia.io/chapters/introduction/getting-started) via environment variable or config file:

```bash
export GLADIA_API_KEY="your-key-here"
```

Or create a config file:
- Linux: `~/.config/sirtea/config.toml`
- macOS: `~/Library/Application Support/sirtea/config.toml`
- Windows: `%APPDATA%\sirtea\config.toml`

```toml
gladia_api_key = "your-key-here"
```

See `config.example.toml` for other options you can set in your configuration file.

### Options

These can be set in the config file or overridden via CLI flags:

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `max_cost` | `--max-cost` | 10.0 | Stop when estimated cost exceeds this (dollars) |
| `parallel` | `--parallel` | 3 | Concurrent transcriptions ([free plan limit](https://docs.gladia.io/chapters/limits-and-specifications/concurrency); paid plans allow 25) |
| `segment_length` | `--segment-length` | 5400 | Max segment length in seconds (1h30m is the default, though Gladia allows [up to 2h15m](https://docs.gladia.io/chapters/limits-and-specifications/supported-formats)) |

## CLI Reference

```
sirtea [OPTIONS] <PATH>...
```

Run `sirtea --help` for full usage information.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
