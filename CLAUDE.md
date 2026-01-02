# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`sirtea` is a CLI tool that transcribes video files to SRT subtitle files using the Gladia speech-to-text API (v2). Accepts any video filename; outputs `<basename>.srt` alongside the input.

For background on why this tool exists, see [the blog post](https://thesquareplanet.com/blog/ai-captioning/).

## Build and Test Commands

```bash
cargo build              # Development build
cargo build --release    # Release build
cargo test               # Run tests
cargo run -- <path>...   # Run with video files or directories
```

## Configuration

Set `GLADIA_API_KEY` environment variable, or create a config file at the XDG config path:
- Linux: `~/.config/sirtea/config.toml`
- macOS: `~/Library/Application Support/sirtea/config.toml`
- Windows: `%APPDATA%\sirtea\config.toml`

See `config.example.toml` for available options (`max_cost`, `parallel`, `segment_length`).

**Note for Claude:** If `sirtea` fails with "No Gladia API key found", ask the user to configure the key in their environment or config file — do not attempt to set it yourself.

## Architecture

Single-file async Rust application (`src/main.rs`) that:

1. **Discovers files** - Accepts video files or directories; recurses directories with `walkdir`
2. **Probes media** - Uses `symphonia` to read audio track metadata (duration, sample rate)
3. **Extracts audio** - Uses ffmpeg to re-encode to Opus in OGG container, streaming directly to Gladia
4. **Handles long videos** - Splits videos exceeding `DEFAULT_MAX_SEGMENT_LENGTH` (90 minutes) into chunks, finding natural sentence boundaries (gaps after punctuation) for clean splits
5. **Transcribes via Gladia v2** - Uploads audio, initiates transcription, polls for results (up to `DEFAULT_CONCURRENT_TRANSCRIBES` parallel requests)
6. **Cost control** - Tracks estimated cost and skips videos that would exceed `DEFAULT_MAX_COST`
7. **Outputs SRT** - Writes captions with timestamps; skips if `.srt` already exists

### Key Constants

- `DEFAULT_MAX_SEGMENT_LENGTH`: 5400 seconds (1h30m) — stays under Gladia's 135-minute limit with margin
- `DEFAULT_CONCURRENT_TRANSCRIBES`: 3 — matches Gladia's free tier concurrency limit
- `DEFAULT_MAX_COST`: $10 — safety limit to prevent runaway spending
- `PRICE_PER_SECOND`: $0.0001694 — used for cost estimation

### CLI Flags

- `--dry-run`: Estimate cost without transcribing
- `--max-cost`, `--parallel`, `--segment-length`: Override config/defaults
- `-q`/`--quiet`: Errors only (no progress bars)
- `-v`/`--verbose`: Print split point details when segmenting
- `-x`/`--one-file-system`: Don't cross filesystem boundaries when recursing

## External Dependencies

Requires `ffmpeg` and `ffprobe` in PATH for audio extraction and delay detection.
