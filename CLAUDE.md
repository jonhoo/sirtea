# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A CLI tool that transcribes video files to SRT subtitle files using the Gladia speech-to-text API. Videos must have filenames in ISO 8601 UTC datetime format (e.g., `2024-01-15T14:30:00Z.mkv`).

## Build and Test Commands

```bash
cargo build              # Development build
cargo build --release    # Release build (with debug symbols)
cargo test               # Run tests
cargo run -- <video>...  # Run with video files
```

## Configuration

Copy `config.example.toml` to `config.toml` and add your Gladia API key.

## Architecture

Single-file async Rust application (`src/main.rs`) that:

1. **Parses video files** - Expects filenames as UTC datetimes, uses symphonia to read audio track metadata
2. **Extracts audio** - Uses ffmpeg/ffprobe to extract audio, re-encodes to Opus in OGG container
3. **Handles long videos** - Splits videos exceeding ~55 minutes (`MAX_SEGMENT_LENGTH`) into chunks, finding natural sentence boundaries for splits
4. **Transcribes via Gladia** - Streams audio directly to the Gladia API (up to `CONCURRENT_TRANSCRIBES` parallel requests)
5. **Cost control** - Tracks estimated cost and stops when `MAX_PRICE` limit would be exceeded
6. **Outputs SRT** - Writes captions with timestamps to `<date>.srt` files; skips if .srt already exists

## External Dependencies

Requires `ffmpeg` and `ffprobe` in PATH for audio extraction and delay detection.
