# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`sirtea` is a CLI tool that transcribes video files to SRT subtitle files
using NVIDIA's Parakeet speech-to-text model, run fully locally via ONNX
Runtime (through the `transcribe-rs` crate). Accepts any video filename;
outputs `<basename>.srt` alongside the input.

For background on why this tool exists, see [the blog post](https://thesquareplanet.com/blog/ai-captioning/)
(written when the tool still used the Gladia cloud API; the segmentation
approach it describes is unchanged).

## Build and Test Commands

```bash
cargo build              # Development build
cargo build --release    # Release build
cargo test               # Run tests
cargo run -- <path>...   # Run with video files or directories
```

## Configuration

Configuration is optional. Config file at the XDG config path:
- Linux: `~/.config/sirtea/config.toml`
- macOS: `~/Library/Application Support/sirtea/config.toml`
- Windows: `%APPDATA%\sirtea\config.toml`

See `config.example.toml` for available options (`model_path`,
`segment_length`).

The Parakeet model files (~670 MB) are auto-downloaded on first run from
HuggingFace (`istupakov/parakeet-tdt-0.6b-v3-onnx`) into the per-user data
dir (Linux: `~/.local/share/sirtea/models/`), unless `model_path`/`--model`
points at an existing model directory.

## Architecture

Single-file async Rust application (`src/main.rs`) that:

1. **Discovers files** - Accepts video files or directories; recurses directories with `walkdir`
2. **Probes media** - Uses `symphonia` to read audio track metadata (duration, sample rate)
3. **Ensures the model** - Downloads the int8 Parakeet ONNX model on first run (atomic: downloads to a `.tmp` dir, renames into place)
4. **Extracts audio** - Uses ffmpeg to decode to raw 16 kHz mono f32 PCM, buffered in memory per segment
5. **Handles long videos** - Splits videos exceeding `DEFAULT_MAX_SEGMENT_LENGTH` into chunks (the Parakeet ONNX export accepts at most ~200s of audio per inference call), preferring sentence-final punctuation for clean splits (Parakeet's token timestamps are contiguous, so there are no silence gaps to detect)
6. **Transcribes locally** - Runs Parakeet inference via `transcribe-rs` with `TimestampGranularity::Word`, one video at a time; uses the GPU automatically via ONNX Runtime's WebGPU execution provider (selected explicitly in `main` — transcribe-rs's `Auto` mode never picks WebGPU), falling back to CPU when no GPU/Vulkan stack is available
7. **Builds cues** - `build_cues` re-groups word timestamps into subtitle-sized cues (at most two 42-char lines, ≤7s): sentences never merge, over-long sentences split via a Knuth-Plass-style DP preferring clause punctuation and inferred pauses (contiguous timestamps absorb silence into the preceding word, so inflated word durations reveal pauses), and cue display times are trimmed so captions don't linger through silence
8. **Outputs SRT** - Writes captions with timestamps, wrapping cue text into at most two balanced lines (`balance_lines`); skips if `.srt` already exists

### Key Constants

- `DEFAULT_MAX_SEGMENT_LENGTH`: 195 seconds (3m15s) — just under the ~200s per-inference limit baked into the ONNX export's positional-embedding table
- `ESTIMATED_REALTIME_FACTOR`: 6 — measured CPU throughput, used only for `--dry-run` estimates
- `MODEL_REPO` / `MODEL_FILES`: the pinned HuggingFace model repo and the exact int8 files downloaded from it
- `MAX_LINE_CHARS` / `MAX_CUE_SECS`: the subtitle envelope (two 42-char lines, ≤7s per cue); the cue-shaping constants block in `main.rs` documents the full cost model (boundary costs, short-cue penalties, pause inference)

### CLI Flags

- `--dry-run`: List what would be transcribed (durations + estimated processing time)
- `--model`, `--segment-length`: Override config/defaults
- `-q`/`--quiet`: Errors only (no progress bars)
- `-v`/`--verbose`: Print split point details when segmenting
- `-x`/`--one-file-system`: Don't cross filesystem boundaries when recursing

## External Dependencies

Requires `ffmpeg` and `ffprobe` in PATH for audio extraction and delay
detection. ONNX Runtime is linked in at build time by the `ort` crate
(its build script downloads prebuilt binaries).
