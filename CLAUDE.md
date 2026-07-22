# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`sirtea` is a CLI tool that transcribes video files to SRT subtitle files
using NVIDIA's Parakeet speech-to-text model, run fully locally via
transcribe.cpp/ggml (through the `transcribe-cpp` crate). Accepts any video
filename; outputs `<basename>.srt` alongside the input.

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

The Parakeet model (a single ~740 MB GGUF file) is auto-downloaded on first
run from HuggingFace (`handy-computer/parakeet-tdt-0.6b-v3-gguf`) into the
per-user data dir (Linux: `~/.local/share/sirtea/models/`), unless
`model_path`/`--model` points at an existing `.gguf` file (or a directory
containing the default one).

## Architecture

Single-file Rust application (`src/main.rs`) that:

1. **Discovers files** - Accepts video files or directories; recurses directories with `walkdir`
2. **Probes media** - Uses `symphonia` to read audio track metadata (duration, sample rate)
3. **Ensures the model** - Downloads the Q8_0 Parakeet GGUF model on first run (atomic: downloads to a `.tmp` dir, renames into place)
4. **Extracts audio** - Uses ffmpeg to decode to raw 16 kHz mono f32 PCM, buffered in memory per segment
5. **Handles long videos** - Splits videos exceeding the segment length into chunks (default: the model's per-inference-call audio limit from `Session::limits()`), preferring sentence-final punctuation for clean splits (Parakeet's token timestamps are contiguous, so there are no silence gaps to detect in the *transcript*). Independently, long silences detected in the *PCM* (`find_long_silence`: ≥3s below −45 dBFS) truncate the inference window and become sample-exact re-anchor points — Parakeet has been observed to nondeterministically collapse a long silence out of its timeline, shifting everything after it several seconds early, and re-anchoring bounds that to a single segment
6. **Transcribes locally** - Runs Parakeet inference via `transcribe-cpp` with `TimestampKind::Word`, one video at a time (transcribe.cpp allows only one in-flight run per loaded model anyway); the GPU backend is compiled in per-OS (Vulkan on Linux, Metal on macOS — see the target-specific dependency tables in `Cargo.toml`), falling back to CPU at runtime when no usable GPU is present
7. **Builds cues** - `build_cues` re-groups word timestamps into subtitle-sized cues (at most two 42-char lines, ≤7s): sentences never merge, over-long sentences split via a Knuth-Plass-style DP preferring clause punctuation and inferred pauses (contiguous timestamps absorb silence into the preceding word, so inflated word durations reveal pauses), and cue display times are trimmed so captions don't linger through silence
8. **Outputs SRT** - Writes captions with timestamps, wrapping cue text into at most two balanced lines (`balance_lines`); skips if `.srt` already exists

### Key Constants

- `FALLBACK_MAX_SEGMENT_LENGTH`: 195 seconds (3m15s) — the segment length used only when the model reports no practical per-call audio limit (the real default comes from `Session::limits()` at startup)
- `ESTIMATED_REALTIME_FACTOR`: 6 — measured CPU throughput, used only for `--dry-run` estimates
- `MODEL_REPO` / `MODEL_FILE`: the pinned HuggingFace model repo and the exact Q8_0 GGUF file downloaded from it
- `MAX_LINE_CHARS` / `MAX_CUE_SECS`: the subtitle envelope (two 42-char lines, ≤7s per cue); the cue-shaping constants block in `main.rs` documents the full cost model (boundary costs, short-cue penalties, pause inference)
- `SILENCE_RMS_DBFS` / `MIN_SPLIT_SILENCE_SECS`: the long-silence re-anchoring detector (its constants block in `main.rs` documents the measured thresholds and the Parakeet timestamp-collapse bug it guards against)

### CLI Flags

- `--dry-run`: List what would be transcribed (durations + estimated processing time)
- `--model`, `--segment-length`: Override config/defaults
- `-q`/`--quiet`: Errors only (no progress bars)
- `-v`/`--verbose`: Print split point details when segmenting
- `-x`/`--one-file-system`: Don't cross filesystem boundaries when recursing

## External Dependencies

Requires `ffmpeg` and `ffprobe` in PATH for audio extraction and delay
detection. The transcribe.cpp C++ core is compiled and statically linked at
build time by the `transcribe-cpp-sys` crate, which needs `cmake` and a C++
toolchain; the Linux Vulkan backend additionally needs the Vulkan headers,
SPIRV headers, and `glslc` at build time.
