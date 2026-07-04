# caption

Generate SRT subtitle files from video using local speech-to-text.

Transcription runs fully locally using NVIDIA's [Parakeet] model via [ONNX
Runtime]; it is free, private (no audio ever leaves your machine), and fast.
When built from source, inference automatically uses your GPU — AMD and
NVIDIA alike, via ONNX Runtime's WebGPU backend (measured ~9x realtime on a
Radeon 7900 XT) — and falls back to CPU otherwise (~6x realtime measured on
a 32-core desktop CPU). On first run, the model files (~670 MB) are
downloaded from [HuggingFace][model].

[Parakeet]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
[ONNX Runtime]: https://onnxruntime.ai/
[model]: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx

## Why this tool?

Speech-to-text models can only transcribe so much audio in one go — the
Parakeet ONNX export accepts at most ~200 seconds per inference call. This
tool handles arbitrarily long videos by:

1. Splitting audio into segments (3m15s by default)
2. Finding natural sentence boundaries for clean splits
3. Stitching captions back together with correct timestamps

For the full story, see [the blog post](https://thesquareplanet.com/blog/ai-captioning/)
(written when this tool still used a cloud transcription API, but the
segmenting-and-stitching approach it describes is unchanged).

## Installation

### Pre-built binaries

Download from the [releases page](https://github.com/jonhoo/sirtea/releases).

**Note: the pre-built binaries are CPU-only.** GPU inference requires a
separate shared library (`libwebgpu_dawn.so`) that the release installers
cannot deliver, so the pre-built binaries are built without GPU support to
stay self-contained. For maximum performance, build from source.

### From source

```bash
cargo install sirtea
```

Source builds enable GPU inference (via ONNX Runtime's WebGPU backend) by
default. On targets without WebGPU-enabled ONNX Runtime prebuilts (e.g.
aarch64 Linux), the `webgpu` feature fails to link; build without it:

```bash
cargo install sirtea --no-default-features
```

### Requirements

- [ffmpeg and ffprobe](https://ffmpeg.org/download.html) in your PATH
- ~700 MB of disk for the auto-downloaded model, and ~5 GB of RAM
  during inference

## Usage

```bash
sirtea video.mkv                   # Transcribe a single video
sirtea lecture1.mp4 lecture2.mp4   # Transcribe multiple videos
sirtea --dry-run *.mkv             # List what would be transcribed
sirtea -q video.mp4                # Quiet mode (errors only)
```

Output SRT files are created alongside the input videos (e.g., `video.mkv` →
`video.srt`). Videos with existing `.srt` files are skipped.

## Caveats

- **Captions are verbatim.** Fillers ("uh", "um") and stutter repetitions
  are transcribed as spoken, and reading speed mirrors the speaker — a fast
  talker yields fast captions.
- **The model is imperfect.** Parakeet occasionally mis-transcribes; in
  rare cases it has been seen to drop a spoken clause or emit a brief
  word-repetition loop.
- Because of both of the above, it is often worthwhile to pass the raw
  transcript through an LLM afterwards to tidy up technical terms, fix
  inconsistent spellings, and remove stutters.

## Configuration

Configuration is optional. To override defaults, create a config file:
- Linux: `~/.config/sirtea/config.toml`
- macOS: `~/Library/Application Support/sirtea/config.toml`
- Windows: `%APPDATA%\sirtea\config.toml`

See `config.example.toml` for a template.

### Options

These can be set in the config file or overridden via CLI flags:

| Option | CLI flag | Default | Description |
|--------|----------|---------|-------------|
| `model_path` | `--model` | auto-download | Directory holding the [Parakeet model files][model]; when unset, they are downloaded on first run into the per-user data dir (e.g. `~/.local/share/sirtea/models/` on Linux) |
| `segment_length` | `--segment-length` | 195 | Max segment length in seconds. The bundled model accepts at most ~200 seconds per inference call, so the default sits just under that ceiling; lowering it mainly reduces memory use |

> **Upgrading from the Gladia-based version?** Transcription is now local, so
> the `gladia_api_key`, `max_cost`, and `parallel` settings are gone; remove
> them from your config file (sirtea will point them out as errors otherwise).

## CLI Reference

```
sirtea [OPTIONS] <PATH>...
```

Run `sirtea --help` for full usage information.

## License

The code is licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

The Parakeet model weights downloaded on first run are NVIDIA's, licensed
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
