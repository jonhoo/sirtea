# caption

Generate SRT subtitle files from video using local speech-to-text.

Transcription runs fully locally using NVIDIA's [Parakeet] model via
[transcribe.cpp] (ggml); it is free, private (no audio ever leaves your
machine), and fast (measured ~160x realtime on a Radeon 7900 XT, ~19x on a
32-core desktop CPU). Inference automatically uses your GPU — Vulkan on
Linux (AMD and NVIDIA alike), Metal on macOS — and falls back to CPU when
no usable GPU is present. On first run, the model (~740 MB, a single GGUF file) is
downloaded from [HuggingFace][model].

[Parakeet]: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
[transcribe.cpp]: https://github.com/handy-computer/transcribe.cpp
[model]: https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v3-gguf

## Why this tool?

Speech-to-text models can only transcribe so much audio in one go — the
model states its per-inference-call limit, and sirtea asks it at startup.
This tool handles arbitrarily long videos by:

1. Splitting audio into segments the model can accept in one call
2. Finding natural sentence boundaries for clean splits
3. Stitching captions back together with correct timestamps

For the full story, see [the blog post](https://thesquareplanet.com/blog/ai-captioning/)
(written when this tool still used a cloud transcription API, but the
segmenting-and-stitching approach it describes is unchanged).

## Installation

### Pre-built binaries

Download from the [releases page](https://github.com/jonhoo/sirtea/releases).

### From source

```bash
cargo install sirtea
```

Building from source compiles the transcribe.cpp C++ core, which needs
`cmake` and a C++ toolchain. On Linux, the (always-on) Vulkan GPU backend
additionally needs the Vulkan development packages at build time: the
Vulkan headers, SPIRV headers, and the `glslc` shader compiler (on Arch:
`vulkan-headers`, `spirv-headers`, `shaderc`; on Debian/Ubuntu the
`vulkan-sdk` or `libvulkan-dev` + `spirv-headers` + `glslc` packages). On
macOS the Metal backend needs only Xcode's toolchain. Machines without a
usable GPU at *runtime* are fine either way: inference falls back to CPU
automatically (the chosen backend is printed at startup).

If linking fails with `undefined symbol: cblas_sgemm`, your system's BLAS
exposes only the Fortran interface (e.g. Arch's netlib `blas` package
without `cblas` on the link line); build without system BLAS instead —
it only affects the lightweight host-side kernels:

```bash
TRANSCRIBE_CMAKE_ARGS=-DTRANSCRIBE_USE_SYSTEM_BLAS=OFF cargo install sirtea
```

### Requirements

- [ffmpeg and ffprobe](https://ffmpeg.org/download.html) in your PATH
- ~740 MB of disk for the auto-downloaded model

If you previously used the ONNX-based version of sirtea, the old model
directory (`~/.local/share/sirtea/models/parakeet-tdt-0.6b-v3-int8` on
Linux) is no longer used and can be deleted to reclaim ~670 MB.

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
| `model_path` | `--model` | auto-download | Path to the [Parakeet GGUF model file][model] (any quantization; a directory containing the default file also works); when unset, it is downloaded on first run into the per-user data dir (e.g. `~/.local/share/sirtea/models/` on Linux) |
| `segment_length` | `--segment-length` | model limit | Max segment length in seconds. Defaults to the model's own per-inference-call audio limit, queried at startup; values above it are rejected, and lowering it mainly reduces memory use |

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
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (repackaged in
GGUF form by the [transcribe.cpp] authors).

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
