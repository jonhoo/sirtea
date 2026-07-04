fn main() {
    // ONNX Runtime's WebGPU execution provider is backed by Dawn, which the
    // `ort` crate ships as a separate shared library (libwebgpu_dawn.so /
    // .dylib / .dll) and copies next to the produced binary. Linux and macOS
    // don't search the binary's own directory for shared libraries by
    // default, so bake that into the rpath; without this the binary aborts
    // at startup with "error while loading shared libraries".
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
}
