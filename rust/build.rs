// build.rs — for the `yolo-cli-bundled` binary, resolves the ONNX file to embed.
//
// Reads the `YOLO_MODEL_ONNX` env var at build time; if unset, defaults to
// `../models/yolov8n.onnx` (relative to this crate's root, i.e. the repo's
// `models/` directory).  Emits `embedded_model.rs` under OUT_DIR that defines
// MODEL_BYTES (via include_bytes!) and MODEL_ID (filename stem for diagnostics).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=YOLO_MODEL_ONNX");

    let default = "../models/yolov8n.onnx";
    let raw = env::var("YOLO_MODEL_ONNX").unwrap_or_else(|_| default.to_string());

    let src_path = Path::new(&raw);
    let abs = if src_path.is_absolute() {
        src_path.to_path_buf()
    } else {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        manifest_dir.join(src_path)
    };

    if !abs.exists() {
        panic!(
            "YOLO_MODEL_ONNX points to {:?} which does not exist (resolved from {:?})",
            abs, raw
        );
    }

    println!("cargo:rerun-if-changed={}", abs.display());

    // Copy the ONNX into OUT_DIR so include_bytes! is path-stable regardless
    // of where YOLO_MODEL_ONNX points (relative paths are resolved against the
    // crate root above, and absolute paths would break reproducibility).
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_path = out_dir.join("embedded.onnx");
    fs::copy(&abs, &target_path).unwrap_or_else(|e| {
        panic!("failed to copy {:?} -> {:?}: {}", abs, target_path, e)
    });

    let model_id = abs.file_stem().and_then(|s| s.to_str()).unwrap_or("model");

    let rs = format!(
        "pub const MODEL_BYTES: &[u8] = include_bytes!(\"{}\");\n\
         pub const MODEL_ID: &str = {:?};\n",
        target_path.display().to_string().replace('\\', "\\\\"),
        model_id,
    );
    fs::write(out_dir.join("embedded_model.rs"), rs).unwrap();

    // Re-expose for user-facing output so `yolo-cli-bundled --version` can show it.
    println!("cargo:rustc-env=BUNDLED_MODEL_ID={}", model_id);
}
