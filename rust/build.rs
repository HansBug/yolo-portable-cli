// build.rs — only does real work when the `bundled` Cargo feature is active.
//
// When `CARGO_FEATURE_BUNDLED` is set (implied by `--features bundled`), this
// reads `YOLO_MODEL_ONNX` (default: `../models/yolov8n.onnx` relative to the
// crate root), copies the ONNX bytes into OUT_DIR, and emits `embedded_model.rs`
// with `MODEL_BYTES` (via include_bytes!) and `MODEL_ID` (filename stem).
//
// When the feature is off, the script exits without producing embedded_model.rs.
// That's fine because `yolo-cli-bundled`'s source file is gated behind
// `required-features = ["bundled"]` in Cargo.toml, so cargo won't try to compile
// it and thus won't evaluate its `include!(...)` of embedded_model.rs.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BUNDLED");
    println!("cargo:rerun-if-env-changed=YOLO_MODEL_ONNX");

    let feature_on = env::var_os("CARGO_FEATURE_BUNDLED").is_some();
    if !feature_on {
        // No work to do. The `yolo-cli-bundled` bin is gated out via required-features.
        return;
    }

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
    println!("cargo:rustc-env=BUNDLED_MODEL_ID={}", model_id);
}
