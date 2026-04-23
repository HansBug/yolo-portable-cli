//! yolo-cli-bundled — bundled variant: the ONNX is embedded at compile time
//! via `include_bytes!`, so there is no `--model` flag and no runtime file I/O
//! for the model.  This produces a single, truly portable executable.
//!
//! Build with `YOLO_MODEL_ONNX` pointing at the .onnx to embed; it defaults to
//! `models/yolov8n.onnx` in the repo root.
//!
//! Usage:
//!     yolo-cli-bundled [--conf F] [--iou F] [--json] IMG [IMG ...]

use std::env;
use std::process::ExitCode;

use yolo_cli::{
    emit_json, emit_tables, infer_images, load_model_from_bytes, parse_args, Mode,
};

include!(concat!(env!("OUT_DIR"), "/embedded_model.rs"));

fn print_help(bin: &str) {
    eprintln!(
        "usage: {bin} [--conf 0.25] [--iou 0.45] [--json] IMG [IMG ...]\n\
         \n\
         Bundled YOLO ONNX detection CLI (pure Rust, tract backend, model\n\
         embedded at compile time).  The binary is fully self-contained — the\n\
         ONNX graph AND weights are baked in via include_bytes!.  For a variant\n\
         that takes --model at runtime, use yolo-cli.\n\
         \n\
         Embedded model: {} ({} bytes)\n\
         \n\
         Flags:\n\
           --conf  F      confidence threshold (default 0.25)\n\
           --iou   F      NMS IoU threshold   (default 0.45)\n\
           --json         emit JSON\n\
           --help, -h     print help\n\
           --version, -V  print version",
        MODEL_ID,
        MODEL_BYTES.len()
    );
}

fn main() -> ExitCode {
    let argv: Vec<String> = env::args().collect();
    let bin = argv.first().map(String::as_str).unwrap_or("yolo-cli-bundled");

    let parsed = match parse_args(&argv, Mode::Bundled) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            print_help(bin);
            return ExitCode::from(2);
        }
    };

    if parsed.help {
        print_help(bin);
        return ExitCode::SUCCESS;
    }
    if parsed.version {
        println!(
            "yolo-cli-bundled {} (bundled; model={}, {} bytes)",
            env!("CARGO_PKG_VERSION"),
            MODEL_ID,
            MODEL_BYTES.len()
        );
        return ExitCode::SUCCESS;
    }

    let handle = match load_model_from_bytes(MODEL_BYTES) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: load embedded model ({}): {e}", MODEL_ID);
            return ExitCode::from(1);
        }
    };

    let results = match infer_images(&handle, &parsed.common.images, parsed.common.conf, parsed.common.iou) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    if parsed.common.json {
        emit_json(&results);
    } else {
        emit_tables(&results);
    }
    ExitCode::SUCCESS
}
