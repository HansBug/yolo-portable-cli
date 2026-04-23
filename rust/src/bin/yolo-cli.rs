//! yolo-cli — free variant: loads an ONNX model at runtime via --model.
//!
//! Usage:
//!     yolo-cli --model MODEL.onnx [--conf F] [--iou F] [--json] IMG [IMG ...]

use std::env;
use std::process::ExitCode;

use yolo_cli::{
    emit_json, emit_tables, infer_images, load_model_from_path, parse_args, Mode,
};

fn print_help(bin: &str) {
    eprintln!(
        "usage: {bin} --model MODEL.onnx [--conf 0.25] [--iou 0.45] [--json] IMG [IMG ...]\n\
         \n\
         Free-form YOLO ONNX detection CLI (pure Rust, tract backend).  The\n\
         model file is loaded at runtime — ship this binary + the .onnx and\n\
         use any model (v5u/v8/v9/v10/v11/v12).  Class labels come from the\n\
         ONNX `names` metadata entry; fallback is `class_<id>`.\n\
         \n\
         Flags:\n\
           --model FILE   path to the .onnx file (required)\n\
           --conf  F      confidence threshold (default 0.25)\n\
           --iou   F      NMS IoU threshold   (default 0.45)\n\
           --json         emit JSON for `[{{\"image\":..., \"detections\":[{{\"label\":...,\"score\":...,\"bbox\":[x0,y0,x1,y1]}}]}}]`\n\
           --help, -h     print help\n\
           --version, -V  print version"
    );
}

fn main() -> ExitCode {
    let argv: Vec<String> = env::args().collect();
    let bin = argv.first().map(String::as_str).unwrap_or("yolo-cli");

    let parsed = match parse_args(&argv, Mode::Free) {
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
        println!("yolo-cli {} (free; runtime --model)", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    let model_path = parsed.model.expect("parse_args guarantees --model in Free mode");
    let handle = match load_model_from_path(&model_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: load {:?}: {e}", model_path);
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
