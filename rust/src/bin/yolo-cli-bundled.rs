//! yolo-cli-bundled — bundled variant: the ONNX is embedded at compile time via
//! `include_bytes!`, so there's no `--model` flag and no runtime file I/O for
//! the model.  This produces a single, truly portable executable.
//!
//! Build with the `bundled` feature + `YOLO_MODEL_ONNX` env var:
//!     YOLO_MODEL_ONNX=../models/yolov8n.onnx \
//!       cargo build --release --features bundled --bin yolo-cli-bundled

use std::env;
use std::process::ExitCode;

use yolo_cli::{
    emit_info, emit_json, emit_tables, infer_images, load_model_from_bytes, parse_args, Mode,
};

include!(concat!(env!("OUT_DIR"), "/embedded_model.rs"));

/// Inline thread-local that holds conf/iou actually in effect, so --help can show them
/// as user-visible defaults. For bundled we print 0.25/0.45 unless the ONNX metadata
/// carried explicit threshold keys (currently Ultralytics doesn't, but the hook is live).
fn print_help(bin: &str, eff_conf: f32, eff_iou: f32) {
    eprintln!(
        "usage: {bin} [--conf {conf}] [--iou {iou}] [--json] [--info] IMG [IMG ...]\n\
         \n\
         Bundled YOLO ONNX detection CLI (pure Rust, tract backend, model\n\
         embedded at compile time).  The binary is fully self-contained — the\n\
         ONNX graph AND weights are baked in via include_bytes!.  For a variant\n\
         that takes --model at runtime, use `yolo-cli`.\n\
         \n\
         Embedded model: {} ({} bytes)\n\
         \n\
         Flags:\n\
           --conf  F      confidence threshold (default {conf})\n\
           --iou   F      NMS IoU threshold   (default {iou})\n\
           --json         emit JSON for `[{{\"image\":..., \"detections\":[{{\"label\":...,\"score\":...,\"bbox\":[x0,y0,x1,y1]}}]}}]`\n\
           --info         print embedded model metadata (imgsz, stride, output\n\
                          shape/format, class labels, defaults); no images needed\n\
           --help, -h     print help\n\
           --version, -V  print version",
        MODEL_ID,
        MODEL_BYTES.len(),
        conf = eff_conf,
        iou  = eff_iou,
    );
}

fn main() -> ExitCode {
    let argv: Vec<String> = env::args().collect();
    let bin = argv.first().map(String::as_str).unwrap_or("yolo-cli-bundled");

    // Peek at model metadata first so --help can display the actual effective defaults.
    let handle = match load_model_from_bytes(MODEL_BYTES, format!("<embedded:{MODEL_ID}>")) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: load embedded model ({}): {e}", MODEL_ID);
            return ExitCode::from(1);
        }
    };
    let eff_conf_default = handle.meta.default_conf.unwrap_or(0.25);
    let eff_iou_default = handle.meta.default_iou.unwrap_or(0.45);

    let parsed = match parse_args(&argv, Mode::Bundled) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            print_help(bin, eff_conf_default, eff_iou_default);
            return ExitCode::from(2);
        }
    };

    if parsed.help {
        print_help(bin, eff_conf_default, eff_iou_default);
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
    if parsed.info {
        emit_info(&handle);
        return ExitCode::SUCCESS;
    }

    let conf = if parsed.conf_explicit { parsed.common.conf } else { eff_conf_default };
    let iou = if parsed.iou_explicit { parsed.common.iou } else { eff_iou_default };

    let results = match infer_images(&handle, &parsed.common.images, conf, iou) {
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
