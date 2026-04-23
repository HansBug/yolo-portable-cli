//! yolo-cli — free variant: loads an ONNX model at runtime via --model.

use std::env;
use std::process::ExitCode;

use yolo_cli::{
    emit_info, emit_json, emit_tables, infer_images, load_model_from_path, parse_args, Mode,
};

fn print_help(bin: &str) {
    eprintln!(
        "usage: {bin} --model MODEL.onnx [--conf F] [--iou F] [--json] [--info] IMG [IMG ...]\n\
         \n\
         Free-form YOLO ONNX detection CLI (pure Rust, tract backend).  The\n\
         model file is loaded at runtime — ship this binary + the .onnx and\n\
         use any YOLO model that Ultralytics can export to ONNX (v5u/v8/v9/v10/v11/v12).\n\
         \n\
         Flags:\n\
           --model FILE   path to the .onnx file (required, except with --help/--version)\n\
           --conf  F      confidence threshold (see default-resolution below)\n\
           --iou   F      NMS IoU threshold   (see default-resolution below)\n\
           --json         emit JSON for `[{{\"image\":..., \"detections\":[{{\"label\":...,\"score\":...,\"bbox\":[x0,y0,x1,y1]}}]}}]`\n\
           --info         load the model and print metadata (imgsz, stride, output\n\
                          shape/format, all class labels, defaults); no images needed\n\
           --help, -h     print help\n\
           --version, -V  print version\n\
         \n\
         Default resolution for --conf / --iou:\n\
           1. If the user passes --conf/--iou, that value wins.\n\
           2. Otherwise, if the ONNX metadata_props carries a \"conf\" /\n\
              \"conf_threshold\" / \"iou\" / \"iou_threshold\" entry, the CLI uses it.\n\
              (Ultralytics does not currently embed these; the hook is future-facing.)\n\
           3. Otherwise, built-in defaults: conf=0.25, iou=0.45 (same as Ultralytics'\n\
              Python predict() defaults).\n\
         \n\
         Input image size and class labels are always read from the ONNX metadata\n\
         (`imgsz` and `names` keys). If the model lacks them, imgsz falls back\n\
         to 640x640 and labels fall back to `class_<id>`."
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

    if parsed.info {
        emit_info(&handle);
        return ExitCode::SUCCESS;
    }

    // Resolve effective thresholds: user > metadata > built-in default.
    let conf = if parsed.conf_explicit {
        parsed.common.conf
    } else {
        handle.meta.default_conf.unwrap_or(parsed.common.conf)
    };
    let iou = if parsed.iou_explicit {
        parsed.common.iou
    } else {
        handle.meta.default_iou.unwrap_or(parsed.common.iou)
    };

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
