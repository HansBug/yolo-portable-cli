//! Shared logic for the `yolo-cli` (free) and `yolo-cli-bundled` binaries.
//!
//! Handles ONNX loading + metadata, output-shape auto-detection for YOLOv5/v7
//! (with-objectness), YOLOv8-12 (class-only), and post-NMS exports, letterbox
//! preprocessing, per-class NMS, and both human-facing table and JSON output.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::DimLike;

// ============================================================================
// Detection type (public shape of the CLI result)
// ============================================================================

#[derive(Debug, Clone)]
pub struct Det {
    pub cls: u32,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
}

// ============================================================================
// CLI argument parsing (shared pieces)
// ============================================================================

pub struct CommonArgs {
    pub images: Vec<PathBuf>,
    pub conf: f32,
    pub iou: f32,
    pub json: bool,
}

pub struct ParsedArgs {
    pub common: CommonArgs,
    /// Only set for the free (--model) binary; None when the binary has an embedded model.
    pub model: Option<PathBuf>,
    /// Help was requested → caller should print help and exit 0.
    pub help: bool,
    /// Version was requested → caller should print version and exit 0.
    pub version: bool,
}

pub enum Mode {
    Free,
    Bundled,
}

pub fn parse_args(argv: &[String], mode: Mode) -> Result<ParsedArgs, String> {
    let mut images: Vec<PathBuf> = Vec::new();
    let mut conf: f32 = 0.25;
    let mut iou: f32 = 0.45;
    let mut json = false;
    let mut model: Option<PathBuf> = None;
    let mut help = false;
    let mut version = false;

    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].clone();
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let mut take_value = |name: &str| -> Result<String, String> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("flag {name} requires a value"))
        };
        match flag.as_str() {
            "--help" | "-h" => help = true,
            "--version" | "-V" => version = true,
            "--json" => json = true,
            "--conf" => conf = take_value("--conf")?.parse().map_err(|e| format!("--conf: {e}"))?,
            "--iou" => iou = take_value("--iou")?.parse().map_err(|e| format!("--iou: {e}"))?,
            "--model" => match mode {
                Mode::Free => model = Some(PathBuf::from(take_value("--model")?)),
                Mode::Bundled => {
                    return Err(
                        "--model is not accepted by yolo-cli-bundled (the ONNX was baked in at build time); use `yolo-cli` for a runtime model".into(),
                    )
                }
            },
            x if x.starts_with("--") => return Err(format!("unknown flag: {x}")),
            _ => images.push(PathBuf::from(&a)),
        }
        i += 1;
    }

    if help || version {
        return Ok(ParsedArgs {
            common: CommonArgs { images, conf, iou, json },
            model,
            help,
            version,
        });
    }

    match mode {
        Mode::Free => {
            if model.is_none() {
                return Err("missing --model MODEL.onnx (required by the free yolo-cli)".into());
            }
        }
        Mode::Bundled => {
            if model.is_some() {
                return Err("--model is not accepted by yolo-cli-bundled".into());
            }
        }
    }
    if images.is_empty() {
        return Err("no input images given".into());
    }
    Ok(ParsedArgs {
        common: CommonArgs { images, conf, iou, json },
        model,
        help,
        version,
    })
}

// ============================================================================
// Metadata parsing
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    V8Anchors,
    V5Anchors,
    PostNms,
}

pub struct Meta {
    pub names: BTreeMap<u32, String>,
    pub imgsz: (u32, u32),
    pub end2end: bool,
}

pub fn parse_names(raw: &str) -> BTreeMap<u32, String> {
    let mut out = BTreeMap::new();
    let s = raw.trim();
    let s = s.strip_prefix('{').unwrap_or(s);
    let s = s.strip_suffix('}').unwrap_or(s);
    let b = s.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        while i < b.len() && matches!(b[i], b' ' | b',' | b'\n' | b'\t' | b'\r') { i += 1; }
        if i >= b.len() { break; }
        let ns = i;
        while i < b.len() && b[i].is_ascii_digit() { i += 1; }
        if i == ns { break; }
        let key: u32 = match std::str::from_utf8(&b[ns..i]).ok().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => break,
        };
        while i < b.len() && b[i] != b'\'' && b[i] != b'"' { i += 1; }
        if i >= b.len() { break; }
        let q = b[i];
        i += 1;
        let vs = i;
        while i < b.len() && b[i] != q { i += 1; }
        if i > b.len() { break; }
        let v = std::str::from_utf8(&b[vs..i]).unwrap_or("").to_string();
        out.insert(key, v);
        i = i.saturating_add(1);
    }
    out
}

pub fn parse_imgsz(raw: &str) -> Option<(u32, u32)> {
    let s = raw.trim();
    let s = s.strip_prefix('[').unwrap_or(s);
    let s = s.strip_suffix(']').unwrap_or(s);
    let mut parts = s.split(',').map(|p| p.trim().parse::<u32>().ok());
    let h = parts.next().flatten()?;
    let w = parts.next().flatten().unwrap_or(h);
    Some((h, w))
}

pub fn read_meta(proto: &tract_onnx::pb::ModelProto) -> Meta {
    let mut names = BTreeMap::new();
    let mut imgsz = (640u32, 640u32);
    let mut end2end = false;
    for p in &proto.metadata_props {
        match p.key.as_str() {
            "names" => names = parse_names(&p.value),
            "imgsz" => {
                if let Some(v) = parse_imgsz(&p.value) {
                    imgsz = v;
                }
            }
            "end2end" => end2end = matches!(p.value.as_str(), "True" | "true" | "1"),
            _ => {}
        }
    }
    Meta { names, imgsz, end2end }
}

pub fn label_of(names: &BTreeMap<u32, String>, cls: u32) -> String {
    names.get(&cls).cloned().unwrap_or_else(|| format!("class_{cls}"))
}

pub fn detect_format(
    out_shape: &[usize],
    nc_hint: Option<usize>,
    end2end_hint: bool,
) -> TractResult<(OutputFormat, usize)> {
    match out_shape {
        [_b, d1, d2] => {
            if end2end_hint && *d2 == 6 {
                return Ok((OutputFormat::PostNms, nc_hint.unwrap_or(0)));
            }
            if *d2 == 6 && *d1 <= 1000 {
                return Ok((OutputFormat::PostNms, nc_hint.unwrap_or(0)));
            }
            if let Some(nc) = nc_hint {
                if *d1 == 4 + nc {
                    return Ok((OutputFormat::V8Anchors, nc));
                }
                if *d2 == 5 + nc {
                    return Ok((OutputFormat::V5Anchors, nc));
                }
            }
            if d1 < d2 {
                let nc = (*d1).saturating_sub(4);
                if nc > 0 {
                    return Ok((OutputFormat::V8Anchors, nc));
                }
            }
            if d2 < d1 {
                let nc = (*d2).saturating_sub(5);
                if nc > 0 {
                    return Ok((OutputFormat::V5Anchors, nc));
                }
            }
            Err(anyhow::anyhow!("cannot detect YOLO output format from shape {:?}", out_shape))
        }
        _ => Err(anyhow::anyhow!("unexpected output rank: shape {:?}", out_shape)),
    }
}

// ============================================================================
// Preprocessing & decoding
// ============================================================================

pub struct Preprocessed {
    pub tensor: Tensor,
    pub orig_w: u32,
    pub orig_h: u32,
    pub scale: f32,
    pub pad_x: u32,
    pub pad_y: u32,
}

pub fn preprocess(path: &Path, target_h: u32, target_w: u32) -> TractResult<Preprocessed> {
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("image::open {:?}: {e}", path))?
        .to_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());
    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;
    let pad_x = (target_w - new_w) / 2;
    let pad_y = (target_h - new_h) / 2;
    let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);
    let mut tensor = Tensor::zero::<f32>(&[1, 3, target_h as usize, target_w as usize])?;
    let slice = tensor.as_slice_mut::<f32>()?;
    let plane = (target_h as usize) * (target_w as usize);
    for v in slice.iter_mut() {
        *v = 114.0 / 255.0;
    }
    for (x, y, px) in resized.enumerate_pixels() {
        let [r, g, b] = px.0;
        let xi = (x + pad_x) as usize;
        let yi = (y + pad_y) as usize;
        let idx = yi * (target_w as usize) + xi;
        slice[idx] = r as f32 / 255.0;
        slice[plane + idx] = g as f32 / 255.0;
        slice[2 * plane + idx] = b as f32 / 255.0;
    }
    Ok(Preprocessed { tensor, orig_w, orig_h, scale, pad_x, pad_y })
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let aa = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let bb = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let u = aa + bb - inter;
    if u <= 0.0 { 0.0 } else { inter / u }
}

fn nms(mut cands: Vec<Det>, iou_th: f32) -> Vec<Det> {
    cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<Det> = Vec::new();
    for c in cands {
        let mut drop = false;
        for k in &kept {
            if k.cls == c.cls && iou(&k.bbox, &c.bbox) > iou_th {
                drop = true;
                break;
            }
        }
        if !drop {
            kept.push(c);
        }
    }
    kept
}

fn unletterbox(
    cx: f32, cy: f32, w: f32, h: f32,
    pad_x: u32, pad_y: u32, scale: f32,
    orig_w: u32, orig_h: u32,
) -> [f32; 4] {
    let x0 = ((cx - w / 2.0) - pad_x as f32) / scale;
    let y0 = ((cy - h / 2.0) - pad_y as f32) / scale;
    let x1 = ((cx + w / 2.0) - pad_x as f32) / scale;
    let y1 = ((cy + h / 2.0) - pad_y as f32) / scale;
    [
        x0.clamp(0.0, orig_w as f32),
        y0.clamp(0.0, orig_h as f32),
        x1.clamp(0.0, orig_w as f32),
        y1.clamp(0.0, orig_h as f32),
    ]
}

pub fn decode(
    out: &tract_ndarray::ArrayViewD<f32>,
    fmt: OutputFormat,
    nc: usize,
    names: &BTreeMap<u32, String>,
    pre: &Preprocessed,
    conf_th: f32,
    iou_th: f32,
) -> Vec<Det> {
    let mut cands: Vec<Det> = Vec::new();
    match fmt {
        OutputFormat::V8Anchors => {
            let s = out.shape();
            let num_anchors = s[2];
            for a in 0..num_anchors {
                let cx = out[[0, 0, a]];
                let cy = out[[0, 1, a]];
                let w = out[[0, 2, a]];
                let h = out[[0, 3, a]];
                let (mut best, mut best_c) = (0f32, 0u32);
                for c in 0..nc {
                    let s = out[[0, 4 + c, a]];
                    if s > best {
                        best = s;
                        best_c = c as u32;
                    }
                }
                if best >= conf_th {
                    let bbox = unletterbox(cx, cy, w, h, pre.pad_x, pre.pad_y, pre.scale, pre.orig_w, pre.orig_h);
                    cands.push(Det { cls: best_c, label: label_of(names, best_c), score: best, bbox });
                }
            }
        }
        OutputFormat::V5Anchors => {
            let s = out.shape();
            let num_anchors = s[1];
            for a in 0..num_anchors {
                let cx = out[[0, a, 0]];
                let cy = out[[0, a, 1]];
                let w = out[[0, a, 2]];
                let h = out[[0, a, 3]];
                let obj = out[[0, a, 4]];
                if obj < conf_th {
                    continue;
                }
                let (mut best, mut best_c) = (0f32, 0u32);
                for c in 0..nc {
                    let cls = out[[0, a, 5 + c]];
                    if cls > best {
                        best = cls;
                        best_c = c as u32;
                    }
                }
                let score = best * obj;
                if score >= conf_th {
                    let bbox = unletterbox(cx, cy, w, h, pre.pad_x, pre.pad_y, pre.scale, pre.orig_w, pre.orig_h);
                    cands.push(Det { cls: best_c, label: label_of(names, best_c), score, bbox });
                }
            }
        }
        OutputFormat::PostNms => {
            let s = out.shape();
            let k = s[1];
            for i in 0..k {
                let x1 = out[[0, i, 0]];
                let y1 = out[[0, i, 1]];
                let x2 = out[[0, i, 2]];
                let y2 = out[[0, i, 3]];
                let score = out[[0, i, 4]];
                let cls = out[[0, i, 5]] as u32;
                if score < conf_th { continue; }
                let ux1 = (x1 - pre.pad_x as f32) / pre.scale;
                let uy1 = (y1 - pre.pad_y as f32) / pre.scale;
                let ux2 = (x2 - pre.pad_x as f32) / pre.scale;
                let uy2 = (y2 - pre.pad_y as f32) / pre.scale;
                let bbox = [
                    ux1.clamp(0.0, pre.orig_w as f32),
                    uy1.clamp(0.0, pre.orig_h as f32),
                    ux2.clamp(0.0, pre.orig_w as f32),
                    uy2.clamp(0.0, pre.orig_h as f32),
                ];
                cands.push(Det { cls, label: label_of(names, cls), score, bbox });
            }
        }
    }
    match fmt {
        OutputFormat::PostNms => cands,
        _ => nms(cands, iou_th),
    }
}

// ============================================================================
// Top-level: model load + inference loop
// ============================================================================

pub struct ModelHandle {
    pub meta: Meta,
    pub format: OutputFormat,
    pub nc: usize,
    pub names: BTreeMap<u32, String>,
    pub runnable: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

pub fn load_model_from_bytes(bytes: &[u8]) -> TractResult<ModelHandle> {
    let onnx = tract_onnx::onnx();
    let proto = onnx.proto_model_for_read(&mut std::io::Cursor::new(bytes))?;
    build_from_proto(&proto)
}

pub fn load_model_from_path(path: &Path) -> TractResult<ModelHandle> {
    let onnx = tract_onnx::onnx();
    let mut f = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("open {:?}: {e}", path))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| anyhow::anyhow!("read {:?}: {e}", path))?;
    let proto = onnx.proto_model_for_read(&mut std::io::Cursor::new(&buf))?;
    build_from_proto(&proto)
}

fn build_from_proto(proto: &tract_onnx::pb::ModelProto) -> TractResult<ModelHandle> {
    let meta = read_meta(proto);
    let onnx = tract_onnx::onnx();
    let (h, w) = (meta.imgsz.0 as i32, meta.imgsz.1 as i32);
    let typed = onnx
        .model_for_proto_model(proto)?
        .with_input_fact(
            0,
            InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, h, w)),
        )?
        .into_optimized()?;

    let out_outlet = typed.output_outlets()?[0];
    let out_fact = typed.outlet_fact(out_outlet)?;
    let out_shape: Vec<usize> = out_fact
        .shape
        .iter()
        .map(|d| d.to_usize().unwrap_or(0))
        .collect();
    let nc_hint = if meta.names.is_empty() { None } else { Some(meta.names.len()) };
    let (format, nc) = detect_format(&out_shape, nc_hint, meta.end2end)?;

    let mut names = meta.names.clone();
    if names.is_empty() {
        for c in 0..(nc as u32) {
            names.insert(c, format!("class_{c}"));
        }
    }

    let runnable = typed.into_runnable()?;
    Ok(ModelHandle { meta, format, nc, names, runnable })
}

pub fn infer_images(
    handle: &ModelHandle,
    images: &[PathBuf],
    conf_th: f32,
    iou_th: f32,
) -> TractResult<Vec<(PathBuf, Vec<Det>)>> {
    let (h, w) = (handle.meta.imgsz.0, handle.meta.imgsz.1);
    let mut results = Vec::with_capacity(images.len());
    for img in images {
        let pre = preprocess(img, h, w)?;
        let outputs = handle
            .runnable
            .run(tvec!(pre.tensor.clone().into()))?;
        let arr = outputs[0].to_array_view::<f32>()?;
        let dets = decode(&arr, handle.format, handle.nc, &handle.names, &pre, conf_th, iou_th);
        results.push((img.clone(), dets));
    }
    Ok(results)
}

// ============================================================================
// Output emitters
// ============================================================================

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn emit_json(per_image: &[(PathBuf, Vec<Det>)]) {
    print!("[");
    for (i, (path, dets)) in per_image.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("{{\"image\":{},\"detections\":[", escape_json_str(&path.to_string_lossy()));
        for (j, d) in dets.iter().enumerate() {
            if j > 0 { print!(","); }
            print!(
                "{{\"label\":{},\"score\":{:.4},\"bbox\":[{:.2},{:.2},{:.2},{:.2}]}}",
                escape_json_str(&d.label),
                d.score,
                d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
            );
        }
        print!("]}}");
    }
    println!("]");
}

pub fn emit_tables(per_image: &[(PathBuf, Vec<Det>)]) {
    for (img_idx, (path, dets)) in per_image.iter().enumerate() {
        if img_idx > 0 { println!(); }
        println!("== {} ==", path.display());
        if dets.is_empty() {
            println!("  (no detections)");
            continue;
        }
        let mut by_label: BTreeMap<String, Vec<&Det>> = BTreeMap::new();
        for d in dets {
            by_label.entry(d.label.clone()).or_default().push(d);
        }
        for (label, group) in &by_label {
            let n = group.len();
            println!();
            println!("{label} ({n} detection{}):", if n == 1 { "" } else { "s" });
            let hdr = ["score", "x0", "y0", "x1", "y1"];
            let mut rows: Vec<[String; 5]> = Vec::with_capacity(n);
            for d in group {
                rows.push([
                    format!("{:.3}", d.score),
                    format!("{:.1}", d.bbox[0]),
                    format!("{:.1}", d.bbox[1]),
                    format!("{:.1}", d.bbox[2]),
                    format!("{:.1}", d.bbox[3]),
                ]);
            }
            let mut w = [0usize; 5];
            for (i, hh) in hdr.iter().enumerate() { w[i] = hh.len(); }
            for r in &rows {
                for (i, c) in r.iter().enumerate() {
                    if c.len() > w[i] { w[i] = c.len(); }
                }
            }
            let mut header_line = String::from("  ");
            for (i, hh) in hdr.iter().enumerate() {
                if i > 0 { header_line.push_str("  "); }
                header_line.push_str(&format!("{:>width$}", hh, width = w[i]));
            }
            let mut rule_line = String::from("  ");
            for (i, wi) in w.iter().enumerate() {
                if i > 0 { rule_line.push_str("  "); }
                for _ in 0..*wi { rule_line.push('-'); }
            }
            println!("{header_line}");
            println!("{rule_line}");
            for r in &rows {
                let mut line = String::from("  ");
                for (i, c) in r.iter().enumerate() {
                    if i > 0 { line.push_str("  "); }
                    line.push_str(&format!("{:>width$}", c, width = w[i]));
                }
                println!("{line}");
            }
        }
    }
}
