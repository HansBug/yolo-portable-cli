use std::env;
use std::path::Path;
use std::time::Instant;

use tract_onnx::prelude::*;

const COCO: [&str; 80] = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
];

fn main() -> TractResult<()> {
    let t_start = Instant::now();
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <model.onnx> <image> [conf=0.25] [iou=0.45]", args[0]);
        std::process::exit(2);
    }
    let model_path = &args[1];
    let image_path = &args[2];
    let conf_th: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.25);
    let iou_th: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.45);

    let t_load0 = Instant::now();
    // YOLOv8 exported with default dynamic=False has fixed shape [1,3,640,640]
    let input_fact = InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 640, 640));
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(0, input_fact)?
        .into_optimized()?
        .into_runnable()?;
    let t_load = t_load0.elapsed();

    let t_pre0 = Instant::now();
    let img = image::open(Path::new(image_path))?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();
    // Letterbox to 640x640: preserve aspect, pad with (114,114,114).
    let scale = (640.0 / orig_w as f32).min(640.0 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;
    let pad_x = (640 - new_w) / 2;
    let pad_y = (640 - new_h) / 2;
    let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);
    let mut input = Tensor::zero::<f32>(&[1, 3, 640, 640])?;
    let input_slice = input.as_slice_mut::<f32>()?;
    let plane = 640 * 640;
    // prefill with 114/255
    for v in input_slice.iter_mut() { *v = 114.0 / 255.0; }
    for (x, y, px) in resized.enumerate_pixels() {
        let [r, g, b] = px.0;
        let xi = (x + pad_x) as usize;
        let yi = (y + pad_y) as usize;
        let i = yi * 640 + xi;
        input_slice[0 * plane + i] = r as f32 / 255.0;
        input_slice[1 * plane + i] = g as f32 / 255.0;
        input_slice[2 * plane + i] = b as f32 / 255.0;
    }
    let t_pre = t_pre0.elapsed();

    let t_inf0 = Instant::now();
    let result = model.run(tvec!(input.into()))?;
    let t_inf = t_inf0.elapsed();

    let t_post0 = Instant::now();
    // YOLOv8 output: [1, 84, 8400]  (4 bbox + 80 classes)
    let out = result[0].to_array_view::<f32>()?;
    let shape = out.shape().to_vec();
    let num_boxes = shape[2];
    let num_classes = shape[1] - 4;

    let mut candidates: Vec<(f32, usize, [f32; 4])> = Vec::new();
    for b in 0..num_boxes {
        let cx = out[[0, 0, b]];
        let cy = out[[0, 1, b]];
        let w = out[[0, 2, b]];
        let h = out[[0, 3, b]];
        let mut best = 0f32;
        let mut best_c = 0usize;
        for c in 0..num_classes {
            let s = out[[0, 4 + c, b]];
            if s > best { best = s; best_c = c; }
        }
        if best >= conf_th {
            // un-letterbox: subtract pad, divide by scale
            let x1 = (cx - w / 2.0 - pad_x as f32) / scale;
            let y1 = (cy - h / 2.0 - pad_y as f32) / scale;
            let x2 = (cx + w / 2.0 - pad_x as f32) / scale;
            let y2 = (cy + h / 2.0 - pad_y as f32) / scale;
            let x1 = x1.max(0.0).min(orig_w as f32);
            let y1 = y1.max(0.0).min(orig_h as f32);
            let x2 = x2.max(0.0).min(orig_w as f32);
            let y2 = y2.max(0.0).min(orig_h as f32);
            candidates.push((best, best_c, [x1, y1, x2, y2]));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // class-agnostic NMS
    let mut kept: Vec<(f32, usize, [f32; 4])> = Vec::new();
    for cand in candidates {
        let mut suppress = false;
        for k in &kept {
            if k.1 == cand.1 && iou(&k.2, &cand.2) > iou_th {
                suppress = true;
                break;
            }
        }
        if !suppress { kept.push(cand); }
    }
    let t_post = t_post0.elapsed();
    let t_total = t_start.elapsed();

    // Stable machine-readable format: "CLS_ID\tCLASS_NAME\tSCORE\tX1\tY1\tX2\tY2"
    for (score, cls, b) in &kept {
        println!("{}\t{}\t{:.3}\t{:.1}\t{:.1}\t{:.1}\t{:.1}",
            cls, COCO.get(*cls).copied().unwrap_or("?"), score, b[0], b[1], b[2], b[3]);
    }
    eprintln!(
        "detections={}  load={:.1}ms  pre={:.1}ms  infer={:.1}ms  post={:.1}ms  total={:.1}ms",
        kept.len(),
        t_load.as_secs_f64() * 1000.0,
        t_pre.as_secs_f64() * 1000.0,
        t_inf.as_secs_f64() * 1000.0,
        t_post.as_secs_f64() * 1000.0,
        t_total.as_secs_f64() * 1000.0,
    );
    Ok(())
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let inter = w * h;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}
