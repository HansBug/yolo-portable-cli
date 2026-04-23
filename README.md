# yolo-portable-cli

A cross-platform, single-binary YOLOv8 ONNX inference CLI written in Rust on top of the pure-Rust
[`tract`](https://github.com/sonos/tract) runtime.

- **No C/C++ runtime dependency.** Just `cargo build`. No `libonnxruntime.so` or `libncnn.dylib` to ship.
- **Truly portable single executable.** Linux (musl static), macOS (universal-ready), Windows (+crt-static) — one file per platform.
- **Fast cold start.** < 10 ms from `exec` to `main` (the model-load cost is the only non-trivial part, ~400 ms for yolov8n on CPU).
- **Correctness validated in CI** against Ultralytics' own `.predict()` on three COCO test images, across six platform/arch combinations.

## Platforms in CI

| OS               | Arch    | Runner               | Target triple                     | Linking       |
|------------------|---------|----------------------|-----------------------------------|---------------|
| Linux            | x86_64  | `ubuntu-22.04`       | `x86_64-unknown-linux-musl`       | static (musl) |
| Linux            | aarch64 | `ubuntu-22.04-arm`   | `aarch64-unknown-linux-musl`      | static (musl) |
| macOS (Intel)    | x86_64  | `macos-13`           | `x86_64-apple-darwin`             | system libs   |
| macOS (Apple Si) | aarch64 | `macos-14`           | `aarch64-apple-darwin`            | system libs   |
| Windows          | x86_64  | `windows-2022`       | `x86_64-pc-windows-msvc`          | `+crt-static` |
| Windows          | aarch64 | `windows-11-arm`     | `aarch64-pc-windows-msvc`         | `+crt-static` |

Every CI job builds the binary, runs it against three fixture images (`bus.jpg`, `zidane.jpg`, `dog.jpg`),
and compares detections against reference JSON generated once by Ultralytics' own ONNX Runtime path.
Match criteria: **all reference boxes found, same class, IoU ≥ 0.95, |score Δ| ≤ 0.04**.

## Build

```bash
# Native host
cargo build --release --manifest-path rust/Cargo.toml

# Linux musl static (zero glibc dep)
rustup target add x86_64-unknown-linux-musl
cargo build --release --manifest-path rust/Cargo.toml --target x86_64-unknown-linux-musl

# Windows with static CRT
cargo build --release --manifest-path rust/Cargo.toml --target x86_64-pc-windows-msvc
```

## Run

```bash
./rust/target/release/yolo-cli models/yolov8n.onnx assets/bus.jpg
# 0 person 0.890 670.4 380.6 809.9 879.7
# 0 person 0.883 221.7 407.4 343.8 856.2
# ...
# stderr: detections=5  load=463ms  pre=67ms  infer=880ms  post=5ms  total=1421ms
```

Output format per line (tab-separated): `cls_id  class_name  score  x1  y1  x2  y2`.

## Verify correctness

```bash
python3 scripts/verify.py rust/target/release/yolo-cli models/yolov8n.onnx \
    assets/bus.jpg assets/zidane.jpg assets/dog.jpg
```

Uses only the Python stdlib; reference detections are committed under `tests/reference/`.

## Why ONNX + tract and not alternatives?

| Backend | Pure? | Needs to ship? | Fits "single portable binary"? |
|--|--|--|--|
| **tract** (used here) | pure Rust | nothing | **yes** |
| ort / onnxruntime-c | C++ | `libonnxruntime.{so,dll,dylib}` ~30 MB | no (multi-file) or binary grows to 40+ MB |
| ncnn, MNN | C++ | `libncnn`, etc. | no (multi-file); ncnn also wants Vulkan/protobuf at build |
| TensorRT / OpenVINO / CoreML | vendor | full vendor SDK | no (vendor-locked) |

## Model export (one-time, needs Ultralytics)

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12 simplify=True imgsz=640
# → yolov8n.onnx (12.3 MB)
```

Key export flags:
- `dynamic=False` (default): fixed `[1,3,640,640]` input → tract can fully specialize
- `simplify=True`: remove constant-folded graph dead weight
- **Do not pass `nms=True`**: NMS is implemented in Rust (≈ 5 ms); embedding it in ONNX pulls in ops tract handles less well.

## Preprocessing notes

YOLOv8 expects **letterbox** preprocessing (preserve aspect ratio + 114-value padding), and outputs a
`[1, 84, 8400]` tensor laid out as `[cx, cy, w, h, cls0, cls1, ..., cls79]` (no objectness score).
Both are implemented in `rust/src/main.rs`.

## License

MIT.
