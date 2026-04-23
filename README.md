# yolo-portable-cli

A cross-platform, single-binary YOLO ONNX detection CLI written in Rust on top of the pure-Rust
[`tract`](https://github.com/sonos/tract) runtime.

- **No C/C++ runtime dependency.** Just `cargo build`. No `libonnxruntime.so`, no `libncnn.dylib`, no vendor runtime to ship alongside.
- **Two build flavors** — pick per use case:
  - `yolo-cli` (free): loads an ONNX file at runtime via `--model`.
  - `yolo-cli-bundled`: the ONNX (graph + weights) is embedded at compile time via `include_bytes!`, producing a truly self-contained executable with no `--model` flag.
- **Universal YOLO output decoder.** Auto-detects YOLOv5/v7 (anchors-major, with objectness), YOLOv8/v9/v10/v11/v12 (features-major, class-only), and post-NMS / end2end (`[K,6]`) layouts.
- **Labels read from ONNX metadata.** Ultralytics embeds `names` in the ONNX `metadata_props`; the CLI parses it at load time. No hard-coded COCO list, no `--labels` flag.
- **Correctness validated in CI** against Ultralytics' own `.predict()` across **8 platform/arch combinations × 6 YOLO versions = 48 jobs**.

## CI matrix — 48 combinations

### Platforms (minimum-version long-term-free runners)

| OS        | Arch    | Runner              | Target triple                     | Linking          |
|-----------|---------|---------------------|-----------------------------------|------------------|
| Linux     | x86_64  | `ubuntu-22.04`      | `x86_64-unknown-linux-gnu`        | glibc dyn        |
| Linux     | x86_64  | `ubuntu-22.04`      | `x86_64-unknown-linux-musl`       | musl static      |
| Linux     | aarch64 | `ubuntu-22.04-arm`  | `aarch64-unknown-linux-gnu`       | glibc dyn        |
| Linux     | aarch64 | `ubuntu-22.04-arm`  | `aarch64-unknown-linux-musl`      | musl static      |
| macOS 15  | x86_64  | `macos-15-intel`    | `x86_64-apple-darwin`             | system libs      |
| macOS 14  | aarch64 | `macos-14`          | `aarch64-apple-darwin`            | system libs      |
| Windows   | x86_64  | `windows-2022`      | `x86_64-pc-windows-msvc`          | `+crt-static`    |
| Windows   | aarch64 | `windows-11-arm`    | `aarch64-pc-windows-gnullvm`      | llvm-mingw + static |

Notes:
- Linux is built **twice**: once against glibc (Ubuntu/Debian/Fedora deployment target) and once against musl (Alpine / any-distro static deployment). The musl build has no `.so` dependency.
- macOS Intel uses `macos-15-intel` because `macos-13` was retired 2025-12 and `macos-15-intel` is the last free Intel macOS image (Apple is ending Intel support around Fall 2027).
- Windows ARM64 uses the `pc-windows-gnullvm` triple rather than `pc-windows-msvc`, because tract-linalg's ARM64 assembly is in GAS syntax which MSVC's `armasm64.exe` cannot parse. `llvm-mingw` is downloaded at CI time to provide the LLVM toolchain (`aarch64-w64-mingw32-clang` + `lld`).

### Model versions exported by Ultralytics 8.4.x (v5 through v12)

`yolov5nu`, `yolov8n`, `yolov9t`, `yolov10n`, `yolo11n`, `yolo12n`

All exported with `yolo export format=onnx opset=12 simplify=True imgsz=640`.

Each (platform × model) cell in CI:
1. Builds **both** `yolo-cli` and `yolo-cli-bundled`. `yolo-cli-bundled`'s build.rs reads `YOLO_MODEL_ONNX=../models/<matrix.model>.onnx` and bakes that specific ONNX into the binary via `include_bytes!`.
2. Runs both binaries against three COCO fixture images (`bus.jpg`, `zidane.jpg`, `dog.jpg`) with `--json` and diffs against `tests/reference/<model>/<image>.json`.
3. Uploads both binaries as an artifact named `yolo-cli-<platform>-<model>`.

**Match criteria** per detection (determined empirically from the tract-vs-ORT fp32 drift observed across v5-v12): same class, IoU ≥ 0.95, |Δscore| ≤ 0.05; threshold-edge detections (score < 0.30) are allowed to be absent or extra on one side.

## Build

```bash
# Free variant (runtime --model) — same binary works for any Ultralytics-exported ONNX.
cargo build --release --manifest-path rust/Cargo.toml --bin yolo-cli

# Bundled variant with a specific model baked in.
YOLO_MODEL_ONNX=../models/yolov8n.onnx \
  cargo build --release --manifest-path rust/Cargo.toml --bin yolo-cli-bundled

# Truly portable Linux static (works on Alpine, CentOS 7, anything).
rustup target add x86_64-unknown-linux-musl
cargo build --release --manifest-path rust/Cargo.toml --target x86_64-unknown-linux-musl

# Windows with static CRT (no VC++ redistributable needed).
rustup target add x86_64-pc-windows-msvc
cargo build --release --manifest-path rust/Cargo.toml --target x86_64-pc-windows-msvc
```

## Run

```bash
# Free
yolo-cli --model MODEL.onnx [--conf 0.25] [--iou 0.45] [--json] IMG [IMG ...]

# Bundled (model is embedded at compile time; no --model flag)
yolo-cli-bundled             [--conf 0.25] [--iou 0.45] [--json] IMG [IMG ...]
```

### Pretty (default) — one aligned table per label per image

```
== assets/bus.jpg ==

bus (1 detection):
  score    x0     y0     x1     y1
  -----  ----  -----  -----  -----
  0.836  31.4  231.2  800.8  777.6

person (4 detections):
  score     x0     y0     x1     y1
  -----  -----  -----  -----  -----
  0.895  671.0  384.5  810.0  879.9
  0.882  221.5  407.2  343.8  856.2
  0.874   50.3  397.7  244.4  905.4
  0.423    0.5  548.9   59.4  868.3
```

### Structured — `--json`

```json
[{"image":"assets/bus.jpg","detections":[
  {"label":"person","score":0.8955,"bbox":[671.03,384.54,810.00,879.87]},
  {"label":"person","score":0.8819,"bbox":[221.49,407.21,343.78,856.19]},
  {"label":"person","score":0.8738,"bbox":[50.25,397.71,244.43,905.37]},
  {"label":"bus","score":0.8358,"bbox":[31.37,231.17,800.81,777.62]},
  {"label":"person","score":0.4229,"bbox":[0.48,548.93,59.42,868.27]}
]}]
```

## Output-shape auto-detection

The binary reads the ONNX output fact and picks a decoder automatically:

| Shape                   | Layout                                            | Applies to                         |
|-------------------------|---------------------------------------------------|------------------------------------|
| `[1, 4+nc, N]`          | features-major, no objectness                     | YOLOv8 / v9 / v10 / v11 / v12 raw  |
| `[1, N, 5+nc]`          | anchors-major, `[cx,cy,w,h,obj, cls..]`           | YOLOv5 / v7 raw (original repo)    |
| `[1, K, 6]`             | `[x1,y1,x2,y2,score,cls]` (already NMS'd)         | YOLOv10 end2end, `nms=True` export |

`nc` is read from the `names` metadata dict (Ultralytics embeds it as a
Python-dict repr: `{0: 'person', 1: 'bicycle', ...}`). If the model lacks
metadata, `nc` is inferred from the output shape and labels fall back to
`class_<id>`.

## Export a model

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12 simplify=True imgsz=640
# → yolov8n.onnx (12.3 MB)
```

## License

MIT.
