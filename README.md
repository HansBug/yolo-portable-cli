# yolo-portable-cli

A cross-platform, single-binary YOLO ONNX detection CLI written in Rust on top
of the pure-Rust [`tract`](https://github.com/sonos/tract) runtime.

## ⚠ Input requirement: the ONNX must be an Ultralytics-exported model

This tool **is not a general ONNX object-detection runner**. It expects the
ONNX file to be produced by the [Ultralytics](https://docs.ultralytics.com/)
framework's `yolo export` command (or the equivalent
`YOLO(...).export(format='onnx')` Python API), i.e. a `yolov5u` / `yolov8` /
`yolov9` / `yolov10` / `yolo11` / `yolo12` detection model. The CLI relies on
two hard contracts:

1. **Metadata present.** Ultralytics embeds `names`, `imgsz`, `task`, `stride`,
   `end2end` etc. in `metadata_props`. The CLI uses these to size the input
   correctly, pick a decoder, and produce human-readable class labels.
   Without them, the binary falls back to `640×640` and `class_<id>` labels
   and behaviour quality depends entirely on whether the fallbacks match your
   model. **Strongly recommend using official Ultralytics exports only.**
2. **One of three supported output layouts:**
   - `[1, 4+nc, N]` — YOLOv8 / v9 / v10 / v11 / v12 raw (features-major, no objectness)
   - `[1, N, 5+nc]` — YOLOv5 / v7 raw (anchors-major, with objectness)
   - `[1, K, 6]` — post-NMS / v10 end2end (`[x1,y1,x2,y2,score,cls]`)

   Anything else (non-detection heads like `segment`/`pose`/`classify`,
   custom multi-output heads, foreign ONNX converters such as keras2onnx,
   non-YOLO detectors like DETR/RT-DETR, quantised int8 graphs with extra
   QuantizeLinear/DequantizeLinear wrappers, etc.) is **not supported** and
   the CLI will refuse with an error.

If your ONNX wasn't produced by `yolo export` on an upstream Ultralytics
model, this tool is not the right fit — look at
[onnxruntime](https://onnxruntime.ai/) or a model-specific CLI instead.

## Features

- **No C/C++ runtime dependency.** Just `cargo build`. No `libonnxruntime.so`,
  no `libncnn.dylib`, no vendor runtime to ship alongside.
- **Two build flavors** sharing a single library:
  - `yolo-cli` (free): loads an ONNX file at runtime via `--model`.
  - `yolo-cli-bundled`: ONNX (graph + weights) embedded at compile time via
    `include_bytes!`, producing a truly self-contained executable with no
    `--model` flag. Only built when the `bundled` Cargo feature is enabled.
- **Universal YOLO output decoder.** Auto-detects the three supported layouts
  listed above, all from a single binary.
- **Labels + imgsz read from ONNX metadata.** No hard-coded COCO list, no
  `--labels` flag; each CI-tested YOLO version drives the table header and
  the JSON output purely from its own `metadata_props`.
- **Correctness validated in CI** against Ultralytics' own `.predict()` across
  **8 platform/arch combinations × 6 YOLO versions**, with both variants
  (free and bundled) built and verified in separate stages.

## Quickstart

```bash
# Grab a prebuilt binary for your platform from the latest CI run's artifacts:
#   https://github.com/HansBug/yolo-portable-cli/actions
#
# Or build yourself — see "Build guide" below.

# Free: loads any YOLO ONNX at runtime.
yolo-cli --model yolov8n.onnx --info                       # inspect the model
yolo-cli --model yolov8n.onnx bus.jpg zidane.jpg           # pretty table per label per image
yolo-cli --model yolov8n.onnx --json --conf 0.3 bus.jpg    # structured JSON

# Bundled: same UX but no --model; the ONNX is baked in.
yolo-cli-bundled --info
yolo-cli-bundled --json bus.jpg zidane.jpg dog.jpg
```

## Build guide

### Prerequisites

| Tool                | Minimum version | Notes                                              |
|---------------------|-----------------|----------------------------------------------------|
| Rust (rustup)       | stable (≥ 1.76) | `rustup install stable`                            |
| (Linux musl target) | —               | `sudo apt-get install musl-tools`                  |
| (Windows ARM64)     | —               | LLVM-mingw, see the Windows ARM64 section          |
| (optional) ONNX     | —               | Only for `yolo-cli-bundled`; pre-exported files live in `models/` |

No Python, no Ultralytics, no ONNX Runtime are required at build time.

### 1. Build only the free variant (any YOLO ONNX at runtime)

The free variant **ignores** `YOLO_MODEL_ONNX` entirely. Its build.rs is a no-op
because the `bundled` feature is off by default.

```bash
# Native host
cargo build --release --manifest-path rust/Cargo.toml --bin yolo-cli
# → rust/target/release/yolo-cli[.exe]

# Explicit target (e.g. Linux musl static, works on Alpine / any distro)
rustup target add x86_64-unknown-linux-musl
sudo apt-get install -y musl-tools                          # host only
cargo build --release --manifest-path rust/Cargo.toml \
    --target x86_64-unknown-linux-musl --bin yolo-cli
```

### 2. Build only the bundled variant (with a specific ONNX baked in)

The bundled binary is gated behind the `bundled` Cargo feature
(`required-features = ["bundled"]`) and reads `YOLO_MODEL_ONNX` at build time.
That env var may be relative to `rust/` (default: `../models/yolov8n.onnx`) or
absolute.

```bash
# Bundle yolov8n (default)
YOLO_MODEL_ONNX=../models/yolov8n.onnx \
    cargo build --release --manifest-path rust/Cargo.toml \
    --features bundled --bin yolo-cli-bundled

# Bundle a different YOLO version
YOLO_MODEL_ONNX=../models/yolo12n.onnx \
    cargo build --release --manifest-path rust/Cargo.toml \
    --features bundled --bin yolo-cli-bundled

# Or an absolute path pointing at any Ultralytics-exported ONNX
YOLO_MODEL_ONNX=/abs/path/to/my_custom_model.onnx \
    cargo build --release --manifest-path rust/Cargo.toml \
    --features bundled --bin yolo-cli-bundled
```

### 3. Build both at once

```bash
YOLO_MODEL_ONNX=../models/yolov8n.onnx \
    cargo build --release --manifest-path rust/Cargo.toml \
    --features bundled --bins
# → rust/target/release/yolo-cli[.exe]          (free, model-agnostic)
# → rust/target/release/yolo-cli-bundled[.exe]  (bundled with yolov8n)
```

### Platform-specific build recipes

#### Linux — glibc dynamic (Ubuntu/Debian/Fedora target)

```bash
cargo build --release --manifest-path rust/Cargo.toml --bin yolo-cli
# Resulting binary requires glibc ≥ 2.35 on Ubuntu 22.04 as-built.
# Build on an older distro (e.g. Ubuntu 20.04) to lower the floor.
```

#### Linux — musl static (runs on Alpine, any Linux distro)

```bash
rustup target add x86_64-unknown-linux-musl
sudo apt-get install -y musl-tools

cargo build --release --manifest-path rust/Cargo.toml \
    --target x86_64-unknown-linux-musl --bin yolo-cli
# Resulting binary has zero .so dependencies, works on any Linux.
```

For aarch64 Linux musl:
```bash
rustup target add aarch64-unknown-linux-musl
sudo apt-get install -y musl-tools     # on an arm64 host, musl-gcc covers aarch64
cargo build --release --manifest-path rust/Cargo.toml \
    --target aarch64-unknown-linux-musl --bin yolo-cli
```

#### macOS — Intel + Apple Silicon

```bash
rustup target add x86_64-apple-darwin aarch64-apple-darwin
cargo build --release --manifest-path rust/Cargo.toml \
    --target x86_64-apple-darwin --bin yolo-cli
cargo build --release --manifest-path rust/Cargo.toml \
    --target aarch64-apple-darwin --bin yolo-cli

# Optional: produce a universal binary
lipo -create -output yolo-cli \
     rust/target/x86_64-apple-darwin/release/yolo-cli \
     rust/target/aarch64-apple-darwin/release/yolo-cli
```

#### Windows — x86_64 MSVC (static CRT, no VC++ redistributable needed)

```bash
rustup target add x86_64-pc-windows-msvc
cargo build --release --manifest-path rust/Cargo.toml \
    --target x86_64-pc-windows-msvc --bin yolo-cli
```

The repo's `rust/.cargo/config.toml` already sets
`target-feature=+crt-static` for Windows MSVC targets, so the produced `.exe`
has no `VCRUNTIME*.dll` dependency.

#### Windows — aarch64 (via gnullvm / LLVM toolchain)

tract-linalg emits GAS-syntax ARM64 assembly, which MSVC's `armasm64` can't
parse.  Use the `pc-windows-gnullvm` triple plus
[mstorsjo/llvm-mingw](https://github.com/mstorsjo/llvm-mingw) for a working
LLVM toolchain (clang + lld):

```powershell
# Download llvm-mingw for the host arch (aarch64 on a Windows-on-ARM machine)
curl.exe -LO https://github.com/mstorsjo/llvm-mingw/releases/download/20250730/llvm-mingw-20250730-ucrt-aarch64.zip
7z x llvm-mingw-*.zip -oC:\llvm-mingw
$env:PATH = "C:\llvm-mingw\llvm-mingw-20250730-ucrt-aarch64\bin;$env:PATH"

rustup target add aarch64-pc-windows-gnullvm
cargo build --release --manifest-path rust/Cargo.toml `
    --target aarch64-pc-windows-gnullvm --bin yolo-cli
```

### Export an ONNX for the bundled build

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12 simplify=True imgsz=640
#                                        ^^^^^^^^^ ^^^^^^^^^^^^^ ^^^^^^^^^^^
#                                        critical  critical      must match
```

Keep `dynamic=False` (default), don't pass `nms=True` — NMS is done by the
Rust CLI in ~5 ms and tract's coverage for NMS-related ops is narrower than
for the dense backbone.

Ultralytics 8.3+ supports v5u, v8, v9, v10, v11, v12 (plus older v3).  The
CLI auto-detects the output layout from the graph + metadata, so a single
`yolo-cli` binary works with any of them.

## Usage

```bash
# Free
yolo-cli --model MODEL.onnx [--conf F] [--iou F] [--json] [--info] IMG [IMG ...]

# Bundled (model baked in; no --model)
yolo-cli-bundled             [--conf F] [--iou F] [--json] [--info] IMG [IMG ...]
```

### Inspecting a model — `--info`

```
$ yolo-cli --model models/yolov8n.onnx --info
source       : models/yolov8n.onnx
size         : 12851087 bytes
version      : 8.4.41
task         : detect
input imgsz  : 640x640  (HxW)
stride       : 32
output shape : [1, 84, 8400]
output format: V8Anchors ([1, 4+nc, N])
end2end      : false
default conf : 0.25 (CLI default)
default iou  : 0.45 (CLI default)
classes (80):
    0: person
    1: bicycle
    ...
   79: toothbrush
```

### Default-threshold resolution

`--conf` and `--iou` resolve in this order:

1. Explicit `--conf F` / `--iou F` on the command line wins.
2. Otherwise, if the ONNX `metadata_props` has a `conf`/`conf_threshold`/`score_threshold`
   or `iou`/`iou_threshold`/`nms_iou_threshold` entry, that value is used.
   Ultralytics' current exports don't embed these; this hook is future-facing.
3. Otherwise, built-in defaults `conf=0.25`, `iou=0.45` (matching Ultralytics'
   Python `predict()` defaults).

Run `--info` to see which source is in effect for the current model.

### Pretty tables (default)

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

### Structured JSON — `--json`

```json
[{"image":"assets/bus.jpg","detections":[
  {"label":"person","score":0.8955,"bbox":[671.03,384.54,810.00,879.87]},
  {"label":"person","score":0.8819,"bbox":[221.49,407.21,343.78,856.19]},
  {"label":"person","score":0.8738,"bbox":[50.25,397.71,244.43,905.37]},
  {"label":"bus","score":0.8358,"bbox":[31.37,231.17,800.81,777.62]},
  {"label":"person","score":0.4229,"bbox":[0.48,548.93,59.42,868.27]}
]}]
```

## CI matrix

The workflow at `.github/workflows/ci.yml` runs four stages, totalling
**8 + 48 + 48 + 48 = 152 jobs**:

| Stage           | Cells             | Purpose                                                             |
|-----------------|-------------------|---------------------------------------------------------------------|
| `free-build`    | 8 (platform)      | Build `yolo-cli` once per platform, upload artifact                 |
| `free-verify`   | 48 (platform×model) | Pull free artifact, run it against each model, diff with reference  |
| `bundled-build` | 48 (platform×model) | Build `yolo-cli-bundled` with `YOLO_MODEL_ONNX=<model>.onnx`, upload |
| `bundled-verify`| 48 (platform×model) | **Clean env** — sparse-checkout (no `models/`), no Rust, just run binary + verify |

### Platforms — minimum long-term-free runners

| OS        | Arch    | Runner              | Target triple                     | Linking              |
|-----------|---------|---------------------|-----------------------------------|----------------------|
| Linux     | x86_64  | `ubuntu-22.04`      | `x86_64-unknown-linux-gnu`        | glibc dynamic        |
| Linux     | x86_64  | `ubuntu-22.04`      | `x86_64-unknown-linux-musl`       | musl static          |
| Linux     | aarch64 | `ubuntu-22.04-arm`  | `aarch64-unknown-linux-gnu`       | glibc dynamic        |
| Linux     | aarch64 | `ubuntu-22.04-arm`  | `aarch64-unknown-linux-musl`      | musl static          |
| macOS 15  | x86_64  | `macos-15-intel`    | `x86_64-apple-darwin`             | system libs          |
| macOS 14  | aarch64 | `macos-14`          | `aarch64-apple-darwin`            | system libs          |
| Windows   | x86_64  | `windows-2022`      | `x86_64-pc-windows-msvc`          | `+crt-static`        |
| Windows   | aarch64 | `windows-11-arm`    | `aarch64-pc-windows-gnullvm`      | llvm-mingw + static  |

### Models in the matrix

`yolov5nu`, `yolov8n`, `yolov9t`, `yolov10n`, `yolo11n`, `yolo12n` — all exported
with `yolo export format=onnx opset=12 simplify=True imgsz=640`.

### Match criteria

The `bundled-verify` stage runs with **no Rust, no cargo, no `models/` directory**
(sparse checkout excludes it) — proving each binary is 100% self-contained.

Reference detections in `tests/reference/<model>/<image>.json` come from
Ultralytics' own ONNX Runtime path.  Match criteria: same class, IoU ≥ 0.95,
|Δscore| ≤ 0.05.  Threshold-edge detections (score < 0.30) are allowed to
be absent/extra on one side — those are numerical fp32 accumulation drifts,
not model-behaviour regressions.

## Output-shape auto-detection

| Shape           | Layout                                        | Applies to                         |
|-----------------|-----------------------------------------------|------------------------------------|
| `[1, 4+nc, N]`  | features-major, no objectness                 | YOLOv8 / v9 / v10 / v11 / v12 raw  |
| `[1, N, 5+nc]`  | anchors-major, `[cx,cy,w,h,obj, cls..]`       | YOLOv5 / v7 raw (original repo)    |
| `[1, K, 6]`     | `[x1,y1,x2,y2,score,cls]` (already NMS'd)     | YOLOv10 end2end, `nms=True` export |

Detected automatically from the optimised graph's output fact + the `end2end`
metadata hint.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
