# AGENTS.md

> This file is the project guide for agents working on `yolo-portable-cli`
> (Claude, Codex, Devin, Cursor, etc.). `CLAUDE.md` at the repo root is a
> **symlink to this file** — editing either one edits both. Do not create a
> separate `CLAUDE.md`; if it ever drifts into a standalone file, consolidate
> back into `AGENTS.md` and re-create the symlink with `ln -sf AGENTS.md CLAUDE.md`.

## What this project is

A pure-Rust YOLO detection CLI that ships as a **single, self-contained
executable** (no `.so` / `.dll` / `.dylib` to carry). Two variants built from
the same library:

- `yolo-cli` (free): loads any Ultralytics-exported ONNX at runtime via `--model`.
- `yolo-cli-bundled`: ONNX embedded at compile time via `include_bytes!`;
  no runtime model file needed. Gated behind the `bundled` Cargo feature.

Everything runs through [`tract`](https://github.com/sonos/tract) as the
pure-Rust ONNX runtime. There is deliberately no dependency on
`onnxruntime`/`ort`/`ncnn`/etc. — doing so would make single-file distribution
impossible (or at least add a 30-MB C++ `.so` to every ship).

## Hard input contract — do not loosen

The CLI expects **Ultralytics-exported ONNX only** (`yolo export format=onnx`
on a v5u/v8/v9/v10/v11/v12 detection model). Two concrete promises we rely on:

1. `metadata_props` carries `names`, `imgsz`, `task`, `stride`, `end2end` entries.
2. The single output tensor has one of three shapes (see `src/lib.rs`
   `OutputFormat`):

   | Shape           | Layout                                          | Applies to           |
   |-----------------|-------------------------------------------------|----------------------|
   | `[1, 4+nc, N]`  | features-major, no objectness                   | v8 / v9 / v10 / v11 / v12 raw |
   | `[1, N, 5+nc]`  | anchors-major, `[cx,cy,w,h,obj, cls..]`         | v5 / v7 raw          |
   | `[1, K, 6]`     | `[x1,y1,x2,y2,score,cls]` (already NMS'd)       | v10 end2end, `nms=True` |

If an input doesn't fit these contracts, the CLI should error clearly — don't
add silent fallbacks that produce garbage detections. Non-detection heads
(`segment`, `pose`, `classify`), RT-DETR/DETR, and non-Ultralytics converters
(keras2onnx etc.) are **explicitly out of scope**.

## Layout

```
rust/                         — Cargo crate
  Cargo.toml                  — features `bundled` gates the second binary
  .cargo/config.toml          — +crt-static for musl+windows targets
  build.rs                    — no-op unless CARGO_FEATURE_BUNDLED; then embeds YOLO_MODEL_ONNX
  src/
    lib.rs                    — all inference logic: metadata parse, output-shape
                                 auto-detect, letterbox preprocess, class-aware NMS,
                                 emit_info / emit_tables / emit_json, load_model_from_*
    bin/
      yolo-cli.rs             — free variant entrypoint
      yolo-cli-bundled.rs     — bundled variant entrypoint; include!s OUT_DIR/embedded_model.rs

models/                       — committed ONNX files (12 MB each) for 6 YOLO versions
                                 — yolov5nu, yolov8n, yolov9t, yolov10n, yolo11n, yolo12n
                                 — .pt files are gitignored; regenerate with `yolo export`
assets/                       — bus.jpg, zidane.jpg, dog.jpg
tests/reference/<model>/<image>.json  — ground-truth detections from Ultralytics+ORT
scripts/
  gen_reference.py            — (host tool) regenerate tests/reference/; needs ultralytics
  verify.py                   — (CI tool) stdlib-only; runs binary, diffs against reference
.github/workflows/
  ci.yml                      — 152-job PR/push matrix: 8 platforms × (free build+verify, bundled build+verify per model)
  release.yml                 — triggered on release:published; builds+verifies free for 8 platforms and uploads to the release
```

## Supported CI runners (minimum long-term-free)

See the platform table in `README.md`. Key constraints that have bitten us:

- **macOS Intel**: `macos-13` was retired 2025-12. Use `macos-15-intel`.
  Apple is retiring Intel macOS around Fall 2027; after that, expect x86_64
  Apple builds to require self-hosted runners.
- **Windows ARM64**: `tract-linalg` emits GAS-flavored ARM64 assembly, which
  MSVC's `armasm64` cannot parse. Use target `aarch64-pc-windows-gnullvm` and
  install `llvm-mingw` at CI time (see both workflows).
- **Linux musl static**: `sudo apt-get install -y musl-tools` gives a working
  host-arch `musl-gcc`. Don't need a cross toolchain because
  `ubuntu-22.04-arm` *is* an arm64 host.
- **glibc floor** (if you ever ship a dynamic-linked Linux build): binaries
  built on ubuntu-22.04 demand glibc ≥ 2.35. If broader distro reach matters,
  use the musl target (already shipped per-platform in CI).

## Build recipes

```bash
# Free (no env vars)
cargo build --release --manifest-path rust/Cargo.toml --bin yolo-cli

# Bundled — needs --features bundled + YOLO_MODEL_ONNX (env var is
# ignored for the free build; don't cargo-check errors if you only build free)
YOLO_MODEL_ONNX=../models/yolov8n.onnx \
  cargo build --release --manifest-path rust/Cargo.toml \
  --features bundled --bin yolo-cli-bundled

# Both at once
YOLO_MODEL_ONNX=../models/yolov8n.onnx \
  cargo build --release --manifest-path rust/Cargo.toml \
  --features bundled --bins

# Linux musl static (works on Alpine / CentOS 7 / anywhere)
rustup target add x86_64-unknown-linux-musl
sudo apt-get install -y musl-tools
cargo build --release --manifest-path rust/Cargo.toml \
    --target x86_64-unknown-linux-musl --bin yolo-cli
```

## Local correctness check

```bash
# Build free debug quickly
cargo build --bin yolo-cli --manifest-path rust/Cargo.toml

# Against every committed YOLO
for m in yolov5nu yolov8n yolov9t yolov10n yolo11n yolo12n; do
  python3 scripts/verify.py \
    rust/target/debug/yolo-cli $m \
    assets/bus.jpg assets/zidane.jpg assets/dog.jpg
done
```

Match criteria in `scripts/verify.py`:
- same class
- IoU ≥ 0.95
- `|Δscore| ≤ 0.05`
- threshold-edge detections (`score < 0.30`) may be absent/extra on either side
  — this is pure fp32 accumulation drift between tract and ORT, not a real
  regression. If you ever see this tolerance being exercised on high-score
  detections, investigate rather than widen further.

## Regenerating references

Only needed if you re-export a model (ONNX bytes change) or add new images.
Requires Ultralytics + PyTorch in a venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics onnx onnxsim
python3 scripts/gen_reference.py            # regenerates all (model × image)
python3 scripts/gen_reference.py yolov8n    # subset by model id
```

Then `git add tests/reference/` and commit.

## Invariants you must preserve

1. **Pure-Rust inference.** Do not introduce deps that bind to
   `onnxruntime`/`ncnn`/`mnn`/`openvino`/etc. That defeats the single-binary
   story. Also don't split runtime loading into multiple files.
2. **No runtime file I/O for models in `yolo-cli-bundled`.** The ONNX bytes
   live in `.rodata`; no `load_from_path` shortcut should sneak in.
3. **Metadata-driven preprocessing.** `imgsz` comes from
   `ModelProto.metadata_props` (fallback 640×640). Letterbox with
   114-value padding is required; a naive resize will silently produce
   wrong bboxes on non-square inputs.
4. **Don't embed NMS into the ONNX.** `yolo export ... nms=True` pulls in
   ops that tract's coverage is weaker on; NMS in Rust is ~5 ms and trivial
   to audit. Keep it in `src/lib.rs::nms`.
5. **Dual-binary, one library.** Keep all decoder logic in `src/lib.rs` so
   the two bins can't diverge.
6. **Output contract stable.** `--json` emits
   `[{"image":..., "detections":[{"label":..., "score":..., "bbox":[x0,y0,x1,y1]}]}]`.
   Scripts downstream (ours and others') rely on this; don't rename fields or
   unwrap the outer list.
7. **tract version pinning.** Cargo.lock is committed for reproducibility.
   When bumping tract, re-run the full CI matrix; tract-linalg interactions
   with MSVC/gnullvm/musl are the most likely to break.

## Things that will trip new contributors

- **YOLOv8 output is `[1, 4+nc, N]` not the v5-style `[1, N, 5+nc]`.** The
  first 4 rows are `cx, cy, w, h`; class logits start at row 4. No
  objectness score. If you wire up a new decoder, pay attention.
- **YOLOv10 is `end2end=True` by default in Ultralytics exports** —
  metadata key `end2end: 'True'`, output shape `[1, 300, 6]` already-NMS'd.
  `src/lib.rs::detect_format` routes it to `PostNms`. Don't re-run NMS.
- **The `names` metadata value is Python dict repr**, e.g.
  `{0: 'person', 1: 'bicycle', ..., 9: 'traffic light'}`. Some class names
  have spaces; `parse_names()` is a tiny state machine — don't swap it for
  a naive `split(',').split(':')` that would break on `'traffic light'`.
- **Build-time env var is tied to the Cargo feature.** build.rs exits
  without producing `embedded_model.rs` unless `CARGO_FEATURE_BUNDLED` is
  set. If you try to build the bundled binary without the feature, cargo
  skips it silently (via `required-features`) — that's correct behaviour,
  not a bug.
- **GitHub Actions sparse-checkout in `bundled-verify`** is deliberate —
  it proves the binary doesn't secretly need `models/`. If you find
  yourself wanting to un-sparse it, you've probably regressed the self-
  containment property; fix the binary, not the checkout.

## Release process

Versions live in `rust/Cargo.toml`. To cut a release (e.g., v0.0.2):

1. Bump `rust/Cargo.toml` → `version = "0.0.2"`; `cargo generate-lockfile`.
2. Commit + push.
3. `git tag v0.0.2 && git push origin v0.0.2`.
4. `gh release create v0.0.2 --title "v0.0.2" --notes "..."`.
5. This fires `.github/workflows/release.yml` (8 platform build + verify +
   upload). Watch with `gh run watch` or the Actions tab. Every platform
   must upload its archive before the release is considered done.
6. Confirm
   `https://github.com/HansBug/yolo-portable-cli/releases/latest/download/yolo-cli-<platform>.(tar.gz|zip)`
   returns 200 for all 8 platform strings.

If a platform fails: fix the underlying issue, `gh release delete v0.0.2`,
`git tag -d v0.0.2 && git push --delete origin v0.0.2`, and redo steps 1-5
with the same (or next) patch.

## Quick-start for a new agent session

```bash
# Read the entry points first
cat AGENTS.md                         # this file — invariants + layout
cat README.md                         # user-facing contract
cat rust/src/lib.rs | head -80        # type declarations and detect_format
cat .github/workflows/ci.yml          # what CI enforces

# Smoke test
cargo build --bin yolo-cli --manifest-path rust/Cargo.toml
python3 scripts/verify.py rust/target/debug/yolo-cli yolov8n \
    assets/bus.jpg assets/zidane.jpg assets/dog.jpg
```

If that prints three `[PASS]` lines, the tree is healthy.
