"""Verify yolo-cli / yolo-cli-bundled output against committed reference detections.

Runs on any OS with just Python 3 stdlib.

Usage:
    python scripts/verify.py <binary> <model_id> <image1> [image2 ...]

<model_id> is e.g. "yolov8n".  The reference file for (model, image) is read
from tests/reference/<model_id>/<image_stem>.json.  If the binary filename
contains "bundled", no --model flag is passed (the binary has the ONNX baked
in); otherwise --model models/<model_id>.onnx is passed.

Exits 0 on PASS, non-zero on FAIL.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Tolerances chosen from empirical tract-vs-ort drift across v5/v8/v9/v10/v11/v12.
# Near-threshold detections may appear in one runtime and not the other because
# of fp32 accumulation-order drift; treat mismatches below NEAR_THRESH as noise.
IOU_MIN = 0.95
SCORE_TOL = 0.05
NEAR_THRESH = 0.30  # drop from "unmatched" accounting if score < this value


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def compare(ref, got):
    used = set()
    pairs = []
    worst_iou = 1.0
    worst_dconf = 0.0
    for i, r in enumerate(ref):
        best_j, best_iou = -1, -1.0
        for j, g in enumerate(got):
            if j in used or g["label"] != r["label"]:
                continue
            io = iou(r["bbox"], g["bbox"])
            if io > best_iou:
                best_iou, best_j = io, j
        if (best_j >= 0 and best_iou >= IOU_MIN and
                abs(got[best_j]["score"] - r["score"]) <= SCORE_TOL):
            pairs.append((i, best_j, best_iou))
            used.add(best_j)
            worst_iou = min(worst_iou, best_iou)
            worst_dconf = max(worst_dconf, abs(got[best_j]["score"] - r["score"]))
    um_ref_all = [i for i in range(len(ref)) if i not in {p[0] for p in pairs}]
    um_got_all = [j for j in range(len(got)) if j not in used]
    # Ignore near-threshold unmatched detections — these are threshold-edge
    # numerical cases, not model behaviour regressions.
    um_ref = [i for i in um_ref_all if ref[i]["score"] >= NEAR_THRESH]
    um_got = [j for j in um_got_all if got[j]["score"] >= NEAR_THRESH]
    ok = not um_ref and not um_got
    return ok, pairs, um_ref, um_got, worst_iou, worst_dconf


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <binary> <model_id> <image> [image ...]", file=sys.stderr)
        sys.exit(2)
    binary = sys.argv[1]
    model_id = sys.argv[2]
    images = sys.argv[3:]

    is_bundled = "bundled" in Path(binary).name
    cmd = [binary]
    if not is_bundled:
        cmd += ["--model", str(ROOT / "models" / f"{model_id}.onnx")]
    cmd += ["--conf", "0.25", "--iou", "0.45", "--json", *[str(p) for p in images]]

    print(f"exec: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] binary exited {proc.returncode}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)
    per_image = json.loads(proc.stdout)
    got_by_stem = {Path(e["image"]).stem: e["detections"] for e in per_image}

    ref_dir = ROOT / "tests" / "reference" / model_id
    all_ok = True
    for img_path in images:
        stem = Path(img_path).stem
        ref_file = ref_dir / f"{stem}.json"
        if not ref_file.exists():
            print(f"[FAIL] no reference for {model_id}/{stem} at {ref_file}", file=sys.stderr)
            all_ok = False
            continue
        ref = json.loads(ref_file.read_text())
        got = got_by_stem.get(stem, [])
        ok, pairs, ur, ug, w_iou, w_dconf = compare(ref, got)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {model_id}/{stem}: {len(pairs)}/{len(ref)} matched, "
              f"worst IoU={w_iou:.4f}, worst |dconf|={w_dconf:.4f}, "
              f"unmatched ref={len(ur)}, unmatched got={len(ug)}")
        if not ok:
            print("  --- reference ---", file=sys.stderr)
            for r in ref:
                print(f"  {r['label']:<14} {r['score']:.3f}  {r['bbox']}", file=sys.stderr)
            print("  --- actual ---", file=sys.stderr)
            for g in got:
                print(f"  {g['label']:<14} {g['score']:.3f}  {g['bbox']}", file=sys.stderr)
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
