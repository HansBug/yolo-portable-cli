"""Verify a Rust CLI binary's output against committed reference detections.

Runs on any OS with just Python 3 stdlib. No ultralytics/torch needed.

Usage: python scripts/verify.py <binary> <model> <image1> [image2 ...]

For each image, compares binary stdout against tests/reference/<stem>.json.
Exits 0 on PASS, non-zero on FAIL.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "tests" / "reference"

# Tolerances chosen from empirical tract-vs-ort drift on the three test images:
# - IoU >= 0.95 (allows ~5% box-edge wobble from resize filter differences)
# - |score delta| <= 0.04 (allows fp32 accumulation-order drift and softmax-edge jitter)
IOU_MIN = 0.95
SCORE_TOL = 0.04


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def parse_rust(stdout):
    dets = []
    for line in stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        try:
            cls = int(parts[0])
            name = parts[1]
            conf = float(parts[2])
            xyxy = tuple(float(v) for v in parts[3:7])
        except ValueError:
            continue
        dets.append({"cls": cls, "name": name, "conf": conf, "xyxy": xyxy})
    return dets


def compare(ref, got):
    """Return (ok, pairs, unmatched_ref, unmatched_got, worst_iou, worst_dconf)."""
    used = set()
    pairs = []
    worst_iou = 1.0
    worst_dconf = 0.0
    for i, r in enumerate(ref):
        best_j, best_iou = -1, -1.0
        for j, g in enumerate(got):
            if j in used or g["cls"] != r["cls"]:
                continue
            io = iou(r["xyxy"], g["xyxy"])
            if io > best_iou:
                best_iou, best_j = io, j
        if (best_j >= 0 and best_iou >= IOU_MIN and
                abs(got[best_j]["conf"] - r["conf"]) <= SCORE_TOL):
            pairs.append((i, best_j, best_iou))
            used.add(best_j)
            worst_iou = min(worst_iou, best_iou)
            worst_dconf = max(worst_dconf, abs(got[best_j]["conf"] - r["conf"]))
    um_ref = [i for i in range(len(ref)) if i not in {p[0] for p in pairs}]
    um_got = [j for j in range(len(got)) if j not in used]
    ok = not um_ref and not um_got
    return ok, pairs, um_ref, um_got, worst_iou, worst_dconf


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <binary> <model> <image> [image ...]", file=sys.stderr)
        sys.exit(2)
    binary, model = sys.argv[1], sys.argv[2]
    images = sys.argv[3:]
    all_ok = True
    for img_path in images:
        img = Path(img_path)
        ref_file = REF_DIR / (img.stem + ".json")
        if not ref_file.exists():
            print(f"[FAIL] no reference for {img.name} at {ref_file}", file=sys.stderr)
            all_ok = False
            continue
        ref = json.loads(ref_file.read_text())
        for r in ref:
            r["xyxy"] = tuple(r["xyxy"])
        proc = subprocess.run([binary, model, str(img), "0.25", "0.45"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[FAIL] {img.name}: binary returned {proc.returncode}", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            all_ok = False
            continue
        got = parse_rust(proc.stdout)
        ok, pairs, ur, ug, w_iou, w_dconf = compare(ref, got)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {img.name}: {len(pairs)}/{len(ref)} matched, "
              f"worst IoU={w_iou:.4f}, worst |dconf|={w_dconf:.4f}, "
              f"unmatched ref={len(ur)}, unmatched got={len(ug)}")
        if not ok:
            print("  --- reference ---", file=sys.stderr)
            for r in ref:
                print(f"  {r['name']:<14} {r['conf']:.3f}  {r['xyxy']}", file=sys.stderr)
            print("  --- actual ---", file=sys.stderr)
            for g in got:
                print(f"  {g['name']:<14} {g['conf']:.3f}  {g['xyxy']}", file=sys.stderr)
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
