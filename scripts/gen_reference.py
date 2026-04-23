"""Generate reference detections (JSON) for each (model, image) pair.

Do NOT run in CI — this needs Ultralytics + PyTorch. Run locally whenever the
ONNX models change, then commit tests/reference/<model>/<image>.json.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["yolov5nu", "yolov8n", "yolov9t", "yolov10n", "yolo11n", "yolo12n"]
IMAGES = ["bus.jpg", "zidane.jpg", "dog.jpg"]


def main() -> None:
    target_models = sys.argv[1:] or MODELS
    for mid in target_models:
        out_dir = ROOT / "tests" / "reference" / mid
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx = ROOT / "models" / f"{mid}.onnx"
        model = YOLO(str(onnx))
        for name in IMAGES:
            img = ROOT / "assets" / name
            r = model.predict(str(img), conf=0.25, iou=0.45, verbose=False, imgsz=640)[0]
            dets = []
            for box in r.boxes:
                cls = int(box.cls.item())
                dets.append({
                    "cls": cls,
                    "label": r.names[cls],
                    "score": round(float(box.conf.item()), 4),
                    "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                })
            dets.sort(key=lambda d: (-d["score"], d["cls"]))
            out = out_dir / f"{Path(name).stem}.json"
            out.write_text(json.dumps(dets, indent=2))
            print(f"{mid} / {name}: {len(dets)} detections -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
