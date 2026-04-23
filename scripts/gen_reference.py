"""Run Ultralytics once to produce reference detections (JSON) for CI to diff against.

Do NOT run this in CI - it requires the full ultralytics+torch stack.
Run locally whenever you regenerate the model or images, then commit the JSON.
"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolov8n.onnx"
IMAGES = ["bus.jpg", "zidane.jpg", "dog.jpg"]
OUT = ROOT / "tests" / "reference"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    for name in IMAGES:
        img = ROOT / "assets" / name
        r = model.predict(str(img), conf=0.25, iou=0.45, verbose=False, imgsz=640)[0]
        dets = []
        for box in r.boxes:
            cls = int(box.cls.item())
            dets.append({
                "cls": cls,
                "name": r.names[cls],
                "conf": round(float(box.conf.item()), 4),
                "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            })
        dets.sort(key=lambda d: (-d["conf"], d["cls"]))
        out = OUT / (Path(name).stem + ".json")
        out.write_text(json.dumps(dets, indent=2))
        print(f"{img.name}: {len(dets)} detections -> {out}")


if __name__ == "__main__":
    main()
