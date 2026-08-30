"""학습된 YOLOv8 가중치(.pt)를 ONNX 포맷으로 변환한다."""
import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 가중치를 ONNX로 변환")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "models" / "food_yolov8.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    exported_path = model.export(format="onnx", imgsz=args.imgsz)
    print(f"ONNX 모델 저장: {exported_path}")


if __name__ == "__main__":
    main()
