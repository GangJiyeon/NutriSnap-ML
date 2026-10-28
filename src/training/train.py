"""food_dataset.yaml 데이터로 YOLOv8을 파인튜닝한다."""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 음식 이미지 파인튜닝")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "food_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default="yolov8n.pt", help="사전학습 가중치 (경로 또는 ultralytics 모델명)")
    parser.add_argument("--batch", type=int, default=-1, help="배치 크기, -1이면 자동 결정")
    parser.add_argument("--device", default=None, help="예: mps, cpu, 0. 미지정 시 ultralytics가 자동 선택")
    parser.add_argument("--patience", type=int, default=20, help="mAP 개선 없을 때 조기 종료까지의 epoch 수")
    parser.add_argument("--fraction", type=float, default=1.0, help="학습에 사용할 train 데이터 비율 (0~1), 빠른 검증용")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="food_yolov8")
    parser.add_argument("--output-weights", type=Path, default=REPO_ROOT / "models" / "food_yolov8.pt")
    parser.add_argument("--resume", action="store_true", help="project/name의 last.pt에서 중단된 학습을 이어서 진행")
    args = parser.parse_args()
    device = args.device or resolve_device()
    # ultralytics는 상대 project 경로를 자체 runs_dir 설정 기준으로 또 붙여서 해석하므로 절대경로로 고정
    args.project = str(Path(args.project).resolve())

    if args.resume:
        # 에폭/옵티마이저 상태와 학습 설정은 체크포인트에 저장된 값을 그대로 사용한다
        last_weights = Path(args.project) / args.name / "weights" / "last.pt"
        YOLO(str(last_weights)).train(resume=True)
    else:
        model = YOLO(args.weights)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            patience=args.patience,
            fraction=args.fraction,
            project=args.project,
            name=args.name,
        )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.output_weights)
    print(f"최고 성능 가중치를 {args.output_weights}에 저장했습니다 (원본: {best_weights})")


if __name__ == "__main__":
    main()
