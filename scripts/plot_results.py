"""README용 학습 곡선, 클래스별 mAP 그래프 생성.

사용법:
    python scripts/plot_results.py

입력:
    runs/detect/food_yolov8/results.csv  (Ultralytics 학습 로그)
    docs/per_class_metrics.csv           (최종 검증 결과의 클래스별 지표)

출력:
    docs/images/training_curves.png
    docs/images/per_class_map.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "runs" / "detect" / "food_yolov8" / "results.csv"
PER_CLASS_CSV = ROOT / "docs" / "per_class_metrics.csv"
OUT_DIR = ROOT / "docs" / "images"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#2b2b2a"
MUTED = "#8a8a85"
GRID = "#e6e5e0"
CLOSE_MOSAIC_EPOCH = 91  # close_mosaic=10: 마지막 10 epoch는 mosaic 증강 끔

plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"],  # 한글 클래스명용 (Windows)
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def plot_training_curves():
    rows = read_csv(RESULTS_CSV)
    epochs = [int(r["epoch"]) for r in rows]
    loss = lambda split: [sum(float(r[f"{split}/{k}_loss"]) for k in ("box", "cls", "dfl")) for r in rows]
    map50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    map50_95 = [float(r["metrics/mAP50-95(B)"]) for r in rows]

    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    ax_loss.plot(epochs, loss("train"), color=BLUE, lw=2, label="train")
    ax_loss.plot(epochs, loss("val"), color=ORANGE, lw=2, label="validation")
    ax_loss.set_title("Loss (box + cls + dfl)", loc="left")
    ax_loss.legend(frameon=False)

    ax_map.plot(epochs, map50, color=BLUE, lw=2, label="mAP50")
    ax_map.plot(epochs, map50_95, color=ORANGE, lw=2, label="mAP50-95")
    ax_map.set_title("Validation mAP", loc="left")
    ax_map.set_ylim(0.5, 1.0)
    ax_map.legend(frameon=False, loc="lower right")
    ax_map.annotate(f"{map50[-1]:.3f}", (epochs[-1], map50[-1]), xytext=(4, 4),
                    textcoords="offset points", color=INK, fontsize=9)
    ax_map.annotate(f"{map50_95[-1]:.3f}", (epochs[-1], map50_95[-1]), xytext=(4, -12),
                    textcoords="offset points", color=INK, fontsize=9)

    for ax in (ax_loss, ax_map):
        ax.set_xlabel("epoch")
        ax.axvline(CLOSE_MOSAIC_EPOCH, color=MUTED, lw=1, ls="--")
        ax.text(CLOSE_MOSAIC_EPOCH - 1, ax.get_ylim()[1], "mosaic off", ha="right", va="top",
                color=MUTED, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", facecolor="white")
    plt.close(fig)


def plot_per_class_map():
    rows = sorted(read_csv(PER_CLASS_CSV), key=lambda r: float(r["map50"]))
    labels = [f"{r['class']}  (n={r['val_images']})" for r in rows]
    map50 = [float(r["map50"]) for r in rows]
    map50_95 = [float(r["map50_95"]) for r in rows]
    y = range(len(rows))

    # 값이 0.58~1.0에 몰려 있어 축을 0에서 시작하지 않는 dot plot 사용
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    ax.hlines(y, map50_95, map50, color=GRID, lw=2, zorder=1)
    ax.scatter(map50_95, y, color=ORANGE, s=36, zorder=2, label="mAP50-95", edgecolors="white", linewidths=1)
    ax.scatter(map50, y, color=BLUE, s=36, zorder=3, label="mAP50", edgecolors="white", linewidths=1)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.set_xlim(0.55, 1.01)
    ax.grid(axis="y", visible=False)
    ax.set_title("Per-class validation mAP (sorted by mAP50, n = validation images)", loc="left")
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_map.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves()
    plot_per_class_map()
    print(f"saved to {OUT_DIR}")
