"""AIHub 음식 이미지 JSON 라벨을 YOLO 포맷(txt)으로 변환한다."""
import argparse
import json
import logging
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SPLIT_MAP = {
    "training": "train",
    "validation": "val",
}


def find_role_dir(split_root: Path, keyword: str) -> Path | None:
    matches = [p for p in split_root.iterdir() if p.is_dir() and keyword in p.name]
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("%s 안에 '%s' 포함 폴더가 여러 개 있어 첫 번째를 사용함: %s", split_root, keyword, matches)
    return sorted(matches)[0]


def class_name_from_folder(folder_name: str) -> str:
    name = folder_name.strip()
    if name.endswith("json"):
        name = name[: -len("json")].strip()
    return name


def collect_classes(input_root: Path) -> dict[str, int]:
    names = set()
    for split_dir_name in SPLIT_MAP:
        split_root = input_root / split_dir_name
        if not split_root.is_dir():
            continue
        labels_root = find_role_dir(split_root, "라벨")
        if labels_root is None:
            continue
        for class_dir in labels_root.iterdir():
            if class_dir.is_dir():
                names.add(class_name_from_folder(class_dir.name))
    return {name: idx for idx, name in enumerate(sorted(names))}


def parse_point(value: str) -> tuple[float, float]:
    x_str, y_str = value.split(",")
    return float(x_str), float(y_str)


def clamp01(value: float, *, context: str) -> float:
    if -1e-3 <= value <= 1 + 1e-3:
        return min(1.0, max(0.0, value))
    raise ValueError(f"{context} 값이 [0,1] 범위를 벗어남: {value}")


def convert_json_to_lines(data: list[dict], class_id: int, *, json_path: Path) -> list[str]:
    lines = []
    for item in data:
        try:
            xc, yc = parse_point(item["Point(x,y)"])
            w = float(item["W"])
            h = float(item["H"])
            xc = clamp01(xc, context="x_center")
            yc = clamp01(yc, context="y_center")
            w = clamp01(w, context="width")
            h = clamp01(h, context="height")
        except (KeyError, ValueError) as exc:
            logger.warning("잘못된 항목 skip (%s): %s", json_path, exc)
            continue
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def process_split(
    split_root: Path,
    output_root: Path,
    out_split: str,
    class_to_id: dict[str, int],
    *,
    copy: bool,
    strict: bool,
) -> dict[str, int]:
    labels_root = find_role_dir(split_root, "라벨")
    images_root = find_role_dir(split_root, "원천")
    if labels_root is None or images_root is None:
        raise SystemExit(f"{split_root} 아래에서 라벨/원천 폴더를 찾을 수 없습니다.")

    out_images_dir = output_root / "images" / out_split
    out_labels_dir = output_root / "labels" / out_split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "boxes": 0, "missing_images": 0, "skipped_boxes": 0}

    for class_dir in sorted(labels_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_name_from_folder(class_dir.name)
        class_id = class_to_id[class_name]
        image_class_dir = images_root / class_name
        if not image_class_dir.is_dir():
            msg = f"{class_dir} 대응 이미지 폴더 없음: {image_class_dir}"
            if strict:
                raise SystemExit(msg)
            logger.warning(msg)
            continue

        for json_path in sorted(class_dir.glob("*.json")):
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not data:
                logger.warning("빈 json, skip: %s", json_path)
                continue

            code_name = data[0].get("Code Name") or f"{json_path.stem}.jpg"
            image_path = image_class_dir / code_name
            if not image_path.is_file():
                msg = f"이미지 없음, skip: {image_path}"
                if strict:
                    raise SystemExit(msg)
                logger.warning(msg)
                stats["missing_images"] += 1
                continue

            lines = convert_json_to_lines(data, class_id, json_path=json_path)
            stats["skipped_boxes"] += len(data) - len(lines)
            if not lines:
                logger.warning("유효한 박스 없음, skip: %s", json_path)
                continue

            label_path = out_labels_dir / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            link_or_copy(image_path, out_images_dir / image_path.name, copy=copy)

            stats["images"] += 1
            stats["boxes"] += len(lines)

    return stats


def write_dataset_yaml(yaml_path: Path, output_root: Path, class_to_id: dict[str, int]) -> None:
    names = {idx: name for name, idx in sorted(class_to_id.items(), key=lambda kv: kv[1])}
    payload = {
        # ultralytics는 상대경로를 yaml 위치가 아니라 자체 datasets_dir 설정 기준으로 해석하므로 절대경로 사용
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="AIHub JSON 라벨을 YOLO 포맷으로 변환")
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--classes-yaml", type=Path, default=Path("data/food_dataset.yaml"))
    parser.add_argument("--copy", action="store_true", help="심볼릭 링크 대신 이미지를 복사")
    parser.add_argument("--strict", action="store_true", help="이미지/라벨 짝이 안 맞으면 즉시 에러")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    class_to_id = collect_classes(args.input)
    if not class_to_id:
        raise SystemExit(f"{args.input} 아래에서 클래스를 찾을 수 없습니다.")
    logger.info("클래스 %d개 발견", len(class_to_id))

    for split_dir_name, out_split in SPLIT_MAP.items():
        split_root = args.input / split_dir_name
        if not split_root.is_dir():
            logger.info("%s 스플릿 없음, 스킵", split_dir_name)
            continue
        stats = process_split(split_root, args.output, out_split, class_to_id, copy=args.copy, strict=args.strict)
        logger.info(
            "[%s] 이미지 %d개, 박스 %d개 변환 완료 (이미지 없음 skip: %d, 잘못된 박스 skip: %d)",
            out_split,
            stats["images"],
            stats["boxes"],
            stats["missing_images"],
            stats["skipped_boxes"],
        )

    write_dataset_yaml(args.classes_yaml, args.output, class_to_id)
    logger.info("클래스 목록 저장: %s", args.classes_yaml)


if __name__ == "__main__":
    main()
