"""통합 식품영양성분DB.xlsx -> tb_foods 적재용 CSV 변환.

AIHub 음식 이미지 클래스가 조리식품뿐 아니라 과일/가공식품(예: 바나나우유)까지
섞여 있어 DB군으로 필터링하지 않고 전체를 사용한다. 같은 식품명이 여러 번
나오면 '전국(대표)' 행을 우선 사용하고 없으면 나머지 행의 평균을 낸다.

사용법:
    python scripts/build_nutrition_csv.py

출력:
    data/processed/tb_foods.csv (컬럼: name,kcal,carb,protein,fat,sugar,is_custom,created_by)

Supabase 적재:
    \\COPY tb_foods(name,kcal,carb,protein,fat,sugar,is_custom,created_by)
    FROM 'data/processed/tb_foods.csv' WITH (FORMAT csv, HEADER true);
"""

import csv
from pathlib import Path

import openpyxl

SOURCE_XLSX = Path(__file__).resolve().parent.parent / "data" / "통합 식품영양성분DB.xlsx"
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "tb_foods.csv"

COL_NAME = 5
COL_REGION = 7
COL_ENERGY_KCAL = 15
COL_PROTEIN = 19
COL_FAT = 20
COL_CARB = 21
COL_SUGAR = 22

NATIONAL_REPRESENTATIVE = "전국(대표)"

# tb_foods 영양성분 컬럼이 numeric(5,1)이라 10000 이상은 저장이 안 됨.
# 소스 xlsx에 대용량(kg 단위 벌크 제품)이나 오타로 보이는 이상치가 섞여있어 걸러냄.
MAX_REASONABLE_VALUE = 9999


def to_number(value) -> float | None:
    if value is None or value == "-":
        return None
    number = float(value)
    if abs(number) >= MAX_REASONABLE_VALUE:
        return None
    return number


def load_food_rows() -> dict[str, list[dict]]:
    wb = openpyxl.load_workbook(SOURCE_XLSX, read_only=True)
    ws = wb["원본"]

    by_name: dict[str, list[dict]] = {}
    for row in ws.iter_rows(min_row=5, values_only=True):
        name = row[COL_NAME]
        if not name:
            continue
        by_name.setdefault(name, []).append(
            {
                "region": row[COL_REGION],
                "kcal": to_number(row[COL_ENERGY_KCAL]),
                "protein": to_number(row[COL_PROTEIN]),
                "fat": to_number(row[COL_FAT]),
                "carb": to_number(row[COL_CARB]),
                "sugar": to_number(row[COL_SUGAR]),
            }
        )
    return by_name


def average(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 2)


def pick_representative(entries: list[dict]) -> dict:
    national = [e for e in entries if e["region"] == NATIONAL_REPRESENTATIVE]
    if national:
        return national[0]
    return {
        "kcal": average([e["kcal"] for e in entries]),
        "protein": average([e["protein"] for e in entries]),
        "fat": average([e["fat"] for e in entries]),
        "carb": average([e["carb"] for e in entries]),
        "sugar": average([e["sugar"] for e in entries]),
    }


def main() -> None:
    by_name = load_food_rows()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "kcal", "carb", "protein", "fat", "sugar", "is_custom", "created_by"])
        for name, entries in sorted(by_name.items()):
            rep = pick_representative(entries)
            writer.writerow(
                [
                    name,
                    rep["kcal"],
                    rep["carb"],
                    rep["protein"],
                    rep["fat"],
                    rep["sugar"],
                    "false",
                    "",
                ]
            )

    print(f"{len(by_name)}개 식품명 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
