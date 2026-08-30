"""AIHub 클래스명이 xlsx 식품명과 정확히 안 맞는 경우를 위한 별칭 행 생성.

data/food_name_aliases.csv에 등록된 (aihub_name -> xlsx_name) 매핑을 이용해
xlsx에서 영양성분을 찾아, AIHub 클래스명을 그대로 name으로 쓰는 tb_foods 행을
추가로 만든다. build_nutrition_csv.py가 이미 xlsx_name으로 된 행을 tb_foods에
넣어두므로, 이 스크립트는 predict.py가 반환하는 food_name으로 직접 조회 가능하게
하는 별칭(동의어) 행만 보충한다.

새 AIHub 카테고리를 추가할 때마다 매칭 안 되는 클래스명이 있으면
data/food_name_aliases.csv에 한 줄 추가하고 이 스크립트를 다시 실행한다.

사용법:
    python scripts/apply_food_aliases.py

출력:
    data/processed/tb_foods_aliases.csv

Supabase 적재:
    \\COPY tb_foods(name,kcal,carb,protein,fat,sugar,is_custom,created_by)
    FROM 'data/processed/tb_foods_aliases.csv' WITH (FORMAT csv, HEADER true);
"""

import csv
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent.parent
ALIASES_CSV = ROOT / "data" / "food_name_aliases.csv"
SOURCE_XLSX = ROOT / "data" / "통합 식품영양성분DB.xlsx"
OUTPUT_CSV = ROOT / "data" / "processed" / "tb_foods_aliases.csv"

COL_NAME = 5
COL_REGION = 7
COL_ENERGY_KCAL = 15
COL_PROTEIN = 19
COL_FAT = 20
COL_CARB = 21
COL_SUGAR = 22

NATIONAL_REPRESENTATIVE = "전국(대표)"
MAX_REASONABLE_VALUE = 9999


def to_number(value) -> float | None:
    if value is None or value == "-":
        return None
    number = float(value)
    if abs(number) >= MAX_REASONABLE_VALUE:
        return None
    return number


def load_aliases() -> dict[str, str]:
    with open(ALIASES_CSV, encoding="utf-8") as f:
        return {row["aihub_name"]: row["xlsx_name"] for row in csv.DictReader(f)}


def find_nutrition(xlsx_names: set[str]) -> dict[str, dict]:
    wb = openpyxl.load_workbook(SOURCE_XLSX, read_only=True)
    ws = wb["원본"]

    found: dict[str, dict] = {}
    for row in ws.iter_rows(min_row=5, values_only=True):
        name = row[COL_NAME]
        if name not in xlsx_names or name in found:
            continue
        if row[COL_REGION] != NATIONAL_REPRESENTATIVE:
            continue
        found[name] = {
            "kcal": to_number(row[COL_ENERGY_KCAL]),
            "carb": to_number(row[COL_CARB]),
            "protein": to_number(row[COL_PROTEIN]),
            "fat": to_number(row[COL_FAT]),
            "sugar": to_number(row[COL_SUGAR]),
        }
    return found


def main() -> None:
    aliases = load_aliases()
    nutrition_by_xlsx_name = find_nutrition(set(aliases.values()))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "kcal", "carb", "protein", "fat", "sugar", "is_custom", "created_by"])
        for aihub_name, xlsx_name in aliases.items():
            nutrition = nutrition_by_xlsx_name.get(xlsx_name)
            if nutrition is None:
                print(f"경고: '{xlsx_name}' 을(를) 전국(대표) 행에서 못 찾음 (건너뜀)")
                continue
            writer.writerow(
                [
                    aihub_name,
                    nutrition["kcal"],
                    nutrition["carb"],
                    nutrition["protein"],
                    nutrition["fat"],
                    nutrition["sugar"],
                    "false",
                    "",
                ]
            )

    print(f"{len(aliases)}개 별칭 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
