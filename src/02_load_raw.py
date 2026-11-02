import re
from pathlib import Path

import duckdb
import pandas as pd

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/parcel.duckdb")
PREFIX = "대분류_착지물동량 "

files = sorted(RAW_DIR.glob("*.csv"))
found = {re.search(r"(\d{6})\.csv$", f.name).group(1) for f in files}
expected = set(pd.period_range("2018-01", "2023-12", freq="M").strftime("%Y%m"))

missing = sorted(expected - found)
if missing:
    raise SystemExit(f"빠진 파일이 있습니다: {missing}")
print(f"파일 {len(files)}개 확인")
def read_csv_any_encoding(path):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 확인할 수 없습니다: {path}")


frames = []
first_columns = None
for path in files:
    df = read_csv_any_encoding(path)
    if first_columns is None:
        first_columns = list(df.columns)
    elif list(df.columns) != first_columns:
        raise SystemExit(f"컬럼이 다릅니다: {path.name}\n{list(df.columns)}")
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"전체 행 수: {len(raw):,}")
raw = raw.rename(columns={
    "배송년월일": "date",
    "송하인_시명": "origin_name",
    "송하인_시코드": "origin_code",
    "수하인_구명": "gu_name",
    "수하인_구코드": "gu_code",
})
raw = raw.drop(columns=["수하인_시명", "수하인_시코드"])
raw.columns = [c.replace(PREFIX, "").strip() for c in raw.columns]

raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d")
raw["origin_code"] = raw["origin_code"].replace({"42": "51"})
raw["origin_name"] = raw["origin_name"].replace({"강원도": "강원특별자치도"})

id_cols = ["date", "origin_name", "origin_code", "gu_name", "gu_code"]
volume_cols = [c for c in raw.columns if c not in id_cols]
raw[volume_cols] = raw[volume_cols].apply(pd.to_numeric).astype("float64")

print(f"품목: {volume_cols}")
print(f"물량 빈 값 수: {int(raw[volume_cols].isna().sum().sum()):,}")
con = duckdb.connect(str(DB_PATH))

con.execute("""
    CREATE OR REPLACE TABLE parcel_wide AS
    SELECT * REPLACE (CAST(date AS DATE) AS date)
    FROM raw
""")

con.execute("""
    CREATE OR REPLACE TABLE parcel_long AS
    SELECT *
    FROM (
        UNPIVOT parcel_wide
        ON COLUMNS(* EXCLUDE (date, origin_name, origin_code, gu_name, gu_code))
        INTO NAME category VALUE volume
    )
""")

print(con.execute("SELECT COUNT(*) FROM parcel_long").fetchone())
con.close()