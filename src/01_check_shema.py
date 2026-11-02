from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

files = sorted(RAW_DIR.glob("*.csv"))
print(f"CSV 파일 수: {len(files)}")

def read_csv_any_encoding(path):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 확인할 수 없습니다: {path}")

for path in files:
    df, encoding = read_csv_any_encoding(path)

    print("\n" + "=" * 60)
    print(f"파일: {path.name} | 인코딩: {encoding} | 크기: {df.shape}")
    print(f"컬럼: {list(df.columns)}")

    print("\n[앞 5행]")
    print(df.head().to_string())

    print("\n[컬럼별 NULL 비율]")
    print(df.isna().mean().round(4).to_string())

    print("\n[컬럼별 고유값]")
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"- {col}: {n_unique}개")
        if n_unique <= 30:
            print(f"  {sorted(df[col].dropna().unique())}")