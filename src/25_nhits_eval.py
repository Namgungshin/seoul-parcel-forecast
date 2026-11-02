import sys

import numpy as np
import pandas as pd

from forecast_common import wape

rng = np.random.default_rng(0)
base = pd.read_csv("reports/check_pipeline_2021_2022.csv", parse_dates=["origin", "target"])
base = base[base["kind"] == "일반·모델"]


def load(files):
    """시드별 파일을 읽어 같은 날의 중앙값으로 합친다."""
    runs = pd.concat([pd.read_csv(f, parse_dates=["origin", "target"]) for f in files])
    pred = runs.groupby(["series", "origin", "target"], as_index=False)[["p50", "q80", "q84"]].median()
    merged = base.merge(pred, on=["series", "origin", "target"], how="inner")
    expected = base[base["origin"].isin(pred["origin"])]
    assert len(merged) == len(expected), "빠진 날이 있습니다"
    return merged


def week_bootstrap(part, model_col, base_col, draws=2000):
    g = part.assign(
        err_m=(part["actual"] - part[model_col]).abs(),
        err_b=(part["actual"] - part[base_col]).abs(),
    ).groupby("origin")[["err_m", "err_b", "actual"]].sum()
    idx = rng.integers(0, len(g), size=(draws, len(g)))
    diff = (g["err_m"].to_numpy()[idx].sum(axis=1) - g["err_b"].to_numpy()[idx].sum(axis=1)) \
        / g["actual"].to_numpy()[idx].sum(axis=1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return lo, hi, (diff < 0).mean()


mode, files = sys.argv[1], sys.argv[2:]
res = load(files)

if mode == "select":  # 2021년으로 설정과 패션 분위수를 고른다
    assert res["target"].max() < pd.Timestamp("2022-01-01")
    for name, part in res.groupby("series"):
        print(f"{name} {len(part)}일  WAPE {wape(part['actual'], part['p50']):.4f}  "
              f"커버리지 q80 {(part['actual'] <= part['q80']).mean():.3f}  q84 {(part['actual'] <= part['q84']).mean():.3f}")

if mode == "judge":  # 2022년으로 판정한다
    assert res["target"].min() >= pd.Timestamp("2022-01-01")
    for name, part in res.groupby("series"):
        print(f"\n[{name}] {len(part)}일")
        for label, col in [("N-HiTS", "p50"), ("LightGBM", "pipeline"), ("단순 기준선", "bar")]:
            print(f"  {label:10s} WAPE {wape(part['actual'], part[col]):.4f}")
        for other, col in [("LightGBM", "pipeline"), ("단순 기준선", "bar")]:
            lo, hi, p = week_bootstrap(part, "p50", col)
            print(f"  N-HiTS − {other}: [{lo:+.4f}, {hi:+.4f}]  N-HiTS가 나은 비율 {p:.2f}  채택 조건 {'충족' if hi < 0 else '미충족'}")
        print(f"  커버리지 q80 {(part['actual'] <= part['q80']).mean():.3f}  q84 {(part['actual'] <= part['q84']).mean():.3f}"
              f"  LightGBM P80 {part['covered'].mean():.3f}")