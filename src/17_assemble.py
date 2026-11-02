import numpy as np
import pandas as pd
from forecast_common import wape, bias

pd.set_option("display.width", 160)

rules = pd.read_csv("reports/backtest_rules.csv", parse_dates=["origin", "target"])
feat = pd.read_csv("reports/features.csv", parse_dates=["origin", "target"])
lgbm = pd.read_csv("reports/lgbm_backtest.csv", parse_dates=["origin", "target"])
keys = ["series", "origin", "target"]

after = feat[feat["after_off"] == 1]


def after_off_ratio(series, origin):
    past = after[(after["series"] == series) & (after["target"] <= origin)]  # 그 시점에 끝난 사례만
    return past["y_ratio"].median() if len(past) >= 3 else 1.0
normal = rules[rules["segment"] == "일반"].merge(feat[keys + ["base", "after_off"]], on=keys, how="left")
normal = normal.merge(lgbm[keys + ["lgbm_full"]], on=keys, how="left")
print("일반일 행:", len(normal), " 기준값 빈 행:", normal["base"].isna().sum())


def predict_normal(row):
    if row["after_off"] == 1:
        return row["base"] * after_off_ratio(row["series"], row["origin"])
    return row["lgbm_full"]


normal["final2"] = normal.apply(predict_normal, axis=1)
assert normal["final2"].notna().all(), "예측이 비어 있는 일반일이 있습니다"

rules = rules.merge(normal[keys + ["final2"]], on=keys, how="left")
rules["pipeline"] = rules["final2"].fillna(rules["final"])
rules.to_csv("reports/backtest_pipeline.csv", index=False)

cols = {"naive7": "naive7", "clean4": "clean4(처음)", "final": "규칙 적용", "pipeline": "규칙 + 모델"}
for name, part in rules.groupby("series"):
    print(f"\n===== {name}: {len(part)}일 =====")
    for col, label in cols.items():
        by_year = "  ".join(f"{y}: {wape(g['actual'], g[col]):.3f}" for y, g in part.groupby("year"))
        print(f"{label:12s} WAPE {wape(part['actual'], part[col]):.3f}   Bias {bias(part['actual'], part[col]):+.3f}   {by_year}")
    by_seg = part.groupby("segment").apply(
        lambda g: pd.Series({
            "days": len(g),
            "규칙": wape(g["actual"], g["final"]),
            "규칙+모델": wape(g["actual"], g["pipeline"]),
        }),
        include_groups=False,
    )
    print(by_seg.round(3).to_string())

rng = np.random.default_rng(0)


def week_bootstrap(part, model_col, base_col, draws=2000):
    g = part.assign(
        err_m=(part["actual"] - part[model_col]).abs(),
        err_b=(part["actual"] - part[base_col]).abs(),
    ).groupby("origin")[["err_m", "err_b", "actual"]].sum()
    idx = rng.integers(0, len(g), size=(draws, len(g)))
    m = g["err_m"].to_numpy()[idx].sum(axis=1)
    b = g["err_b"].to_numpy()[idx].sum(axis=1)
    a = g["actual"].to_numpy()[idx].sum(axis=1)
    diff = (m - b) / a
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return lo, hi, (diff < 0).mean()


print("\n[주 단위 부트스트랩: 규칙+모델 WAPE − 규칙 적용 WAPE, 95% 구간]")
for name, part in rules.groupby("series"):
    lo, hi, p = week_bootstrap(part, "pipeline", "final")
    print(f"{name:8s} [{lo:+.4f}, {hi:+.4f}]  모델이 나은 비율 {p:.2f}")