import numpy as np
import pandas as pd
import lightgbm as lgb
from forecast_common import load_daily, clean_median, wape, bias

pd.set_option("display.width", 160)

df = pd.read_csv("reports/features.csv", parse_dates=["origin", "target"])

daily = load_daily()
is_off_day = (daily["is_holiday"] == 1) | (daily["is_big_holiday"] == 1)
daily["prev_off"] = is_off_day.shift(1, fill_value=False)
daily["next_off"] = is_off_day.shift(-1, fill_value=False)
daily["is_clean2"] = daily["is_clean"] & ~daily["prev_off"] & ~daily["next_off"]

BARS = {"total": 6, "fashion": 2}
FULL = ["dow", "month", "next_off", "days_to_big", "days_since_big",
        "lag1", "lag2", "lag3", "lag4", "level_ratio"]
NO_MONTH = [f for f in FULL if f != "month"]

for name in ["total", "fashion"]:
    sub = df[(df["series"] == name) & (df["after_off"] == 0)].sort_values("y_ratio")
    print(f"\n[{name}] y_ratio 가장 낮은 5행 / 가장 높은 5행")
    print(sub[["target", "dow", "y_ratio"]].head(5).to_string(index=False))
    print(sub[["target", "dow", "y_ratio"]].tail(5).to_string(index=False))

def fit_predict(train, test, features):
    model = lgb.LGBMRegressor(
        objective="l1", n_estimators=300, learning_rate=0.03, num_leaves=7,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=0, verbose=-1,
    )
    model.fit(train[features], train["y_ratio"])
    return model.predict(test[features])


results = []
for name in ["total", "fashion"]:
    part = df[(df["series"] == name) & (df["after_off"] == 0)]
    test_origins = sorted(part.loc[part["in_backtest"], "origin"].unique())
    for origin in test_origins:
        train = part[part["target"] <= origin]  # 그 시점에 실제 값이 알려진 행만
        test = part[part["origin"] == origin]
        out = test[["series", "origin", "target", "actual", "base"]].copy()
        out["lgbm_full"] = test["base"] * fit_predict(train, test, FULL)
        out["lgbm_no_month"] = test["base"] * fit_predict(train, test, NO_MONTH)
        results.append(out)

res = pd.concat(results, ignore_index=True)
res["year"] = res["target"].dt.year

for n in range(1, 9):
    res[f"clean2_n{n}"] = [
        clean_median(daily[s], daily["is_clean2"], t, o, n=n, max_back=20)
        for s, t, o in zip(res["series"], res["target"], res["origin"])
    ]

for n in range(1, 9):
    res[f"clean2_n{n}"] = [
        clean_median(daily[s], daily["is_clean2"], t, o, n=n, max_back=20)
        for s, t, o in zip(res["series"], res["target"], res["origin"])
    ]

cols = [f"clean2_n{n}" for n in range(1, 9)] + ["lgbm_full", "lgbm_no_month"]
for name, part in res.groupby("series"):
    print(f"\n===== {name}: {len(part)}일 (기준선 n={BARS[name]}) =====")
    for c in cols:
        by_year = "  ".join(f"{y}: {wape(g['actual'], g[c]):.4f}" for y, g in part.groupby("year"))
        print(f"{c:14s} WAPE {wape(part['actual'], part[c]):.4f}   Bias {bias(part['actual'], part[c]):+.3f}   {by_year}")

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
    diff = (m - b) / a  # 음수면 모델이 더 낫다
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return lo, hi, (diff < 0).mean()


print("\n[주 단위 부트스트랩: 모델 WAPE − 기준선 WAPE, 95% 구간]")
for name, part in res.groupby("series"):
    for model_col in ["lgbm_full", "lgbm_no_month"]:
        for base_col in [f"clean2_n{BARS[name]}", "clean2_n4"]:
            lo, hi, p_better = week_bootstrap(part, model_col, base_col)
            print(f"{name:8s} {model_col:14s} vs {base_col:10s}  [{lo:+.4f}, {hi:+.4f}]  모델이 나은 비율 {p_better:.2f}")

res.to_csv("reports/lgbm_backtest.csv", index=False)