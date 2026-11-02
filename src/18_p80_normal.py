import numpy as np
import pandas as pd
import lightgbm as lgb

pd.set_option("display.width", 160)

df = pd.read_csv("reports/features.csv", parse_dates=["origin", "target"])
p50 = pd.read_csv("reports/lgbm_backtest.csv", parse_dates=["origin", "target"])[["series", "origin", "target", "lgbm_full"]]

FULL = ["dow", "month", "next_off", "days_to_big", "days_since_big",
        "lag1", "lag2", "lag3", "lag4", "level_ratio"]
ALPHA = 0.8


def fit_quantile(train, test):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=ALPHA, n_estimators=300, learning_rate=0.03,
        num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=0, verbose=-1,
    )
    model.fit(train[FULL], train["y_ratio"])
    return model.predict(test[FULL])


results = []
for name in ["total", "fashion"]:
    part = df[(df["series"] == name) & (df["after_off"] == 0)]
    for origin in sorted(part.loc[part["in_backtest"], "origin"].unique()):
        train = part[part["target"] <= origin]
        test = part[part["origin"] == origin]
        out = test[["series", "origin", "target", "dow", "actual", "base"]].copy()
        out["p80"] = test["base"] * fit_quantile(train, test)
        results.append(out)

res = pd.concat(results, ignore_index=True).merge(p50, on=["series", "origin", "target"], how="left")
res["crossed"] = res["p80"] < res["lgbm_full"]          # P80이 P50보다 낮게 나온 행
res["p80"] = np.maximum(res["p80"], res["lgbm_full"])   # 상한은 중앙값 아래로 내려가지 않게
res["covered"] = res["actual"] <= res["p80"]
res["width"] = res["p80"] / res["lgbm_full"] - 1
res["year"] = res["target"].dt.year
res["요일"] = res["dow"].map(lambda i: ["월", "화", "수", "목", "금", "토", "일"][int(i)])
res.to_csv("reports/p80_normal.csv", index=False)

for name, part in res.groupby("series"):
    print(f"\n===== {name}: {len(part)}일, P80 =====")
    print(f"커버리지 {part['covered'].mean():.3f} (목표 0.80)   "
          f"평균 폭(P80/P50−1) {part['width'].mean():.3f}   P50 아래로 나온 행 {int(part['crossed'].sum())}")
    print("\n[연도별 커버리지]")
    print(part.groupby("year")["covered"].mean().round(3).to_string())
    print("\n[요일별 커버리지 / 평균 폭]")
    print(part.groupby("요일").agg(days=("covered", "size"), coverage=("covered", "mean"),
                                  width=("width", "mean")).round(3).to_string())