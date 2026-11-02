import numpy as np
import pandas as pd
import lightgbm as lgb

pd.set_option("display.width", 160)

df = pd.read_csv("reports/features.csv", parse_dates=["origin", "target"])
p50 = pd.read_csv("reports/lgbm_backtest.csv", parse_dates=["origin", "target"])[["series", "origin", "target", "lgbm_full"]]

FULL = ["dow", "month", "next_off", "days_to_big", "days_since_big",
        "lag1", "lag2", "lag3", "lag4", "level_ratio"]
ALPHAS = [0.80, 0.84, 0.88]
SERIES = "fashion"


def fit_quantile(train, test, alpha):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=300, learning_rate=0.03,
        num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=0, verbose=-1,
    )
    model.fit(train[FULL], train["y_ratio"])
    return model.predict(test[FULL])


part = df[(df["series"] == SERIES) & (df["after_off"] == 0)]
rows = []
for origin in sorted(part.loc[part["in_backtest"], "origin"].unique()):
    train = part[part["target"] <= origin]
    test = part[part["origin"] == origin]
    out = test[["series", "origin", "target", "actual", "base"]].copy()
    for alpha in ALPHAS:
        out[f"p{round(alpha * 100)}"] = test["base"] * fit_quantile(train, test, alpha)
    rows.append(out)

res = pd.concat(rows, ignore_index=True).merge(p50, on=["series", "origin", "target"], how="left")
res["year"] = res["target"].dt.year
res.to_csv("reports/p80_fashion_calibrated.csv", index=False)

print(f"\n===== {SERIES}: 알파별 P80 커버리지 =====")
print("alpha   전체   2021   2022   평균 폭")
cov2021 = {}
for alpha in ALPHAS:
    upper = np.maximum(res[f"p{round(alpha * 100)}"], res["lgbm_full"])
    covered = res["actual"] <= upper
    by_year = covered.groupby(res["year"]).mean()
    width = (upper / res["lgbm_full"] - 1).mean()
    cov2021[alpha] = by_year[2021]
    print(f"{alpha:.2f}   {covered.mean():.3f}  {by_year[2021]:.3f}  {by_year[2022]:.3f}  {width:.3f}")

best = min(ALPHAS, key=lambda a: (abs(cov2021[a] - 0.80), a))
print(f"\n2021년 커버리지가 0.80에 가장 가까운 알파: {best:.2f} (이 값만 2022년 검증에 씁니다)")