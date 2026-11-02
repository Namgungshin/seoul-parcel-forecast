import numpy as np
import pandas as pd
import lightgbm as lgb

pd.set_option("display.width", 160)

res = pd.read_csv("reports/lgbm_backtest.csv", parse_dates=["origin", "target"])
feat = pd.read_csv("reports/features.csv", parse_dates=["origin", "target"])
res = res.merge(
    feat[["series", "origin", "target", "dow", "days_to_big", "days_since_big"]],
    on=["series", "origin", "target"], how="left",
)
print("병합 후 비어 있는 행:", res["dow"].isna().sum())

res["bar"] = np.where(res["series"] == "total", res["clean2_n6"], res["clean2_n1"])
res["err_bar"] = (res["actual"] - res["bar"]).abs()
res["err_model"] = (res["actual"] - res["lgbm_full"]).abs()
res["요일"] = res["dow"].map(lambda i: ["월", "화", "수", "목", "금", "토", "일"][int(i)])
res["near_big"] = np.where((res["days_to_big"] <= 10) | (res["days_since_big"] <= 10), "명절 10일 이내", "그 외")

def gain_table(part, key):
    g = part.groupby(key).agg(
        days=("actual", "size"), actual=("actual", "sum"),
        err_bar=("err_bar", "sum"), err_model=("err_model", "sum"),
    )
    g["wape_bar"] = g["err_bar"] / g["actual"]
    g["wape_model"] = g["err_model"] / g["actual"]
    g["gain_share"] = (g["err_bar"] - g["err_model"]) / (g["err_bar"] - g["err_model"]).sum()
    return g[["days", "wape_bar", "wape_model", "gain_share"]].round(3)


for name, part in res.groupby("series"):
    print(f"\n########## {name}: 기준선 {'n=6' if name == 'total' else 'n=1'} 대비")
    print("\n[명절 10일 이내 vs 그 외]")
    print(gain_table(part, "near_big").to_string())
    print("\n[요일별]")
    print(gain_table(part, "요일").to_string())


FULL = ["dow", "month", "next_off", "days_to_big", "days_since_big",
        "lag1", "lag2", "lag3", "lag4", "level_ratio"]

for name in ["total", "fashion"]:
    sub = feat[(feat["series"] == name) & (feat["after_off"] == 0)]
    model = lgb.LGBMRegressor(
        objective="l1", n_estimators=300, learning_rate=0.03, num_leaves=7,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=0, verbose=-1,
    )
    model.fit(sub[FULL], sub["y_ratio"])
    imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=FULL)
    print(f"\n[{name}] 피처 중요도(gain 비율)")
    print((imp / imp.sum()).round(3).sort_values(ascending=False).to_string())


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


fashion = res[res["series"] == "fashion"]
lo, hi, p = week_bootstrap(fashion, "lgbm_full", "clean2_n1")
print(f"\n패션 lgbm_full vs clean2_n1  [{lo:+.4f}, {hi:+.4f}]  모델이 나은 비율 {p:.2f}")
