import numpy as np
import pandas as pd

pd.set_option("display.width", 160)

keys = ["series", "origin", "target"]
pipe = pd.read_csv("reports/backtest_pipeline.csv", parse_dates=["origin", "target"])
hol = pd.read_csv("reports/holiday_ratio.csv", parse_dates=["target"])
pre = pd.read_csv("reports/prepost_ratio.csv", parse_dates=["target"])
feat = pd.read_csv("reports/features.csv", parse_dates=["origin", "target"])
p80_total = pd.read_csv("reports/p80_normal.csv", parse_dates=["origin", "target"])
p80_fashion = pd.read_csv("reports/p80_fashion_calibrated.csv", parse_dates=["origin", "target"])

tot = p80_total[p80_total["series"] == "total"][keys + ["p80"]].rename(columns={"p80": "p80_model"})
fas = p80_fashion[keys].copy()
fas["p80_model"] = np.maximum(p80_fashion["p84"], p80_fashion["lgbm_full"])
model_p80 = pd.concat([tot, fas], ignore_index=True)

pipe = pipe.merge(model_p80, on=keys, how="left")
pipe = pipe.merge(feat[keys + ["base", "after_off"]], on=keys, how="left")

hol_group = hol.set_index(["series", "target"])["group"]
pre_group = pre.set_index(["series", "target"])[["group", "pool"]]

def p80_of(row):
    s, o, t = row["series"], row["origin"], row["target"]
    if row["segment"] == "일반":
        if row["after_off"] == 1:
            past = feat[(feat["series"] == s) & (feat["after_off"] == 1) & (feat["target"] <= o)]
            ratio = past["y_ratio"].quantile(0.8) if len(past) >= 3 else 1.0
            return row["base"] * ratio
        return row["p80_model"]
    if row["segment"] in ("공휴일", "명절 연휴"):
        group = hol_group.get((s, t))
        past = hol[(hol["series"] == s) & (hol["group"] == group) & (hol["target"] <= o)]
        return row["clean4"] * past["ratio"].quantile(0.8)  # 과거 사례가 없으면 NaN
    group, pool = pre_group.loc[(s, t)]
    past = pre[(pre["series"] == s) & (pre["target"] <= o)]
    same = past[past["group"] == group]
    if len(same) < 2:  # 칸이 비면 평일/주말을 합쳐서 본다
        same = past[past["pool"] == pool]
    return row["clean4"] * same["ratio"].quantile(0.8)


pipe["p80"] = pipe.apply(p80_of, axis=1)
print("P80이 비어 있는 행:", pipe["p80"].isna().sum())
pipe["p80"] = np.maximum(pipe["p80"], pipe["pipeline"])
pipe["covered"] = pipe["actual"] <= pipe["p80"]
pipe["kind"] = np.where(
    pipe["segment"] != "일반", pipe["segment"],
    np.where(pipe["after_off"] == 1, "일반·휴일 다음 날", "일반·모델"),
)
pipe.to_csv("reports/backtest_p80.csv", index=False)

for name, part in pipe.groupby("series"):
    over80 = part["p80"].sum() / part["actual"].sum() - 1
    over50 = part["pipeline"].sum() / part["actual"].sum() - 1
    print(f"\n===== {name}: {len(part)}일 =====")
    print(f"P80 커버리지 {part['covered'].mean():.3f}   "
          f"Σ(P80)/Σ(실제)−1 = {over80:+.3f}   Σ(P50)/Σ(실제)−1 = {over50:+.3f}")
    print("\n[연도별 커버리지]")
    print(part.groupby("year")["covered"].mean().round(3).to_string())
    print("\n[날의 종류별 커버리지]")
    print(part.groupby("kind").agg(days=("covered", "size"), coverage=("covered", "mean")).round(3).to_string())