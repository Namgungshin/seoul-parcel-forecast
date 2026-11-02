import pandas as pd
from forecast_common import wape, bias

pd.set_option("display.width", 160)

table = pd.read_csv("reports/holiday_ratio.csv", parse_dates=["target"])
backtest = pd.read_csv("reports/baseline_backtest.csv", parse_dates=["origin", "target"])

group_of = table.set_index(["series", "target"])["group"]
holiday = backtest[backtest["segment"].isin(["공휴일", "명절 연휴"])].copy()
holiday["group"] = [group_of.get((s, t)) for s, t in zip(holiday["series"], holiday["target"])]
print("그룹이 비어 있는 행:", holiday["group"].isna().sum())


def rule_ratio(row, last_n=None):
    past = table[
        (table["series"] == row["series"])
        & (table["group"] == row["group"])
        & (table["target"] <= row["origin"])  # 예측 시점에 이미 끝난 휴일만
    ].sort_values("target")
    if last_n:
        past = past.tail(last_n)
    return past["ratio"].median() if len(past) else 1.0


holiday["rule_all"] = holiday["clean4"] * holiday.apply(rule_ratio, axis=1)
holiday["rule_recent4"] = holiday["clean4"] * holiday.apply(rule_ratio, axis=1, last_n=4)
holiday[["series", "target", "rule_all"]].to_csv("reports/holiday_rule.csv", index=False)
cols = ["clean4", "rule_all", "rule_recent4"]

for name in ["total", "fashion"]:
    part = holiday[holiday["series"] == name]
    print(f"\n===== {name}: 휴일 {len(part)}일 =====")
    for col in cols:
        print(f"{col:13s} WAPE {wape(part['actual'], part[col]):.3f}   Bias {bias(part['actual'], part[col]):+.3f}")

    by_group = part.groupby("group").apply(
        lambda g: pd.Series({"days": len(g), **{c: (g["actual"] - g[c]).abs().mean() for c in cols}}),
        include_groups=False,
    )
    print(f"\n[{name}] 그룹별 하루 평균 절대오차(건)")
    print(by_group.round(0).astype(int).to_string())

    print("\n[전체 721일 WAPE: 휴일만 규칙으로 교체]")
for name in ["total", "fashion"]:
    part = backtest[backtest["series"] == name]
    line = f"{name:8s} clean4 {wape(part['actual'], part['clean4']):.3f}"
    idx = holiday.index[holiday["series"] == name]
    for col in ["rule_all", "rule_recent4"]:
        pred = part["clean4"].copy()
        pred.loc[idx] = holiday.loc[idx, col]
        line += f"   {col} {wape(part['actual'], pred):.3f}"
    print(line)