import pandas as pd
from forecast_common import load_daily, clean_median, wape, bias, FINAL_START

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 300)

daily = load_daily()


def prepost_group(date):
    row = daily.loc[date]
    side = "전" if row["days_to_big"] <= 3 else "후"
    dist = int(row["days_to_big"] if side == "전" else row["days_since_big"])
    weekend = "주말" if row["is_weekend"] else "평일"
    return f"{side}{dist}", f"{side}{dist}·{weekend}"


targets = daily.index[(daily["segment"] == "명절 전후 3일") & (daily.index < FINAL_START)]

rows = []
for target in targets:
    origin = target - pd.Timedelta(days=target.day_of_week + 1)
    pool, group = prepost_group(target)
    for series in ["total", "fashion"]:
        try:
            base = clean_median(daily[series], daily["is_clean"], target, origin)
        except KeyError:
            continue
        rows.append({
            "series": series, "target": target, "pool": pool, "group": group,
            "actual": daily.loc[target, series], "clean4": base,
            "ratio": daily.loc[target, series] / base,
        })

table = pd.DataFrame(rows).dropna(subset=["ratio"])
table.to_csv("reports/prepost_ratio.csv", index=False)

print("[전체 물량: 명절 전후 3일 그룹별 비율]")
total = table[table["series"] == "total"]
print(total.groupby(["pool", "group"])["ratio"].agg(["count", "median", "min", "max"]).round(2).to_string())

backtest = pd.read_csv("reports/baseline_backtest.csv", parse_dates=["origin", "target"])
pp = backtest[backtest["segment"] == "명절 전후 3일"].copy()
labels = [prepost_group(t) for t in pp["target"]]
pp["pool"] = [p for p, _ in labels]
pp["group"] = [g for _, g in labels]


def rule_ratio(row):
    past = table[(table["series"] == row["series"]) & (table["target"] <= row["origin"])]
    same = past[past["group"] == row["group"]]
    if len(same) < 2:  # 평일/주말로 나눈 칸이 비면 합쳐서 본다
        same = past[past["pool"] == row["pool"]]
    return same["ratio"].median() if len(same) else 1.0


pp["rule"] = pp["clean4"] * pp.apply(rule_ratio, axis=1)

for name, part in pp.groupby("series"):
    print(f"\n===== {name}: 명절 전후 3일 {len(part)}일 =====")
    print(f"clean4  WAPE {wape(part['actual'], part['clean4']):.3f}   Bias {bias(part['actual'], part['clean4']):+.3f}")
    print(f"rule    WAPE {wape(part['actual'], part['rule']):.3f}   Bias {bias(part['actual'], part['rule']):+.3f}")

rules = pd.read_csv("reports/holiday_rule.csv", parse_dates=["target"])
hol_map = rules.set_index(["series", "target"])["rule_all"]
is_hol = backtest["segment"].isin(["공휴일", "명절 연휴"])

final = backtest["clean4"].copy()
final.loc[is_hol] = [hol_map[(s, t)] for s, t in zip(backtest.loc[is_hol, "series"], backtest.loc[is_hol, "target"])]
final.loc[pp.index] = pp["rule"]
backtest["final"] = final
backtest.to_csv("reports/backtest_rules.csv", index=False)

print("\n[전체 721일 WAPE]")
for name, part in backtest.groupby("series"):
    print(f"\n{name}: clean4 {wape(part['actual'], part['clean4']):.3f} → 규칙 적용 {wape(part['actual'], part['final']):.3f}"
          f"   Bias {bias(part['actual'], part['final']):+.3f}")
    by_seg = part.groupby("segment").apply(
        lambda g: pd.Series({"days": len(g), "clean4": wape(g["actual"], g["clean4"]), "final": wape(g["actual"], g["final"])}),
        include_groups=False,
    )
    print(by_seg.round(3).to_string())