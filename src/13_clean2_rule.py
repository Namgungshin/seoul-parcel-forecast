import pandas as pd
from forecast_common import load_daily, clean_median, wape, FINAL_START

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

daily = load_daily()
is_off_day = (daily["is_holiday"] == 1) | (daily["is_big_holiday"] == 1)
daily["prev_off"] = is_off_day.shift(1, fill_value=False)
daily["next_off"] = is_off_day.shift(-1, fill_value=False)
daily["is_clean2"] = daily["is_clean"] & ~daily["prev_off"] & ~daily["next_off"]  # 휴일 이웃도 기준값에서 제외

tue_to_fri = pd.Series(daily.index.day_of_week.isin([1, 2, 3, 4]), index=daily.index)
daily["after_off_weekday"] = daily["prev_off"] & tue_to_fri  # 휴일 다음 날이 화~금

backtest = pd.read_csv("reports/baseline_backtest.csv", parse_dates=["origin", "target"])
normal = backtest[backtest["segment"] == "일반"].copy()
normal["after_off"] = daily.loc[normal["target"], "after_off_weekday"].to_numpy()
normal["plain"] = ~(daily.loc[normal["target"], "prev_off"] | daily.loc[normal["target"], "next_off"]).to_numpy()
history_targets = daily.index[
    (daily["segment"] == "일반") & daily["after_off_weekday"] & (daily.index < FINAL_START)
]


def ratio_table(name, n):
    rows = []
    for t in history_targets:
        origin = t - pd.Timedelta(days=t.day_of_week + 1)
        try:
            base = clean_median(daily[name], daily["is_clean2"], t, origin, n=n, max_back=20)
        except KeyError:
            continue
        rows.append((t, daily.loc[t, name] / base))
    return pd.DataFrame(rows, columns=["target", "ratio"]).dropna()

ns = [2, 3, 4, 6, 8]
for name in ["total", "fashion"]:
    part = normal[normal["series"] == name].copy()
    print(f"\n########## {name}: 일반 {len(part)}일 (평범한 날 {int(part['plain'].sum())}일)")
    print("  n   기존clean  새clean  새clean+규칙  | 평범한 날만: 기존  새")
    for n in ns:
        part["old"] = [clean_median(daily[name], daily["is_clean"], t, o, n=n, max_back=20)
    for t, o in zip(part["target"], part["origin"])]
        part["new"] = [clean_median(daily[name], daily["is_clean2"], t, o, n=n, max_back=20)
    for t, o in zip(part["target"], part["origin"])]
        table = ratio_table(name, n)

        def factor(row):
            if not row["after_off"]:
                return 1.0
            past = table[table["target"] <= row["origin"]]
            return past["ratio"].median() if len(past) >= 3 else 1.0

        part["rule"] = part["new"] * part.apply(factor, axis=1)
        plain = part[part["plain"]]
        print(f"{n:>3}   {wape(part['actual'], part['old']):.3f}     {wape(part['actual'], part['new']):.3f}"
        f"      {wape(part['actual'], part['rule']):.3f}        |"
        f"   {wape(plain['actual'], plain['old']):.3f}   {wape(plain['actual'], plain['new']):.3f}")
    for name in ["total", "fashion"]:
        t = ratio_table(name, 4)
        t["요일"] = t["target"].dt.day_of_week.map(lambda i: ["월", "화", "수", "목", "금", "토", "일"][i])
        t["ratio"] = t["ratio"].round(2)
    print(f"\n[{name}: 휴일 다음 화~금, 실제 / 새 clean4]")
    print(t.to_string(index=False))
