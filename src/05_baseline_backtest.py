from pathlib import Path

import pandas as pd

from forecast_common import FINAL_START, bias, clean_median, load_daily, wape

pd.set_option("display.width", 160)

daily = load_daily()

BT_START = pd.Timestamp("2021-01-01")
BT_END = pd.Timestamp("2022-12-31")
assert BT_END < FINAL_START

origins = [
    d for d in pd.date_range("2020-12-01", "2022-12-31", freq="D")
    if d.day_of_week == 6
    and d + pd.Timedelta(days=1) >= BT_START
    and d + pd.Timedelta(days=7) <= BT_END
]
print(f"원점(일요일) {len(origins)}개: {origins[0].date()} ~ {origins[-1].date()}")

rows = []
for origin in origins:
    for h in range(1, 8):
        target = origin + pd.Timedelta(days=h)
        assert BT_START <= target <= BT_END
        naive_day = target - pd.Timedelta(days=7)
        assert naive_day <= origin

        for name in ["total", "fashion"]:
            y = daily[name]
            rows.append({
                "series": name,
                "origin": origin,
                "target": target,
                "h": h,
                "actual": y.loc[target],
                "naive7": y.loc[naive_day],
                "clean4": clean_median(y, daily["is_clean"], target, origin),
                "segment": daily.loc[target, "segment"],
            })

result = pd.DataFrame(rows)
assert result["clean4"].notna().all(), "깨끗한 날이 하나도 없는 경우가 있습니다"
print(f"예측 행 수: {len(result):,}")

def summarize(df, group_cols):
    records = []
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        for model in ["naive7", "clean4"]:
            records.append({
                **dict(zip(group_cols, key)),
                "model": model,
                "n": len(g),
                "WAPE": round(wape(g["actual"], g[model]), 3),
                "Bias": round(bias(g["actual"], g[model]), 3),
            })
    return pd.DataFrame(records)


result["year"] = result["target"].dt.year

for name in ["total", "fashion"]:
    part = result[result["series"] == name]
    print(f"\n===== {name} =====")
    print("\n[전체]")
    print(summarize(part.assign(all="all"), ["all"]).to_string(index=False))
    print("\n[구간별]")
    print(summarize(part, ["segment"]).to_string(index=False))
    print("\n[연도별]")
    print(summarize(part, ["year"]).to_string(index=False))

Path("reports").mkdir(exist_ok=True)
result.to_csv("reports/baseline_backtest.csv", index=False, encoding="utf-8-sig")
print("\n저장: reports/baseline_backtest.csv")