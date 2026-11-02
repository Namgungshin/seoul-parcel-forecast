import pandas as pd
from forecast_common import load_daily, clean_median

pd.set_option("display.width", 160)

daily = load_daily()
is_off_day = (daily["is_holiday"] == 1) | (daily["is_big_holiday"] == 1)
daily["prev_off"] = is_off_day.shift(1, fill_value=False)
daily["next_off"] = is_off_day.shift(-1, fill_value=False)
daily["is_clean2"] = daily["is_clean"] & ~daily["prev_off"] & ~daily["next_off"]
tue_to_fri = pd.Series(daily.index.day_of_week.isin([1, 2, 3, 4]), index=daily.index)
daily["after_off_weekday"] = daily["prev_off"] & tue_to_fri

N_BASE = 4
LAGS = 4
BACKTEST_START = pd.Timestamp("2021-01-03")  # 백테스트 첫 예측 시점
origins = pd.date_range("2018-03-04", "2022-12-18", freq="W-SUN")

rows = []
for origin in origins:
    recent = daily.loc[origin - pd.Timedelta(days=13): origin]
    prior = daily.loc[origin - pd.Timedelta(days=27): origin - pd.Timedelta(days=14)]
    for h in range(1, 8):
        target = origin + pd.Timedelta(days=h)
        if daily.loc[target, "segment"] != "일반":
            continue  # 휴일, 명절 전후는 규칙이 맡는다
        for name in ["total", "fashion"]:
            y = daily[name]
            try:
                base = clean_median(y, daily["is_clean2"], target, origin, n=N_BASE)
            except KeyError:
                continue
            if pd.isna(base):
                continue
            row = {
                "series": name, "origin": origin, "target": target,
                "dow": target.day_of_week, "month": target.month,
                "next_off": int(daily.loc[target, "next_off"]),
                "after_off": int(daily.loc[target, "after_off_weekday"]),
                "days_to_big": min(float(daily.loc[target, "days_to_big"]), 30),
                "days_since_big": min(float(daily.loc[target, "days_since_big"]), 30),
                "actual": y.loc[target], "base": base, "y_ratio": y.loc[target] / base,
                "in_backtest": origin >= BACKTEST_START,
            }
            for k in range(1, LAGS + 1):
                past = target - pd.Timedelta(days=7 * k)
                assert past <= origin, "예측 시점 이후 값을 쓰려 했습니다"
                row[f"lag{k}"] = y.loc[past] / base if daily.loc[past, "is_clean2"] else float("nan")
            recent_mean = recent.loc[recent["is_clean2"], name].mean()
            prior_mean = prior.loc[prior["is_clean2"], name].mean()
            row["level_ratio"] = recent_mean / prior_mean if prior_mean > 0 else float("nan")
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("reports/features.csv", index=False)

print("행 수:", df.groupby(["series", "in_backtest"]).size().to_string())
print("\n[결측 비율]")
print(df[["lag1", "lag2", "lag3", "lag4", "level_ratio"]].isna().mean().round(3).to_string())

feats = ["lag1", "lag2", "lag3", "lag4", "level_ratio"]
for name in ["total", "fashion"]:
    sub = df[(df["series"] == name) & (df["after_off"] == 0) & (df["dow"] <= 4)]
    print(f"\n[{name}] 평일 {len(sub)}행: y_ratio 분포")
    print(sub["y_ratio"].describe().round(3).to_string())
    print(f"\n[{name}] y_ratio와의 상관")
    print(sub[feats + ["y_ratio"]].corr()["y_ratio"].drop("y_ratio").round(2).to_string())