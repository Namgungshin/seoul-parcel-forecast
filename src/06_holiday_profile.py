import pandas as pd

from forecast_common import FINAL_START, clean_median, load_daily

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

daily = load_daily()
y = daily["total"]

targets = daily.index[(daily.index >= "2018-04-01") & (daily.index < FINAL_START)]
ratios = []
for target in targets:
    base = clean_median(y, daily["is_clean"], target, origin=target - pd.Timedelta(days=1))
    ratios.append(y.loc[target] / base if base and base > 0 else float("nan"))

profile = daily.loc[targets, ["days_to_big", "days_since_big", "is_weekend", "is_holiday", "segment"]].copy()
profile["ratio"] = ratios
profile = profile[profile["is_holiday"] == 0].dropna(subset=["ratio"])
before = profile[profile["days_to_big"].between(1, 7)]
print("[명절 시작 전] days_to_big 별 평균 비율")
print(before.groupby(["days_to_big", "is_weekend"])["ratio"].agg(["mean", "count"]).round(2).to_string())

after = profile[profile["days_since_big"].between(1, 7)]
print("\n[명절 끝난 후] days_since_big 별 평균 비율")
print(after.groupby(["days_since_big", "is_weekend"])["ratio"].agg(["mean", "count"]).round(2).to_string())