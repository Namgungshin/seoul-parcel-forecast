import pandas as pd
from forecast_common import load_daily, clean_median, FINAL_START

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 300)

daily = load_daily()
names = ["월", "화", "수", "목", "금", "토", "일"]

is_off_day = (daily["is_holiday"] == 1) | (daily["is_big_holiday"] == 1)
tomorrow_weekend = pd.Series((daily.index + pd.Timedelta(days=1)).day_of_week >= 5, index=daily.index)
daily["next_off"] = is_off_day.shift(-1, fill_value=False)           # 내일이 공휴일
daily["next_work"] = ~daily["next_off"] & ~tomorrow_weekend           # 내일이 평일 근무일


def holiday_group(date):
    row = daily.loc[date]
    if row["segment"] == "명절 연휴":
        return "명절·다음날 근무" if row["next_work"] else "명절·연휴 중"
    if date.day_of_week == 6:
        return "일·다음날 휴일" if row["next_off"] else "일·다음날 평일"
    if date.day_of_week == 5:
        return "토"
    if date.day_of_week == 0:
        return "월"
    return "화~금"


targets = daily.index[daily["segment"].isin(["공휴일", "명절 연휴"]) & (daily.index < FINAL_START)]

rows = []
for target in targets:
    origin = target - pd.Timedelta(days=target.day_of_week + 1)  # 그 주를 예측하는 직전 일요일
    for series in ["total", "fashion"]:
        try:
            base = clean_median(daily[series], daily["is_clean"], target, origin)
        except KeyError:  # 2018년 초는 과거 데이터가 모자라 건너뜀
            continue
        rows.append({
            "series": series,
            "target": target,
            "year": target.year,
            "요일": names[target.day_of_week],
            "segment": daily.loc[target, "segment"],
            "group": holiday_group(target),
            "actual": daily.loc[target, series],
            "clean4": base,
            "ratio": daily.loc[target, series] / base,
        })

table = pd.DataFrame(rows).dropna(subset=["ratio"])
table["ratio"] = table["ratio"].round(3)
table.to_csv("reports/holiday_ratio.csv", index=False)

print("[그룹별 비율 요약]")
summary = table.groupby(["series", "group"])["ratio"].agg(["count", "median", "min", "max"]).round(3)
print(summary.to_string())

print("\n[전체 물량: 그룹별 연도 분포]")
total = table[table["series"] == "total"]
print(total.pivot_table(index="group", columns="year", values="ratio", aggfunc="median").round(3).to_string())

