import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

result = pd.read_csv("reports/baseline_backtest.csv", parse_dates=["origin", "target"])
result["abs_err"] = (result["actual"] - result["clean4"]).abs()

for name in ["total", "fashion"]:
    part = result[result["series"] == name]
    by_seg = part.groupby("segment").agg(
        days=("target", "count"),
        actual_sum=("actual", "sum"),
        abs_err_sum=("abs_err", "sum"),
    )
    by_seg["actual_share"] = (by_seg["actual_sum"] / by_seg["actual_sum"].sum()).round(3)
    by_seg["error_share"] = (by_seg["abs_err_sum"] / by_seg["abs_err_sum"].sum()).round(3)
    print(f"\n===== {name}: clean4 오차가 나오는 구간 =====")
    print(by_seg[["days", "actual_share", "error_share"]].to_string())
    
weekday_names = ["월", "화", "수", "목", "금", "토", "일"]

holiday_rows = result[
    (result["series"] == "total") & result["segment"].isin(["공휴일", "명절 연휴"])
].copy()
holiday_rows["요일"] = holiday_rows["target"].dt.day_of_week.map(lambda i: weekday_names[i])
holiday_rows["ratio"] = (holiday_rows["actual"] / holiday_rows["clean4"]).round(3)

print("\n[공휴일·명절 연휴 날짜별: 실제 / 평소 수준]")
columns = ["target", "요일", "segment", "actual", "clean4", "ratio"]
print(holiday_rows.sort_values("target")[columns].to_string(index=False))

print("\n[구간별 비율 요약]")
print(holiday_rows.groupby("segment")["ratio"].agg(["count", "mean", "median", "min", "max"]).round(3).to_string())