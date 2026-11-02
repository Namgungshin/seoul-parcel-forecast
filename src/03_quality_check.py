from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

con = duckdb.connect("data/parcel.duckdb", read_only=True)


def show(title, sql):
    print(f"\n[{title}]")
    print(con.execute(sql).df().to_string(index=False))


show("기간", """
    SELECT MIN(date) AS start_date, MAX(date) AS end_date,
           COUNT(DISTINCT date) AS n_days, COUNT(*) AS n_rows
    FROM parcel_wide
""")

show("빠진 날짜", """
    SELECT CAST(d AS DATE) AS missing_date
    FROM generate_series(DATE '2018-01-01', DATE '2023-12-31', INTERVAL 1 DAY) AS t(d)
    WHERE CAST(d AS DATE) NOT IN (SELECT date FROM parcel_wide)
""")

show("중복 조합 수", """
    SELECT COUNT(*) AS n_duplicates
    FROM (
        SELECT date, origin_code, gu_code
        FROM parcel_wide
        GROUP BY 1, 2, 3
        HAVING COUNT(*) > 1
    )
""")

show("연도별 행 채움률", """
    SELECT year(date) AS yr, COUNT(*) AS n_rows, COUNT(DISTINCT date) AS n_days,
           ROUND(COUNT(*) / (17 * 25.0 * COUNT(DISTINCT date)), 4) AS fill_rate
    FROM parcel_wide
    GROUP BY 1
    ORDER BY 1
""")
show("품목 목록", "SELECT DISTINCT category FROM parcel_long ORDER BY 1")

show("작은 값의 비율", """
    SELECT year(date) AS yr,
           ROUND(AVG(CASE WHEN volume = 0 THEN 1.0 ELSE 0 END), 4) AS v0,
           ROUND(AVG(CASE WHEN volume = 1 THEN 1.0 ELSE 0 END), 4) AS v1,
           ROUND(AVG(CASE WHEN volume = 2 THEN 1.0 ELSE 0 END), 4) AS v2,
           ROUND(AVG(CASE WHEN volume = 3 THEN 1.0 ELSE 0 END), 4) AS v3,
           ROUND(AVG(CASE WHEN volume = 4 THEN 1.0 ELSE 0 END), 4) AS v4,
           ROUND(AVG(CASE WHEN volume = 5 THEN 1.0 ELSE 0 END), 4) AS v5
    FROM parcel_long
    GROUP BY 1
    ORDER BY 1
""")
daily = con.execute("""
    SELECT date,
           SUM(volume) AS total,
           SUM(CASE WHEN category IN ('패션의류', '패션잡화') THEN volume ELSE 0 END) AS fashion
    FROM parcel_long
    GROUP BY date
    ORDER BY date
""").df()
con.close()

daily["date"] = pd.to_datetime(daily["date"])
daily["weekday"] = daily["date"].dt.dayofweek
weekday_names = ["월", "화", "수", "목", "금", "토", "일"]

print("\n[연도별 합계]")
print(daily.groupby(daily["date"].dt.year)[["total", "fashion"]].sum().to_string())

print("\n[요일별 평균]")
by_weekday = daily.groupby("weekday")[["total", "fashion"]].mean().round()
by_weekday.index = weekday_names
print(by_weekday.to_string())

daily["baseline"] = daily.groupby("weekday")["total"].transform(
    lambda s: s.shift(1).rolling(4).median()
)
daily["ratio"] = (daily["total"] / daily["baseline"]).round(3)
daily["요일"] = daily["weekday"].map(lambda i: weekday_names[i])

columns = ["date", "요일", "total", "baseline", "ratio"]
print("\n[평소보다 가장 적은 날 25개]")
print(daily.nsmallest(25, "ratio")[columns].to_string(index=False))
print("\n[평소보다 가장 많은 날 15개]")
print(daily.nlargest(15, "ratio")[columns].to_string(index=False))
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for ax, col, title in zip(axes, ["total", "fashion"], ["서울 도착 전체", "패션의류 + 패션잡화"]):
    ax.plot(daily["date"], daily[col], linewidth=0.5, alpha=0.5, label="일별")
    ax.plot(daily["date"], daily[col].rolling(28).mean(), linewidth=2, label="28일 평균")
    ax.set_title(title)
    ax.legend()

output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(output_dir / "daily_volume.png", dpi=150)
print(f"\n그래프 저장: {output_dir / 'daily_volume.png'}")