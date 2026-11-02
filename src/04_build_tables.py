import duckdb
import holidays
import pandas as pd

DB_PATH = "data/parcel.duckdb"

kr = holidays.country_holidays("KR", years=range(2017, 2025))
holiday_df = pd.DataFrame(
    [(pd.Timestamp(d), name) for d, name in sorted(kr.items())],
    columns=["date", "holiday_name"],
)
holiday_df["date"] = holiday_df["date"].astype("datetime64[ns]")
print(holiday_df[holiday_df["date"].between("2018-01-01", "2023-12-31")].to_string(index=False))

is_seollal = holiday_df["holiday_name"].str.contains("설날|Lunar New Year", regex=True)
is_chuseok = holiday_df["holiday_name"].str.contains("추석|Chuseok", regex=True)
holiday_df["is_big_holiday"] = (is_seollal | is_chuseok).astype(int)

big = holiday_df[holiday_df["is_big_holiday"] == 1]
print("\n[명절로 분류된 날]")
print(big[big["date"].between("2018-01-01", "2023-12-31")].to_string(index=False))

calendar = pd.DataFrame({"date": pd.date_range("2018-01-01", "2023-12-31", freq="D").astype("datetime64[ns]")})
calendar = calendar.merge(holiday_df, on="date", how="left")
calendar["is_holiday"] = calendar["holiday_name"].notna().astype(int)
calendar["is_big_holiday"] = calendar["is_big_holiday"].fillna(0).astype(int)
calendar["day_of_week"] = calendar["date"].dt.day_of_week
calendar["is_weekend"] = (calendar["day_of_week"] >= 5).astype(int)
calendar["is_off"] = ((calendar["is_weekend"] == 1) | (calendar["is_holiday"] == 1)).astype(int)

big_days = big["date"].sort_values().reset_index(drop=True)

next_big = pd.merge_asof(
    calendar[["date"]], big_days.rename("next_big").to_frame(),
    left_on="date", right_on="next_big", direction="forward",
)
prev_big = pd.merge_asof(
    calendar[["date"]], big_days.rename("prev_big").to_frame(),
    left_on="date", right_on="prev_big", direction="backward",
)

calendar["days_to_big"] = (next_big["next_big"] - calendar["date"]).dt.days
calendar["days_since_big"] = (calendar["date"] - prev_big["prev_big"]).dt.days

con = duckdb.connect(DB_PATH)
con.execute("CREATE OR REPLACE TABLE calendar AS SELECT * FROM calendar")

con.execute("""
    CREATE OR REPLACE TABLE daily_target AS
    SELECT
        date,
        CAST(SUM(volume) AS BIGINT) AS total,
        CAST(SUM(CASE WHEN category IN ('패션의류', '패션잡화') THEN volume ELSE 0 END) AS BIGINT) AS fashion
    FROM parcel_long
    GROUP BY date
    ORDER BY date
""")

joined = con.execute("""
    SELECT d.date, d.total, d.fashion, c.day_of_week, c.is_holiday, c.is_big_holiday,
        c.is_off, c.days_to_big, c.days_since_big, c.holiday_name
    FROM daily_target d
    JOIN calendar c USING (date)
    ORDER BY d.date
""").df()
print(f"\n행 수: {len(joined)} (2191이어야 합니다)")
print(joined.groupby(joined["date"].dt.year)[["total", "fashion"]].sum().to_string())

print("\n[쉬는 날 여부별 평균 물량]")
print(joined.groupby("is_off")[["total", "fashion"]].mean().round().to_string())

print("\n[명절 당일과 전후 물량]")
near = joined[(joined["days_to_big"] <= 3) | (joined["days_since_big"] <= 3)]
print(near.groupby("days_to_big")["total"].mean().round().head(8).to_string())

low_dates = ["2018-02-16", "2018-09-24", "2019-02-05", "2019-09-13", "2020-01-24",
        "2020-10-01", "2021-02-12", "2021-09-21", "2022-02-01", "2022-09-10",
        "2023-01-22", "2023-09-29"]
check = joined[joined["date"].isin(pd.to_datetime(low_dates))]
print("\n[1단계의 극저점 날짜가 달력에서 어떻게 분류됐는지]")
print(check[["date", "total", "holiday_name", "is_big_holiday", "is_off"]].to_string(index=False))
con.close()