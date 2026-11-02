import duckdb
import numpy as np
import pandas as pd

from forecast_common import DB_PATH, FINAL_START, load_daily
from forecast_pipeline import prepare_calendar

con = duckdb.connect(DB_PATH, read_only=True)
cat = con.execute("""
    SELECT date AS ds, category AS unique_id, CAST(SUM(volume) AS DOUBLE) AS y
    FROM parcel_long
    GROUP BY date, category
""").df()
con.close()
cat["ds"] = pd.to_datetime(cat["ds"]).astype("datetime64[ns]")

daily = prepare_calendar(load_daily())
targets = daily[["total", "fashion"]].astype(float).rename_axis("ds").reset_index()
targets["ds"] = targets["ds"].astype("datetime64[ns]")
targets = targets.melt(id_vars="ds", var_name="unique_id", value_name="y")
panel = pd.concat([targets, cat], ignore_index=True)

cal = pd.DataFrame(index=daily.index.astype("datetime64[ns]"))
cal.index.name = "ds"
for d in range(7):
    cal[f"dow{d}"] = (daily.index.day_of_week == d).astype(float)
cal["month_sin"] = np.sin(2 * np.pi * daily.index.month / 12)
cal["month_cos"] = np.cos(2 * np.pi * daily.index.month / 12)
cal["off"] = ((daily["is_holiday"] == 1) | (daily["is_big_holiday"] == 1)).astype(float).to_numpy()
cal["prev_off"] = daily["prev_off"].astype(float).to_numpy()
cal["next_off"] = daily["next_off"].astype(float).to_numpy()
cal["to_big"] = daily["days_to_big"].clip(upper=30).fillna(30).to_numpy() / 30
cal["since_big"] = daily["days_since_big"].clip(upper=30).fillna(30).to_numpy() / 30
# LightGBM이 맡던 날만 손실에 넣는다. 휴일과 휴일 다음 화~금은 규칙이 맡는다
cal["available_mask"] = ((daily["segment"] == "일반") & ~daily["after_off_weekday"]).astype(float).to_numpy()

panel = panel.merge(cal.reset_index(), on="ds", how="left")
panel = panel[panel["ds"] < FINAL_START].sort_values(["unique_id", "ds"])  # 2023년은 아예 넣지 않는다
assert panel["ds"].max() < FINAL_START
assert panel.isna().sum().sum() == 0
assert panel.groupby("unique_id").size().nunique() == 1

panel.to_parquet("reports/dl_panel.parquet", index=False)
print(panel.groupby("unique_id")["y"].agg(["size", "mean"]).round(0).to_string())
print("기간:", panel["ds"].min().date(), "~", panel["ds"].max().date())
print("마스크 비율:", round(panel["available_mask"].mean(), 3))