import duckdb
import pandas as pd

DB_PATH = "data/parcel.duckdb"
FINAL_START = pd.Timestamp("2023-01-01")  # 최종 평가 구간. 모델 선택에는 쓰지 않는다


def _segment(row):
    if row["is_big_holiday"] == 1:
        return "명절 연휴"
    if row["is_holiday"] == 1:
        return "공휴일"
    if row["days_to_big"] <= 3 or row["days_since_big"] <= 3:
        return "명절 전후 3일"
    return "일반"


def load_daily():
    con = duckdb.connect(DB_PATH, read_only=True)
    daily = con.execute("""
        SELECT d.date, d.total, d.fashion,
        c.is_holiday, c.is_big_holiday, c.days_to_big, c.days_since_big
        FROM daily_target d
        JOIN calendar c USING (date)
        ORDER BY d.date
    """).df()
    con.close()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")
    daily["segment"] = daily.apply(_segment, axis=1)
    daily["is_clean"] = daily["segment"] == "일반"
    daily["is_weekend"] = daily.index.day_of_week >= 5
    return daily
def clean_median(y, is_clean, target, origin, n=4, max_back=12):
    values = []
    for k in range(1, max_back + 1):
        past = target - pd.Timedelta(days=7 * k)
        assert past <= origin, "예측 시점 이후 데이터를 쓰려 했습니다"
        if is_clean.loc[past]:
            values.append(y.loc[past])
        if len(values) == n:
            break
    return float(pd.Series(values).median()) if values else float("nan")


def wape(actual, pred):
    return (actual - pred).abs().sum() / actual.sum()


def bias(actual, pred):
    return (pred - actual).sum() / actual.sum()
