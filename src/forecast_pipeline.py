"""규칙 + 모델 예측 파이프라인 (05~19단계에서 만든 것을 한 곳에 모은 것).

한 예측 시점(일요일)마다 다음 월~일 7일의 P50과 P80을 만든다.
모든 설정은 2021~2022 백테스트에서 정했고, 여기서는 바꾸지 않는다.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

from forecast_common import clean_median

SERIES = ["total", "fashion"]
FEATURES = ["dow", "month", "next_off", "days_to_big", "days_since_big",
            "lag1", "lag2", "lag3", "lag4", "level_ratio"]
ALPHA = {"total": 0.80, "fashion": 0.84}  # 패션 알파는 2021년 커버리지로 고른 값(18b)
BAR_N = {"total": 6, "fashion": 1}        # 일반일에서 가장 좋았던 단순 기준선의 n(13, 15)
N_BASE = 4
LAGS = 4
FEATURE_START = pd.Timestamp("2018-03-04")
LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1,
)


def prepare_calendar(daily):
    daily = daily.copy()
    is_off_day = (daily["is_holiday"] == 1) | (daily["is_big_holiday"] == 1)
    tomorrow_weekend = pd.Series((daily.index + pd.Timedelta(days=1)).day_of_week >= 5, index=daily.index)
    daily["prev_off"] = is_off_day.shift(1, fill_value=False)
    daily["next_off"] = is_off_day.shift(-1, fill_value=False)
    daily["next_work"] = ~daily["next_off"] & ~tomorrow_weekend
    daily["is_clean2"] = daily["is_clean"] & ~daily["prev_off"] & ~daily["next_off"]
    tue_to_fri = pd.Series(daily.index.day_of_week.isin([1, 2, 3, 4]), index=daily.index)
    daily["after_off_weekday"] = daily["prev_off"] & tue_to_fri
    return daily


def holiday_group(daily, date):
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


def prepost_group(daily, date):
    row = daily.loc[date]
    side = "전" if row["days_to_big"] <= 3 else "후"
    dist = int(row["days_to_big"] if side == "전" else row["days_since_big"])
    weekend = "주말" if row["is_weekend"] else "평일"
    return f"{side}{dist}", f"{side}{dist}·{weekend}"


def build_ratio_tables(daily, last_date):
    """공휴일·명절 연휴, 명절 전후 3일의 (실제 / clean4) 비율표. last_date까지의 날만 쓴다."""
    hol_rows, pre_rows = [], []
    for target in daily.index[daily.index <= last_date]:
        segment = daily.loc[target, "segment"]
        if segment not in ("공휴일", "명절 연휴", "명절 전후 3일"):
            continue
        origin = target - pd.Timedelta(days=target.day_of_week + 1)
        for series in SERIES:
            try:
                base = clean_median(daily[series], daily["is_clean"], target, origin)
            except KeyError:
                continue
            if pd.isna(base):
                continue
            ratio = daily.loc[target, series] / base
            if segment == "명절 전후 3일":
                pool, group = prepost_group(daily, target)
                pre_rows.append({"series": series, "target": target, "pool": pool, "group": group, "ratio": ratio})
            else:
                hol_rows.append({"series": series, "target": target,
                                 "group": holiday_group(daily, target), "ratio": round(ratio, 3)})
    return pd.DataFrame(hol_rows), pd.DataFrame(pre_rows)


def build_feature_rows(daily, origins):
    rows = []
    for origin in origins:
        recent = daily.loc[origin - pd.Timedelta(days=13): origin]
        prior = daily.loc[origin - pd.Timedelta(days=27): origin - pd.Timedelta(days=14)]
        for h in range(1, 8):
            target = origin + pd.Timedelta(days=h)
            if daily.loc[target, "segment"] != "일반":
                continue
            for name in SERIES:
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
                }
                for k in range(1, LAGS + 1):
                    past = target - pd.Timedelta(days=7 * k)
                    assert past <= origin, "예측 시점 이후 값을 쓰려 했습니다"
                    row[f"lag{k}"] = y.loc[past] / base if daily.loc[past, "is_clean2"] else float("nan")
                recent_mean = recent.loc[recent["is_clean2"], name].mean()
                prior_mean = prior.loc[prior["is_clean2"], name].mean()
                row["level_ratio"] = recent_mean / prior_mean if prior_mean > 0 else float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def fit_predict(train, test, series, kind):
    params = dict(LGB_PARAMS)
    if kind == "p50":
        params.update(objective="l1")
    else:
        params.update(objective="quantile", alpha=ALPHA[series])
    model = lgb.LGBMRegressor(**params)
    model.fit(train[FEATURES], train["y_ratio"])
    return model.predict(test[FEATURES])


def _ratio_stats(past):
    if not len(past):
        return 1.0, 1.0
    return past["ratio"].median(), past["ratio"].quantile(0.8)


def run_pipeline(daily, origins):
    """origins의 각 일요일에 대해 다음 7일의 예측을 만든다. 그 시점에 끝난 사례와 값만 쓴다."""
    last_target = origins[-1] + pd.Timedelta(days=7)
    feats = build_feature_rows(daily, pd.date_range(FEATURE_START, origins[-1], freq="W-SUN"))
    hol, pre = build_ratio_tables(daily, last_target)
    after = feats[feats["after_off"] == 1]
    model_feats = feats[feats["after_off"] == 0]

    rows = []
    for origin in origins:
        model_pred = {}
        for series in SERIES:
            part = model_feats[model_feats["series"] == series]
            train = part[part["target"] <= origin]  # 그 시점에 실제 값이 알려진 행만
            test = part[part["origin"] == origin]
            if not len(test):
                continue
            p50 = test["base"].to_numpy() * fit_predict(train, test, series, "p50")
            p80 = test["base"].to_numpy() * fit_predict(train, test, series, "p80")
            for t, a, b in zip(test["target"], p50, p80):
                model_pred[(series, t)] = (a, b)

        for h in range(1, 8):
            target = origin + pd.Timedelta(days=h)
            segment = daily.loc[target, "segment"]
            for series in SERIES:
                y = daily[series]
                clean4 = clean_median(y, daily["is_clean"], target, origin)
                row = {
                    "series": series, "origin": origin, "target": target, "h": h, "segment": segment,
                    "actual": y.loc[target], "naive7": y.loc[target - pd.Timedelta(days=7)],
                    "clean4": clean4, "base": np.nan, "bar": np.nan,
                }
                if segment == "일반":
                    base = clean_median(y, daily["is_clean2"], target, origin, n=N_BASE)
                    row["base"] = base
                    if daily.loc[target, "after_off_weekday"]:
                        past = after[(after["series"] == series) & (after["target"] <= origin)]
                        if len(past) >= 3:
                            med, q80 = past["y_ratio"].median(), past["y_ratio"].quantile(0.8)
                        else:
                            med, q80 = 1.0, 1.0
                        p50, p80 = base * med, base * q80
                        kind = "일반·휴일 다음 날"
                    else:
                        p50, p80 = model_pred[(series, target)]
                        row["bar"] = clean_median(y, daily["is_clean2"], target, origin,
                                                  n=BAR_N[series], max_back=20)
                        kind = "일반·모델"
                    rules_only = clean4
                elif segment in ("공휴일", "명절 연휴"):
                    group = holiday_group(daily, target)
                    past = hol[(hol["series"] == series) & (hol["group"] == group) & (hol["target"] <= origin)]
                    med, q80 = _ratio_stats(past)
                    p50, p80 = clean4 * med, clean4 * q80
                    kind, rules_only = segment, p50
                else:  # 명절 전후 3일
                    pool, group = prepost_group(daily, target)
                    past = pre[(pre["series"] == series) & (pre["target"] <= origin)]
                    same = past[past["group"] == group]
                    if len(same) < 2:  # 칸이 비면 평일/주말을 합쳐서 본다
                        same = past[past["pool"] == pool]
                    med, q80 = _ratio_stats(same)
                    p50, p80 = clean4 * med, clean4 * q80
                    kind, rules_only = segment, p50
                row.update(kind=kind, rules_only=rules_only, pipeline=p50, p80=max(p80, p50))
                rows.append(row)

    pred = pd.DataFrame(rows)
    pred["covered"] = pred["actual"] <= pred["p80"]
    return pred
