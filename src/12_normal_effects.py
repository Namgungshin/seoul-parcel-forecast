import pandas as pd
from forecast_common import load_daily, clean_median, wape, bias

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

daily = load_daily()
is_off_day = (daily["is_holiday"] == 1) | (daily["is_big_holiday"] == 1)
daily["prev_off"] = is_off_day.shift(1, fill_value=False)
daily["next_off"] = is_off_day.shift(-1, fill_value=False)

backtest = pd.read_csv("reports/baseline_backtest.csv", parse_dates=["origin", "target"])
normal = backtest[backtest["segment"] == "일반"].copy()
normal["요일"] = normal["target"].dt.day_of_week.map(lambda i: ["월", "화", "수", "목", "금", "토", "일"][i])
normal["prev_off"] = daily.loc[normal["target"], "prev_off"].to_numpy()
normal["next_off"] = daily.loc[normal["target"], "next_off"].to_numpy()
normal["tag"] = "평범"
normal.loc[normal["next_off"], "tag"] = "내일 휴일"
normal.loc[normal["prev_off"], "tag"] = "어제 휴일"  # 둘 다면 어제 휴일 우선
def breakdown(part, key):
    out = part.groupby(key).apply(
        lambda g: pd.Series({
            "days": len(g),
            "actual_share": g["actual"].sum(),
            "wape": wape(g["actual"], g["clean4"]),
            "bias": bias(g["actual"], g["clean4"]),
            "error_share": (g["actual"] - g["clean4"]).abs().sum(),
        }),
        include_groups=False,
    )
    out["actual_share"] = out["actual_share"] / out["actual_share"].sum()
    out["error_share"] = out["error_share"] / out["error_share"].sum()
    return out.round(3)


for name in ["total", "fashion"]:
    part = normal[normal["series"] == name]
    print(f"\n########## {name}")
    print("\n[태그별]")
    print(breakdown(part, "tag").to_string())
    print("\n[태그 × 요일]")
    print(breakdown(part, ["tag", "요일"]).to_string())
    print("\n[일반일 WAPE: 같은 요일의 최근 n번 중앙값, 연도별]")
for name in ["total", "fashion"]:
    part = normal[normal["series"] == name].copy()
    for n in [2, 3, 4, 6, 8]:
        part[f"n{n}"] = [
            clean_median(daily[name], daily["is_clean"], t, o, n=n, max_back=20)
            for t, o in zip(part["target"], part["origin"])
        ]
    for year, g in part.groupby("year"):
        line = f"{name:8s}{year}"
        for n in [2, 3, 4, 6, 8]:
            line += f"   n={n}: {wape(g['actual'], g[f'n{n}']):.3f}"
        print(line)