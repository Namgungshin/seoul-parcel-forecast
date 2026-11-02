import pandas as pd
from forecast_common import load_daily, clean_median, wape, bias

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

daily = load_daily()
backtest = pd.read_csv("reports/baseline_backtest.csv", parse_dates=["origin", "target"])
normal = backtest[backtest["segment"] == "일반"].copy()
normal["요일"] = normal["target"].dt.day_of_week.map(lambda i: ["월", "화", "수", "목", "금", "토", "일"][i])


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
    print(f"\n########## {name}: 일반일 {len(part)}일, clean4 WAPE {wape(part['actual'], part['clean4']):.3f}")
    for key in ["요일", "h", "year"]:
        print(f"\n[{name}] {key}별")
        print(breakdown(part, key).to_string())
        print("\n[일반일 WAPE: 같은 요일의 최근 n번 중앙값]")
for name in ["total", "fashion"]:
    part = normal[normal["series"] == name]
    line = f"{name:8s}"
    for n in [2, 3, 4, 6, 8]:
        pred = [
            clean_median(daily[name], daily["is_clean"], t, o, n=n, max_back=20)
            for t, o in zip(part["target"], part["origin"])
        ]
        line += f"   n={n}: {wape(part['actual'], pd.Series(pred, index=part.index)):.3f}"
    print(line)