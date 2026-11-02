"""21단계: 2023년 최종 평가.

사용법
  python src/21_final_2023.py check   # 2021~2022를 같은 코드로 다시 돌려 이전 숫자가 재현되는지 확인
  python src/21_final_2023.py final   # 2023년 최종 평가 (한 번만. 다시 돌리려면 --again)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_common import FINAL_START, bias, load_daily, wape
from forecast_pipeline import prepare_calendar, run_pipeline
from sim_common import print_tables

pd.set_option("display.width", 160)

mode = sys.argv[1] if len(sys.argv) > 1 else "check"
assert mode in ("check", "final"), "check 또는 final을 지정하세요"

if mode == "check":
    origins = list(pd.date_range("2021-01-03", "2022-12-18", freq="W-SUN"))
    assert origins[-1] + pd.Timedelta(days=7) < FINAL_START, "check 모드가 2023년을 건드립니다"
    out_path = Path("reports/check_pipeline_2021_2022.csv")
    period = "year"
else:
    origins = list(pd.date_range("2023-01-01", "2023-12-24", freq="W-SUN"))
    out_path = Path("reports/final_2023_pred.csv")
    period = "quarter"
    if out_path.exists() and "--again" not in sys.argv:
        sys.exit("2023년 최종 평가는 이미 실행했습니다. 정말 다시 돌리려면 --again을 붙이세요.")

daily = prepare_calendar(load_daily())
print(f"모드 {mode}: 예측 시점 {len(origins)}개 ({origins[0].date()} ~ {origins[-1].date()})")

pred = run_pipeline(daily, origins)
pred["year"] = pred["target"].dt.year
pred["quarter"] = pred["target"].dt.year.astype(str) + "Q" + pred["target"].dt.quarter.astype(str)
pred.to_csv(out_path, index=False)
print(f"예측 행 수 {len(pred)} (비어 있는 예측 {int(pred[['pipeline', 'p80']].isna().sum().sum())}), 저장: {out_path}")

rng = np.random.default_rng(0)


def week_bootstrap(part, model_col, base_col, draws=2000):
    g = part.assign(
        err_m=(part["actual"] - part[model_col]).abs(),
        err_b=(part["actual"] - part[base_col]).abs(),
    ).groupby("origin")[["err_m", "err_b", "actual"]].sum()
    idx = rng.integers(0, len(g), size=(draws, len(g)))
    m = g["err_m"].to_numpy()[idx].sum(axis=1)
    b = g["err_b"].to_numpy()[idx].sum(axis=1)
    a = g["actual"].to_numpy()[idx].sum(axis=1)
    diff = (m - b) / a
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return lo, hi, (diff < 0).mean()


for name, part in pred.groupby("series"):
    print(f"\n===== {name}: {len(part)}일 =====")
    for col, label in [("naive7", "naive7"), ("clean4", "clean4(처음)"),
                       ("rules_only", "규칙만"), ("pipeline", "규칙 + 모델")]:
        by = "  ".join(f"{k}: {wape(g['actual'], g[col]):.3f}" for k, g in part.groupby(period))
        print(f"{label:12s} WAPE {wape(part['actual'], part[col]):.3f}   Bias {bias(part['actual'], part[col]):+.3f}   {by}")

    print("\n[날의 종류별 WAPE]")
    by_kind = part.groupby("kind").apply(
        lambda g: pd.Series({
            "days": len(g),
            "naive7": wape(g["actual"], g["naive7"]),
            "규칙만": wape(g["actual"], g["rules_only"]),
            "규칙+모델": wape(g["actual"], g["pipeline"]),
            "P80 커버리지": g["covered"].mean(),
        }),
        include_groups=False,
    )
    print(by_kind.round(3).to_string())

    over80 = part["p80"].sum() / part["actual"].sum() - 1
    over50 = part["pipeline"].sum() / part["actual"].sum() - 1
    by_period = "  ".join(f"{k}: {g['covered'].mean():.3f}" for k, g in part.groupby(period))
    print(f"\nP80 커버리지 {part['covered'].mean():.3f}   Σ(P80)/Σ(실제)−1 = {over80:+.3f}   "
          f"Σ(P50)/Σ(실제)−1 = {over50:+.3f}\n  {by_period}")
    rule_days = part[part["kind"] != "일반·모델"]
    print(f"  모델이 아닌 날(규칙 날) {len(rule_days)}일의 P80 커버리지 {rule_days['covered'].mean():.3f}")

    model_days = part[part["kind"] == "일반·모델"]
    bar_label = f"clean2_n{ {'total': 6, 'fashion': 1}[name] }"
    print(f"\n[일반·모델 {len(model_days)}일] 모델 {wape(model_days['actual'], model_days['pipeline']):.4f}"
          f"   가장 좋은 단순 기준선({bar_label}) {wape(model_days['actual'], model_days['bar']):.4f}"
          f"   모델의 기준값(clean2_n4) {wape(model_days['actual'], model_days['base']):.4f}")
    lo, hi, p = week_bootstrap(model_days, "pipeline", "bar")
    print(f"  주 단위 부트스트랩 (모델 − 기준선) [{lo:+.4f}, {hi:+.4f}]  모델이 나은 비율 {p:.2f}")
    lo, hi, p = week_bootstrap(part, "pipeline", "rules_only")
    print(f"  전체 {len(part)}일 부트스트랩 (규칙+모델 − 규칙만) [{lo:+.4f}, {hi:+.4f}]  모델이 나은 비율 {p:.2f}")

if mode == "final":
    # θ는 2021~2022 백테스트 구간으로 고정 (결과를 보고 바꾸지 않는다)
    window = daily.loc["2021-01-04":"2022-12-25"]
    for name, part in pred.groupby("series"):
        normal_weekdays = window[(window["segment"] == "일반") & (window.index.day_of_week < 5)]
        theta_by_n = {n: normal_weekdays[name].mean() / n for n in (30, 100, 300)}
        print_tables(part, f"2023 {name}, θ는 2021~2022 기준", theta_by_n)
