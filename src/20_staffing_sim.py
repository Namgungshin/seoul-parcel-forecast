import pandas as pd

from sim_common import SHOW, print_tables, run

pd.set_option("display.width", 160)

pipe = pd.read_csv("reports/backtest_p80.csv", parse_dates=["origin", "target"])

for name, part in pipe.groupby("series"):
    print_tables(part, f"{name}, 적체 상환 포함")

# 비교용: 적체를 갚을 계획이 없는 첫 설계 (적체가 끝없이 쌓여 현실적이지 않음)
print("\n########## 비교용: 적체 상환 없음 (첫 설계) ##########")
for name, part in pipe.groupby("series"):
    table = run(part, 100, clear_days=None)
    print(f"\n===== {name}, 적체 상환 없음, 100명 =====")
    print(table[SHOW].round(1).to_string())
