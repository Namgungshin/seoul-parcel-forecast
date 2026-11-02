import numpy as np
import pandas as pd

# 아래 값은 모두 가정입니다 (실제 처리량·비용 데이터 없음)
ALPHAS = [1, 2, 5]   # θ건이 하루 밀리면 α인일(人日)의 비용
CLEAR_DAYS = 7       # 월요일 시점의 이월 적체를 그 주 7일에 나눠 갚도록 계획에 포함
POLICIES = {
    "naive7": "naive7",
    "clean4": "clean4",
    "P50(모델)": "pipeline",
    "P80(모델)": "p80",
    "oracle(실제값)": "actual",
}


def simulate(dates, actual, forecast, theta, clear_days=CLEAR_DAYS):
    """매일 인원 = 올림((예측 + 갚을 적체) / θ). 용량을 넘는 물량은 다음 날로 이월된다.

    clear_days가 None이면 적체를 갚을 계획이 없다(적체가 계속 쌓일 수 있음).
    """
    backlog, idle, extra = 0.0, 0.0, 0.0
    staffs, backlogs = [], []
    for d, a, f in zip(dates, actual, forecast):
        if clear_days and d.day_of_week == 0:  # 매주 월요일에 지난주 끝의 적체를 보고 계획을 세운다
            extra = backlog / clear_days
        s = np.ceil((f + extra) / theta)
        work = backlog + a                      # 오늘 처리할 물량 = 이월분 + 오늘 도착분
        capacity = s * theta
        idle += max(0.0, capacity - work)
        backlog = max(0.0, work - capacity)
        staffs.append(s)
        backlogs.append(backlog)
    return np.array(staffs), np.array(backlogs), idle


def run(part, n_staff, theta=None, clear_days=CLEAR_DAYS):
    """한 시리즈의 정책별 시뮬레이션 표. theta를 주지 않으면 이 구간의 일반 평일 평균 물량 / n_staff."""
    part = part.sort_values("target")
    if theta is None:
        weekday_normal = part[(part["segment"] == "일반") & (part["target"].dt.day_of_week < 5)]
        theta = weekday_normal["actual"].mean() / n_staff
    dates = list(part["target"])
    actual = part["actual"].to_numpy()
    rows = []
    for label, col in POLICIES.items():
        staff, backlog, idle = simulate(dates, actual, part[col].to_numpy(), theta, clear_days)
        backlog_person_days = backlog.sum() / theta
        row = {
            "policy": label,
            "labor": staff.sum(),
            "idle%": idle / (staff.sum() * theta) * 100,
            "late_days%": (backlog > 0).mean() * 100,
            "avg_backlog%": backlog.mean() / (theta * n_staff) * 100,
        }
        for a in ALPHAS:
            row[f"cost_a{a}"] = staff.sum() + a * backlog_person_days
        rows.append(row)
    out = pd.DataFrame(rows).set_index("policy")
    out["labor_vs_oracle%"] = (out["labor"] / out.loc["oracle(실제값)", "labor"] - 1) * 100
    for a in ALPHAS:
        out[f"cost_a{a}_vs_oracle%"] = (out[f"cost_a{a}"] / out.loc["oracle(실제값)", f"cost_a{a}"] - 1) * 100
    return out


SHOW = ["labor_vs_oracle%", "idle%", "late_days%", "avg_backlog%"] + [f"cost_a{a}_vs_oracle%" for a in ALPHAS]


def print_tables(part, label, theta_by_n=None, clear_days=CLEAR_DAYS):
    """n_staff 100은 전체 표, 30과 300은 P50과 P80만 보여 준다."""
    for n_staff in [100, 30, 300]:
        theta = None if theta_by_n is None else theta_by_n[n_staff]
        table = run(part, n_staff, theta, clear_days)
        if n_staff != 100:
            table = table.loc[["P50(모델)", "P80(모델)"]]
        print(f"\n===== {label}: 평일 평균 물량을 {n_staff}명이 처리한다고 가정 ({len(part)}일) =====")
        print(table[SHOW].round(1).to_string())
