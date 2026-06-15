from __future__ import annotations

import csv
import json
import os
from typing import Optional

import numpy as np

from .models import (
    ScheduleResult,
    ScheduleProblem,
    SLOT_MINUTES,
    SLOTS_PER_HOUR,
    VEHICLE_CONFIG,
    VehicleType,
)


def _slot_time_str(slot_idx: int) -> str:
    hour = slot_idx // SLOTS_PER_HOUR
    minute = (slot_idx % SLOTS_PER_HOUR) * SLOT_MINUTES
    return f"{hour:02d}:{minute:02d}"


def print_terminal_report(result: ScheduleResult) -> None:
    problem = result.problem
    sessions = problem.sessions
    num_sessions = result.num_sessions
    num_slots = result.num_slots
    hours_per_slot = SLOT_MINUTES / 60.0

    print("=" * 70)
    print(f"  充电排程报表  |  策略: {result.strategy_name}")
    print("=" * 70)

    peak_load = float(np.max(result.slot_total_load_kw))
    peak_slot = int(np.argmax(result.slot_total_load_kw))
    valley_load = float(np.min(result.slot_total_load_kw[result.slot_total_load_kw > 0])) \
        if np.any(result.slot_total_load_kw > 0) else 0.0
    peak_valley_diff = peak_load - valley_load

    total_cost = result.get_total_cost()
    valley_ratio = result.get_valley_energy_ratio()

    num_fully_charged = sum(
        1 for i in range(num_sessions)
        if result.get_session_completion_ratio(i) >= 0.999
    )
    avg_completion = (
        sum(result.get_session_completion_ratio(i) for i in range(num_sessions))
        / num_sessions if num_sessions > 0 else 0.0
    )

    total_energy = float(np.sum(result.slot_total_load_kw) * hours_per_slot)

    print()
    print("【系统指标】")
    print(f"  全天峰值负荷:      {peak_load:.2f} kW (@ {_slot_time_str(peak_slot)})")
    print(f"  谷段最低负荷:      {valley_load:.2f} kW")
    print(f"  峰谷差:            {peak_valley_diff:.2f} kW")
    print(f"  总充电电量:        {total_energy:.2f} kWh")
    print(f"  总电费:            ¥{total_cost:.2f}")
    print(f"  谷段电量占比:      {valley_ratio * 100:.1f}%")
    print(f"  充满车辆数:        {num_fully_charged}/{num_sessions}")
    print(f"  平均完成度:        {avg_completion * 100:.1f}%")

    print()
    print("【各车充电明细】")
    header = f"  {'ID':<10} {'类型':<8} {'到达':<6} {'离开':<6} "
    header += f"{'需求kWh':>8} {'实充kWh':>8} {'完成率':>7} {'上限kW':>7}"
    print(header)
    print("  " + "-" * 70)

    for i, sess in enumerate(sessions):
        delivered = result.get_session_energy(i)
        ratio = result.get_session_completion_ratio(i)
        vtype_name = VEHICLE_CONFIG[sess.vehicle_type]["name"]
        arr_str = _slot_time_str(sess.arrival_slot)
        dep_str = _slot_time_str(sess.effective_departure_slot)
        line = (
            f"  {sess.session_id:<10} {vtype_name:<8} {arr_str:<6} {dep_str:<6} "
            f"{sess.required_energy_kwh:>8.2f} {delivered:>8.2f} "
            f"{ratio*100:>6.1f}% {sess.max_power_kw:>7.2f}"
        )
        if sess.actual_departure_slot is not None:
            line += "  [提前离开]"
        if sess.charger_max_power_factor < 0.99:
            line += f"  [桩功率×{sess.charger_max_power_factor:.1f}]"
        print(line)

    print()
    print("【负荷曲线（每小时）】")
    hourly = np.zeros(24)
    for h in range(24):
        s = h * SLOTS_PER_HOUR
        e = s + SLOTS_PER_HOUR
        hourly[h] = float(np.mean(result.slot_total_load_kw[s:e]))
    max_h = float(np.max(hourly)) if np.max(hourly) > 0 else 1.0
    for h in range(24):
        bar_len = int(hourly[h] / max_h * 30)
        bar = "█" * bar_len
        print(f"  {h:02d}:00  {hourly[h]:>6.2f} kW  {bar}")

    print()
    print("=" * 70)


def export_csv(result: ScheduleResult, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    problem = result.problem
    hours_per_slot = SLOT_MINUTES / 60.0

    load_path = os.path.join(output_dir, f"{result.strategy_name}_load_curve.csv")
    with open(load_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "slot_index", "time", "total_load_kw", "capacity_limit_kw",
            "price_yuan_per_kwh",
        ])
        for s in range(result.num_slots):
            writer.writerow([
                s,
                _slot_time_str(s),
                f"{result.slot_total_load_kw[s]:.4f}",
                f"{problem.grid_profile.get_capacity(s):.4f}",
                f"{problem.grid_profile.get_price(s):.4f}",
            ])

    detail_path = os.path.join(output_dir, f"{result.strategy_name}_session_detail.csv")
    with open(detail_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = [
            "session_id", "vehicle_type", "arrival_slot", "arrival_time",
            "departure_slot", "departure_time", "required_kwh",
            "delivered_kwh", "completion_ratio", "max_power_kw",
        ]
        for s in range(result.num_slots):
            header.append(f"slot_{s}_{_slot_time_str(s)}_kw")
        writer.writerow(header)

        for i, sess in enumerate(problem.sessions):
            delivered = result.get_session_energy(i)
            ratio = result.get_session_completion_ratio(i)
            row = [
                sess.session_id,
                VEHICLE_CONFIG[sess.vehicle_type]["name"],
                sess.arrival_slot,
                _slot_time_str(sess.arrival_slot),
                sess.effective_departure_slot,
                _slot_time_str(sess.effective_departure_slot),
                f"{sess.required_energy_kwh:.4f}",
                f"{delivered:.4f}",
                f"{ratio:.4f}",
                f"{sess.max_power_kw:.4f}",
            ]
            for s in range(result.num_slots):
                row.append(f"{result.power_matrix[i, s]:.4f}")
            writer.writerow(row)

    metrics_path = os.path.join(output_dir, f"{result.strategy_name}_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        peak_load = float(np.max(result.slot_total_load_kw))
        writer.writerow(["peak_load_kw", f"{peak_load:.4f}"])
        writer.writerow(["total_cost_yuan", f"{result.get_total_cost():.4f}"])
        writer.writerow(["valley_energy_ratio", f"{result.get_valley_energy_ratio():.4f}"])
        writer.writerow([
            "num_fully_charged",
            sum(1 for i in range(result.num_sessions)
                if result.get_session_completion_ratio(i) >= 0.999),
        ])
        writer.writerow(["num_sessions", result.num_sessions])

    return {
        "load_curve": load_path,
        "session_detail": detail_path,
        "metrics": metrics_path,
    }


def export_json(result: ScheduleResult, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result.strategy_name}_schedule.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    return path
