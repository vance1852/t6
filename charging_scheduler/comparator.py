from __future__ import annotations

import os
from typing import List, Dict

import numpy as np

from .models import ScheduleProblem, ScheduleResult
from .engine import SchedulingEngine


def compare_strategies(
    problem: ScheduleProblem,
    strategy_names: List[str],
    output_dir: str = ".",
) -> Dict[str, ScheduleResult]:
    engine = SchedulingEngine()
    results: Dict[str, ScheduleResult] = {}

    for name in strategy_names:
        results[name] = engine.run(problem, name)

    print_comparison_table(results)
    return results


def print_comparison_table(results: Dict[str, ScheduleResult]) -> None:
    names = list(results.keys())
    if not names:
        print("无数据可对比")
        return

    print()
    print("=" * 90)
    print("  多策略对比报表")
    print("=" * 90)

    header = f"  {'策略':<15} {'峰值kW':>10} {'峰谷差kW':>10} {'电费¥':>10} "
    header += f"{'谷段占比%':>10} {'充满数':>8} {'平均完成%':>10}"
    print(header)
    print("  " + "-" * 88)

    for name in names:
        r = results[name]
        peak = float(np.max(r.slot_total_load_kw))
        loads_pos = r.slot_total_load_kw[r.slot_total_load_kw > 0]
        valley = float(np.min(loads_pos)) if len(loads_pos) > 0 else 0.0
        peak_valley = peak - valley
        cost = r.get_total_cost()
        valley_ratio = r.get_valley_energy_ratio()
        n_full = sum(1 for i in range(r.num_sessions)
                     if r.get_session_completion_ratio(i) >= 0.999)
        avg_ratio = (
            sum(r.get_session_completion_ratio(i) for i in range(r.num_sessions))
            / r.num_sessions if r.num_sessions > 0 else 0.0
        )
        line = (
            f"  {name:<15} {peak:>10.2f} {peak_valley:>10.2f} {cost:>10.2f} "
            f"{valley_ratio*100:>9.1f}% {n_full:>5}/{r.num_sessions:<2} "
            f"{avg_ratio*100:>9.1f}%"
        )
        print(line)

    print("  " + "-" * 88)

    best_peak = min(results.keys(), key=lambda n: float(np.max(results[n].slot_total_load_kw)))
    best_cost = min(results.keys(), key=lambda n: results[n].get_total_cost())
    best_full = max(results.keys(), key=lambda n: sum(
        1 for i in range(results[n].num_sessions)
        if results[n].get_session_completion_ratio(i) >= 0.999
    ))
    best_valley = max(results.keys(), key=lambda n: results[n].get_valley_energy_ratio())

    print()
    print("  各项最优:")
    print(f"    最低峰值负荷:    {best_peak}  ({np.max(results[best_peak].slot_total_load_kw):.2f} kW)")
    print(f"    最低总电费:      {best_cost}  (¥{results[best_cost].get_total_cost():.2f})")
    print(f"    最多车辆充满:    {best_full}  ({sum(1 for i in range(results[best_full].num_sessions) if results[best_full].get_session_completion_ratio(i)>=0.999)}辆)")
    print(f"    最高谷段占比:    {best_valley}  ({results[best_valley].get_valley_energy_ratio()*100:.1f}%)")
    print("=" * 90)
