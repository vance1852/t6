from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .models import (
    ScheduleResult,
    SLOT_MINUTES,
)


@dataclass
class ValidationViolation:
    severity: str
    category: str
    slot_idx: Optional[int]
    session_idx: Optional[int]
    message: str

    def __str__(self) -> str:
        parts = [f"[{self.severity.upper()}] {self.category}"]
        if self.slot_idx is not None:
            hour = self.slot_idx // 4
            minute = (self.slot_idx % 4) * 15
            parts.append(f"时间片 {self.slot_idx} ({hour:02d}:{minute:02d})")
        if self.session_idx is not None:
            parts.append(f"会话 #{self.session_idx}")
        parts.append(self.message)
        return " - ".join(parts)


@dataclass
class ValidationReport:
    violations: List[ValidationViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warning")

    def summary(self) -> str:
        return (
            f"校验结果: {'通过' if self.is_valid else '未通过'} | "
            f"错误: {self.error_count} | 警告: {self.warning_count}"
        )


def validate_schedule(result: ScheduleResult, tol: float = 1e-6) -> ValidationReport:
    report = ValidationReport()
    problem = result.problem
    sessions = problem.sessions
    num_sessions = len(sessions)
    num_slots = problem.num_slots
    hours_per_slot = SLOT_MINUTES / 60.0

    for slot_idx in range(num_slots):
        cap = problem.grid_profile.get_capacity(slot_idx)
        total = float(result.slot_total_load_kw[slot_idx])
        if total > cap + tol:
            report.violations.append(ValidationViolation(
                severity="error",
                category="容量越限",
                slot_idx=slot_idx,
                session_idx=None,
                message=f"总负荷 {total:.3f}kW 超过台区容量上限 {cap:.3f}kW (超出 {total-cap:.3f}kW)",
            ))

    for si in range(num_sessions):
        session = sessions[si]
        max_p = session.max_power_kw
        for slot_idx in range(num_slots):
            p = float(result.power_matrix[si, slot_idx])
            if p > max_p + tol:
                report.violations.append(ValidationViolation(
                    severity="error",
                    category="车辆功率越限",
                    slot_idx=slot_idx,
                    session_idx=si,
                    message=f"功率 {p:.3f}kW 超过车辆上限 {max_p:.3f}kW",
                ))
            if p < -tol:
                report.violations.append(ValidationViolation(
                    severity="error",
                    category="负功率",
                    slot_idx=slot_idx,
                    session_idx=si,
                    message=f"功率为负 {p:.6f}kW",
                ))

    for si in range(num_sessions):
        session = sessions[si]
        for slot_idx in range(num_slots):
            p = float(result.power_matrix[si, slot_idx])
            if p > tol and not (
                session.arrival_slot <= slot_idx < session.effective_departure_slot
            ):
                report.violations.append(ValidationViolation(
                    severity="error",
                    category="时间违规",
                    slot_idx=slot_idx,
                    session_idx=si,
                    message=(
                        f"在站时间外充电: 功率 {p:.3f}kW, "
                        f"到达 {session.arrival_slot}, 离开 {session.effective_departure_slot}"
                    ),
                ))

    for si in range(num_sessions):
        session = sessions[si]
        delivered = float(np.sum(result.power_matrix[si]) * hours_per_slot)
        if delivered > session.required_energy_kwh + tol + 0.01:
            report.violations.append(ValidationViolation(
                severity="warning",
                category="过充",
                slot_idx=None,
                session_idx=si,
                message=(
                    f"充电量 {delivered:.3f}kWh 超过需求 {session.required_energy_kwh:.3f}kWh"
                ),
            ))

    total_power_check = np.sum(result.power_matrix, axis=0)
    if not np.allclose(total_power_check, result.slot_total_load_kw, atol=tol):
        diff = np.max(np.abs(total_power_check - result.slot_total_load_kw))
        report.violations.append(ValidationViolation(
            severity="error",
            category="数据不一致",
            slot_idx=None,
            session_idx=None,
            message=f"功率矩阵之和与总负荷不一致，最大偏差 {diff:.6f}kW",
        ))

    for si in range(num_sessions):
        session = sessions[si]
        if not session.is_feasible:
            max_e = session.max_possible_energy_kwh
            report.violations.append(ValidationViolation(
                severity="warning",
                category=" infeasible_demand",
                slot_idx=None,
                session_idx=si,
                message=(
                    f"需求电量 {session.required_energy_kwh:.2f}kWh 超过在站最大可充 "
                    f"{max_e:.2f}kWh，无法充满（即使无其他车辆）"
                ),
            ))

    return report
