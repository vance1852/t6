from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from ..models import (
    ChargingSession,
    ScheduleProblem,
    ScheduleResult,
    SLOTS_PER_DAY,
    SLOT_MINUTES,
)


class BaseSchedulingStrategy(ABC):
    name: str = "base"

    def __init__(self):
        pass

    def solve(self, problem: ScheduleProblem) -> ScheduleResult:
        num_sessions = len(problem.sessions)
        num_slots = problem.num_slots
        power_matrix = np.zeros((num_sessions, num_slots), dtype=np.float64)

        self._allocate(power_matrix, problem)

        slot_total_load = np.sum(power_matrix, axis=0)

        return ScheduleResult(
            problem=problem,
            strategy_name=self.name,
            power_matrix=power_matrix,
            slot_total_load_kw=slot_total_load,
        )

    @abstractmethod
    def _allocate(
        self, power_matrix: np.ndarray, problem: ScheduleProblem
    ) -> None:
        raise NotImplementedError

    def _session_order(self, problem: ScheduleProblem) -> List[int]:
        return list(range(len(problem.sessions)))

    def _remaining_energy(
        self, power_matrix: np.ndarray, session_idx: int, session: ChargingSession
    ) -> float:
        hours_per_slot = SLOT_MINUTES / 60.0
        delivered = float(np.sum(power_matrix[session_idx])) * hours_per_slot
        return max(0.0, session.required_energy_kwh - delivered)

    def _allocate_to_slot(
        self,
        power_matrix: np.ndarray,
        slot_total_load: np.ndarray,
        problem: ScheduleProblem,
        session_idx: int,
        slot_idx: int,
        desired_kw: float,
    ) -> float:
        session = problem.sessions[session_idx]
        if slot_idx < session.arrival_slot or slot_idx >= session.effective_departure_slot:
            return 0.0

        capacity_remaining = problem.grid_profile.get_capacity(slot_idx) - slot_total_load[slot_idx]
        if capacity_remaining <= 1e-9:
            return 0.0

        current_power = power_matrix[session_idx, slot_idx]
        max_for_vehicle = session.max_power_kw - current_power
        if max_for_vehicle <= 1e-9:
            return 0.0

        remaining_energy = self._remaining_energy(power_matrix, session_idx, session)
        hours_per_slot = SLOT_MINUTES / 60.0
        max_for_energy = remaining_energy / hours_per_slot if hours_per_slot > 0 else 0.0
        if max_for_energy <= 1e-9:
            return 0.0

        alloc = min(desired_kw, capacity_remaining, max_for_vehicle, max_for_energy)
        if alloc > 1e-9:
            power_matrix[session_idx, slot_idx] += alloc
            slot_total_load[slot_idx] += alloc
            return alloc
        return 0.0
