from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseSchedulingStrategy
from ..models import (
    ChargingSession,
    ScheduleProblem,
    SLOT_MINUTES,
)


class EDFStrategy(BaseSchedulingStrategy):
    name = "edf"

    def _session_order(self, problem: ScheduleProblem) -> List[int]:
        indices = list(range(len(problem.sessions)))
        indices.sort(key=lambda i: (
            problem.sessions[i].effective_departure_slot,
            problem.sessions[i].arrival_slot,
            problem.sessions[i].session_id,
        ))
        return indices

    def _allocate(
        self, power_matrix: np.ndarray, problem: ScheduleProblem
    ) -> None:
        num_slots = problem.num_slots
        slot_total_load = np.zeros(num_slots, dtype=np.float64)
        hours_per_slot = SLOT_MINUTES / 60.0
        num_sessions = len(problem.sessions)

        remaining = np.array([
            problem.sessions[i].required_energy_kwh
            for i in range(num_sessions)
        ], dtype=np.float64)

        for slot_idx in range(num_slots):
            active = []
            for si in range(num_sessions):
                s = problem.sessions[si]
                if (s.arrival_slot <= slot_idx < s.effective_departure_slot
                        and remaining[si] > 1e-9):
                    urgency = s.effective_departure_slot - slot_idx
                    active.append((urgency, si))

            active.sort(key=lambda x: x[0])

            for _, si in active:
                session = problem.sessions[si]
                if remaining[si] <= 1e-9:
                    continue
                desired = min(session.max_power_kw, remaining[si] / hours_per_slot)
                alloc = self._allocate_to_slot(
                    power_matrix, slot_total_load, problem, si, slot_idx, desired
                )
                remaining[si] -= alloc * hours_per_slot
