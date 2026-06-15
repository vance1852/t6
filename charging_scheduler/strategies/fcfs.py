from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseSchedulingStrategy
from ..models import (
    ChargingSession,
    ScheduleProblem,
    SLOT_MINUTES,
)


class FCFSStrategy(BaseSchedulingStrategy):
    name = "fcfs"

    def _session_order(self, problem: ScheduleProblem) -> List[int]:
        indices = list(range(len(problem.sessions)))
        indices.sort(key=lambda i: (
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

        order = self._session_order(problem)

        for si in order:
            session = problem.sessions[si]
            for slot_idx in session.available_slots:
                remaining = self._remaining_energy(power_matrix, si, session)
                if remaining <= 1e-9:
                    break
                desired = min(session.max_power_kw, remaining / hours_per_slot)
                self._allocate_to_slot(
                    power_matrix, slot_total_load, problem, si, slot_idx, desired
                )
