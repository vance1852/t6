from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import BaseSchedulingStrategy
from ..models import (
    ChargingSession,
    ScheduleProblem,
    SLOT_MINUTES,
)


class ValleyFillStrategy(BaseSchedulingStrategy):
    name = "valley_fill"

    def _slot_order_for_session(
        self, session: ChargingSession, problem: ScheduleProblem
    ) -> List[int]:
        slots = session.available_slots
        slots_with_price = [
            (problem.grid_profile.get_price(s),
             -problem.grid_profile.get_capacity(s),
             s)
            for s in slots
        ]
        slots_with_price.sort()
        return [s for (_, _, s) in slots_with_price]

    def _allocate(
        self, power_matrix: np.ndarray, problem: ScheduleProblem
    ) -> None:
        num_slots = problem.num_slots
        slot_total_load = np.zeros(num_slots, dtype=np.float64)
        hours_per_slot = SLOT_MINUTES / 60.0
        num_sessions = len(problem.sessions)

        urgency = []
        for si in range(num_sessions):
            s = problem.sessions[si]
            available = len(s.available_slots)
            max_possible = s.max_power_kw * available * hours_per_slot
            if max_possible <= 1e-9:
                urg = float('inf')
            else:
                urg = s.required_energy_kwh / max_possible
            urgency.append((-urg, s.effective_departure_slot, s.arrival_slot, si))

        urgency.sort()

        for _, _, _, si in urgency:
            session = problem.sessions[si]
            ordered_slots = self._slot_order_for_session(session, problem)
            for slot_idx in ordered_slots:
                remaining = self._remaining_energy(power_matrix, si, session)
                if remaining <= 1e-9:
                    break
                desired = min(session.max_power_kw, remaining / hours_per_slot)
                self._allocate_to_slot(
                    power_matrix, slot_total_load, problem, si, slot_idx, desired
                )
