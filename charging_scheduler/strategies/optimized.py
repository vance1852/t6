from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import BaseSchedulingStrategy
from ..models import (
    ChargingSession,
    ScheduleProblem,
    SLOT_MINUTES,
    SLOTS_PER_DAY,
)


class OptimizedStrategy(BaseSchedulingStrategy):
    name = "optimized"

    def __init__(self, fairness_weight: float = 0.3, peak_weight: float = 0.4,
                 cost_weight: float = 0.3):
        super().__init__()
        self.fairness_weight = fairness_weight
        self.peak_weight = peak_weight
        self.cost_weight = cost_weight

    def _slot_score(
        self, slot_idx: int, session: ChargingSession,
        problem: ScheduleProblem, current_load: np.ndarray,
        urgency_ratio: float,
    ) -> float:
        price = problem.grid_profile.get_price(slot_idx)
        capacity = problem.grid_profile.get_capacity(slot_idx)
        current = current_load[slot_idx]

        remaining_cap = capacity - current
        capacity_util = current / capacity if capacity > 1e-9 else 0.0

        price_range = self._get_price_range(problem)
        norm_price = (
            (price - price_range[0]) / (price_range[1] - price_range[0])
            if price_range[1] > price_range[0] else 0.5
        )

        capacity_penalty = capacity_util ** 2

        time_to_deadline = session.effective_departure_slot - slot_idx
        total_available = session.effective_departure_slot - session.arrival_slot
        time_pressure = 1.0 - (time_to_deadline / max(total_available, 1))

        score = (
            -self.cost_weight * norm_price
            - self.peak_weight * capacity_penalty
            + self.fairness_weight * urgency_ratio * time_pressure
        )

        if remaining_cap < 0.5:
            score -= 10.0

        return score

    def _get_price_range(self, problem: ScheduleProblem) -> Tuple[float, float]:
        prices = [tp.price for tp in problem.grid_profile.tariffs]
        if not prices:
            return (0.0, 1.0)
        return (min(prices), max(prices))

    def _allocate(
        self, power_matrix: np.ndarray, problem: ScheduleProblem
    ) -> None:
        num_slots = problem.num_slots
        num_sessions = len(problem.sessions)
        slot_total_load = np.zeros(num_slots, dtype=np.float64)
        hours_per_slot = SLOT_MINUTES / 60.0

        remaining = np.array([
            problem.sessions[i].required_energy_kwh
            for i in range(num_sessions)
        ], dtype=np.float64)

        max_possible = np.array([
            problem.sessions[i].max_possible_energy_kwh
            for i in range(num_sessions)
        ], dtype=np.float64)

        urgency = np.zeros(num_sessions, dtype=np.float64)
        for i in range(num_sessions):
            if max_possible[i] <= 1e-9:
                urgency[i] = 1.0
            else:
                urgency[i] = min(1.0, remaining[i] / max_possible[i])

        num_rounds = 20
        for round_idx in range(num_rounds):
            updated = False
            for slot_idx in range(num_slots):
                candidates: List[Tuple[float, int]] = []
                for si in range(num_sessions):
                    s = problem.sessions[si]
                    if (s.arrival_slot <= slot_idx < s.effective_departure_slot
                            and remaining[si] > 1e-9):
                        score = self._slot_score(
                            slot_idx, s, problem, slot_total_load, urgency[si]
                        )
                        candidates.append((score, si))

                if not candidates:
                    continue

                candidates.sort(key=lambda x: -x[0])

                for _, si in candidates:
                    session = problem.sessions[si]
                    if remaining[si] <= 1e-9:
                        continue
                    desired = min(
                        session.max_power_kw,
                        remaining[si] / hours_per_slot * 0.5,
                    )
                    alloc = self._allocate_to_slot(
                        power_matrix, slot_total_load, problem,
                        si, slot_idx, desired,
                    )
                    if alloc > 1e-9:
                        remaining[si] -= alloc * hours_per_slot
                        updated = True
                        for i in range(num_sessions):
                            if max_possible[i] > 1e-9:
                                urgency[i] = min(1.0, remaining[i] / max_possible[i])

            if not updated or np.all(remaining <= 1e-9):
                break

        for si in range(num_sessions):
            if remaining[si] <= 1e-9:
                continue
            session = problem.sessions[si]
            slots = sorted(session.available_slots,
                           key=lambda s: (
                               problem.grid_profile.get_price(s),
                               slot_total_load[s],
                           ))
            for slot_idx in slots:
                if remaining[si] <= 1e-9:
                    break
                desired = min(session.max_power_kw, remaining[si] / hours_per_slot)
                alloc = self._allocate_to_slot(
                    power_matrix, slot_total_load, problem,
                    si, slot_idx, desired,
                )
                remaining[si] -= alloc * hours_per_slot
