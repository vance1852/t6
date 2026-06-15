from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from .models import (
    ChargingSession,
    GridProfile,
    ScheduleProblem,
    TariffPeriod,
    VehicleType,
    SLOTS_PER_DAY,
    SLOTS_PER_HOUR,
    SLOT_MINUTES,
    VEHICLE_CONFIG,
)


def build_default_grid_profile() -> GridProfile:
    capacity = np.full(SLOTS_PER_DAY, 100.0, dtype=np.float64)

    for hour in range(24):
        start_slot = hour * SLOTS_PER_HOUR
        end_slot = start_slot + SLOTS_PER_HOUR
        if 8 <= hour < 12:
            capacity[start_slot:end_slot] = 80.0
        elif 17 <= hour < 22:
            capacity[start_slot:end_slot] = 50.0
        elif 0 <= hour < 6:
            capacity[start_slot:end_slot] = 120.0
        elif 22 <= hour < 24 or 6 <= hour < 8:
            capacity[start_slot:end_slot] = 90.0

    tariffs = [
        TariffPeriod("peak", 8, 11, 1.2),
        TariffPeriod("peak", 18, 21, 1.2),
        TariffPeriod("flat", 11, 18, 0.8),
        TariffPeriod("flat", 21, 22, 0.8),
        TariffPeriod("valley", 22, 6, 0.35),
        TariffPeriod("valley", 6, 8, 0.5),
    ]
    return GridProfile(capacity_limits_kw=capacity, tariffs=tariffs)


def generate_sessions(
    num_ev: int = 15,
    num_ebike: int = 40,
    seed: Optional[int] = None,
    early_leave_prob: float = 0.1,
    fault_prob: float = 0.05,
) -> List[ChargingSession]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    sessions: List[ChargingSession] = []
    session_id_counter = 0

    ev_evening_peak_hours = list(range(17, 22))
    ev_morning_peak_hours = list(range(7, 10))
    ev_other_hours = list(range(0, 24))

    for _ in range(num_ev):
        r = random.random()
        if r < 0.6:
            arrival_hour = random.choice(ev_evening_peak_hours) + random.random()
        elif r < 0.85:
            arrival_hour = random.choice(ev_morning_peak_hours) + random.random()
        else:
            arrival_hour = random.choice(ev_other_hours) + random.random()

        arrival_slot = int(arrival_hour * SLOTS_PER_HOUR)
        arrival_slot = max(0, min(arrival_slot, SLOTS_PER_DAY - 1))

        stay_hours = random.uniform(2.0, 10.0)
        if r < 0.6:
            stay_hours = random.uniform(5.0, 10.0)

        expected_departure_slot = min(
            arrival_slot + int(stay_hours * SLOTS_PER_HOUR),
            SLOTS_PER_DAY,
        )
        if expected_departure_slot <= arrival_slot:
            expected_departure_slot = arrival_slot + SLOTS_PER_HOUR

        actual_departure_slot = None
        if random.random() < early_leave_prob:
            early_slots = random.randint(
                SLOTS_PER_HOUR,
                max(SLOTS_PER_HOUR, (expected_departure_slot - arrival_slot) // 2),
            )
            actual_departure_slot = max(
                arrival_slot + SLOTS_PER_HOUR,
                expected_departure_slot - early_slots,
            )

        required_kwh = random.uniform(10.0, 50.0)

        charger_factor = 1.0
        if random.random() < fault_prob:
            charger_factor = random.uniform(0.3, 0.7)

        sessions.append(
            ChargingSession(
                session_id=f"EV-{session_id_counter:04d}",
                vehicle_type=VehicleType.EV,
                arrival_slot=arrival_slot,
                expected_departure_slot=expected_departure_slot,
                actual_departure_slot=actual_departure_slot,
                required_energy_kwh=required_kwh,
                charger_max_power_factor=charger_factor,
            )
        )
        session_id_counter += 1

    ebike_peak_hours = list(range(17, 23)) + list(range(6, 9))
    for _ in range(num_ebike):
        r = random.random()
        if r < 0.7:
            arrival_hour = random.choice(ebike_peak_hours) + random.random()
        else:
            arrival_hour = random.uniform(0, 24)

        arrival_slot = int(arrival_hour * SLOTS_PER_HOUR)
        arrival_slot = max(0, min(arrival_slot, SLOTS_PER_DAY - 1))

        stay_hours = random.uniform(2.0, 12.0)
        expected_departure_slot = min(
            arrival_slot + int(stay_hours * SLOTS_PER_HOUR),
            SLOTS_PER_DAY,
        )
        if expected_departure_slot <= arrival_slot:
            expected_departure_slot = arrival_slot + SLOTS_PER_HOUR

        actual_departure_slot = None
        if random.random() < early_leave_prob:
            early_slots = random.randint(
                SLOTS_PER_HOUR,
                max(SLOTS_PER_HOUR, (expected_departure_slot - arrival_slot) // 2),
            )
            actual_departure_slot = max(
                arrival_slot + SLOTS_PER_HOUR,
                expected_departure_slot - early_slots,
            )

        required_kwh = random.uniform(0.5, 2.5)

        charger_factor = 1.0
        if random.random() < fault_prob:
            charger_factor = random.uniform(0.4, 0.8)

        sessions.append(
            ChargingSession(
                session_id=f"EB-{session_id_counter:04d}",
                vehicle_type=VehicleType.EBIKE,
                arrival_slot=arrival_slot,
                expected_departure_slot=expected_departure_slot,
                actual_departure_slot=actual_departure_slot,
                required_energy_kwh=required_kwh,
                charger_max_power_factor=charger_factor,
            )
        )
        session_id_counter += 1

    return sessions


def generate_problem(
    num_ev: int = 15,
    num_ebike: int = 40,
    seed: Optional[int] = None,
    overload_factor: float = 1.0,
) -> ScheduleProblem:
    sessions = generate_sessions(num_ev, num_ebike, seed)
    grid = build_default_grid_profile()

    if overload_factor > 1.0:
        for s in sessions:
            s.required_energy_kwh *= overload_factor

    return ScheduleProblem(sessions=sessions, grid_profile=grid)


def slot_to_hhmm(slot_idx: int) -> str:
    hour = slot_idx // SLOTS_PER_HOUR
    minute = (slot_idx % SLOTS_PER_HOUR) * SLOT_MINUTES
    return f"{hour:02d}:{minute:02d}"
