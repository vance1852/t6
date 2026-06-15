from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np


class VehicleType(Enum):
    EV = "ev"
    EBIKE = "ebike"


VEHICLE_CONFIG = {
    VehicleType.EV: {"max_power_kw": 7.0, "name": "电动汽车"},
    VehicleType.EBIKE: {"max_power_kw": 0.5, "name": "电动自行车"},
}

SLOT_MINUTES = 15
SLOTS_PER_HOUR = 60 // SLOT_MINUTES
SLOTS_PER_DAY = 24 * SLOTS_PER_HOUR


@dataclass
class TariffPeriod:
    name: str
    start_hour: int
    end_hour: int
    price: float

    def contains_slot(self, slot_idx: int) -> bool:
        hour = slot_idx / SLOTS_PER_HOUR
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        else:
            return hour >= self.start_hour or hour < self.end_hour


@dataclass
class ChargingSession:
    session_id: str
    vehicle_type: VehicleType
    arrival_slot: int
    expected_departure_slot: int
    actual_departure_slot: Optional[int]
    required_energy_kwh: float
    charger_max_power_factor: float = 1.0

    @property
    def max_power_kw(self) -> float:
        base = VEHICLE_CONFIG[self.vehicle_type]["max_power_kw"]
        return base * self.charger_max_power_factor

    @property
    def effective_departure_slot(self) -> int:
        return (
            self.actual_departure_slot
            if self.actual_departure_slot is not None
            else self.expected_departure_slot
        )

    @property
    def available_slots(self) -> List[int]:
        return list(range(self.arrival_slot, self.effective_departure_slot))

    @property
    def max_possible_energy_kwh(self) -> float:
        hours_per_slot = SLOT_MINUTES / 60.0
        return self.max_power_kw * len(self.available_slots) * hours_per_slot

    @property
    def is_feasible(self) -> bool:
        return self.required_energy_kwh <= self.max_possible_energy_kwh + 1e-9

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vehicle_type"] = self.vehicle_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChargingSession":
        return cls(
            session_id=d["session_id"],
            vehicle_type=VehicleType(d["vehicle_type"]),
            arrival_slot=d["arrival_slot"],
            expected_departure_slot=d["expected_departure_slot"],
            actual_departure_slot=d.get("actual_departure_slot"),
            required_energy_kwh=d["required_energy_kwh"],
            charger_max_power_factor=d.get("charger_max_power_factor", 1.0),
        )


@dataclass
class GridProfile:
    capacity_limits_kw: np.ndarray
    tariffs: List[TariffPeriod]

    def get_price(self, slot_idx: int) -> float:
        slot_idx = slot_idx % SLOTS_PER_DAY
        for tp in self.tariffs:
            if tp.contains_slot(slot_idx):
                return tp.price
        return self.tariffs[0].price if self.tariffs else 0.5

    def get_capacity(self, slot_idx: int) -> float:
        return float(self.capacity_limits_kw[slot_idx % SLOTS_PER_DAY])

    def to_dict(self) -> dict:
        return {
            "capacity_limits_kw": self.capacity_limits_kw.tolist(),
            "tariffs": [asdict(t) for t in self.tariffs],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridProfile":
        return cls(
            capacity_limits_kw=np.array(d["capacity_limits_kw"], dtype=np.float64),
            tariffs=[TariffPeriod(**t) for t in d["tariffs"]],
        )


@dataclass
class ScheduleProblem:
    sessions: List[ChargingSession]
    grid_profile: GridProfile
    num_slots: int = SLOTS_PER_DAY

    def to_dict(self) -> dict:
        return {
            "sessions": [s.to_dict() for s in self.sessions],
            "grid_profile": self.grid_profile.to_dict(),
            "num_slots": self.num_slots,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduleProblem":
        return cls(
            sessions=[ChargingSession.from_dict(s) for s in d["sessions"]],
            grid_profile=GridProfile.from_dict(d["grid_profile"]),
            num_slots=d.get("num_slots", SLOTS_PER_DAY),
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ScheduleProblem":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ScheduleResult:
    problem: ScheduleProblem
    strategy_name: str
    power_matrix: np.ndarray
    slot_total_load_kw: np.ndarray

    @property
    def num_sessions(self) -> int:
        return len(self.problem.sessions)

    @property
    def num_slots(self) -> int:
        return self.problem.num_slots

    def get_session_energy(self, session_idx: int) -> float:
        hours_per_slot = SLOT_MINUTES / 60.0
        return float(np.sum(self.power_matrix[session_idx]) * hours_per_slot)

    def get_session_completion_ratio(self, session_idx: int) -> float:
        session = self.problem.sessions[session_idx]
        delivered = self.get_session_energy(session_idx)
        required = session.required_energy_kwh
        if required <= 0:
            return 1.0
        return min(delivered / required, 1.0)

    def get_total_cost(self) -> float:
        hours_per_slot = SLOT_MINUTES / 60.0
        total = 0.0
        for s in range(self.num_slots):
            price = self.problem.grid_profile.get_price(s)
            total += self.slot_total_load_kw[s] * price * hours_per_slot
        return float(total)

    def get_valley_energy_ratio(self) -> float:
        if not self.problem.grid_profile.tariffs:
            return 0.0
        prices = [tp.price for tp in self.problem.grid_profile.tariffs]
        min_price = min(prices)
        hours_per_slot = SLOT_MINUTES / 60.0
        valley_energy = 0.0
        total_energy = 0.0
        for s in range(self.num_slots):
            if self.problem.grid_profile.get_price(s) <= min_price + 1e-9:
                valley_energy += self.slot_total_load_kw[s] * hours_per_slot
            total_energy += self.slot_total_load_kw[s] * hours_per_slot
        return valley_energy / total_energy if total_energy > 0 else 0.0

    def to_dict(self) -> dict:
        session_details = []
        for i, sess in enumerate(self.problem.sessions):
            energy = self.get_session_energy(i)
            ratio = self.get_session_completion_ratio(i)
            session_details.append({
                "session_id": sess.session_id,
                "vehicle_type": sess.vehicle_type.value,
                "delivered_energy_kwh": round(energy, 4),
                "required_energy_kwh": sess.required_energy_kwh,
                "completion_ratio": round(ratio, 4),
                "power_by_slot": self.power_matrix[i].tolist(),
            })
        return {
            "strategy_name": self.strategy_name,
            "slot_total_load_kw": self.slot_total_load_kw.tolist(),
            "sessions": session_details,
            "metrics": {
                "peak_load_kw": float(np.max(self.slot_total_load_kw)),
                "valley_energy_ratio": round(self.get_valley_energy_ratio(), 4),
                "total_cost_yuan": round(self.get_total_cost(), 2),
                "num_fully_charged": sum(
                    1 for i in range(self.num_sessions) if self.get_session_completion_ratio(i) >= 0.999
                ),
                "num_sessions": self.num_sessions,
            },
        }

    @classmethod
    def from_dict(cls, d: dict, problem: ScheduleProblem) -> "ScheduleResult":
        num_sessions = len(problem.sessions)
        num_slots = problem.num_slots
        power_matrix = np.zeros((num_sessions, num_slots), dtype=np.float64)
        for i, sd in enumerate(d["sessions"]):
            power_matrix[i] = np.array(sd["power_by_slot"], dtype=np.float64)
        slot_total_load = np.array(d["slot_total_load_kw"], dtype=np.float64)
        return cls(
            problem=problem,
            strategy_name=d["strategy_name"],
            power_matrix=power_matrix,
            slot_total_load_kw=slot_total_load,
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str, problem: ScheduleProblem) -> "ScheduleResult":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), problem)
