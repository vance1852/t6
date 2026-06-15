from __future__ import annotations

from typing import Optional

from .models import ScheduleProblem, ScheduleResult
from .strategies import BaseSchedulingStrategy, STRATEGY_REGISTRY


class SchedulingEngine:
    def __init__(self):
        self._strategies = dict(STRATEGY_REGISTRY)

    def register_strategy(self, name: str, strategy_cls) -> None:
        self._strategies[name] = strategy_cls

    def list_strategies(self) -> list:
        return sorted(self._strategies.keys())

    def run(
        self,
        problem: ScheduleProblem,
        strategy_name: str,
        strategy_kwargs: Optional[dict] = None,
    ) -> ScheduleResult:
        if strategy_name not in self._strategies:
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available: {self.list_strategies()}"
            )
        strategy_cls = self._strategies[strategy_name]
        kwargs = strategy_kwargs or {}
        strategy: BaseSchedulingStrategy = strategy_cls(**kwargs)
        return strategy.solve(problem)
