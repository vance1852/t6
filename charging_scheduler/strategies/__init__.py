from .base import BaseSchedulingStrategy
from .fcfs import FCFSStrategy
from .edf import EDFStrategy
from .valley_fill import ValleyFillStrategy
from .optimized import OptimizedStrategy

STRATEGY_REGISTRY = {
    "fcfs": FCFSStrategy,
    "edf": EDFStrategy,
    "valley_fill": ValleyFillStrategy,
    "optimized": OptimizedStrategy,
}

__all__ = [
    "BaseSchedulingStrategy",
    "FCFSStrategy",
    "EDFStrategy",
    "ValleyFillStrategy",
    "OptimizedStrategy",
    "STRATEGY_REGISTRY",
]
