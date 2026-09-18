"""Lightweight custom UAV simulator: point-mass physics, wind and fault injection."""

from sim.config import DEFAULT_CONFIG, NO_WIND, SimConfig, Wind
from sim.executor import Executor, LiveObstacle, RunEvent, execute, initial_state
from sim.faults import NO_FAULTS, FaultEvent, FaultSchedule, FaultType, Trigger
from sim.state import MissionState, UAVState

__all__ = [
    "DEFAULT_CONFIG",
    "Executor",
    "FaultEvent",
    "FaultSchedule",
    "FaultType",
    "LiveObstacle",
    "MissionState",
    "NO_FAULTS",
    "NO_WIND",
    "RunEvent",
    "SimConfig",
    "Trigger",
    "UAVState",
    "Wind",
    "execute",
    "initial_state",
]
