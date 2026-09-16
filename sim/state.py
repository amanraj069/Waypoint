"""UAV mission state carried through execution and feedback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MissionState:
    x: float
    y: float
    z: float
    velocity: float
    heading: float
    battery_pct: float
    timestamp: float

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)
