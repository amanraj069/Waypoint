"""Perception: what the aircraft can see around it, as structured events.

The monitor never asks the simulator what is out there. It asks a
`PerceptionSource`, which turns the current state into `PerceptionEvent`s, and
the closed loop keeps every obstacle it has been told about. Swapping how the
aircraft sees (the simulator's own obstacle list today, object detection on a
Gazebo camera feed later) is a change of source, not of the monitor or the loop:

    perception = SymbolicPerceptionSource(backend)
    events = perception.get_events(backend.state)

`SymbolicPerceptionSource` is the real implementation for now. It reads the
obstacles the backend knows about (the planned ones and any a fault made
appear) through the `ExecutorBackend` interface alone, so it works unchanged on
any backend that reports its obstacles.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from waypoint.backends import ExecutorBackend
from waypoint.planner.models import Obstacle
from waypoint.sim.state import UAVState

Vector = tuple[float, float, float]


class PerceptionKind(str, Enum):
    OBSTACLE = "OBSTACLE"


class PerceptionSeverity(str, Enum):
    """How close it is: INFO is in view, WARNING is near, CRITICAL is about to matter."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class PerceptionEvent:
    """One thing seen, at one moment.

    `position` is the obstacle's base (x, y, ground), `distance` the horizontal
    gap from the aircraft to its edge. `obstacle` is what the monitor and the
    planner reason about; every source must fill it for an OBSTACLE event.
    """

    kind: PerceptionKind
    t: float
    position: Vector
    severity: PerceptionSeverity
    description: str
    source: str
    distance: float
    obstacle: Obstacle | None = None

    def as_record(self) -> dict:
        record = {
            "type": self.kind.value,
            "t": round(self.t, 3),
            "position": [round(c, 2) for c in self.position],
            "severity": self.severity.value,
            "distance": round(self.distance, 2),
            "description": self.description,
            "source": self.source,
        }
        if self.obstacle is not None:
            record["obstacle"] = self.obstacle.model_dump()
        return record


class PerceptionSource(ABC):
    """Anything that can say what is around the aircraft right now."""

    #: Written into the mission log, so a run says how it saw.
    name: str = "perception"

    @abstractmethod
    def get_events(self, current_state: UAVState) -> list[PerceptionEvent]:
        """Everything perceived at this state. Repeating an event every tick is fine."""


class SymbolicPerceptionSource(PerceptionSource):
    """Perfect perception of the backend's obstacles, optionally within a sensor range.

    `sensor_range` is metres from the aircraft to an obstacle's edge; None sees
    everything the backend knows the moment it knows it, so a fault-injected
    obstacle is perceived on the tick it appears. Ranges are measured from the
    true position, because a sensor sees what is physically near it whatever
    the GPS says.
    """

    name = "symbolic"

    #: Edge distances, in metres, under which an event is CRITICAL or WARNING.
    critical_within = 10.0
    warning_within = 40.0

    def __init__(self, backend: ExecutorBackend, *, sensor_range: float | None = None):
        self.backend = backend
        self.sensor_range = sensor_range

    def get_events(self, current_state: UAVState) -> list[PerceptionEvent]:
        events = []
        for obstacle in self.backend.obstacles:
            gap = max(
                0.0,
                math.hypot(obstacle.x - current_state.x, obstacle.y - current_state.y) - obstacle.radius,
            )
            if self.sensor_range is not None and gap > self.sensor_range:
                continue
            events.append(self._event(obstacle, gap, current_state.timestamp))
        return events

    def _event(self, obstacle: Obstacle, gap: float, t: float) -> PerceptionEvent:
        if gap <= self.critical_within:
            severity = PerceptionSeverity.CRITICAL
        elif gap <= self.warning_within:
            severity = PerceptionSeverity.WARNING
        else:
            severity = PerceptionSeverity.INFO
        top = f", {obstacle.height:g} m tall" if obstacle.height is not None else ""
        return PerceptionEvent(
            kind=PerceptionKind.OBSTACLE,
            t=t,
            position=(obstacle.x, obstacle.y, 0.0),
            severity=severity,
            description=(
                f"'{obstacle.name}' at ({obstacle.x:g}, {obstacle.y:g}), radius {obstacle.radius:g} m{top}, "
                f"{gap:.0f} m from the aircraft"
            ),
            source=self.name,
            distance=gap,
            obstacle=obstacle,
        )
