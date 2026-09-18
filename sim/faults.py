"""Pluggable fault injection.

A `FaultSchedule` is a list of `FaultEvent`s, each with a trigger. The executor
asks the schedule what has become due on this tick and applies the effects:

| Fault                  | Effect on the simulation |
| ---------------------- | ------------------------ |
| `GPS_DRIFT`            | The reported position slides away from the true one at `magnitude` m/s, so the controller flies the aircraft off course while believing it is on track |
| `OBSTACLE_APPEARS`     | An obstacle is added to the live obstacle list mid-flight, the way perception would find one |
| `WIND_GUST`            | A wind vector of `magnitude` m/s from `direction` is added for `duration` seconds |
| `BATTERY_DRAIN_SPIKE`  | An extra `magnitude` % per second of battery drain for `duration` seconds |

Everything is data: a schedule can be written as JSON, replayed exactly, and
diffed between runs.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from planner.models import Obstacle

Vector = tuple[float, float, float]


class FaultType(str, Enum):
    GPS_DRIFT = "GPS_DRIFT"
    OBSTACLE_APPEARS = "OBSTACLE_APPEARS"
    WIND_GUST = "WIND_GUST"
    BATTERY_DRAIN_SPIKE = "BATTERY_DRAIN_SPIKE"


class Trigger(BaseModel):
    """When a fault fires.

    Set `time` for a fixed moment, or one of the conditions to fire on the state
    of the flight. When several are set they must all hold. A trigger with
    nothing set never fires, which keeps a half-written schedule inert rather
    than surprising.
    """

    time: float | None = Field(default=None, ge=0, description="Seconds into the flight.")
    battery_below: float | None = Field(default=None, description="Fires under this battery %.")
    at_waypoint: int | None = Field(
        default=None, description="Fires once the aircraft is flying toward this waypoint index."
    )
    beyond_distance: float | None = Field(
        default=None, description="Fires beyond this distance from home, in metres."
    )
    above_altitude: float | None = Field(default=None, description="Fires above this altitude.")

    @property
    def is_set(self) -> bool:
        return any(
            value is not None
            for value in (
                self.time,
                self.battery_below,
                self.at_waypoint,
                self.beyond_distance,
                self.above_altitude,
            )
        )

    def fires(self, *, t: float, battery_pct: float, waypoint_index: int, distance_from_home: float, altitude: float) -> bool:
        if not self.is_set:
            return False
        if self.time is not None and t < self.time:
            return False
        if self.battery_below is not None and battery_pct >= self.battery_below:
            return False
        if self.at_waypoint is not None and waypoint_index < self.at_waypoint:
            return False
        if self.beyond_distance is not None and distance_from_home < self.beyond_distance:
            return False
        if self.above_altitude is not None and altitude <= self.above_altitude:
            return False
        return True

    def describe(self) -> str:
        parts = []
        if self.time is not None:
            parts.append(f"t >= {self.time:g}s")
        if self.battery_below is not None:
            parts.append(f"battery < {self.battery_below:g}%")
        if self.at_waypoint is not None:
            parts.append(f"waypoint >= {self.at_waypoint}")
        if self.beyond_distance is not None:
            parts.append(f"{self.beyond_distance:g} m from home")
        if self.above_altitude is not None:
            parts.append(f"above {self.above_altitude:g} m")
        return " and ".join(parts) or "never"


class FaultEvent(BaseModel):
    """One thing that goes wrong, and when."""

    type: FaultType
    trigger: Trigger = Field(default_factory=Trigger)
    magnitude: float = Field(default=1.0, description="m/s for drift and gusts, %/s for drain.")
    direction: float = Field(
        default=0.0, description="Compass-style bearing in degrees for drift and gusts (0 = +x)."
    )
    vertical: float = Field(default=0.0, description="Vertical component, m/s, for drift and gusts.")
    duration: float | None = Field(
        default=None, gt=0, description="Seconds the fault lasts; None means for the rest of the flight."
    )
    obstacle: Obstacle | None = Field(default=None, description="Required for OBSTACLE_APPEARS.")
    name: str = ""

    @model_validator(mode="after")
    def _check_payload(self) -> "FaultEvent":
        if self.type is FaultType.OBSTACLE_APPEARS and self.obstacle is None:
            raise ValueError("an OBSTACLE_APPEARS fault needs an `obstacle`")
        return self

    @property
    def label(self) -> str:
        return self.name or self.type.value

    @property
    def vector(self) -> Vector:
        """Magnitude and bearing as a vector, for drift and gust faults."""
        radians = math.radians(self.direction)
        return (
            self.magnitude * math.cos(radians),
            self.magnitude * math.sin(radians),
            self.vertical,
        )

    def describe(self) -> str:
        when = self.trigger.describe()
        lasting = f" for {self.duration:g}s" if self.duration else ""
        if self.type is FaultType.OBSTACLE_APPEARS and self.obstacle:
            what = (
                f"'{self.obstacle.name}' appears at "
                f"({self.obstacle.x:g}, {self.obstacle.y:g}), radius {self.obstacle.radius:g} m"
            )
        elif self.type is FaultType.BATTERY_DRAIN_SPIKE:
            what = f"battery drains an extra {self.magnitude:g}%/s"
        elif self.type is FaultType.WIND_GUST:
            what = f"{self.magnitude:g} m/s gust from bearing {self.direction:g} deg"
        else:
            what = f"GPS drifts {self.magnitude:g} m/s toward bearing {self.direction:g} deg"
        return f"{self.label}: {what}{lasting} (when {when})"


class FaultSchedule(BaseModel):
    """The faults for one run, in the order they were written.

    The schedule is pure data and holds no run state: the executor tracks what
    has already fired, so one schedule can drive many runs.
    """

    events: list[FaultEvent] = Field(default_factory=list)

    @classmethod
    def of(cls, *events: FaultEvent) -> "FaultSchedule":
        return cls(events=list(events))

    def due(
        self,
        *,
        fired: set[int],
        t: float,
        battery_pct: float,
        waypoint_index: int,
        distance_from_home: float,
        altitude: float,
    ) -> list[tuple[int, FaultEvent]]:
        """Every not-yet-fired event whose trigger holds now, with its index."""
        return [
            (index, event)
            for index, event in enumerate(self.events)
            if index not in fired
            and event.trigger.fires(
                t=t,
                battery_pct=battery_pct,
                waypoint_index=waypoint_index,
                distance_from_home=distance_from_home,
                altitude=altitude,
            )
        ]

    def describe(self) -> str:
        if not self.events:
            return "  (no faults)"
        return "\n".join(f"  - {event.describe()}" for event in self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __bool__(self) -> bool:
        return bool(self.events)


NO_FAULTS = FaultSchedule()
