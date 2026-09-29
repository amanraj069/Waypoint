"""What the feedback layer looks at, and what it can conclude."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from waypoint.planner.models import Obstacle, Waypoint

Vector = tuple[float, float, float]


@dataclass(frozen=True)
class TelemetrySnapshot:
    """One moment of the flight, as the aircraft itself would know it.

    Positions are the *reported* ones (what GPS says), because that is all a
    real aircraft has; a drifting GPS is invisible from here, which is the
    point of it being a fault.
    """

    t: float
    position: Vector
    velocity: Vector
    heading: float
    battery_pct: float
    ground: float
    #: Index into `plan_waypoints` of the waypoint being flown toward.
    target_index: int
    subtask: str
    #: The current plan, flattened, and the obstacles it was verified against.
    plan_waypoints: tuple[Waypoint, ...]
    planned_obstacles: tuple[Obstacle, ...]
    #: Everything known now, including obstacles that appeared in flight.
    known_obstacles: tuple[Obstacle, ...]
    active_faults: tuple[str, ...] = ()
    min_separation: float = 0.0

    @property
    def remaining(self) -> tuple[Waypoint, ...]:
        return self.plan_waypoints[self.target_index:]

    @property
    def new_obstacles(self) -> tuple[Obstacle, ...]:
        """Known now, but not when the current plan was verified."""
        return tuple(o for o in self.known_obstacles if o not in self.planned_obstacles)


class TriggerKind(str, Enum):
    OBSTACLE_ON_ROUTE = "OBSTACLE_ON_ROUTE"
    BATTERY_INSUFFICIENT = "BATTERY_INSUFFICIENT"


@dataclass(frozen=True)
class ReplanTrigger:
    """A reason the current plan can no longer be flown as it is."""

    kind: TriggerKind
    t: float
    reason: str
    detail: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {"trigger": self.kind.value, "t": round(self.t, 3), "reason": self.reason, **self.detail}


class FeedbackConfig(BaseModel):
    """When the monitor speaks up.

    The battery model is deliberately simple and backend-independent: metres
    per percent and seconds per percent, the same figures the custom simulator
    drains by, so the estimate is honest there and conservative elsewhere.
    """

    meters_per_battery_pct: float = Field(default=40.0, gt=0)
    seconds_per_battery_pct: float = Field(default=60.0, gt=0)
    cruise_speed: float = Field(default=5.0, gt=0, description="For the time part of the estimate.")
    battery_reserve_pct: float = Field(
        default=8.0, ge=0, description="Must still be left on touchdown, or the plan has to change."
    )
    battery_cooldown_s: float = Field(
        default=20.0, ge=0, description="After a battery replan, give the new plan this long."
    )
