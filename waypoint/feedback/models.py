"""What the feedback layer looks at, and what it can conclude."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from waypoint.feedback.perception import PerceptionEvent
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
    #: Where the current plan took over (the pad, or where an in-flight replan
    #: began): the start of the first leg. None treats the first leg as on track.
    route_start: Vector | None = None
    #: What the perception source reported at this moment.
    perception: tuple[PerceptionEvent, ...] = ()

    @property
    def remaining(self) -> tuple[Waypoint, ...]:
        return self.plan_waypoints[self.target_index:]

    @property
    def leg(self) -> tuple[Vector, Vector] | None:
        """The leg being flown, from the last waypoint (or the route start) to the target."""
        if self.target_index >= len(self.plan_waypoints):
            return None
        target = self.plan_waypoints[self.target_index]
        if self.target_index > 0:
            before = self.plan_waypoints[self.target_index - 1]
            start = (before.x, before.y, before.z)
        elif self.route_start is not None:
            start = self.route_start
        else:
            return None
        return start, (target.x, target.y, target.z)

    @property
    def deviation(self) -> float:
        """Horizontal metres between where the aircraft is and the leg the plan has it on.

        Cross-track, not distance to the waypoint: halfway down a 100 m leg the
        aircraft is 50 m from its target and still exactly where it should be.
        Horizontal only, because a multirotor climbs slower than it flies and so
        sags below a climbing leg by design.
        """
        leg = self.leg
        if leg is None:
            return 0.0
        return horizontal_gap(self.position[:2], *leg)

    @property
    def new_obstacles(self) -> tuple[Obstacle, ...]:
        """Known now, but not when the current plan was verified."""
        return tuple(o for o in self.known_obstacles if o not in self.planned_obstacles)


def horizontal_gap(point, a, b) -> float:
    """Horizontal distance from a point to the segment a-b."""
    px, py = point
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    f = 0.0 if span == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
    return math.hypot(px - (ax + f * dx), py - (ay + f * dy))


class TriggerKind(str, Enum):
    OBSTACLE_ON_ROUTE = "OBSTACLE_ON_ROUTE"
    BATTERY_INSUFFICIENT = "BATTERY_INSUFFICIENT"
    OFF_ROUTE = "OFF_ROUTE"
    #: Only under the runtime-only gate: the guard refused the next command (a waypoint outside the geofence).
    COMMAND_REJECTED = "COMMAND_REJECTED"


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
    battery_trend_window_s: float = Field(
        default=10.0, gt=0, description="How much recent flight the observed drain rate is taken over."
    )
    battery_trend_min_s: float = Field(
        default=5.0, ge=0, description="No trend is trusted from less flight than this."
    )
    battery_trend_tolerance: float = Field(
        default=1.25,
        ge=1,
        description=(
            "Observed drain up to this multiple of the model's is noise; above it, "
            "the rest of the route is costed at the observed rate."
        ),
    )
    deviation_threshold_m: float = Field(
        default=15.0, gt=0, description="Cross-track metres off the planned leg that count as off route."
    )
    deviation_hold_s: float = Field(
        default=3.0,
        ge=0,
        description="Off route for this long, without a break, before replanning (a gust is not a reason).",
    )

    @property
    def history_seconds(self) -> float:
        """How much telemetry history the monitor ever looks at."""
        return max(self.battery_trend_window_s, self.deviation_hold_s)
