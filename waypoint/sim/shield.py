"""The runtime safety shield: a control barrier function filter under the controller.

The verifier proves the *plan* clears everything, along straight legs. The
aircraft does not fly straight legs: wind it cannot cancel, a gust, a drifting
estimate push it off them, and a verified plan can still end in a keep-out
or a wall. The shield closes that gap at run time, every tick, the way
safety filters do under learned or LLM controllers.

Each hazard near the aircraft is a barrier function h(p), positive where the
aircraft may be and zero on the boundary it must not cross:

    obstacle    horizontal distance to its keep-out cylinder (radius plus the
                mission's separation), while below its top
    building    horizontal distance to its footprint plus a margin, while below
                its roof (and, over a roof, height above it)
    no-fly      horizontal distance to the zone plus a margin, at any altitude
    geofence    distance inside the fence, less a margin
    terrain     height above the ground, less a minimum clearance
    ceiling     the altitude limit, less the altitude

Every one asks the commanded velocity v to satisfy grad h . v >= -alpha * h:
far from a boundary nothing changes; close to it, the velocity toward it is
limited in proportion to the room left, so the aircraft slides along the
boundary instead of through it. The filtered velocity is the one closest to
what the controller asked for that meets every constraint at once (the
projection onto the intersection of half-spaces, by Dykstra's algorithm), so
the shield changes as little as it can.

The shield acts on the position the aircraft *believes* (its estimate): like
any onboard filter, it cannot protect against an error it cannot see.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from shapely.geometry import Point
from shapely.ops import nearest_points

from waypoint.planner.models import MissionConstraints, Obstacle
from waypoint.verifier.checks import polygon_of
from waypoint.world.models import World

Vector = tuple[float, float, float]

#: How fast the shield lets the aircraft close on a boundary, per metre of room left (1/s).
ALPHA = 0.7
#: The deceleration counted on when working out how far the aircraft needs to stop (m/s^2):
#: below the airframe's 3 m/s^2, because part of it goes to fighting the wind.
BRAKING = 2.0
#: Hazards farther than this are not looked at.
HORIZON = 45.0
#: Margins, metres, beyond the hard boundary.
OBSTACLE_MARGIN = 0.5
BUILDING_MARGIN = 2.0
ZONE_MARGIN = 2.0
FENCE_MARGIN = 1.0
GROUND_CLEARANCE = 2.0
ROOF_CLEARANCE = 0.5


@dataclass(frozen=True)
class Barrier:
    name: str
    h: float
    gradient: Vector

    def bound(self, alpha: float) -> float:
        return -alpha * self.h


class SafetyShield:
    """Filters the controller's velocity so no barrier is crossed."""

    def __init__(
        self,
        constraints: MissionConstraints | None,
        world: World | None,
        obstacles: Callable[[], list[Obstacle]] | None = None,
        *,
        alpha: float = ALPHA,
        braking: float = BRAKING,
    ):
        self.constraints = constraints
        self.world = world
        self.obstacles = obstacles or (lambda: list(constraints.obstacles) if constraints else [])
        self.alpha = alpha
        self.braking = braking
        self.fence = polygon_of(constraints.geofence, "geofence") if constraints else None
        self.zones = [polygon_of(z, "no-fly zone") for z in constraints.no_fly_zones] if constraints else []

    # -- barriers ---------------------------------------------------------------

    def barriers(self, p: Vector, *, landing: bool = False) -> list[Barrier]:
        x, y, z = p
        here = Point(x, y)
        out: list[Barrier] = []
        separation = self.constraints.min_separation if self.constraints else 0.0

        for o in self.obstacles():
            gap = math.hypot(x - o.x, y - o.y)
            keep = o.radius + separation + OBSTACLE_MARGIN
            if gap - keep > HORIZON:
                continue
            if o.height is not None and z > o.height:
                if gap < keep:  # over it: stay over the top, like a roof
                    out.append(Barrier(f"the top of '{o.name}'", z - o.height - ROOF_CLEARANCE, (0.0, 0.0, 1.0)))
                continue
            if gap < 1e-6:
                continue
            out.append(Barrier(f"'{o.name}'", gap - keep, ((x - o.x) / gap, (y - o.y) / gap, 0.0)))

        if self.world is not None and self.world.buildings:
            for b in self.world.obstructions(here.buffer(HORIZON)):
                if b.covers(x, y):
                    # Over the roof: keep above it (the verifier lets a leg cross anywhere above a roof).
                    # A landing too: pads are surveyed clear of buildings, so coming down onto
                    # a roof is always drift, never the plan.
                    if z > b.roof:
                        out.append(Barrier(f"the roof of '{b.name or 'a building'}'", z - b.roof - ROOF_CLEARANCE, (0.0, 0.0, 1.0)))
                    continue
                if z > b.roof + ROOF_CLEARANCE:
                    continue  # high enough to cross over it
                edge, _ = nearest_points(b.polygon.exterior, here)
                gap = here.distance(b.polygon)
                if gap < 1e-6:
                    continue
                out.append(Barrier(
                    f"'{b.name or 'a building'}'", gap - BUILDING_MARGIN,
                    ((x - edge.x) / gap, (y - edge.y) / gap, 0.0),
                ))

        for number, zone in enumerate(self.zones, 1):
            gap = here.distance(zone)
            if gap > HORIZON or gap < 1e-6:
                continue
            edge, _ = nearest_points(zone.exterior, here)
            out.append(Barrier(f"no-fly zone {number}", gap - ZONE_MARGIN, ((x - edge.x) / gap, (y - edge.y) / gap, 0.0)))

        if self.fence is not None:
            edge, _ = nearest_points(self.fence.exterior, here)
            gap = here.distance(self.fence.exterior)
            inside = self.fence.covers(here)
            if gap < HORIZON and gap > 1e-6:
                # Inside, room grows moving away from the edge; outside, moving toward it.
                sign = 1.0 if inside else -1.0
                out.append(Barrier(
                    "the geofence", sign * gap - FENCE_MARGIN,
                    (sign * (x - edge.x) / gap, sign * (y - edge.y) / gap, 0.0),
                ))

        if self.world is not None and self.world.terrain is not None and not landing:
            ground = self.world.elevation_at(x, y)
            step = 1.0
            gx = (self.world.elevation_at(x + step, y) - self.world.elevation_at(x - step, y)) / (2 * step)
            gy = (self.world.elevation_at(x, y + step) - self.world.elevation_at(x, y - step)) / (2 * step)
            out.append(Barrier("the ground", z - ground - GROUND_CLEARANCE, (-gx, -gy, 1.0)))

        if self.constraints is not None:
            out.append(Barrier("the ceiling", self.constraints.max_altitude - z, (0.0, 0.0, -1.0)))
        return out

    # -- the filter ---------------------------------------------------------------

    def filter(
        self,
        p: Vector,
        desired: Vector,
        *,
        velocity: Vector = (0.0, 0.0, 0.0),
        disturbance: Vector = (0.0, 0.0, 0.0),
        landing: bool = False,
    ) -> tuple[Vector, list[str]]:
        """The velocity to fly instead of `desired`, and the hazards that changed it.

        `velocity` is how the aircraft is moving now. A multirotor cannot stop
        at once, so each barrier's room is reduced by the distance it needs to
        brake from its present speed toward that hazard: the filter starts
        slowing early enough to stop on the boundary rather than coast past it.

        `disturbance` is what is moving the aircraft that the command does not
        account for (wind the autopilot does not cancel, a gust), as estimated
        by `DisturbanceObserver`. The constraint is on the velocity the aircraft
        will actually have, command plus disturbance, so a crosswind toward a
        wall is met by commanding away from it, not by commanding zero.
        """
        constraints = []
        for b in self.barriers(p, landing=landing):
            norm = math.sqrt(_dot(b.gradient, b.gradient)) or 1.0
            closing = max(0.0, -_dot(b.gradient, velocity) / norm)
            room = b.h - norm * closing * closing / (2.0 * self.braking)
            constraints.append((b, b.gradient, -self.alpha * room - _dot(b.gradient, disturbance)))
        violated = [c for c in constraints if _dot(c[1], desired) < c[2] - 1e-9]
        if not violated:
            return desired, []
        v = _project(desired, [(g, bound) for _, g, bound in constraints])
        changed = [b.name for b, g, bound in constraints if _dot(g, desired) < bound - 1e-9]
        return v, changed


class DisturbanceObserver:
    """Estimates the part of the aircraft's motion its commands do not explain.

    Each tick: the velocity it had, less the velocity it was commanded, low-pass
    filtered so a moment of acceleration lag is not mistaken for wind.
    """

    def __init__(self, gain: float = 0.25):
        self.gain = gain
        self.estimate: Vector = (0.0, 0.0, 0.0)
        self._command: Vector | None = None

    def update(self, velocity: Vector) -> Vector:
        if self._command is not None:
            seen = tuple(velocity[i] - self._command[i] for i in range(3))
            self.estimate = tuple((1 - self.gain) * self.estimate[i] + self.gain * seen[i] for i in range(3))
        return self.estimate

    def commanded(self, command: Vector) -> None:
        self._command = command


class GpsDriftObserver:
    """Tracks a drifting GPS by comparing it with the airframe's own sense of motion.

    A GPS fix that drifts moves the reported position without the aircraft
    moving: the velocity implied by successive fixes then disagrees with the
    velocity the airframe measures itself (inertial and optical odometry,
    which do not drift over the tens of seconds a GPS fault lasts). Integrating
    that disagreement estimates the drift, and the shield guards the corrected
    position, so a drifting fix cannot steer the aircraft into something the
    shield cannot see. Disagreements under `floor` (m/s) are noise and ignored.
    """

    def __init__(self, floor: float = 0.05, gain: float = 0.5):
        self.floor = floor
        self.gain = gain
        self.offset: Vector = (0.0, 0.0, 0.0)
        self._rate: Vector = (0.0, 0.0, 0.0)
        self._last: Vector | None = None

    def update(self, reported: Vector, velocity: Vector, dt: float) -> Vector:
        """The best estimate of the true position, given this tick's fix and measured velocity."""
        if self._last is not None and dt > 0:
            seen = tuple((reported[i] - self._last[i]) / dt - velocity[i] for i in range(3))
            self._rate = tuple((1 - self.gain) * self._rate[i] + self.gain * seen[i] for i in range(3))
            if math.sqrt(_dot(self._rate, self._rate)) > self.floor:
                self.offset = tuple(self.offset[i] + self._rate[i] * dt for i in range(3))
        self._last = reported
        return tuple(reported[i] - self.offset[i] for i in range(3))

    @property
    def error(self) -> float:
        return math.sqrt(_dot(self.offset, self.offset))


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _project(v: Vector, halfspaces: list[tuple[Vector, float]], iterations: int = 60) -> Vector:
    """The point nearest `v` in the intersection of {u : g . u >= b}: Dykstra's alternating projections."""
    x = list(v)
    corrections = [[0.0, 0.0, 0.0] for _ in halfspaces]
    for _ in range(iterations):
        moved = 0.0
        for k, (g, b) in enumerate(halfspaces):
            y = [x[i] + corrections[k][i] for i in range(3)]
            norm2 = _dot(g, g)
            if norm2 <= 1e-12:
                continue
            slack = _dot(g, tuple(y)) - b
            projected = y if slack >= 0 else [y[i] - slack * g[i] / norm2 for i in range(3)]
            corrections[k] = [y[i] - projected[i] for i in range(3)]
            moved = max(moved, max(abs(projected[i] - x[i]) for i in range(3)))
            x = projected
        if moved < 1e-6:
            break
    return (x[0], x[1], x[2])
