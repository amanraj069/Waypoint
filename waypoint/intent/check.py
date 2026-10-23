"""Does the plan do what the mission asked? Deterministic, like the safety checks.

`check_intent(plan, intent, ...)` returns `Violation`s of type INTENT, written
like every other violation: what is missing, by how much, and a concrete fix.
A fix is not a hint: when a target is not served, the checker searches for a
viewpoint that would serve it and is itself inside the envelope (fence,
no-fly zones, obstacles, altitude band, ground and roof clearance, clear line
of sight), and names it.

What "serves" means, exactly:

* **CAPTURE / HOVER**: a CAPTURE waypoint (a HOVER target also accepts a
  HOVER) within `reach(target)` of its centre, with a clear line of sight
  to it (no other building, no ground in between; the target's own buildings
  do not count), at the requested height if the mission gave one. Seen from
  above when `overhead`: within the target's middle and above its top.
* **views and sides**: views count as distinct when at least
  `MIN_VIEW_SEPARATION` degrees apart around the target; a side is seen from a
  view within `SIDE_TOLERANCE` degrees of its compass bearing.
* **OVERFLY**: some leg passes over the target, below `max_altitude_agl` over
  its base when the mission asked for a low pass.
* **avoid**: no leg passes within its radius, at any altitude.
* **ordered**: each target is first served after the one before it.
* **landing**: the plan ends with a LAND on the landing pad.

`assess(intent, ...)` asks the same questions of the mission before any plan
exists: whether a viewpoint can serve each target at all, and whether the
shortest conceivable route fits the battery. A mission that fails it cannot be
flown as asked, and is refused with the reason instead of flown wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

from waypoint.intent.models import MissionIntent, Side, Target, TargetAction
from waypoint.planner.models import Action, MissionConstraints, Plan
from waypoint.verifier.checks import lowest_clearance, polygon_of
from waypoint.verifier.models import DEFAULT_CONFIG, Severity, VerifierConfig, Violation, ViolationType
from waypoint.world.models import World

#: The most metres from a target's edge a camera still serves it, for a large target.
VIEW_RANGE = 80.0
#: The least: a point (a field corner, a spot 120 m east) is photographed from close by.
MIN_VIEW_RANGE = 15.0


def reach(target: Target) -> float:
    """How far from its centre a view still serves the target.

    In proportion to its size: twice its radius beyond its edge, at least
    MIN_VIEW_RANGE and at most VIEW_RANGE. A field corner is served from
    within 20 m; a 36 m tomb from within about 54 m; a palace from 80 m out.
    """
    return target.radius + min(VIEW_RANGE, max(MIN_VIEW_RANGE, 2.0 * target.radius))
#: Degrees around a target between two views for them to count as distinct sides.
MIN_VIEW_SEPARATION = 60.0
#: Degrees either side of a compass bearing that still count as that side.
SIDE_TOLERANCE = 50.0
#: Metres above a target's top for a view to count as from above.
OVERHEAD_CLEARANCE = 3.0
#: Metres from the landing pad a LAND may be.
LANDING_TOLERANCE = 3.0


def altitude_tolerance(altitude_agl: float) -> float:
    """How far from a requested height a view may be and still count."""
    return max(5.0, 0.15 * altitude_agl)


@dataclass(frozen=True)
class RoutePoint:
    """A waypoint of the route, or the point it starts from (index -1)."""

    index: int
    subtask: str
    x: float
    y: float
    z: float
    action: Action | None

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def label(self) -> str:
        if self.index < 0:
            return f"the start ({self.x:g}, {self.y:g}, {self.z:g})"
        return f'waypoint {self.index} (subtask "{self.subtask}") at ({self.x:g}, {self.y:g}, {self.z:g})'


def route_of(plan: Plan, start: tuple[float, float, float] | None = None) -> list[RoutePoint]:
    """The plan as points in flying order, led by where it starts."""
    points = []
    if start is not None:
        points.append(RoutePoint(-1, "", *start, None))
    index = 0
    for subtask in plan.subtasks:
        for wp in subtask.waypoints:
            points.append(RoutePoint(index, subtask.name, wp.x, wp.y, wp.z, wp.action))
            index += 1
    return points


# ---------------------------------------------------------------------------
# Seeing a target
# ---------------------------------------------------------------------------


class Sight:
    """Line-of-sight and serving tests against one world, with the target's own buildings excused."""

    def __init__(self, world: World | None):
        self.world = world
        self._own: dict[int, set[int]] = {}

    def own(self, target: Target) -> set[int]:
        """The buildings that are the target itself: they cannot block the view of it."""
        key = id(target)
        if key not in self._own:
            if self.world is None:
                self._own[key] = set()
            else:
                area = Point(target.x, target.y).buffer(target.radius + 2.0)
                self._own[key] = {id(b) for b in self.world.obstructions(area)}
        return self._own[key]

    @staticmethod
    def aim(target: Target) -> tuple[float, float, float]:
        return (target.x, target.y, target.base + max(1.0, 0.6 * target.height))

    def blocker(self, a: tuple[float, float, float], b: tuple[float, float, float], target: Target) -> str | None:
        """What stands between a and b, or None for a clear line of sight."""
        world = self.world
        if world is None:
            return None
        if math.dist(a[:2], b[:2]) > 1e-6 and world.buildings:
            line = LineString([a[:2], b[:2]])
            own = self.own(target)
            for building in world.obstructions(line):
                if id(building) in own:
                    continue
                crossing = line.intersection(building.polygon)
                fractions = [
                    line.project(Point(c)) / line.length
                    for geom in getattr(crossing, "geoms", [crossing])
                    for c in getattr(geom, "coords", [])
                ]
                if not fractions:
                    continue
                lowest = min(a[2] + (b[2] - a[2]) * f for f in (min(fractions), max(fractions)))
                if lowest <= building.roof:
                    return f"'{building.name or 'a building'}' ({building.roof:.0f} m roof)"
        if world.terrain is not None and not target.terrain:
            worst = lowest_clearance(a, b, world, skip_end=min(3.0, 0.1 * math.dist(a, b)))
            if worst is not None and worst[0] < 0:
                return f"the ground at ({worst[1]:.0f}, {worst[2]:.0f})"
        return None

    def view(self, target: Target, p: RoutePoint) -> tuple[bool, float | None, str]:
        """Whether `p` serves `target`, the bearing it sees it from (None: from above), and why not."""
        if p.action is None:
            return False, None, "not a waypoint"
        wanted = (Action.CAPTURE,) if target.action is TargetAction.CAPTURE else (Action.CAPTURE, Action.HOVER)
        if p.action not in wanted:
            return False, None, f"it is a {p.action.value}, not a {' or '.join(a.value for a in wanted)}"
        gap = math.hypot(p.x - target.x, p.y - target.y)
        if target.altitude_agl is not None:
            above = p.z - target.base
            if abs(above - target.altitude_agl) > altitude_tolerance(target.altitude_agl):
                return False, None, f"it is {above:.0f} m above the target's base, not {target.altitude_agl:g} m"
        if target.overhead:
            if gap > max(10.0, 0.5 * target.radius):
                return False, None, f"it is {gap:.0f} m off centre, not overhead"
            if p.z < target.top + OVERHEAD_CLEARANCE:
                return False, None, f"it is at {p.z:.0f} m, not above the {target.top:.0f} m top"
            return True, None, ""
        if gap > reach(target):
            return False, None, f"it is {gap:.0f} m from the centre, beyond the {reach(target):.0f} m a view of it serves"
        blocked = self.blocker(p.xyz, self.aim(target), target)
        if blocked:
            return False, None, f"{blocked} blocks the line of sight"
        if gap < 0.5 * target.radius and p.z >= target.top:
            return True, None, ""
        return True, math.degrees(math.atan2(p.y - target.y, p.x - target.x)) % 360.0, ""

    def passes_over(self, target: Target, a: RoutePoint, b: RoutePoint) -> tuple[bool, str]:
        """Whether the leg a-b flies over the target (low enough, when asked), and why not."""
        reach = max(target.radius, 10.0)
        f, gap = _closest(target.x, target.y, a, b)
        if gap > reach:
            return False, f"passes {gap:.0f} m from its centre"
        if target.max_altitude_agl is not None:
            z = a.z + (b.z - a.z) * f
            if z - target.base > target.max_altitude_agl + altitude_tolerance(target.max_altitude_agl):
                return False, f"crosses it {z - target.base:.0f} m above its base, higher than {target.max_altitude_agl:g} m"
        return True, ""


def _closest(x: float, y: float, a: RoutePoint, b: RoutePoint) -> tuple[float, float]:
    """Where on a-b (as a fraction) it comes closest to (x, y), and how close, horizontally."""
    dx, dy = b.x - a.x, b.y - a.y
    span = dx * dx + dy * dy
    f = 0.0 if span == 0 else max(0.0, min(1.0, ((x - a.x) * dx + (y - a.y) * dy) / span))
    return f, math.hypot(x - (a.x + f * dx), y - (a.y + f * dy))


def _gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


#: A target at least this big (radius, metres) is long enough to be photographed
#: from several points along one side, not only from several sides.
LARGE_TARGET = 30.0


def distinct_views(
    bearings: list[float | None],
    points: list[tuple[float, float]] | None = None,
    target: Target | None = None,
) -> list[float | None]:
    """The views, in order, that each add something new.

    A view is new when it is at least MIN_VIEW_SEPARATION degrees around the
    target from every view before it. For a large target (a dam, a palace
    front) a view is also new when it stands well apart from every view before
    it, at least half the target's radius and 25 m: three photographs along a
    dam's face are three views, though they are all from one side.
    """
    chosen: list[int] = []
    apart = max(25.0, 0.5 * target.radius) if target is not None and target.radius >= LARGE_TARGET else None
    for k, bearing in enumerate(bearings):
        if bearing is None:
            if all(bearings[c] is not None for c in chosen):
                chosen.append(k)
            continue
        new = True
        for c in chosen:
            other = bearings[c]
            if other is None:
                continue
            if _gap(bearing, other) >= MIN_VIEW_SEPARATION:
                continue
            if apart is not None and points is not None and math.dist(points[k], points[c]) >= apart:
                continue
            new = False
            break
        if new:
            chosen.append(k)
    return [bearings[c] for c in chosen]


def compass(bearing: float | None) -> str:
    if bearing is None:
        return "above"
    sides = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return sides[int(((bearing + 22.5) % 360) // 45)]


@dataclass
class Service:
    """How well a route serves one target."""

    target: Target
    first: int | None  # index of the first waypoint that serves it
    bearings: list[float | None]
    near_misses: list[tuple[RoutePoint, str]]
    points: list[tuple[float, float]] | None = None
    #: Every waypoint index that serves it, in order (for an OVERFLY, the end of each leg that crosses it).
    hits: list[int] | None = None

    @property
    def views(self) -> list[float | None]:
        return distinct_views(self.bearings, self.points, self.target)

    def missing_sides(self) -> list[Side]:
        return [
            side for side in self.target.sides
            if not any(b is not None and _gap(b, side.bearing) <= SIDE_TOLERANCE for b in self.bearings)
        ]

    @property
    def satisfied(self) -> bool:
        if self.first is None:
            return False
        if self.target.action is TargetAction.OVERFLY:
            return True
        return len(self.views) >= self.target.views and not self.missing_sides()

    @property
    def score(self) -> float:
        """Fraction of what was asked that the route delivers, 0 to 1."""
        if self.first is None:
            return 0.0
        if self.target.action is TargetAction.OVERFLY:
            return 1.0
        if self.target.sides:
            return (len(self.target.sides) - len(self.missing_sides())) / len(self.target.sides)
        return min(len(self.views), self.target.views) / self.target.views


def serve(target: Target, route: list[RoutePoint], sight: Sight) -> Service:
    """Everything the route does for one target."""
    first = None
    bearings: list[float | None] = []
    points: list[tuple[float, float]] = []
    misses: list[tuple[RoutePoint, str]] = []
    hits: list[int] = []
    if target.action is TargetAction.OVERFLY:
        for a, b in zip(route, route[1:]):
            ok, why = sight.passes_over(target, a, b)
            if ok:
                hits.append(b.index)
            else:
                misses.append((b, why))
        if hits:
            return Service(target, hits[0], [], [], hits=hits)
        misses.sort(key=lambda m: math.hypot(m[0].x - target.x, m[0].y - target.y))
        return Service(target, None, [], misses[:2], hits=[])
    for p in route:
        ok, bearing, why = sight.view(target, p)
        if ok:
            bearings.append(bearing)
            points.append((p.x, p.y))
            hits.append(p.index)
            if first is None:
                first = p.index
        elif p.action in (Action.CAPTURE, Action.HOVER) or math.hypot(p.x - target.x, p.y - target.y) <= reach(target):
            if p.index >= 0:
                misses.append((p, why))
    misses.sort(key=lambda m: math.hypot(m[0].x - target.x, m[0].y - target.y))
    return Service(target, first, bearings, misses[:2], points, hits)


def in_order(services: list[Service]) -> tuple[bool, Service | None, Service | None]:
    """Whether the targets can be served in the order listed, and if not, the pair that cannot.

    Each target needs one waypoint that serves it after the one chosen for the
    target before it; the earliest such waypoint is always the best choice, so
    a single greedy pass decides. A target served before *and* after the one
    it must follow (a town the flight starts in and later crosses again) is in
    order.
    """
    position = -1
    previous = None
    for service in services:
        later = [h for h in (service.hits or []) if h > position]
        if service.first is None:
            continue  # not served at all: reported as such, not as out of order
        if not later:
            return False, previous, service
        position, previous = later[0], service
    return True, None, None


# ---------------------------------------------------------------------------
# Finding a viewpoint that would work
# ---------------------------------------------------------------------------


class Envelope:
    """Where a waypoint may go: the safety envelope, as fast point tests."""

    def __init__(self, constraints: MissionConstraints | None, world: World | None, config: VerifierConfig):
        self.constraints = constraints
        self.world = world
        self.config = config
        self.fence: Polygon | None = polygon_of(constraints.geofence, "geofence") if constraints else None
        self.zones = [polygon_of(z, "no-fly zone") for z in constraints.no_fly_zones] if constraints else []

    def floor_at(self, x: float, y: float) -> float:
        """The lowest a waypoint at (x, y) may be: above the ground, the roofs and the altitude floor."""
        config = self.config
        lowest = self.constraints.min_altitude if self.constraints else 0.0
        if self.world is not None:
            lowest = max(lowest, self.world.elevation_at(x, y) + config.min_ground_clearance + 1.0)
            for building in self.world.buildings_at(x, y):
                lowest = max(lowest, building.roof + config.roof_clearance + 1.0)
        return lowest

    def admits(self, x: float, y: float, z: float) -> bool:
        constraints = self.constraints
        point = Point(x, y)
        if self.fence is not None and (not self.fence.covers(point) or self.fence.exterior.distance(point) < 3.0):
            return False
        if any(zone.buffer(3.0).covers(point) for zone in self.zones):
            return False
        if constraints is not None:
            if z > constraints.max_altitude or z < constraints.min_altitude:
                return False
            for obstacle in constraints.obstacles:
                if obstacle.height is not None and z > obstacle.height + 3.0:
                    continue
                if math.hypot(x - obstacle.x, y - obstacle.y) < obstacle.radius + constraints.min_separation + 3.0:
                    return False
        return z >= self.floor_at(x, y) - 1e-6


def suggest_viewpoint(
    target: Target,
    envelope: Envelope,
    sight: Sight,
    *,
    side: Side | None = None,
    away_from: list[float | None] = (),
) -> tuple[float, float, float] | None:
    """A point that would serve the target and is itself flyable, or None if there is none.

    Prefers the requested side; otherwise the bearing farthest from the views
    already taken, then the nearest stand-off.
    """
    probe = RoutePoint(0, "", 0.0, 0.0, 0.0, Action.CAPTURE if target.action is not TargetAction.HOVER else Action.HOVER)

    def height_at(x: float, y: float) -> float | None:
        if target.altitude_agl is not None:
            z = target.base + target.altitude_agl
        elif target.overhead:
            z = target.top + max(10.0, OVERHEAD_CLEARANCE + 2.0)
        else:
            z = target.base + min(max(20.0, target.height * 0.8 + 5.0), 60.0)
        z = max(z, envelope.floor_at(x, y))
        if target.altitude_agl is not None and abs(z - target.base - target.altitude_agl) > altitude_tolerance(target.altitude_agl):
            return None
        return round(z, 1)

    def works(x: float, y: float) -> tuple[float, float, float] | None:
        z = height_at(x, y)
        if z is None or not envelope.admits(x, y, z):
            return None
        p = RoutePoint(0, "", x, y, z, probe.action)
        ok, bearing, _ = sight.view(target, p)
        if not ok:
            return None
        if side is not None and (bearing is None or _gap(bearing, side.bearing) > SIDE_TOLERANCE):
            return None
        if away_from and target.views > 1 and not target.overhead:
            if bearing is None or any(b is not None and _gap(bearing, b) < MIN_VIEW_SEPARATION for b in away_from):
                return None
        return (round(x, 1), round(y, 1), z)

    if target.overhead:
        for r in (0.0, 0.25 * target.radius, 0.45 * target.radius):
            for angle in range(0, 360, 45):
                hit = works(target.x + r * math.cos(math.radians(angle)), target.y + r * math.sin(math.radians(angle)))
                if hit:
                    return hit
                if r == 0.0:
                    break
        return None

    if side is not None:
        angles = [side.bearing + d for d in (0, 20, -20, 40, -40)]
    else:
        taken = [b for b in away_from if b is not None]
        angles = sorted(
            range(0, 360, 30),
            key=lambda a: -min((_gap(a, b) for b in taken), default=180.0),
        )
    for standoff in (8.0, 15.0, 25.0, 35.0, 50.0, 65.0, 80.0):
        r = target.radius + standoff
        if r > reach(target):
            break
        for angle in angles:
            hit = works(target.x + r * math.cos(math.radians(angle)), target.y + r * math.sin(math.radians(angle)))
            if hit:
                return hit
    return None


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def check_intent(
    plan: Plan,
    intent: MissionIntent,
    *,
    constraints: MissionConstraints | None = None,
    world: World | None = None,
    config: VerifierConfig | None = None,
) -> list[Violation]:
    """Every way the plan falls short of the mission, with a flyable fix for each."""
    if intent is None or intent.empty:
        return []
    config = config or DEFAULT_CONFIG
    sight = Sight(world)
    envelope = Envelope(constraints, world, config)
    route = route_of(plan, tuple(float(v) for v in config.home))
    violations: list[Violation] = []

    def add(text: str, index: int | None = None, fix: dict | None = None) -> None:
        violations.append(Violation(
            type=ViolationType.INTENT, waypoint_index=index, description=text, severity=Severity.ERROR, fix=fix,
        ))

    def capture_at(target: Target, point) -> dict | None:
        if point is None:
            return None
        action = "CAPTURE" if target.action is TargetAction.CAPTURE else "HOVER"
        return {"kind": "add_waypoint", "action": action, "at": list(point), "target": target.name}

    services = [serve(target, route, sight) for target in intent.targets]
    for service in services:
        target = service.target
        if target.action is TargetAction.OVERFLY:
            if service.first is None:
                low = f" no higher than {target.max_altitude_agl:g} m above its base" if target.max_altitude_agl else ""
                miss = f" The nearest leg {service.near_misses[0][1]}." if service.near_misses else ""
                add(
                    f"The mission asks to fly over '{target.name}' (centred at ({target.x:.0f}, {target.y:.0f}), "
                    f"{target.radius * 2:.0f} m across){low}, but no leg of the plan crosses it.{miss} "
                    f"Route a leg across it, e.g. through ({target.x:.0f}, {target.y:.0f})."
                , fix=_overfly_fix(target, envelope))
            continue
        verb = "photograph" if target.action is TargetAction.CAPTURE else "hover at"
        if service.first is None:
            fix = suggest_viewpoint(target, envelope, sight)
            why = ""
            if service.near_misses:
                p, reason = service.near_misses[0]
                why = f" The closest candidate, {p.label}, does not count: {reason}."
            remedy = (
                f"Add a {'CAPTURE' if target.action is TargetAction.CAPTURE else 'HOVER'} waypoint that serves it, "
                f"e.g. at ({fix[0]:g}, {fix[1]:g}, {fix[2]:g}), which is inside the envelope with a clear view."
                if fix else
                "No point inside the envelope serves it; the mission cannot be flown as asked."
            )
            add(
                f"The mission asks to {verb} '{target.name}', but no waypoint in the plan does: a "
                f"{'CAPTURE' if target.action is TargetAction.CAPTURE else 'HOVER or CAPTURE'} has to be within "
                f"{reach(target):.0f} m of its centre ({target.x:.0f}, {target.y:.0f}) with a clear line "
                f"of sight{', directly above its top' if target.overhead else ''}"
                f"{f', {target.altitude_agl:g} m above its base' if target.altitude_agl else ''}.{why} {remedy}",
                fix=capture_at(target, fix),
            )
            continue
        views = service.views
        for side in service.missing_sides():
            fix = suggest_viewpoint(target, envelope, sight, side=side)
            add(
                f"The mission asks to {verb} '{target.name}' from the {side.value} side, but no view in the plan is "
                f"within {SIDE_TOLERANCE:g} deg of {side.value} (it is seen from: "
                f"{', '.join(compass(b) for b in views)}). "
                + (f"Add a CAPTURE on the {side.value} side, e.g. at ({fix[0]:g}, {fix[1]:g}, {fix[2]:g})." if fix
                   else f"No flyable point sees it from the {side.value}; the mission cannot be flown as asked."),
                fix=capture_at(target, fix),
            )
        if not target.sides and len(views) < target.views:
            fix = suggest_viewpoint(target, envelope, sight, away_from=service.bearings)
            add(
                f"The mission asks to {verb} '{target.name}' from {target.views} distinct sides, but the plan sees it "
                f"from only {len(views)} ({', '.join(compass(b) for b in views)}); views count as distinct when at "
                f"least {MIN_VIEW_SEPARATION:g} deg apart around it. "
                + (f"Add a CAPTURE from another side, e.g. at ({fix[0]:g}, {fix[1]:g}, {fix[2]:g})." if fix
                   else "No further flyable side has a clear view; the mission cannot be flown as asked."),
                fix=capture_at(target, fix),
            )

    if intent.ordered:
        ok, before, after = in_order(services)
        if not ok:
            add(
                f"The mission asks for '{before.target.name}' before '{after.target.name}', but the plan never "
                f"serves '{after.target.name}' after it has served '{before.target.name}'. "
                "Reorder the subtasks to follow the mission.",
                after.first,
            )

    for avoid in intent.avoid:
        for a, b in zip(route, route[1:]):
            _, gap = _closest(avoid.x, avoid.y, a, b)
            if gap < avoid.radius:
                add(
                    f"The mission says never to pass over '{avoid.name}', but the leg to {b.label} passes "
                    f"{gap:.0f} m from its centre ({avoid.x:.0f}, {avoid.y:.0f}), inside its {avoid.radius:.0f} m. "
                    "Route around it.",
                    b.index,
                )
                break

    if intent.landing is not None and route[1:]:
        last_point = route[-1]
        lx, ly, lz = intent.landing
        if last_point.action is not Action.LAND or math.hypot(last_point.x - lx, last_point.y - ly) > LANDING_TOLERANCE:
            add(
                f"The flight must end with a LAND on the landing pad at ({lx:g}, {ly:g}, {lz:g}), but the plan ends "
                f"with {last_point.label}{'' if last_point.action is None else ' (' + last_point.action.value + ')'}. "
                f"Make the last waypoint a LAND at ({lx:g}, {ly:g}, {lz:g}).",
                last_point.index,
                fix={"kind": "set_landing", "at": [lx, ly, lz]},
            )
    return violations


def _overfly_fix(target: Target, envelope: Envelope) -> dict | None:
    """A MOVE over the target's centre, as low as allowed, when that is flyable."""
    floor = envelope.floor_at(target.x, target.y)
    z = floor + 2.0
    if target.max_altitude_agl is not None:
        z = min(max(floor, target.base + 0.5 * target.max_altitude_agl), target.base + target.max_altitude_agl)
        if z < floor:
            return None
    if not envelope.admits(target.x, target.y, round(z, 1)):
        return None
    return {"kind": "add_waypoint", "action": "MOVE", "at": [target.x, target.y, round(z, 1)], "target": target.name}


# ---------------------------------------------------------------------------
# Before any plan: can the mission be flown at all?
# ---------------------------------------------------------------------------


def assess(
    intent: MissionIntent,
    *,
    constraints: MissionConstraints,
    world: World | None = None,
    config: VerifierConfig | None = None,
    home: tuple[float, float, float] | None = None,
    landing: tuple[float, float, float] | None = None,
) -> list[str]:
    """Why the mission cannot be flown as asked, or [] if it can (as far as can be known up front)."""
    config = config or DEFAULT_CONFIG
    sight = Sight(world)
    envelope = Envelope(constraints, world, config)
    home = home or (0.0, 0.0, 0.0)
    landing = landing or home
    reasons: list[str] = []
    fence = envelope.fence
    for target in intent.targets:
        outside = fence.exterior.distance(Point(target.x, target.y)) if fence and not fence.covers(Point(target.x, target.y)) else 0.0
        if target.action is TargetAction.OVERFLY:
            if outside > target.radius:
                reasons.append(f"'{target.name}' at ({target.x:.0f}, {target.y:.0f}) lies {outside:.0f} m outside the geofence, so it cannot be flown over")
            continue
        if target.altitude_agl is not None and target.base + target.altitude_agl > constraints.max_altitude + altitude_tolerance(target.altitude_agl):
            reasons.append(
                f"the mission asks for {target.altitude_agl:g} m above '{target.name}' (an altitude of "
                f"{target.base + target.altitude_agl:.0f} m), over the {constraints.max_altitude:g} m ceiling"
            )
            continue
        if outside > reach(target):
            reasons.append(
                f"'{target.name}' at ({target.x:.0f}, {target.y:.0f}) lies {outside:.0f} m outside the geofence, "
                f"beyond the {reach(target) - target.radius:.0f} m from its edge a view inside it could serve"
            )
            continue
        found: list[float | None] = []
        wanted = [None] * (1 if target.overhead else max(target.views - len(target.sides), 0))
        for side in [*target.sides, *wanted]:
            point = suggest_viewpoint(target, envelope, sight, side=side, away_from=found)
            if point is None:
                what = f"from the {side.value}" if side else ("from above" if target.overhead else f"from {target.views} distinct sides")
                reasons.append(f"no flyable point inside the envelope sees '{target.name}' {what} with a clear line of sight")
                break
            ok, bearing, _ = sight.view(target, RoutePoint(0, "", *point, Action.CAPTURE))
            found.append(bearing)

    if intent.targets:
        near = lambda t: t.radius if t.action is TargetAction.OVERFLY else reach(t)  # noqa: E731
        shortest = max(
            max(0.0, math.dist(home[:2], (t.x, t.y)) - near(t)) + max(0.0, math.dist((t.x, t.y), landing[:2]) - near(t))
            for t in intent.targets
        )
        needed = shortest / config.meters_per_battery_pct + config.reserve_battery_pct
        if needed > constraints.battery_budget_pct:
            reasons.append(
                f"even the shortest conceivable route, about {shortest:.0f} m, needs {needed:.0f}% battery, "
                f"over the {constraints.battery_budget_pct:g}% budget"
            )
    return reasons


# ---------------------------------------------------------------------------
# During and after the flight
# ---------------------------------------------------------------------------


def remaining(intent: MissionIntent, flown: list[RoutePoint], world: World | None = None) -> MissionIntent:
    """What is left of the mission once the waypoints in `flown` have been reached."""
    sight = Sight(world)
    left = []
    for target in intent.targets:
        service = serve(target, flown, sight)
        if service.satisfied:
            continue
        if service.first is None or target.action is TargetAction.OVERFLY:
            left.append(target)
            continue
        missing = service.missing_sides()
        views = max(1, target.views - len(service.views)) if not target.sides else max(len(missing), 1)
        left.append(target.model_copy(update={"views": views, "sides": missing}))
    return intent.model_copy(update={"targets": left})


def evaluate(
    intent: MissionIntent,
    flown: list[RoutePoint],
    track: list[tuple[float, float, float]],
    world: World | None = None,
    *,
    planned_landing: tuple[float, float] | None = None,
) -> dict:
    """How much of the mission a flight actually did, from the waypoints it reached and its track.

    `flown` are the waypoints reached, in order; `track` every position, for
    the avoid zones and the touchdown. Whether it landed where the mission
    said is judged by where the flight's last plan put its LAND
    (`planned_landing`), when given: a touchdown blown 20 m off the pad by a
    storm or a drifting GPS is an accuracy problem, reported as
    `touchdown_error_m`, not a different mission. A flight that ended on a
    failsafe somewhere else did fly a different mission, and fails.
    """
    sight = Sight(world)
    services = [serve(t, flown, sight) for t in intent.targets]
    rows = [
        {
            "name": s.target.name,
            "action": s.target.action.value,
            "required_views": max(s.target.views, len(s.target.sides)),
            "views": len(s.views) if s.first is not None else 0,
            "missing_sides": [side.value for side in s.missing_sides()] if s.first is not None else [x.value for x in s.target.sides],
            "served": s.satisfied,
            "score": round(s.score, 3),
        }
        for s in services
    ]
    order_ok = True
    if intent.ordered:
        order_ok = all(s.first is not None for s in services) and in_order(services)[0]
    breached = [
        a.name for a in intent.avoid
        if any(math.hypot(x - a.x, y - a.y) < a.radius for x, y, _ in track)
    ]
    landed_ok = None
    touchdown = None
    if intent.landing is not None and track:
        x, y, _ = track[-1]
        touchdown = round(math.hypot(x - intent.landing[0], y - intent.landing[1]), 2)
        where = planned_landing or (x, y)
        landed_ok = math.hypot(where[0] - intent.landing[0], where[1] - intent.landing[1]) <= max(LANDING_TOLERANCE, 5.0)
    score = sum(r["score"] for r in rows) / len(rows) if rows else 1.0
    return {
        "targets": rows,
        "score": round(score, 4),
        "order_ok": order_ok,
        "avoid_breached": breached,
        "landed_on_pad": landed_ok,
        "touchdown_error_m": touchdown,
        "accomplished": all(r["served"] for r in rows) and order_ok and not breached and landed_ok is not False,
    }
