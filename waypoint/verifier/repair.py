"""Deterministic repair: fix what geometry can fix, before asking the LLM again.

When the verifier rejects a plan, the reasons usually name the remedy: climb
over this roof, route around that no-fly zone, slow down here, add a
photograph from the north. Many of those remedies need no judgement, and
asking a language model for them costs seconds and sometimes several tries.
`repair()` applies them directly:

    MAX_VELOCITY       slow the waypoint to the limit
    TURN_FEASIBILITY   slow through the corner to the speed that makes it flyable
    ALTITUDE           bring the waypoint inside the band
    TERRAIN            raise a low waypoint; lift a low leg over the high ground
                       (a vertical climb out of, or descent onto, a pad)
    BUILDING           move a waypoint out of a footprint; take a leg around the
                       building, or over its roof, whichever is shorter
    NO_FLY_ZONE        move a waypoint out; take a leg around the zone
    MIN_SEPARATION     push a waypoint out of the keep-out; take a leg around the
                       obstacle, or over its top
    GEOFENCE           pull a waypoint back inside; keep a leg inside a concave fence
    LANDING            put a LAND on the ground, and on the landing pad
    INTENT             add the waypoint the intent check proposed (a CAPTURE that
                       serves the target, a pass over it), at the cheapest place in
                       the route

Detours come from a visibility graph over the buffered blockers inside the
fence: the shortest planar path between the leg's ends that touches nothing.

Repair never relaxes anything. Every round runs the full verifier (and the
intent check, when there is an intent) on the result, and a plan is returned
only when it passes; otherwise `plan` is None and the LLM is asked, with the
verifier's reasons, as before. BATTERY has no geometric fix and is left to the
planner (or, in the air, to `trim_for_battery`).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

from waypoint.planner.models import Action, MissionConstraints, Plan, SubTask, Waypoint
from waypoint.verifier.checks import lowest_clearance, polygon_of
from waypoint.verifier.models import (
    DEFAULT_CONFIG,
    SAFETY_TYPES,
    Severity,
    VerificationResult,
    VerifierConfig,
    Violation,
    ViolationType,
)
from waypoint.verifier.safety_verifier import verify
from waypoint.world.landing import find as find_site, survey as survey_site
from waypoint.world.models import World

if TYPE_CHECKING:
    from waypoint.intent.models import MissionIntent

#: Metres of horizontal margin a detour keeps from what it goes around.
DETOUR_MARGIN = 6.0
#: Metres above a roof or obstacle top a climb-over clears it by, beyond the verifier's own clearance.
OVER_MARGIN = 2.0
#: How far around a leg to look for things a detour has to avoid.
CORRIDOR = 250.0
#: Speed for waypoints repair adds.
ADDED_SPEED = 4.0
CAPTURE_SPEED = 2.5


@dataclass
class Repair:
    """What repair did, and the plan it ended with if that plan passes."""

    plan: Plan | None
    actions: list[str] = field(default_factory=list)
    result: VerificationResult | None = None
    rounds: int = 0

    @property
    def repaired(self) -> bool:
        return self.plan is not None


# ---------------------------------------------------------------------------
# The route, editable
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    subtask: str
    wp: Waypoint


def _entries(plan: Plan) -> list[_Entry]:
    return [_Entry(s.name, wp.model_copy()) for s in plan.subtasks for wp in s.waypoints]


def _plan(entries: list[_Entry], mission_text: str) -> Plan:
    """Back to subtasks: consecutive waypoints of the same phase stay together."""
    subtasks: list[SubTask] = []
    for entry in entries:
        if subtasks and subtasks[-1].name == entry.subtask:
            subtasks[-1].waypoints.append(entry.wp)
        else:
            subtasks.append(SubTask(name=entry.subtask, waypoints=[entry.wp]))
    return Plan(mission_text=mission_text, subtasks=subtasks)


def _point(entries: list[_Entry], index: int, home: tuple[float, float, float]) -> tuple[float, float, float]:
    if index < 0:
        return home
    wp = entries[index].wp
    return (wp.x, wp.y, wp.z)


# ---------------------------------------------------------------------------
# Planar detours
# ---------------------------------------------------------------------------


def shortest_path(
    a: tuple[float, float],
    b: tuple[float, float],
    blockers: list[Polygon],
    fence: Polygon | None,
) -> list[tuple[float, float]] | None:
    """The shortest planar path from a to b that enters no blocker and stays in the fence.

    Returns the intermediate corners (empty when the straight line is clear),
    or None when there is no way through. Blockers are expected to be buffered
    already; their vertices are the graph's nodes.
    """
    corridor = LineString([a, b]).buffer(CORRIDOR)
    near = [poly for poly in blockers if poly.intersects(corridor)]
    # One merged, simplified region: one test per edge instead of one per
    # building, and touching buildings become one block with one outline.
    region = unary_union(near).simplify(0.5, preserve_topology=True) if near else None
    solid = prep(region.buffer(-0.05)) if region is not None and not region.is_empty else None
    inside = prep(fence.buffer(0.5)) if fence is not None else None

    def clear(p, q) -> bool:
        if math.dist(p, q) < 1e-6:
            return True
        segment = LineString([p, q])
        if inside is not None and not inside.covers(segment):
            return False
        return solid is None or not solid.intersects(segment)

    if clear(a, b):
        return []
    if region is None:
        return None
    nodes = [a, b]
    for poly in getattr(region, "geoms", [region]):
        for x, y in list(poly.exterior.coords)[:-1]:
            if fence is None or fence.covers(Point(x, y)):
                nodes.append((x, y))
    # A* with edges checked lazily: the graph is small, the checks are not free.
    best = {0: 0.0}
    previous: dict[int, int] = {}
    queue = [(math.dist(a, b), 0.0, 0)]
    done: set[int] = set()
    while queue:
        _, cost, i = heapq.heappop(queue)
        if i in done:
            continue
        done.add(i)
        if i == 1:
            break
        for j in range(len(nodes)):
            if j in done or j == i:
                continue
            step = math.dist(nodes[i], nodes[j])
            if cost + step >= best.get(j, math.inf):
                continue
            if not clear(nodes[i], nodes[j]):
                continue
            best[j] = cost + step
            previous[j] = i
            heapq.heappush(queue, (cost + step + math.dist(nodes[j], b), cost + step, j))
    if 1 not in previous:
        return None
    path = []
    node = previous[1]
    while node != 0:
        path.append(nodes[node])
        node = previous[node]
    return [(round(x, 1), round(y, 1)) for x, y in reversed(path)]


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


class _Repairer:
    def __init__(self, constraints, world, config, intent):
        self.constraints: MissionConstraints = constraints
        self.world: World | None = world
        self.config: VerifierConfig = config
        self.intent = intent
        self.fence = polygon_of(constraints.geofence, "geofence")
        self.zones = [polygon_of(z, "no-fly zone") for z in constraints.no_fly_zones]
        self.home = tuple(float(v) for v in config.home)
        self.actions: list[str] = []

    # -- helpers ---------------------------------------------------------------

    def floor_at(self, x: float, y: float) -> float:
        floor = self.constraints.min_altitude
        if self.world is not None:
            floor = max(floor, self.world.elevation_at(x, y) + self.config.min_ground_clearance + 1.0)
        return floor

    def blockers(self, z_low: float) -> list[Polygon]:
        """What a leg as low as `z_low` somewhere must go around."""
        out = [zone.buffer(DETOUR_MARGIN, quad_segs=2) for zone in self.zones]
        sep = self.constraints.min_separation
        for o in self.constraints.obstacles:
            if o.height is not None and z_low > o.height + 1.0:
                continue
            out.append(Point(o.x, o.y).buffer(o.radius + sep + DETOUR_MARGIN, quad_segs=3))
        if self.world is not None:
            for b in self.world.buildings:
                if b.roof + self.config.roof_clearance >= z_low:
                    out.append(b.polygon.buffer(DETOUR_MARGIN, quad_segs=1, join_style=2))
        return out

    def inside(self, x: float, y: float, margin: float = 3.0) -> tuple[float, float]:
        """The nearest point at least `margin` inside the fence."""
        if self.fence.buffer(-margin).covers(Point(x, y)):
            return x, y
        core = self.fence.buffer(-margin)
        if core.is_empty:
            return x, y
        nearest = core.exterior.interpolate(core.exterior.project(Point(x, y)))
        return round(nearest.x, 1), round(nearest.y, 1)

    def out_of(self, poly: Polygon, x: float, y: float, margin: float) -> tuple[float, float]:
        grown = poly.buffer(margin)
        if not grown.covers(Point(x, y)):
            return x, y
        edge = grown.exterior.interpolate(grown.exterior.project(Point(x, y)))
        return round(edge.x, 1), round(edge.y, 1)

    def interp(self, a, b, points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
        """Give planar corners altitudes along a-b, by distance travelled."""
        legs = [a[:2], *points, b[:2]]
        lengths = [math.dist(p, q) for p, q in zip(legs, legs[1:])]
        span = sum(lengths) or 1.0
        out, run = [], 0.0
        for corner, length in zip(points, lengths):
            run += length
            z = a[2] + (b[2] - a[2]) * run / span
            out.append((corner[0], corner[1], round(max(z, self.floor_at(*corner)), 1)))
        return out

    def insert(self, entries: list[_Entry], at: int, points, subtask: str | None = None, speed: float | None = None) -> None:
        name = subtask or (entries[at].subtask if at < len(entries) else entries[-1].subtask)
        reference = entries[min(at, len(entries) - 1)].wp.speed
        for k, (x, y, z) in enumerate(points):
            entries.insert(at + k, _Entry(name, Waypoint(x=x, y=y, z=z, speed=speed or min(reference, ADDED_SPEED) or ADDED_SPEED)))

    # -- one round --------------------------------------------------------------

    def apply(self, entries: list[_Entry], violations: list[Violation]) -> bool:
        """Apply one fix per waypoint this round; True if anything changed."""
        changed = False
        touched: set[int] = set()
        # Later waypoints first, so inserting points does not shift what is still to fix.
        ordered = sorted(
            violations,
            key=lambda v: (-(v.waypoint_index if v.waypoint_index is not None else -1), v.type.value),
        )
        for v in ordered:
            i = v.waypoint_index
            if v.type is ViolationType.INTENT:
                continue  # after the safety fixes, below
            if i is None or i >= len(entries) or i in touched:
                continue
            fixed = self.fix(entries, i, v)
            if fixed:
                touched.add(i)
                changed = True
        if not changed:
            for v in violations:
                if v.type is ViolationType.INTENT and v.fix:
                    if self.fix_intent(entries, v):
                        changed = True
                        break  # one at a time: each addition changes the route the next is judged on
        return changed

    def fix(self, entries: list[_Entry], i: int, v: Violation) -> bool:
        wp = entries[i].wp
        c = self.constraints
        kind = v.type
        if kind is ViolationType.MAX_VELOCITY and wp.speed > c.max_velocity:
            wp.speed = c.max_velocity
            self.actions.append(f"slowed waypoint {i} to {c.max_velocity:g} m/s")
            return True
        if kind is ViolationType.TURN_FEASIBILITY:
            angle = self._turn(entries, i)
            safe = self.config.speed_for_turn(angle) if angle else None
            if safe and wp.speed > safe and i + 1 < len(entries):
                wp.speed = round(max(1.0, safe * 0.95), 1)
                entries[i + 1].wp.speed = min(entries[i + 1].wp.speed, wp.speed)
                self.actions.append(f"slowed through the {angle:.0f} deg corner at waypoint {i}")
                return True
            return False
        if kind is ViolationType.ALTITUDE:
            if wp.z > c.max_altitude:
                wp.z = c.max_altitude
            elif wp.z < c.min_altitude and wp.action is not Action.LAND:
                wp.z = c.min_altitude
            else:
                return False
            self.actions.append(f"brought waypoint {i} inside the altitude band, to {wp.z:g} m")
            return True
        if kind is ViolationType.LANDING and wp.action is Action.LAND:
            return self._land(entries, i)
        if kind is ViolationType.GEOFENCE:
            if not self.fence.covers(Point(wp.x, wp.y)):
                wp.x, wp.y = self.inside(wp.x, wp.y)
                self.actions.append(f"pulled waypoint {i} back inside the geofence")
                return True
            return self._detour(entries, i, "keep inside the concave fence")
        if kind is ViolationType.NO_FLY_ZONE:
            for zone in self.zones:
                if zone.covers(Point(wp.x, wp.y)):
                    wp.x, wp.y = self.out_of(zone, wp.x, wp.y, DETOUR_MARGIN)
                    self.actions.append(f"moved waypoint {i} out of a no-fly zone")
                    return True
            return self._detour(entries, i, "around a no-fly zone")
        if kind is ViolationType.MIN_SEPARATION:
            for o in c.obstacles:
                keep = o.radius + c.min_separation
                if math.hypot(wp.x - o.x, wp.y - o.y) < keep and not (o.height is not None and wp.z > o.height):
                    wp.x, wp.y = self.out_of(Point(o.x, o.y).buffer(keep, quad_segs=8), wp.x, wp.y, DETOUR_MARGIN)
                    self.actions.append(f"pushed waypoint {i} out of the keep-out round '{o.name}'")
                    return True
            return self._around_or_over(entries, i, "an obstacle")
        if kind is ViolationType.BUILDING:
            if self.world is not None:
                for b in self.world.buildings_at(wp.x, wp.y):
                    over = b.roof + self.config.roof_clearance + 0.5
                    if b.roof < wp.z < over and wp.action is not Action.LAND and over <= c.max_altitude:
                        wp.z = round(over, 1)
                        self.actions.append(f"raised waypoint {i} to {wp.z:g} m, clear of the roof of {b.label()}")
                        return True
                    if wp.z <= b.roof:
                        if wp.action is Action.MOVE and b.roof + self.config.roof_clearance + OVER_MARGIN <= c.max_altitude:
                            wp.z = round(b.roof + self.config.roof_clearance + OVER_MARGIN, 1)
                            self.actions.append(f"raised waypoint {i} over the roof of {b.label()}")
                        else:
                            wp.x, wp.y = self.out_of(b.polygon, wp.x, wp.y, DETOUR_MARGIN)
                            self.actions.append(f"moved waypoint {i} off the footprint of {b.label()}")
                        return True
            return self._around_or_over(entries, i, "a building")
        if kind is ViolationType.TERRAIN:
            if self.world is not None:
                floor = self.floor_at(wp.x, wp.y)
                if wp.z < floor - 1.0 + 1e-6 and wp.action is not Action.LAND and not (i == 0 and self._grounded(wp)):
                    if floor > c.max_altitude:
                        return False
                    wp.z = round(floor, 1)
                    self.actions.append(f"raised waypoint {i} to {wp.z:g} m to clear the ground")
                    return True
            return self._lift(entries, i)
        return False

    def _grounded(self, wp: Waypoint) -> bool:
        ground = self.world.elevation_at(wp.x, wp.y) if self.world else 0.0
        return abs(wp.z - ground) <= 0.5

    def _turn(self, entries, i) -> float | None:
        a, b = _point(entries, i - 1, self.home), _point(entries, i, self.home)
        c = _point(entries, i + 1, self.home) if i + 1 < len(entries) else None
        if c is None:
            return None
        u, w = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
        if math.hypot(*u) < 1e-6 or math.hypot(*w) < 1e-6:
            return None
        return abs(math.degrees(math.atan2(u[0] * w[1] - u[1] * w[0], u[0] * w[0] + u[1] * w[1])))

    def _land(self, entries, i) -> bool:
        wp = entries[i].wp
        target = self.intent.landing if self.intent is not None and self.intent.landing is not None else None
        if target is not None and math.hypot(wp.x - target[0], wp.y - target[1]) > 0.5:
            wp.x, wp.y, wp.z = target
            self.actions.append(f"moved the LAND to the landing pad ({target[0]:g}, {target[1]:g})")
            return True
        if self.world is None:
            return False
        site = survey_site(self.world, wp.x, wp.y)
        if not site.usable:
            nearby = find_site(self.world, wp.x, wp.y)
            if not nearby.usable:
                return False
            wp.x, wp.y, wp.z = nearby.x, nearby.y, round(nearby.elevation, 2)
            self.actions.append(f"moved the LAND {nearby.moved:.0f} m to ground it can be put down on")
            return True
        if abs(wp.z - site.elevation) > 0.01:
            wp.z = round(site.elevation, 2)
            self.actions.append(f"put the LAND at waypoint {i} on the ground, z = {wp.z:g}")
            return True
        return False

    def _lift(self, entries, i) -> bool:
        """A leg that runs too low over the ground: climb before it, or lift its low point."""
        if self.world is None:
            return False
        a, b = _point(entries, i - 1, self.home), _point(entries, i, self.home)
        worst = lowest_clearance(a, b, self.world)
        if worst is None:
            return False
        _, x, y = worst
        need = self.world.elevation_at(x, y) + self.config.min_ground_clearance + OVER_MARGIN
        if need > self.constraints.max_altitude:
            return False
        end = entries[i].wp
        start_grounded = i == 0 or (self.world is not None and abs(a[2] - self.world.elevation_at(a[0], a[1])) <= 0.5)
        steep = b[2] > a[2] + 0.5 and _sags(a, b, end.speed, self.config)
        if steep and not start_grounded and math.dist(a[:2], b[:2]) > 1.0 and end.action is not Action.LAND:
            # Climbing faster than the airframe can: climb in place first, then fly level.
            climb = round(min(max(b[2], need), self.constraints.max_altitude), 1)
            self.insert(entries, i, [(a[0], a[1], climb)])
            self.actions.append(f"climbed to {climb:g} m before leg {i}, which was steeper than the aircraft can climb")
            return True
        if start_grounded and math.dist(a[:2], b[:2]) > 1.0:
            climb = round(max(need, min(b[2], self.constraints.max_altitude)), 1)
            self.insert(entries, i, [(a[0], a[1], climb)])
            self.actions.append(f"climbed straight up to {climb:g} m before leg {i}")
            return True
        if end.action is Action.LAND and math.dist(a[:2], b[:2]) > 1.0:
            hold = round(max(need, a[2]), 1)
            self.insert(entries, i, [(b[0], b[1], hold)])
            self.actions.append(f"crossed to the pad at {hold:g} m before landing")
            return True
        f = (math.dist(a[:2], (x, y)) / max(math.dist(a[:2], b[:2]), 1e-6))
        z_line = a[2] + (b[2] - a[2]) * f
        lift = round(max(need, z_line), 1)
        if abs(x - b[0]) < 0.5 and abs(y - b[1]) < 0.5:
            return False
        self.insert(entries, i, [(round(x, 1), round(y, 1), lift)])
        self.actions.append(f"lifted leg {i} to {lift:g} m over the high ground at ({x:.0f}, {y:.0f})")
        return True

    def _detour(self, entries, i, why: str) -> bool:
        a, b = _point(entries, i - 1, self.home), _point(entries, i, self.home)
        corners = shortest_path(a[:2], b[:2], self.blockers(min(a[2], b[2])), self.fence)
        if not corners:
            return False
        self.insert(entries, i, self.interp(a, b, corners))
        self.actions.append(f"routed leg {i} {why} through {len(corners)} corner(s)")
        return True

    def _around_or_over(self, entries, i, what: str) -> bool:
        """Around, or over the top, whichever adds less flying (climbs count in full)."""
        a, b = _point(entries, i - 1, self.home), _point(entries, i, self.home)
        options = []
        corners = shortest_path(a[:2], b[:2], self.blockers(min(a[2], b[2])), self.fence)
        if corners:
            pts = self.interp(a, b, corners)
            route = [a, *pts, b]
            options.append((sum(math.dist(p, q) for p, q in zip(route, route[1:])), "around", pts))
        top = self._top_along(a, b)
        if top is not None:
            z_over = round(top + OVER_MARGIN, 1)
            if z_over <= self.constraints.max_altitude:
                pts = []
                if a[2] < z_over:
                    pts.append((a[0], a[1], z_over))
                if b[2] < z_over:
                    pts.append((b[0], b[1], z_over))
                if pts:
                    route = [a, *pts, b]
                    options.append((sum(math.dist(p, q) for p, q in zip(route, route[1:])), "over", pts))
        if not options:
            return False
        length, how, pts = min(options, key=lambda o: o[0])
        self.insert(entries, i, pts)
        self.actions.append(f"took leg {i} {how} {what} (+{length - math.dist(a, b):.0f} m)")
        return True

    def _top_along(self, a, b) -> float | None:
        """The altitude a leg would have to hold to clear every roof, obstacle and slope along a-b."""
        line = LineString([a[:2], b[:2]]) if math.dist(a[:2], b[:2]) > 1e-6 else Point(a[:2])
        top = 0.0
        hit = False
        if self.world is not None:
            for building in self.world.obstructions(line.buffer(DETOUR_MARGIN)):
                top = max(top, building.roof + self.config.roof_clearance)
                hit = True
        sep = self.constraints.min_separation
        for o in self.constraints.obstacles:
            if line.distance(Point(o.x, o.y)) < o.radius + sep:
                if o.height is None:
                    return None
                top = max(top, o.height + 1.0)
                hit = True
        if self.world is not None and self.world.terrain is not None:
            steps = max(2, int(line.length / 5)) if line.length else 1
            for k in range(steps + 1):
                p = line.interpolate(k / steps, normalized=True) if line.length else line
                top = max(top, self.floor_at(p.x, p.y))
        return top if hit else None

    def fix_intent(self, entries: list[_Entry], v: Violation) -> bool:
        fix = v.fix or {}
        if fix.get("kind") == "set_landing":
            x, y, z = fix["at"]
            last = entries[-1].wp
            if last.action is Action.LAND:
                last.x, last.y, last.z = x, y, z
            else:
                entries.append(_Entry(entries[-1].subtask, Waypoint(x=x, y=y, z=z, speed=1.5, action=Action.LAND)))
            self.actions.append(f"ended the flight with a LAND on the pad ({x:g}, {y:g})")
            return True
        if fix.get("kind") != "add_waypoint":
            return False
        x, y, z = fix["at"]
        action = Action(fix.get("action", "CAPTURE"))
        # Cheapest insertion, between the climb out and the final approach.
        last_air = len(entries) - (1 if entries and entries[-1].wp.action is Action.LAND else 0)
        best = None
        for at in range(1, max(last_air, 1) + 1):
            p = _point(entries, at - 1, self.home)
            q = _point(entries, at, self.home) if at < len(entries) else p
            cost = math.dist(p, (x, y, z)) + math.dist((x, y, z), q) - math.dist(p, q)
            if best is None or cost < best[0]:
                best = (cost, at)
        if best is None:
            return False
        at = best[1]
        subtask = entries[at - 1].subtask if at - 1 < len(entries) else entries[-1].subtask
        speed = CAPTURE_SPEED if action is not Action.MOVE else ADDED_SPEED
        entries.insert(at, _Entry(subtask, Waypoint(x=x, y=y, z=z, speed=speed, action=action)))
        self.actions.append(
            f"added a {action.value} at ({x:g}, {y:g}, {z:g}) for '{fix.get('target', 'the mission')}' (+{best[0]:.0f} m)"
        )
        return True


def _sags(a, b, speed: float, config: VerifierConfig) -> bool:
    """Whether a climbing leg from a to b is steeper than the aircraft can climb at that speed."""
    dh = math.dist(a[:2], b[:2])
    if dh <= 1e-6:
        return False
    return (b[2] - a[2]) > config.max_climb_rate * dh / max(speed, 1e-6)


def repair(
    plan: Plan,
    constraints: MissionConstraints,
    *,
    world: World | None = None,
    config: VerifierConfig | None = None,
    intent: "MissionIntent | None" = None,
    result: VerificationResult | None = None,
    rounds: int = 6,
) -> Repair:
    """Fix the plan geometrically until it passes, or say it could not.

    `result` is the verdict already in hand for `plan`, to save one verify.
    """
    config = config or DEFAULT_CONFIG
    repairer = _Repairer(constraints, world, config, intent)
    entries = _entries(plan)
    if not entries:
        return Repair(plan=None)
    current = result or verify(plan, constraints, config, world, intent=intent)
    for round_ in range(1, rounds + 1):
        if current.passed:
            fixed = _plan(entries, plan.mission_text)
            return Repair(plan=fixed, actions=repairer.actions, result=current, rounds=round_ - 1)
        todo = [v for v in current.violations if v.blocking or v.type is ViolationType.TURN_FEASIBILITY]
        if not repairer.apply(entries, todo):
            break
        current = verify(_plan(entries, plan.mission_text), constraints, config, world, intent=intent)
    if current.passed:
        return Repair(plan=_plan(entries, plan.mission_text), actions=repairer.actions, result=current, rounds=rounds)
    return Repair(plan=None, actions=repairer.actions, result=current, rounds=rounds)


def route_length(plan: Plan, start: tuple[float, float, float]) -> float:
    points = [tuple(start), *((w.x, w.y, w.z) for w in plan.waypoints())]
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def trim_for_battery(
    plan: Plan,
    constraints: MissionConstraints,
    *,
    world: World | None = None,
    config: VerifierConfig | None = None,
    landing: tuple[float, float, float] | None = None,
) -> Repair:
    """Not enough battery in the air: give up mission phases, cheapest loss first, until the rest flies.

    Each round drops the one phase (a subtask that does not land) whose
    removal shortens the route most, then repairs the geometry of what is
    left. When nothing is left to drop, the last resort is straight to the
    landing pad. The result passes the full safety verifier or is None;
    what the mission lost is in `actions`, for the log.
    """
    config = config or DEFAULT_CONFIG
    start = tuple(float(v) for v in config.home)
    actions: list[str] = []
    candidate = plan
    for _ in range(len(plan.subtasks) + 1):
        fixed = repair(candidate, constraints, world=world, config=config)
        if fixed.repaired:
            return Repair(plan=fixed.plan, actions=actions + fixed.actions, result=fixed.result)
        droppable = [
            k for k, s in enumerate(candidate.subtasks)
            if not any(w.action is Action.LAND for w in s.waypoints) and len(candidate.subtasks) > 1
        ]
        if not droppable:
            break
        before = route_length(candidate, start)

        def without(k: int) -> Plan:
            return candidate.model_copy(update={"subtasks": [s for j, s in enumerate(candidate.subtasks) if j != k]})

        best = max(droppable, key=lambda k: before - route_length(without(k), start))
        actions.append(
            f"dropped '{candidate.subtasks[best].name}' to save {before - route_length(without(best), start):.0f} m"
        )
        candidate = without(best)
    if landing is not None:
        lx, ly, lz = landing
        cruise = max(start[2], constraints.min_altitude)
        direct = Plan(mission_text=plan.mission_text, subtasks=[SubTask(name="return_and_land", waypoints=[
            Waypoint(x=lx, y=ly, z=cruise, speed=min(8.0, constraints.max_velocity)),
            Waypoint(x=lx, y=ly, z=lz, speed=1.5, action=Action.LAND),
        ])])
        fixed = repair(direct, constraints, world=world, config=config)
        if fixed.repaired:
            return Repair(plan=fixed.plan, actions=[*actions, "abandoned what was left: straight to the landing pad", *fixed.actions], result=fixed.result)
    return Repair(plan=None, actions=actions)


def relaxed(result: VerificationResult) -> VerificationResult:
    """The same verdict with the mission's requirements made advisory: only safety blocks.

    Used in the air as a last resort, when no plan both serves everything left
    and stays safe: the mission gives way, safety never does.
    """
    violations = [
        v.model_copy(update={"severity": Severity.WARNING}) if v.type is ViolationType.INTENT and v.blocking else v
        for v in result.violations
    ]
    return VerificationResult(
        passed=not any(v.blocking and v.type in SAFETY_TYPES for v in violations), violations=violations,
    )
