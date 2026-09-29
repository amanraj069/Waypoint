"""The individual deterministic safety checks.

Every function here is pure geometry and arithmetic over a `Plan` and a
`MissionConstraints`: same inputs, same violations, every time. No model, no
sampling, no tolerance that depends on anything but the numbers given.

Waypoint indices are 0-based into the *flattened* plan (`Plan.waypoints()`),
which is the order the aircraft flies. The aircraft is assumed to start at
`config.home`, so the first leg is home -> waypoint 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

from waypoint.planner.models import Action, GeoPoint, MissionConstraints, Obstacle, Plan, Waypoint
from waypoint.verifier.models import Severity, Violation, ViolationType, VerifierConfig
from waypoint.world.landing import find as find_site, survey as survey_site
from waypoint.world.models import Building, World

# Slack for floating point comparisons only -- not a safety margin.
EPS = 1e-9


@dataclass(frozen=True)
class FlatWaypoint:
    """A waypoint together with where it sits in the flattened plan."""

    index: int
    subtask: str
    wp: Waypoint

    @property
    def xy(self) -> tuple[float, float]:
        return (self.wp.x, self.wp.y)

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.wp.x, self.wp.y, self.wp.z)

    @property
    def label(self) -> str:
        return (
            f'waypoint {self.index} (subtask "{self.subtask}") at '
            f"({self.wp.x:g}, {self.wp.y:g}, {self.wp.z:g})"
        )


@dataclass(frozen=True)
class Leg:
    """A straight segment the aircraft flies, ending at `end`."""

    start: tuple[float, float, float]
    start_label: str
    end: FlatWaypoint

    @property
    def index(self) -> int:
        return self.end.index

    @property
    def line_xy(self) -> LineString:
        start_xy = (self.start[0], self.start[1])
        if math.dist(start_xy, self.end.xy) <= EPS:
            # A zero-length LineString is invalid in shapely; a point behaves
            # identically for the distance and intersection tests we run.
            return Point(start_xy)
        return LineString([start_xy, self.end.xy])

    @property
    def length_3d(self) -> float:
        return math.dist(self.start, self.end.xyz)

    @property
    def length_2d(self) -> float:
        return math.dist((self.start[0], self.start[1]), self.end.xy)

    @property
    def label(self) -> str:
        return f"the leg from {self.start_label} to {self.end.label}"


def flatten(plan: Plan) -> list[FlatWaypoint]:
    """The plan as the flat, ordered waypoint sequence the aircraft flies."""
    flat: list[FlatWaypoint] = []
    for subtask in plan.subtasks:
        for wp in subtask.waypoints:
            flat.append(FlatWaypoint(index=len(flat), subtask=subtask.name, wp=wp))
    return flat


def legs_of(flat: list[FlatWaypoint], config: VerifierConfig) -> list[Leg]:
    """Every straight segment flown, starting with home -> waypoint 0."""
    legs: list[Leg] = []
    previous = tuple(float(v) for v in config.home)
    previous_label = f"{config.start_label} ({previous[0]:g}, {previous[1]:g}, {previous[2]:g})"
    for fw in flat:
        legs.append(Leg(start=previous, start_label=previous_label, end=fw))
        previous = fw.xyz
        previous_label = fw.label
    return legs


def polygon_of(points: list[GeoPoint], name: str) -> Polygon:
    """Build a shapely polygon, refusing anything that is not a real polygon."""
    poly = Polygon([(p.x, p.y) for p in points])
    if not poly.is_valid or poly.is_empty or poly.area <= 0:
        raise ValueError(
            f"{name} is not a valid polygon (self-intersecting, collinear or empty): "
            + ", ".join(f"({p.x:g}, {p.y:g})" for p in points)
        )
    return poly


# --------------------------------------------------------------------------
# Checks. Each returns zero or more violations and never raises on plan data.
# --------------------------------------------------------------------------


def check_geofence(
    flat: list[FlatWaypoint],
    legs: list[Leg],
    constraints: MissionConstraints,
    config: VerifierConfig,
) -> list[Violation]:
    """Every waypoint -- and every leg between them -- must stay inside the fence."""
    fence = polygon_of(constraints.geofence, "geofence")
    violations: list[Violation] = []
    inside: dict[int, bool] = {}

    for fw in flat:
        point = Point(fw.xy)
        inside[fw.index] = bool(fence.covers(point))
        if not inside[fw.index]:
            violations.append(
                Violation(
                    type=ViolationType.GEOFENCE,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} is outside the geofence, {point.distance(fence):.1f} m "
                        "beyond its boundary. Move this waypoint back inside the geofence "
                        "polygon, leaving a few metres of margin."
                    ),
                    severity=Severity.ERROR,
                )
            )
        elif config.geofence_margin > 0:
            clearance = point.distance(fence.exterior)
            if clearance < config.geofence_margin:
                violations.append(
                    Violation(
                        type=ViolationType.GEOFENCE,
                        waypoint_index=fw.index,
                        description=(
                            f"{fw.label} is only {clearance:.1f} m inside the geofence, less "
                            f"than the {config.geofence_margin:g} m margin. Pull it further "
                            "from the boundary."
                        ),
                        severity=Severity.WARNING,
                    )
                )

    home_inside = bool(fence.covers(Point(config.home[0], config.home[1])))
    for leg in legs:
        start_inside = home_inside if leg.index == 0 else inside.get(leg.index - 1, False)
        # A leg whose endpoint is already flagged needs no second report.
        if not (start_inside and inside.get(leg.index, False)):
            continue
        if not fence.covers(leg.line_xy):
            violations.append(
                Violation(
                    type=ViolationType.GEOFENCE,
                    waypoint_index=leg.index,
                    description=(
                        f"{leg.label} leaves the geofence even though both ends are inside "
                        "it -- the fence is concave along that line. Insert an intermediate "
                        "waypoint so the flown path stays inside the fence."
                    ),
                    severity=Severity.ERROR,
                )
            )
    return violations


def check_no_fly_zones(
    flat: list[FlatWaypoint], legs: list[Leg], constraints: MissionConstraints
) -> list[Violation]:
    """No waypoint may sit in a no-fly zone and no leg may cross one."""
    violations: list[Violation] = []
    for number, vertices in enumerate(constraints.no_fly_zones, 1):
        zone = polygon_of(vertices, f"no-fly zone {number}")
        occupied = {fw.index for fw in flat if zone.covers(Point(fw.xy))}
        for fw in flat:
            if fw.index in occupied:
                violations.append(
                    Violation(
                        type=ViolationType.NO_FLY_ZONE,
                        waypoint_index=fw.index,
                        description=(
                            f"{fw.label} is inside no-fly zone {number} "
                            f"({_render(vertices)}). Move this waypoint out of the zone."
                        ),
                        severity=Severity.CRITICAL,
                    )
                )
        for leg in legs:
            start_occupied = leg.index - 1 in occupied
            if leg.index in occupied or start_occupied:
                continue  # already reported as a waypoint inside the zone
            if leg.line_xy.intersects(zone):
                violations.append(
                    Violation(
                        type=ViolationType.NO_FLY_ZONE,
                        waypoint_index=leg.index,
                        description=(
                            f"{leg.label} crosses no-fly zone {number} "
                            f"({_render(vertices)}). Route around the zone by inserting "
                            "intermediate waypoints instead of flying straight through it."
                        ),
                        severity=Severity.CRITICAL,
                    )
                )
    return violations


def check_altitude(flat: list[FlatWaypoint], constraints: MissionConstraints) -> list[Violation]:
    """Every airborne waypoint must sit inside the altitude envelope."""
    violations: list[Violation] = []
    for fw in flat:
        z = fw.wp.z
        if z < 0:
            violations.append(
                Violation(
                    type=ViolationType.ALTITUDE,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} has a negative altitude (z = {z:g} m), which is below "
                        "the ground. Set z to a positive altitude inside the "
                        f"{constraints.min_altitude:g}-{constraints.max_altitude:g} m envelope."
                    ),
                    severity=Severity.ERROR,
                )
            )
            continue
        if _on_the_ground(fw):
            continue  # takeoff and landing waypoints are allowed at z = 0
        if z < constraints.min_altitude - EPS:
            violations.append(
                Violation(
                    type=ViolationType.ALTITUDE,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} is {constraints.min_altitude - z:.1f} m below the "
                        f"{constraints.min_altitude:g} m minimum altitude. Raise it to at "
                        f"least {constraints.min_altitude:g} m, or make it the LAND waypoint "
                        "if it is meant to touch down."
                    ),
                    severity=Severity.ERROR,
                )
            )
        elif z > constraints.max_altitude + EPS:
            violations.append(
                Violation(
                    type=ViolationType.ALTITUDE,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} is {z - constraints.max_altitude:.1f} m above the "
                        f"{constraints.max_altitude:g} m ceiling. Lower it to at most "
                        f"{constraints.max_altitude:g} m."
                    ),
                    severity=Severity.ERROR,
                )
            )
    return violations


def check_velocity(legs: list[Leg], constraints: MissionConstraints) -> list[Violation]:
    """Commanded speed, and the speed a steep leg implies, must stay under the limit."""
    limit = constraints.max_velocity
    violations: list[Violation] = []
    for leg in legs:
        wp = leg.end.wp
        if wp.speed > limit + EPS:
            violations.append(
                Violation(
                    type=ViolationType.MAX_VELOCITY,
                    waypoint_index=leg.index,
                    description=(
                        f"{leg.end.label} commands {wp.speed:g} m/s, which is "
                        f"{wp.speed - limit:.1f} m/s above the {limit:g} m/s limit. Reduce "
                        f"its speed to at most {limit:g} m/s."
                    ),
                    severity=Severity.ERROR,
                )
            )
            continue
        # A steep climb flown at the commanded *ground* speed means a higher
        # speed along the actual path.
        ground = leg.length_2d
        if ground <= EPS:
            continue
        implied = wp.speed * leg.length_3d / ground
        if implied > limit + EPS:
            climb = abs(leg.end.wp.z - leg.start[2])
            safe = wp.speed * limit / implied
            violations.append(
                Violation(
                    type=ViolationType.MAX_VELOCITY,
                    waypoint_index=leg.index,
                    description=(
                        f"{leg.label} climbs {climb:.1f} m over only {ground:.1f} m of "
                        f"ground, so the commanded {wp.speed:g} m/s ground speed implies "
                        f"{implied:.1f} m/s along the flown path, above the {limit:g} m/s "
                        f"limit. Slow this waypoint to at most {safe:.1f} m/s or spread the "
                        "climb over a longer leg."
                    ),
                    severity=Severity.ERROR,
                )
            )
    return violations


def check_turns(flat: list[FlatWaypoint], config: VerifierConfig) -> list[Violation]:
    """Flag corners too sharp to fly at the speed commanded through them."""
    violations: list[Violation] = []
    points = [tuple(float(v) for v in config.home)] + [fw.xyz for fw in flat]

    for position, fw in enumerate(flat):
        if position + 1 >= len(flat):
            break  # nothing to turn into
        nxt = flat[position + 1]
        if fw.wp.action in (Action.HOVER, Action.LAND):
            continue  # the aircraft stops there, so it can turn on the spot
        incoming = _heading_vector(points[position], fw.xyz)
        outgoing = _heading_vector(fw.xyz, nxt.xyz)
        if incoming is None or outgoing is None:
            continue  # a pure climb or descent has no heading to change
        angle = _angle_between(incoming, outgoing)
        speed = max(fw.wp.speed, nxt.wp.speed)
        allowed = config.max_turn_at(speed)
        if angle <= allowed + EPS:
            continue
        safe_speed = config.speed_for_turn(angle)
        remedy = (
            f"Slow to at most {safe_speed:.1f} m/s through that waypoint, or insert an "
            "intermediate waypoint to widen the turn."
            if safe_speed is not None
            else (
                f"No speed makes a turn that sharp flyable (the airframe limit is "
                f"{config.max_turn_angle_deg:g} deg), so insert intermediate waypoints to "
                "break it into gentler turns."
            )
        )
        violations.append(
            Violation(
                type=ViolationType.TURN_FEASIBILITY,
                waypoint_index=fw.index,
                description=(
                    f"The route turns {angle:.0f} deg at {fw.label} while flying "
                    f"{speed:g} m/s; at that speed the aircraft can only turn "
                    f"{allowed:.0f} deg between legs. {remedy}"
                ),
                severity=Severity.WARNING,
            )
        )
    return violations


def check_battery(
    legs: list[Leg],
    flat: list[FlatWaypoint],
    constraints: MissionConstraints,
    config: VerifierConfig,
) -> list[Violation]:
    """Estimate energy with a linear model and reject a route that cannot be flown."""
    distance = sum(leg.length_3d for leg in legs)
    hover_seconds = sum(
        config.hover_seconds_per_waypoint for fw in flat if fw.wp.action is Action.HOVER
    )
    needed = (
        distance / config.meters_per_battery_pct
        + hover_seconds / config.hover_seconds_per_battery_pct
    )
    required = needed + config.reserve_battery_pct
    if required <= constraints.battery_budget_pct + EPS:
        return []

    over = required - constraints.battery_budget_pct
    reserve = (
        f" plus a {config.reserve_battery_pct:g}% reserve" if config.reserve_battery_pct else ""
    )
    hover = f" and hovers {hover_seconds:.0f} s" if hover_seconds else ""
    return [
        Violation(
            type=ViolationType.BATTERY,
            waypoint_index=None,
            description=(
                f"The route flies {distance:.0f} m{hover}, which needs about {needed:.1f}% "
                f"battery{reserve} on a model of {config.meters_per_battery_pct:g} m per 1% "
                f"and {config.hover_seconds_per_battery_pct:g} s of hovering per 1%. That "
                f"exceeds the {constraints.battery_budget_pct:g}% budget by {over:.1f}%. "
                f"Shorten the route by at least {over * config.meters_per_battery_pct:.0f} m "
                "-- merge or drop waypoints, or fly a tighter pattern."
            ),
            severity=Severity.ERROR,
        )
    ]


def check_separation(
    flat: list[FlatWaypoint], legs: list[Leg], constraints: MissionConstraints
) -> list[Violation]:
    """Keep every waypoint and every leg outside each obstacle's keep-out radius."""
    violations: list[Violation] = []
    for obstacle in constraints.obstacles:
        clearance = obstacle.radius + constraints.min_separation
        if clearance <= 0:
            continue
        centre = Point(obstacle.x, obstacle.y)
        breached: set[int] = set()

        for fw in flat:
            if _clears_vertically(obstacle, fw.wp.z):
                continue
            gap = centre.distance(Point(fw.xy))
            if gap >= clearance - EPS:
                continue
            breached.add(fw.index)
            violations.append(
                Violation(
                    type=ViolationType.MIN_SEPARATION,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} is {gap:.1f} m from the centre of '{obstacle.name}' at "
                        f"({obstacle.x:g}, {obstacle.y:g}), inside its {clearance:.1f} m "
                        f"keep-out radius ({obstacle.radius:g} m obstacle + "
                        f"{constraints.min_separation:g} m separation). Move it at least "
                        f"{clearance - gap:.1f} m further out{_over_the_top(obstacle)}."
                    ),
                    severity=Severity.CRITICAL,
                )
            )

        for leg in legs:
            if leg.index in breached or leg.index - 1 in breached:
                continue  # already reported at the waypoint
            if _clears_vertically(obstacle, leg.start[2]) and _clears_vertically(
                obstacle, leg.end.wp.z
            ):
                continue
            gap = centre.distance(leg.line_xy)
            if gap >= clearance - EPS:
                continue
            violations.append(
                Violation(
                    type=ViolationType.MIN_SEPARATION,
                    waypoint_index=leg.index,
                    description=(
                        f"{leg.label} passes within {gap:.1f} m of '{obstacle.name}' at "
                        f"({obstacle.x:g}, {obstacle.y:g}), inside its {clearance:.1f} m "
                        f"keep-out radius. Insert a waypoint that routes the path around "
                        f"it with at least {clearance:.1f} m of clearance"
                        f"{_over_the_top(obstacle)}."
                    ),
                    severity=Severity.CRITICAL,
                )
            )
    return violations


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _render(vertices: list[GeoPoint]) -> str:
    return ", ".join(f"({p.x:g}, {p.y:g})" for p in vertices)


def _on_the_ground(fw: FlatWaypoint) -> bool:
    """True for the launch waypoint and for a landing, which may sit at z = 0."""
    return abs(fw.wp.z) <= EPS and (fw.index == 0 or fw.wp.action is Action.LAND)


def _clears_vertically(obstacle: Obstacle, z: float) -> bool:
    return obstacle.height is not None and z > obstacle.height + EPS


def _over_the_top(obstacle: Obstacle) -> str:
    if obstacle.height is None:
        return ""
    return f", or fly above its {obstacle.height:g} m top"


def _heading_vector(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> tuple[float, float] | None:
    """The horizontal direction of travel, or None if the leg is purely vertical."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) <= EPS:
        return None
    return (dx, dy)


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Heading change in degrees: 0 means straight on, 180 means a reversal."""
    dot = a[0] * b[0] + a[1] * b[1]
    cross = a[0] * b[1] - a[1] * b[0]
    return abs(math.degrees(math.atan2(cross, dot)))


# --------------------------------------------------------------------------
# The world: buildings standing in the way, and ground that is not flat.
# These only run when a World is supplied; with none, the sky is empty.
# --------------------------------------------------------------------------


def check_buildings(
    flat: list[FlatWaypoint], legs: list[Leg], world: World | None
) -> list[Violation]:
    """No waypoint may sit inside a building, and no leg may fly through one."""
    if not world or not world.buildings:
        return []

    violations: list[Violation] = []
    occupied: set[int] = set()

    for fw in flat:
        building = world.collides(*fw.xyz)
        if building is None:
            continue
        occupied.add(fw.index)
        violations.append(
            Violation(
                type=ViolationType.BUILDING,
                waypoint_index=fw.index,
                description=(
                    f"{fw.label} is inside {building.label()}, whose roof is at "
                    f"{building.roof:.0f} m. Move the waypoint off the footprint, or raise it "
                    f"above {building.roof:.0f} m to cross over the top."
                ),
                severity=Severity.CRITICAL,
            )
        )

    for leg in legs:
        if leg.index in occupied or leg.index - 1 in occupied:
            continue  # already reported at the waypoint
        for building, lowest in _legs_through(leg, world):
            violations.append(
                Violation(
                    type=ViolationType.BUILDING,
                    waypoint_index=leg.index,
                    description=(
                        f"{leg.label} passes through {building.label()}: over the footprint the "
                        f"route is as low as {lowest:.0f} m, below the {building.roof:.0f} m roof. "
                        "Route around the building, or climb above its roof before crossing it."
                    ),
                    severity=Severity.CRITICAL,
                )
            )
    return violations


def _legs_through(leg: Leg, world: World) -> list[tuple[Building, float]]:
    """Buildings this leg flies into, with the route's lowest altitude over each.

    The leg climbs or descends linearly, so its lowest point over a footprint is
    at one end of the crossing -- no sampling needed to be exact.
    """
    line = leg.line_xy
    if line.geom_type != "LineString":
        return []

    hits = []
    for building in world.obstructions(line):
        crossing = line.intersection(building.polygon)
        if crossing.is_empty:
            continue
        fractions = [
            line.project(Point(coord)) / line.length
            for geom in getattr(crossing, "geoms", [crossing])
            for coord in getattr(geom, "coords", [])
        ]
        if not fractions:
            continue
        start_z, end_z = leg.start[2], leg.end.wp.z
        altitudes = [start_z + (end_z - start_z) * f for f in (min(fractions), max(fractions))]
        lowest = min(altitudes)
        if lowest <= building.roof:
            hits.append((building, lowest))
    return hits


def check_terrain(
    flat: list[FlatWaypoint],
    legs: list[Leg],
    world: World | None,
    config: VerifierConfig,
) -> list[Violation]:
    """Stay above the ground, by a margin, all the way along the route."""
    if not world or not world.terrain:
        return []

    violations: list[Violation] = []
    grounded: set[int] = set()

    for fw in flat:
        ground = world.elevation_at(fw.wp.x, fw.wp.y)
        clearance = fw.wp.z - ground
        if _on_the_ground(fw) or abs(clearance) <= 0.5 and (fw.index == 0 or fw.wp.action is Action.LAND):
            continue
        if clearance < 0:
            grounded.add(fw.index)
            violations.append(
                Violation(
                    type=ViolationType.TERRAIN,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} is {abs(clearance):.1f} m underground -- the ground there is "
                        f"at {ground:.1f} m. Remember z is height above the launch pad, not above "
                        f"the ground below, so raise it to at least "
                        f"{ground + config.min_ground_clearance:.1f} m."
                    ),
                    severity=Severity.CRITICAL,
                )
            )
        elif clearance < config.min_ground_clearance:
            grounded.add(fw.index)
            violations.append(
                Violation(
                    type=ViolationType.TERRAIN,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} clears the ground by only {clearance:.1f} m (the ground "
                        f"there is at {ground:.1f} m), under the "
                        f"{config.min_ground_clearance:g} m minimum. Raise it to at least "
                        f"{ground + config.min_ground_clearance:.1f} m."
                    ),
                    severity=Severity.ERROR,
                )
            )

    for leg in legs:
        if leg.index in grounded or leg.index - 1 in grounded:
            continue
        worst = _lowest_clearance(leg, world, config.terrain_sample_step)
        if worst is None or worst[0] >= config.min_ground_clearance:
            continue
        clearance, x, y = worst
        problem = "flies into the hillside" if clearance < 0 else "skims the ground"
        violations.append(
            Violation(
                type=ViolationType.TERRAIN,
                waypoint_index=leg.index,
                description=(
                    f"{leg.label} {problem}: at ({x:.0f}, {y:.0f}) the ground rises to "
                    f"{world.elevation_at(x, y):.1f} m and the route clears it by only "
                    f"{clearance:.1f} m. Raise the leg, or route around the high ground."
                ),
                severity=Severity.CRITICAL if clearance < 0 else Severity.ERROR,
            )
        )
    return violations


def _lowest_clearance(
    leg: Leg, world: World, step: float
) -> tuple[float, float, float] | None:
    """Walk the leg and return the worst (clearance, x, y) over the ground."""
    length = leg.length_3d
    if length <= EPS:
        return None
    # Ignore the very ends: a landing leg is meant to touch the ground.
    steps = max(2, int(length / step))
    worst: tuple[float, float, float] | None = None
    start, end = leg.start, leg.end.xyz
    for i in range(1, steps):
        f = i / steps
        x = start[0] + (end[0] - start[0]) * f
        y = start[1] + (end[1] - start[1]) * f
        z = start[2] + (end[2] - start[2]) * f
        clearance = z - world.elevation_at(x, y)
        if worst is None or clearance < worst[0]:
            worst = (clearance, x, y)
    return worst


def check_landing(
    flat: list[FlatWaypoint], world: World | None, config: VerifierConfig
) -> list[Violation]:
    """A LAND waypoint has to be somewhere you could actually put it down.

    Two separate things go wrong here. The waypoint can be at the wrong height
    -- hovering above the ground or buried under it, because z is measured from
    the datum and not from the ground below. And the ground itself can be
    unusable: a patch that slopes away under the skids tips a multirotor over,
    however accurately it was flown.
    """
    if not world:
        return []

    violations: list[Violation] = []
    for fw in flat:
        if fw.wp.action is not Action.LAND:
            continue

        site = survey_site(world, fw.wp.x, fw.wp.y)
        above = fw.wp.z - site.elevation
        if abs(above) > config.landing_tolerance:
            where = "above" if above > 0 else "below"
            violations.append(
                Violation(
                    type=ViolationType.LANDING,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} is a LAND waypoint {abs(above):.1f} m {where} the ground, "
                        f"which is at {site.elevation:.1f} m there. Put it at "
                        f"z = {site.elevation:.1f} so it touches down."
                    ),
                    severity=Severity.ERROR,
                )
            )

        if not site.usable:
            nearby = find_site(world, fw.wp.x, fw.wp.y)
            remedy = (
                f"The nearest ground that would do is ({nearby.x:.0f}, {nearby.y:.0f}), "
                f"{nearby.moved:.0f} m away."
                if nearby.usable
                else "There is nowhere within 120 m that would do; land somewhere else entirely."
            )
            violations.append(
                Violation(
                    type=ViolationType.LANDING,
                    waypoint_index=fw.index,
                    description=(
                        f"{fw.label} cannot be landed on: {site.why_not()}. {remedy}"
                    ),
                    severity=Severity.CRITICAL,
                )
            )
    return violations
