"""The inputs to one flight, from what a person types.

A scenario is a world, a plan, a wind and a fault schedule. Every entry point
(the CLI scripts, the prompt, the API) turns short specs like `lodhi`,
`sample` or `4,0,0` into those objects the same way, so they all live here
rather than in any one script.
"""

from __future__ import annotations

import json
from pathlib import Path

from waypoint.paths import FAULTS_DIR, PLANS_DIR, WORLDS_DIR
from waypoint.planner import DEFAULT_CONSTRAINTS, GeoPoint, MissionConstraints, Obstacle, Plan
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, Trigger, Wind
from waypoint.world.landing import Site, find as find_site
from waypoint.world.models import World

#: Headroom above the highest thing in a world when the ceiling has to be raised.
CEILING_MARGIN = 40.0


def sample_schedule(plan: Plan) -> FaultSchedule:
    """A demo schedule aimed at whatever plan we were given.

    One fault of each kind. Two fire on flight conditions and two on the clock
    or the battery, so a run exercises both kinds of trigger.
    """
    waypoints = plan.waypoints()
    midpoint = waypoints[len(waypoints) // 2] if waypoints else None
    turn = max(1, len(waypoints) // 2)

    events = [
        FaultEvent(
            type=FaultType.WIND_GUST,
            trigger=Trigger(time=12.0),
            magnitude=9.0,
            direction=90.0,
            duration=6.0,
            name="squall",
        ),
        FaultEvent(
            type=FaultType.BATTERY_DRAIN_SPIKE,
            trigger=Trigger(battery_below=97.0),
            magnitude=0.8,
            duration=15.0,
            name="cell fault",
        ),
        FaultEvent(
            type=FaultType.GPS_DRIFT,
            trigger=Trigger(at_waypoint=turn),
            magnitude=0.6,
            direction=200.0,
            duration=18.0,
            name="gps slip",
        ),
    ]
    if midpoint is not None:
        events.append(
            FaultEvent(
                type=FaultType.OBSTACLE_APPEARS,
                trigger=Trigger(at_waypoint=max(1, turn - 1)),
                obstacle=Obstacle(
                    x=midpoint.x * 0.6,
                    y=midpoint.y * 0.6,
                    radius=10.0,
                    height=40.0,
                    name="mobile crane",
                ),
                name="crane spotted",
            )
        )
    return FaultSchedule(events=events)


def load_faults(spec: str, plan: Plan) -> FaultSchedule:
    """`none`, `sample`, a schedule in data/faults by name, or a JSON path."""
    if spec == "none":
        return FaultSchedule()
    if spec == "sample":
        return sample_schedule(plan)
    return FaultSchedule.model_validate_json(_find(spec, FAULTS_DIR).read_text())


def parse_wind(spec: str | None, turbulence: float = 0.0) -> Wind:
    """`x,y,z` in m/s (missing components are 0), or calm."""
    if not spec:
        return Wind(turbulence=turbulence)
    parts = [float(value) for value in spec.split(",")]
    while len(parts) < 3:
        parts.append(0.0)
    return Wind(vector=(parts[0], parts[1], parts[2]), turbulence=turbulence)


def load_world(spec: str | None) -> World | None:
    """A world by name, or a path to one.

    A name is tried as a saved world in data/worlds first (`lodhi`,
    `gateway_of_india`, anything `scripts/make_world.py` wrote), then as a
    built-in generator (`delhi`, `lodhi`, `chowk`, `munnar`, and the older
    `city`/`park`/`canyon`/`hills` aliases). `none` means open space.
    """
    if not spec or spec == "none":
        return None
    from waypoint.world.generate import BUILDERS, build  # noqa: PLC0415 -- keeps startup cheap

    saved = WORLDS_DIR / f"{spec}.json"
    if saved.exists():
        return World.model_validate_json(saved.read_text())
    if spec in BUILDERS:
        return build(spec)
    return World.model_validate_json(Path(spec).read_text())


def load_plan(spec: str | Path) -> Plan:
    """A plan in data/plans by name (`lodhi_garden_tour`), or a JSON path."""
    path = _find(str(spec), PLANS_DIR)
    payload = json.loads(path.read_text())
    if "subtasks" not in payload:
        raise ValueError(f"{path} does not hold a Plan (no 'subtasks' key)")
    return Plan.model_validate(payload)


def fence_around(world: World, margin: float = 20.0) -> list[GeoPoint]:
    """A geofence that contains the whole world, inset by a margin."""
    minx, miny, maxx, maxy = world.bounds
    return [
        GeoPoint(x=minx + margin, y=miny + margin),
        GeoPoint(x=maxx - margin, y=miny + margin),
        GeoPoint(x=maxx - margin, y=maxy - margin),
        GeoPoint(x=minx + margin, y=maxy - margin),
    ]


def constraints_for(world: World | None, max_altitude: float | None) -> tuple[MissionConstraints, list[str]]:
    """The envelope to fly this world in, and what had to be changed to get it.

    Two things the default envelope cannot know: a real place is bigger than the
    practice area, and a place with a 225 m hill in it cannot be flown under a
    120 m ceiling. Both are adjusted here rather than failing every mission
    before it starts, and both are reported so the change is visible.
    """
    constraints = DEFAULT_CONSTRAINTS
    notes: list[str] = []
    if max_altitude:
        constraints = constraints.model_copy(update={"max_altitude": max_altitude})
    if world is None:
        return constraints, notes

    constraints = constraints.model_copy(update={"geofence": fence_around(world)})
    minx, miny, maxx, maxy = world.bounds
    notes.append(f"geofence widened to the world: x {minx:.0f}..{maxx:.0f}, y {miny:.0f}..{maxy:.0f} m")

    highest = max(
        [world.tallest.roof if world.tallest else 0.0]
        + [world.terrain.relief[1] if world.terrain else 0.0]
    )
    # Round first, then compare: raising a 120 m ceiling to 120 m is not a
    # change, and reporting it as one makes the page look like it is guessing.
    needed = round(highest + CEILING_MARGIN, 0)
    if needed > constraints.max_altitude:
        constraints = constraints.model_copy(update={"max_altitude": needed})
        notes.append(
            f"ceiling raised to {constraints.max_altitude:.0f} m: the highest thing here "
            f"is at {highest:.0f} m, and nothing could be flown over it otherwise"
        )
    return constraints, notes


def resolve_point(world: World | None, spec: str) -> tuple[float, float, str]:
    """`x,y` in metres, or the name (or part of it) of a landmark in the world.

    Returns the point and what to call it.
    """
    spec = spec.strip()
    try:
        x, y = (float(v) for v in spec.split(","))
        return x, y, f"({x:g}, {y:g})"
    except ValueError:
        pass
    if world is None:
        raise ValueError(f"no world to look up {spec!r} in; give the point as x,y")
    wanted = spec.lower()
    matches = [lm for lm in world.landmarks if wanted in lm.name.lower()]
    exact = [lm for lm in matches if lm.name.lower() == wanted]
    if exact:
        matches = exact
    if not matches:
        raise ValueError(f"no landmark called {spec!r}; /landmarks lists them, or give x,y")
    if len(matches) > 1:
        names = ", ".join(repr(lm.name) for lm in matches[:5])
        raise ValueError(f"{spec!r} could be {names}; be more specific")
    return matches[0].x, matches[0].y, matches[0].name


def pad_at(world: World | None, x: float, y: float, role: str) -> Site:
    """A surveyed pad at (x, y), moved to the nearest flat, clear ground if it has to be.

    Raises ValueError, saying why, when nothing within 120 m will do.
    """
    site = find_site(world, float(x), float(y))
    if not site.usable:
        raise ValueError(
            f"the aircraft cannot {role} at ({x:.0f}, {y:.0f}): {site.why_not()}, "
            "and there is nowhere within 120 m that would do"
        )
    return site


def _find(spec: str, directory: Path) -> Path:
    """A bare name means a file in `directory`; anything else is a path."""
    named = directory / f"{spec}.json"
    return named if named.exists() else Path(spec)
