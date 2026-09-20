#!/usr/bin/env python3
"""Fly a verified plan through the simulator and watch it in Gazebo.

    # fly a saved plan with the sample fault schedule, and watch it
    python scripts/simulate.py --plan /tmp/plan.json --gazebo

    # plan it live with Gemini first (spends tokens)
    python scripts/simulate.py "survey the north field" --gazebo

    # a calm, fault-free reference run, written but not watched
    python scripts/simulate.py --plan /tmp/plan.json --faults none

The plan is verified before it flies -- the simulator is not a way around the
safety gate -- and the finished run is written to runs/<mission>_<stamp>.json,
which `scripts/gazebo_view.py` replays.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Both the project root (for the packages) and scripts/ (for the sibling
# modules), so this works whether it is run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner import (  # noqa: E402
    DEFAULT_CONSTRAINTS,
    GeoPoint,
    MissionConstraints,
    Obstacle,
    Plan,
    PlannerError,
    plan as make_plan,
)
from orchestrator.loop import MissionReport, format_state  # noqa: E402
from orchestrator.runlog import write_run  # noqa: E402
from sim import (  # noqa: E402
    Executor,
    FaultEvent,
    FaultSchedule,
    FaultType,
    SimConfig,
    Trigger,
    Wind,
)
from verifier import verify  # noqa: E402
from world.models import World  # noqa: E402

RULE = "=" * 78


def sample_schedule(plan: Plan) -> FaultSchedule:
    """A demo schedule aimed at whatever plan we were given.

    Two faults fire on the clock and two on flight conditions, so a run
    exercises both kinds of trigger.
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


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def show_plan(plan: Plan, from_file: Path | None) -> None:
    """Print the hierarchy the planner produced: phases, then their waypoints."""
    if from_file is not None:
        print(f"{RULE}\nPLAN\n{RULE}")
        print(f"  loaded from {from_file}")
        if plan.mission_text:
            print(f'  mission: "{plan.mission_text}"')
        print()

    total = 0
    for number, subtask in enumerate(plan.subtasks, 1):
        print(f'  {number}. {subtask.name}  ({len(subtask.waypoints)} waypoint(s))')
        for wp in subtask.waypoints:
            print(
                f"       [{total}] ({wp.x:g}, {wp.y:g}, {wp.z:g}) at {wp.speed:g} m/s  "
                f"{wp.action.value}"
            )
            total += 1
    print(f"\n  {len(plan.subtasks)} subtask(s), {total} waypoint(s)")


def load_faults(spec: str, plan: Plan) -> FaultSchedule:
    if spec == "none":
        return FaultSchedule()
    if spec == "sample":
        return sample_schedule(plan)
    return FaultSchedule.model_validate_json(Path(spec).read_text())


def parse_wind(spec: str | None, turbulence: float) -> Wind:
    if not spec:
        return Wind(turbulence=turbulence)
    parts = [float(value) for value in spec.split(",")]
    while len(parts) < 3:
        parts.append(0.0)
    return Wind(vector=(parts[0], parts[1], parts[2]), turbulence=turbulence)


def python_command() -> str:
    """How to invoke this interpreter from the project directory.

    Printing a bare "python" is a lie on a machine where the virtualenv is not
    activated, which is most of them.
    """
    exe = Path(sys.executable)
    try:
        return str(exe.relative_to(Path.cwd()))
    except ValueError:
        return str(exe)

def load_world(spec: str | None) -> World | None:
    """A built-in world by name, or a path to one from scripts/make_world.py.

    The names are the keys of `world.generate.BUILDERS` -- `delhi`, `lodhi`,
    `chowk`, `munnar`, and the older `city`/`park`/`canyon`/`hills` aliases.
    """
    if not spec or spec == "none":
        return None
    from world.generate import BUILDERS, build  # noqa: PLC0415

    if spec in BUILDERS:
        return build(spec)
    return World.model_validate_json(Path(spec).read_text())


def fence_around(world: World, margin: float = 20.0) -> list[GeoPoint]:
    """A geofence that contains the whole world, inset by a margin."""
    minx, miny, maxx, maxy = world.bounds
    return [
        GeoPoint(x=minx + margin, y=miny + margin),
        GeoPoint(x=maxx - margin, y=miny + margin),
        GeoPoint(x=maxx - margin, y=maxy - margin),
        GeoPoint(x=minx + margin, y=maxy - margin),
    ]


def load_plan(path: Path) -> Plan:
    payload = json.loads(path.read_text())
    if "subtasks" not in payload:
        raise ValueError(f"{path} does not hold a Plan (no 'subtasks' key)")
    return Plan.model_validate(payload)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mission", nargs="?", help="Mission text to plan with the live API.")
    parser.add_argument("--plan", type=Path, help="Fly this saved plan JSON instead.")
    parser.add_argument("--save", type=Path, help="Write the generated plan here to re-fly later.")
    parser.add_argument("--faults", default="sample", help="'sample', 'none', or a schedule JSON.")
    parser.add_argument(
        "--world",
        help="Environment: 'delhi', 'lodhi', 'chowk', 'munnar', or a worlds/*.json file.",
    )
    parser.add_argument(
        "--keep-geofence",
        action="store_true",
        help="Keep the default geofence instead of widening it to the world.",
    )
    parser.add_argument(
        "--max-altitude", type=float, help="Raise the altitude ceiling, e.g. to clear a skyline."
    )
    parser.add_argument("--wind", help="Steady wind as 'x,y,z' in m/s.")
    parser.add_argument("--turbulence", type=float, default=0.0, help="Gustiness std-dev, m/s.")
    parser.add_argument("--dt", type=float, default=0.1, help="Tick length in seconds.")
    parser.add_argument("--blind", action="store_true", help="No wind compensation: fly blind.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--quiet", action="store_true", help="Suppress per-second telemetry.")
    parser.add_argument("--force", action="store_true", help="Fly even if verification fails.")
    parser.add_argument("--gazebo", action="store_true", help="Watch this run fly in Gazebo.")
    parser.add_argument("--speed", type=float, default=1.0, help="Gazebo playback speed.")
    args = parser.parse_args(argv)

    if not args.mission and not args.plan:
        parser.error("give a mission to plan, or --plan with a saved plan file")

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    constraints: MissionConstraints = DEFAULT_CONSTRAINTS
    world = load_world(args.world)
    if args.max_altitude:
        constraints = constraints.model_copy(update={"max_altitude": args.max_altitude})
    if world and not args.keep_geofence:
        # A city is bigger than the default practice area; fence the world
        # instead, or every mission fails before it starts.
        constraints = constraints.model_copy(update={"geofence": fence_around(world)})
    if world:
        summary = world.summary()
        print(f"{RULE}\nWORLD\n{RULE}")
        print(f"  {world.name} ({world.source}): {summary['buildings']} buildings, tallest roof "
              f"{summary['tallest_roof']:g} m, ground {summary['elevation_min']:g}.."
              f"{summary['elevation_max']:g} m")
        fence = "kept as given" if args.keep_geofence else "widened to the world bounds"
        print(f"  geofence: {fence}\n")

    if args.plan:
        plan = load_plan(args.plan)
    else:
        print(f"{RULE}\nPLAN\n{RULE}")
        print(f'  mission: "{args.mission}"')
        print(f"  constraints:\n{indent(constraints.describe())}")
        if world:
            print(f"  environment:\n{indent(world.describe())}")
        print("\n  asking the planner to decompose it...\n")
        try:
            plan = make_plan(args.mission, constraints, world=world)
        except PlannerError as exc:
            print(f"Planning failed: {exc}", file=sys.stderr)
            return 1

    show_plan(plan, args.plan)
    if args.save:
        args.save.write_text(plan.model_dump_json(indent=2) + "\n")
        print(f"\n  plan saved to {args.save} -- re-fly it with --plan, no tokens needed")

    print(f"\n{RULE}\nVERIFY\n{RULE}")
    result = verify(plan, constraints, world=world)
    for violation in result.violations:
        print(f"  [{violation.severity.value}] [{violation.type.value}] {violation.description}")
    if not result.passed and not args.force:
        print("\nPlan rejected -- not flying it. Re-run with --force to fly it anyway.")
        return 1
    print(f"  verdict: {'PASSED' if result.passed else 'FAILED (forced)'}")

    faults = load_faults(args.faults, plan)
    wind = parse_wind(args.wind, args.turbulence)
    config = SimConfig(dt=args.dt, wind_compensation=0.0 if args.blind else 0.75)

    print(f"\n{RULE}\nFLY\n{RULE}")
    print(f"  wind: {wind.vector} m/s, turbulence {wind.turbulence:g}")
    print("  faults:")
    print(faults.describe())
    print()

    executor = Executor(
        plan, constraints=constraints, config=config, wind=wind, faults=faults, world=world
    )
    report = MissionReport(
        mission_text=plan.mission_text or (args.mission or "simulated mission"),
        plan=plan,
        verification=result,
        constraints=constraints,
        sim_config=config,
        wind=wind,
        faults=faults,
        world=world,
    )

    next_print = 0.0
    for state in executor.run():
        report.states.append(state)
        if not args.quiet and state.timestamp >= next_print:
            print(f"  {format_state(state)}")
            next_print = state.timestamp + 1.0

    report.events = executor.event_records()
    report.obstacles = executor.obstacle_records()
    report.distance_flown = executor.distance_flown
    report.crashed = executor.crashed

    print(f"\n{RULE}\nFLIGHT LOG\n{RULE}")
    for event in report.events:
        print(f"  t={event['t']:7.1f}s  {event['type']:14} {event['description']}")

    print(
        f"\n  {len(report.states)} ticks | {report.duration:.1f}s | "
        f"{report.distance_flown:.0f} m flown | {report.battery_remaining:.1f}% battery left | "
        f"{'landed' if report.completed else 'CRASHED' if report.crashed else 'did not land cleanly'}"
    )

    report.run_file = write_run(report, runs_dir=args.runs_dir)
    print(f"\n  run written to {report.run_file}")

    if args.gazebo:
        from gazebo.launch import GazeboError, play  # noqa: PLC0415 -- optional path
        from orchestrator.runlog import read_run  # noqa: PLC0415

        try:
            return play(read_run(report.run_file), speed=args.speed)
        except GazeboError as exc:
            print(f"Gazebo: {exc}", file=sys.stderr)
            return 1
    print(f"  watch it:  {python_command()} scripts/gazebo_view.py {report.run_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
