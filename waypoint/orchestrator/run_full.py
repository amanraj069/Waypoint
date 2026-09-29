"""The closed loop from the command line, watched in Gazebo.

    python -m waypoint.orchestrator.run_full \\
        --mission "survey the north field, then come home" \\
        --faults crane_ahead --gazebo

Prints each phase as it happens (PLANNING, VERIFYING, EXECUTING, REPLAN
TRIGGERED, REPLANNING, RE-VERIFYING, RESUMING, ...), then writes two files to
runs/: the run file Gazebo replays, and the mission log as JSON Lines.
`--gazebo` opens the finished flight in Gazebo: the replacement route appears
at the moment it took over, and the trail shows the aircraft follow it.

`--backend custom` (the default) is the built-in simulator; `px4` is reserved
for PX4 SITL with Gazebo as its physics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from waypoint.backends import BACKENDS, BackendError, make_backend
from waypoint.orchestrator.loop import run_mission
from waypoint.paths import RUNS_DIR
from waypoint.planner import DEFAULT_CONSTRAINTS, Plan
from waypoint.scenario import fence_around, load_faults, load_plan, load_world, parse_wind


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m waypoint.orchestrator.run_full",
        description="Plan, verify, fly and replan a UAV mission in a closed loop.",
    )
    parser.add_argument("--mission", required=True, help="The mission, in plain English.")
    parser.add_argument(
        "--faults", default="none",
        help="'none', 'sample', a name in data/faults (e.g. crane_ahead), or a schedule JSON.",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="custom", help="What flies it.")
    parser.add_argument("--world", help="A name in data/worlds (lodhi, munnar, ...) or a world JSON.")
    parser.add_argument(
        "--plan",
        help="Start from a saved plan (a name in data/plans, or a JSON path) instead of asking "
             "the planner. Replans in flight still go to the planner.",
    )
    parser.add_argument("--wind", help="Steady wind as 'x,y,z' in m/s.")
    parser.add_argument("--turbulence", type=float, default=0.0, help="Gustiness std-dev, m/s.")
    parser.add_argument("--max-altitude", type=float, help="Raise the altitude ceiling.")
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Corrections to ask for after each verifier rejection.",
    )
    parser.add_argument(
        "--max-replans", type=int, default=3, help="In-flight replans allowed before aborting."
    )
    parser.add_argument(
        "--check-every", type=int, default=1, help="Ticks between feedback checks (10 ticks = 1 s)."
    )
    parser.add_argument("--runs-dir", type=Path, help=f"Where to write the run (default {RUNS_DIR}).")
    parser.add_argument("--quiet", action="store_true", help="Print nothing but errors.")
    parser.add_argument("--gazebo", action="store_true", help="Watch the flight in Gazebo afterwards.")
    parser.add_argument("--speed", type=float, default=1.0, help="Gazebo playback speed.")
    parser.add_argument("--plot", action="store_true", help="Also write a Plotly 3D plot of the flight.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    world = load_world(args.world)
    constraints = DEFAULT_CONSTRAINTS
    if world:
        constraints = constraints.model_copy(update={"geofence": fence_around(world)})
    if args.max_altitude:
        constraints = constraints.model_copy(update={"max_altitude": args.max_altitude})

    planner = None
    saved = load_plan(args.plan) if args.plan else Plan()
    if args.plan:
        planner = lambda mission, c, **_: saved  # noqa: E731 -- the one plan, whatever is asked
    # `sample` aims its faults at a plan, so it needs one; a named schedule does not.
    faults = load_faults(args.faults, saved)

    try:
        backend = make_backend(args.backend, world=world, wind=parse_wind(args.wind, args.turbulence))
    except BackendError as exc:
        print(f"backend: {exc}", file=sys.stderr)
        return 2

    mission_log = run_mission(
        args.mission,
        constraints,
        faults,
        backend,
        world=world,
        planner=planner,
        max_replan_attempts=args.max_attempts,
        max_total_replans=args.max_replans,
        check_every=args.check_every,
        verbose=not args.quiet,
        **({"runs_dir": args.runs_dir} if args.runs_dir else {}),
    )
    report = mission_log.report

    if args.plot and report is not None and report.run_file:
        from waypoint.orchestrator.plotting import render  # noqa: PLC0415 -- plotly is slow to import
        from waypoint.orchestrator.runlog import read_run  # noqa: PLC0415

        try:
            print(f"    plot: {render(read_run(report.run_file), report.run_file, open_browser=False)}")
        except ValueError as exc:
            print(f"    nothing to plot: {exc}", file=sys.stderr)

    if args.gazebo and report is not None and report.run_file and report.states:
        from waypoint.gazebo.launch import GazeboError, play  # noqa: PLC0415 -- optional
        from waypoint.orchestrator.runlog import read_run  # noqa: PLC0415

        try:
            play(read_run(report.run_file), speed=args.speed)
        except GazeboError as exc:
            print(f"Gazebo: {exc}", file=sys.stderr)
            return 1

    return 0 if mission_log.outcome == "completed" else 1



if __name__ == "__main__":
    raise SystemExit(main())
