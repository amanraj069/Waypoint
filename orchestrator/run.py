"""CLI entrypoint: python -m orchestrator.run --mission "survey the north field"."""

from __future__ import annotations

import argparse
import logging
import sys

from orchestrator.loop import run_mission
from planner import PlannerError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator.run",
        description="Run a UAV mission through the plan -> verify -> execute loop.",
    )
    parser.add_argument("--mission", required=True, help="Natural language mission description.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tick output.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render the flight path as an interactive 3D plot and open it in a browser.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="With --plot, write the HTML but do not open a browser.",
    )
    parser.add_argument(
        "--gazebo",
        action="store_true",
        help="Watch the flown mission in Gazebo once it lands.",
    )
    args = parser.parse_args(argv)

    # So model-shift warnings from the planner reach the operator.
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    try:
        report = run_mission(args.mission, verbose=not args.quiet)
    except PlannerError as exc:
        print(f"\nPlanning failed: {exc}", file=sys.stderr)
        return 1

    if args.plot:
        # Imported here so a plain run does not pay for plotly's import.
        from orchestrator.plotting import render
        from orchestrator.runlog import build_run

        try:
            out = render(
                build_run(report),
                report.run_file,
                open_browser=not args.no_open,
            )
        except ValueError as exc:
            print(f"\nNothing to plot: {exc}", file=sys.stderr)
            return 1
        print(f"  plot written to {out}")

    if args.gazebo and report.run_file and report.states:
        from gazebo.launch import GazeboError, play
        from orchestrator.runlog import read_run

        try:
            play(read_run(report.run_file))
        except GazeboError as exc:
            print(f"\nGazebo: {exc}", file=sys.stderr)
            return 1

    return 0 if report.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
