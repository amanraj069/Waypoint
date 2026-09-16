"""CLI entrypoint: python -m orchestrator.run --mission "survey the north field"."""

from __future__ import annotations

import argparse

from orchestrator.loop import run_mission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator.run",
        description="Run a UAV mission through the plan -> verify -> execute loop.",
    )
    parser.add_argument("--mission", required=True, help="Natural language mission description.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tick output.")
    args = parser.parse_args(argv)

    report = run_mission(args.mission, verbose=not args.quiet)
    return 0 if report.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
