#!/usr/bin/env python3
"""Render a saved mission run as an interactive 3D flight path.

    python scripts/visualize.py runs/survey-the-north-field_20260916T120000.json
    python scripts/visualize.py --latest

Writes a standalone HTML file beside the run file and opens it in the default
browser. Pass --no-open to skip the browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.plotting import render  # noqa: E402
from orchestrator.runlog import RUNS_DIR, read_run  # noqa: E402


def latest_run(runs_dir: Path) -> Path:
    runs = sorted(runs_dir.glob("*.json"))
    if not runs:
        raise SystemExit(f"No run files in {runs_dir}/. Fly a mission first.")
    return runs[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/visualize.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_file", nargs="?", help="Path to a runs/*.json file.")
    parser.add_argument(
        "--latest", action="store_true", help="Use the most recent run in runs/."
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Write the HTML without opening a browser."
    )
    args = parser.parse_args(argv)

    if args.latest or not args.run_file:
        run_file = latest_run(RUNS_DIR)
    else:
        run_file = Path(args.run_file)

    try:
        run = read_run(run_file)
        out = render(run, run_file, open_browser=not args.no_open)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not visualize {run_file}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
