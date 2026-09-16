#!/usr/bin/env python3
"""Manual smoke test of the real planner against the live Gemini API.

    python scripts/try_planner.py "survey the north field and return to base"

Requires GEMINI_API_KEY in .env or the environment. This spends real tokens.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner import DEFAULT_CONSTRAINTS, PlannerError, plan  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    mission_text = argv[0]
    print(f"Mission:     {mission_text}", file=sys.stderr)
    print(f"Constraints: {DEFAULT_CONSTRAINTS.describe()}\n", file=sys.stderr)

    try:
        result = plan(mission_text, DEFAULT_CONSTRAINTS)
    except PlannerError as exc:
        print(f"Planning failed: {exc}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2))
    print(
        f"\n{len(result.subtasks)} subtask(s), {len(result.waypoints())} waypoint(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
