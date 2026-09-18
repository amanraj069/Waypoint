#!/usr/bin/env python3
"""Run the real verifier against a real plan, then against an impossible envelope.

    # plan live with Gemini (spends tokens), then verify twice
    python scripts/try_verifier.py "survey the north field and return to base"

    # keep the plan so the next run costs nothing
    python scripts/try_verifier.py "survey the north field" --save /tmp/plan.json
    python scripts/try_verifier.py --plan /tmp/plan.json

The first pass verifies against the normal operating envelope. The second pass
shrinks the geofence around its centre until the plan cannot possibly satisfy
it, which is the point: a deterministic verifier should turn that into a
specific, quotable list of violations rather than a shrug.

`--hazards` also drops a no-fly zone across the middle of the route and an
obstacle on its furthest waypoint, to exercise those checks on a real plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapely.affinity import scale  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from planner import (  # noqa: E402
    DEFAULT_CONSTRAINTS,
    GeoPoint,
    MissionConstraints,
    Obstacle,
    Plan,
    PlannerError,
    plan as make_plan,
)
from verifier import DEFAULT_CONFIG, VerificationResult, verify  # noqa: E402

RULE = "=" * 78


def shrink_geofence(constraints: MissionConstraints, factor: float) -> MissionConstraints:
    """Scale the geofence about its own centre, keeping it a valid polygon."""
    fence = Polygon([(p.x, p.y) for p in constraints.geofence])
    smaller = scale(fence, xfact=factor, yfact=factor, origin="centroid")
    vertices = [GeoPoint(x=x, y=y) for x, y in list(smaller.exterior.coords)[:-1]]
    return constraints.model_copy(update={"geofence": vertices})


def add_hazards(constraints: MissionConstraints, plan: Plan) -> MissionConstraints:
    """Put a no-fly zone across the middle of the route and an obstacle on its far end."""
    waypoints = plan.waypoints()
    if not waypoints:
        return constraints
    middle = waypoints[len(waypoints) // 2]
    far = max(waypoints, key=lambda wp: wp.x**2 + wp.y**2)
    zone = [
        GeoPoint(x=middle.x - 20, y=middle.y - 20),
        GeoPoint(x=middle.x + 20, y=middle.y - 20),
        GeoPoint(x=middle.x + 20, y=middle.y + 20),
        GeoPoint(x=middle.x - 20, y=middle.y + 20),
    ]
    return constraints.model_copy(
        update={
            "no_fly_zones": [zone],
            "obstacles": [Obstacle(x=far.x, y=far.y, radius=5.0, name="comms tower")],
            "min_separation": 15.0,
        }
    )


def show(title: str, constraints: MissionConstraints, result: VerificationResult) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")
    print(constraints.describe())
    verdict = "PASSED" if result.passed else "FAILED"
    print(f"\nverdict: {verdict}  ({len(result.blocking)} blocking, {len(result.warnings)} warning)")
    for violation in result.violations:
        index = "plan" if violation.waypoint_index is None else f"wp {violation.waypoint_index}"
        print(f"\n  [{violation.severity.value}] [{violation.type.value}] ({index})")
        print(f"    {violation.description}")


def load_plan(path: Path) -> Plan:
    payload = json.loads(path.read_text())
    if "subtasks" not in payload:
        raise ValueError(f"{path} does not hold a Plan (no 'subtasks' key)")
    return Plan.model_validate(payload)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mission", nargs="?", help="Mission text to plan with the live API.")
    parser.add_argument("--plan", type=Path, help="Verify this saved plan JSON instead of calling the API.")
    parser.add_argument("--save", type=Path, help="Write the generated plan here for re-runs.")
    parser.add_argument(
        "--shrink",
        type=float,
        default=0.2,
        help="Scale factor for the second geofence (default 0.2 -- a fifth of the original).",
    )
    parser.add_argument(
        "--hazards",
        action="store_true",
        help="Also add a no-fly zone and an obstacle to the shrunken envelope.",
    )
    args = parser.parse_args(argv)

    if not args.mission and not args.plan:
        parser.error("give a mission to plan, or --plan with a saved plan file")

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    constraints = DEFAULT_CONSTRAINTS

    if args.plan:
        plan = load_plan(args.plan)
        print(f"Plan:    {args.plan}", file=sys.stderr)
    else:
        print(f"Mission: {args.mission}", file=sys.stderr)
        try:
            plan = make_plan(args.mission, constraints)
        except PlannerError as exc:
            print(f"Planning failed: {exc}", file=sys.stderr)
            return 1
        if args.save:
            args.save.write_text(plan.model_dump_json(indent=2) + "\n")
            print(f"Plan saved to {args.save}", file=sys.stderr)

    print(f"\n{RULE}\nPLAN\n{RULE}")
    for subtask in plan.subtasks:
        print(f"  subtask \"{subtask.name}\":")
        for wp in subtask.waypoints:
            print(f"    ({wp.x:g}, {wp.y:g}, {wp.z:g}) at {wp.speed:g} m/s, {wp.action.value}")
    print(f"\n  {len(plan.subtasks)} subtask(s), {len(plan.waypoints())} waypoint(s)")

    show("1. NORMAL ENVELOPE", constraints, verify(plan, constraints, DEFAULT_CONFIG))

    tight = shrink_geofence(constraints, args.shrink)
    if args.hazards:
        tight = add_hazards(tight, plan)
    title = f"2. GEOFENCE SHRUNK TO {args.shrink:g}x" + (" + HAZARDS" if args.hazards else "")
    show(title, tight, verify(plan, tight, DEFAULT_CONFIG))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
