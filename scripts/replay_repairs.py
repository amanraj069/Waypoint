"""Replay every plan the verifier rejected in past experiment batches through deterministic repair.

    python scripts/replay_repairs.py experiments/runs/<stamp> [more dirs...]

For each mission log in the result directories, every `plan_proposed` whose
`verification` failed is rebuilt, along with the scenario it was planned for
(world, envelope, pads), and handed to `waypoint.verifier.repair`. Prints how
many repair turned into a passing plan, what it did, and how long it took,
next to how the LLM's own corrections fared in the same logs. No LLM is called.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from waypoint.experiments.spec import Batch, prepare
from waypoint.planner.models import Obstacle, Plan
from waypoint.verifier import verify
from waypoint.verifier.models import DEFAULT_CONFIG
from waypoint.verifier.repair import repair

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--missions", type=Path, default=ROOT / "experiments" / "missions.json")
    parser.add_argument("--ground-only", action="store_true", help="Only plans proposed on the ground.")
    args = parser.parse_args(argv)

    specs = {m.id: m for m in Batch.load(args.missions).missions}
    scenarios = {}
    tried = fixed = 0
    times, kinds, unfixed, longer = [], Counter(), Counter(), []
    for result in args.results:
        for log in sorted(result.glob("*/*/**/verified/*.log.jsonl")):
            mission_id = log.relative_to(result).parts[1]
            spec = specs.get(mission_id)
            if spec is None:
                continue
            if mission_id not in scenarios:
                scenarios[mission_id] = prepare(spec)
            scenario = scenarios[mission_id]
            entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            proposals = {e["plan_id"]: e for e in entries if e["kind"] == "plan_proposed"}
            for check in (e for e in entries if e["kind"] == "verification" and not e["passed"]):
                proposal = proposals.get(check["plan_id"])
                if proposal is None or (args.ground_only and proposal.get("t") is not None):
                    continue
                start = tuple(proposal["start"]) if proposal.get("start") else scenario.home
                in_air = proposal.get("t") is not None
                plan = Plan.model_validate({"subtasks": [{"name": "route", "waypoints": proposal["waypoints"]}]})
                config = DEFAULT_CONFIG.model_copy(update={"home": start})
                constraints = scenario.constraints
                if in_air:
                    # Rebuild the in-flight envelope: the obstacles perception had reported by then.
                    seen = [e["obstacle"] for e in entries if e["kind"] == "perception" and e.get("obstacle") and e["t"] <= proposal["t"] + 1e-6]
                    constraints = constraints.model_copy(
                        update={"obstacles": [*constraints.obstacles, *(Obstacle(**o) for o in seen)]}
                    )
                verdict = verify(plan, constraints, config, scenario.world)
                if verdict.passed:
                    continue  # rejected in flight for a budget that is not rebuilt here
                tried += 1
                began = time.perf_counter()
                outcome = repair(plan, constraints, world=scenario.world, config=config, result=verdict)
                times.append((time.perf_counter() - began) * 1000)
                for v in verdict.blocking:
                    kinds[v.type.value] += 1
                if outcome.repaired:
                    fixed += 1
                    longer.append(100.0 * (_length(outcome.plan, start) / max(_length(plan, start), 1e-6) - 1.0))
                else:
                    for v in outcome.result.blocking if outcome.result else verdict.blocking:
                        unfixed[v.type.value] += 1
    print(f"rejected plans replayed: {tried}")
    if tried:
        print(f"repaired into a passing plan: {fixed} ({fixed / tried:.0%})")
        print(f"repair time: median {statistics.median(times):.0f} ms, p95 {sorted(times)[int(0.95 * (len(times) - 1))]:.0f} ms")
        if longer:
            print(f"route length added by repair: median {statistics.median(longer):.1f}%, p95 {sorted(longer)[int(0.95 * (len(longer) - 1))]:.1f}%")
        print("violations in the rejected plans:", dict(kinds.most_common()))
        print("left unfixed:", dict(unfixed.most_common()))
    return 0


def _length(plan: Plan, start) -> float:
    points = [tuple(start), *((w.x, w.y, w.z) for w in plan.waypoints())]
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


if __name__ == "__main__":
    sys.exit(main())
