"""Fly the plans of past experiment batches again, with the runtime shield on and off.

    python scripts/shield_trial.py experiments/runs/<stamp> [more dirs...]

For every flight pair in the result directories, the first plan the verifier
passed (and, separately, the first plan as the LLM wrote it, which the
ungated baseline flew) is flown open loop, with the mission's own wind and
fault schedule, once without the shield and once with it. No planner is
called and nothing replans, so any difference is the shield's alone. Prints
crashes, keep-out breaches, completion and flight time for each.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from waypoint.experiments.spec import Batch, prepare
from waypoint.planner.models import Plan
from waypoint.sim import FaultEvent, FaultSchedule, SimConfig, Wind
from waypoint.sim.executor import Executor

ROOT = Path(__file__).resolve().parents[1]


def fly(plan: Plan, scenario, faults: FaultSchedule, shield: bool) -> dict:
    executor = Executor(
        plan, constraints=scenario.constraints, config=SimConfig(home=scenario.home), wind=scenario.wind,
        faults=faults, world=scenario.world, shield=shield,
    )
    states = list(executor.run())
    events = executor.events
    landed = any(e.type == "LANDED" for e in events) and not executor.crashed
    why = None
    bad = next((e for e in events if e.type in ("COLLISION", "SEPARATION")), None)
    if bad is not None:
        at = min(states, key=lambda s: abs(s.timestamp - bad.t))
        spawned = [e.t for e in events if e.type == "FAULT" and (e.payload or {}).get("obstacle")]
        if at.gps_error > 3.0:
            why = "gps"  # the estimate the shield steers on was off by metres
        elif bad.type == "SEPARATION" and any(0 <= bad.t - t <= 1.0 for t in spawned):
            why = "spawned"  # the obstacle appeared already inside its keep-out
        else:
            why = "other"
    return {
        "why": why,
        "crashed": executor.crashed,
        "breach": any(e.type == "SEPARATION" for e in events),
        "landed": landed,
        "timeouts": sum(e.type == "LEG_TIMEOUT" for e in events),
        "t": states[-1].timestamp if states else 0.0,
        "shield_ticks": executor.shield_ticks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--missions", type=Path, default=ROOT / "experiments" / "missions.json")
    parser.add_argument(
        "--calm", action="store_true",
        help="No wind and no faults: a verified plan that still crashes or breaks a keep-out is one the "
             "verifier should have refused (its false negatives).",
    )
    args = parser.parse_args(argv)
    specs = {m.id: m for m in Batch.load(args.missions).missions}
    tallies = {(kind, shield): [] for kind in ("verified", "llm-first") for shield in (False, True)}
    for result in args.results:
        manifest = json.loads((result / "manifest.json").read_text())
        for flight in manifest["flights"]:
            spec = specs.get(flight["mission_id"])
            log_rel = (flight.get("verified") or {}).get("log")
            if spec is None or not log_rel:
                continue
            scenario = prepare(spec, seed=manifest["config"].get("seed", 0), repeat=flight.get("repeat", 0))
            faults = FaultSchedule(events=[FaultEvent(**e) for e in flight.get("faults") or []])
            if args.calm:
                faults = FaultSchedule()
                scenario.wind = Wind()
            entries = [json.loads(line) for line in (result / log_rel).read_text().splitlines() if line.strip()]
            proposals = {e["plan_id"]: e for e in entries if e["kind"] == "plan_proposed" and e.get("t") is None}
            checks = [e for e in entries if e["kind"] == "verification" and e["plan_id"] in proposals]
            if not checks:
                continue
            first = proposals[checks[0]["plan_id"]]
            passing = next((proposals[c["plan_id"]] for c in checks if c["passed"]), None)
            for kind, proposal in (("verified", passing), ("llm-first", first)):
                if proposal is None:
                    continue
                plan = Plan.model_validate({"subtasks": [{"name": "route", "waypoints": proposal["waypoints"]}]})
                for shield in (False, True):
                    tallies[(kind, shield)].append(fly(plan, scenario, faults, shield))
    print(f"{'plans':<10} {'shield':<7} {'n':>4} {'crash':>7} {'breach':>7} {'landed':>7} {'timeouts':>9} {'median t':>9}")
    for (kind, shield), rows in tallies.items():
        if not rows:
            continue
        n = len(rows)
        rate = lambda key: f"{100 * sum(r[key] for r in rows) / n:5.1f}%"  # noqa: E731
        print(
            f"{kind:<10} {'on' if shield else 'off':<7} {n:>4} {rate('crashed'):>7} {rate('breach'):>7} "
            f"{rate('landed'):>7} {sum(r['timeouts'] for r in rows):>9} {statistics.median(r['t'] for r in rows):>8.1f}s"
        )
    print("\nwhy a shielded flight still crashed or breached (gps: estimate off by > 3 m; "
          "spawned: obstacle appeared inside its keep-out; other):")
    for (kind, shield), rows in tallies.items():
        if shield:
            causes = {}
            for r in rows:
                if r["why"]:
                    causes[r["why"]] = causes.get(r["why"], 0) + 1
            print(f"  {kind}: {causes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
