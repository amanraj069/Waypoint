"""Ground truth: what each mission really asked for, and how a flight measures up to it.

The batch file's `expect` entries are written by hand and never shown to a
planner. Here they are grounded in the world (`waypoint.intent.ground`, the
same grounding the extractor's output gets) with the scenario's landing pad,
and every flight is scored against them from its run file alone: the
waypoints it actually reached, with their actions, and its track. So intent
success means the same thing for every condition, including the ones that
never read an intent, and for result directories flown before intent existed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from waypoint.intent import MissionIntent, RoutePoint, evaluate, ground
from waypoint.intent.check import reach
from waypoint.planner.models import Action
from waypoint.world.models import World


@dataclass
class Truth:
    intent: MissionIntent
    feasible: bool | None
    world: World | None


def load_truth(missions_path: Path | str | None) -> dict[str, Truth]:
    """Ground truth for every mission of a batch file that has an `expect`; {} when there is none."""
    from waypoint.experiments.spec import Batch, prepare  # noqa: PLC0415 -- spec imports the runner's world

    if not missions_path or not Path(missions_path).exists():
        return {}
    truths: dict[str, Truth] = {}
    for spec in Batch.load(Path(missions_path)).missions:
        if spec.expect is None:
            continue
        try:
            scenario = prepare(spec)
        except ValueError:
            continue
        intent = ground(spec.expect.payload(), scenario.world, landing=scenario.landing)
        truths[spec.id] = Truth(intent=intent, feasible=spec.expect.feasible, world=scenario.world)
    return truths


def flown(run: dict) -> tuple[list[RoutePoint], list[tuple[float, float, float]]]:
    """The waypoints a run reached (with their actions, where it reached them) and its track."""
    points = []
    for event in run.get("events") or []:
        if event.get("type") != "WAYPOINT":
            continue
        payload = event.get("payload") or {}
        action = payload.get("action")
        if action is None:  # runs from before the action was recorded: it is in the description
            text = event.get("description", "")
            action = next((a.value for a in Action if f"({a.value})" in text), "MOVE")
        x, y, z = payload.get("position") or (0.0, 0.0, 0.0)
        points.append(RoutePoint(len(points), "", x, y, z, Action(action)))
    track = [(t["x"], t["y"], t["z"]) for t in run.get("ticks") or [] if "x" in t]
    return points, track


def score(truth: Truth, run: dict) -> dict | None:
    """How much of what the mission asked for the flight did (`waypoint.intent.evaluate`)."""
    if not run:
        return None
    points, track = flown(run)
    return evaluate(truth.intent, points, track, truth.world, planned_landing=planned_landing(run))


def planned_landing(run: dict) -> tuple[float, float] | None:
    """Where the last plan that flew put its LAND, if it ended with one."""
    plans = run.get("plans") or []
    waypoints = (plans[-1].get("waypoints") if plans else None) or run.get("waypoints") or []
    last = waypoints[-1] if waypoints else None
    if last and last.get("action") == "LAND":
        return (last["x"], last["y"])
    return None


def extraction_match(truth: MissionIntent, entry: dict) -> dict:
    """How the intent a flight read compares with the ground truth.

    Targets are matched by landmark name, or, for points, by position within
    the target's reach. `exact` needs the same targets, served the same way
    (action, views, sides, from above), the same order flag and the same avoid
    zones; `target_f1` scores the target sets alone.
    """
    read = entry.get("targets") or []
    matched = []
    unmatched = list(read)
    for want in truth.targets:
        hit = None
        for got in unmatched:
            same_name = want.landmark and got.get("landmark") and got.get("name", "").lower() == want.name.lower()
            near = not want.landmark and math.hypot(got.get("x", 0) - want.x, got.get("y", 0) - want.y) <= reach(want)
            if same_name or near:
                hit = got
                break
        if hit is not None:
            unmatched.remove(hit)
            matched.append((want, hit))
    precision = len(matched) / len(read) if read else (1.0 if not truth.targets else 0.0)
    recall = len(matched) / len(truth.targets) if truth.targets else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    same_way = all(
        got.get("action") == want.action.value
        and int(got.get("views", 1)) == want.views
        and sorted(got.get("sides") or []) == sorted(s.value for s in want.sides)
        and bool(got.get("overhead")) == want.overhead
        for want, got in matched
    )
    avoid_read = sorted(a.get("name", "").lower() for a in entry.get("avoid") or [])
    avoid_true = sorted(a.name.lower() for a in truth.avoid)
    ordered_ok = bool(entry.get("ordered")) == truth.ordered or len(truth.targets) < 2
    exact = (
        len(matched) == len(truth.targets) == len(read) and same_way and ordered_ok and avoid_read == avoid_true
    )
    return {"exact": exact, "target_f1": round(f1, 4)}
