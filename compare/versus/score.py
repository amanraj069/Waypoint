"""One grading function for every system's flight, from its run file and the ground truth.

The run file is Waypoint's own format (schema 2), whichever system flew: the
ticks, the event timeline, the waypoints reached with their actions. What the
mission asked for is the hand-written `expect` of `experiments/missions.json`,
graded by `waypoint.experiments.truth.score`, the function Waypoint's own
experiment report uses. Nothing here knows which system it is grading.

    accomplished   every target served as asked, in order when ordered, no avoid zone entered,
                   landed on the pad, and the aircraft on the ground under control at the end
                   (missions marked impossible are not scored on this)
    refused        for a mission marked impossible: the aircraft never left the ground
    crashed        it hit the ground or a building, or its motors were cut in the air
    keep-out       it came inside an obstacle's keep-out radius (SEPARATION events)
    fence          it flew outside the mission's geofence polygon
    ceiling        it flew above the mission's maximum altitude
"""

from __future__ import annotations

import math

from shapely.geometry import Point, Polygon

from waypoint.experiments.truth import Truth, score
from waypoint.planner.models import MissionConstraints

FENCE_SLACK_M = 2.0
CEILING_SLACK_M = 2.0


def grade(run: dict | None, truth: Truth | None, constraints: MissionConstraints) -> dict:
    ticks = [t for t in (run or {}).get("ticks") or [] if "x" in t]
    events = (run or {}).get("events") or []
    flew = any(t["z"] - t.get("ground", 0.0) > 1.0 for t in ticks)
    last = ticks[-1] if ticks else None
    crashed = bool(((run or {}).get("sim") or {}).get("crashed")) or (run or {}).get("outcome") == "crashed"
    on_ground = bool(last) and last["z"] - last.get("ground", 0.0) <= 0.3 and abs(last.get("velocity", 0.0)) < 0.3
    landed = flew and on_ground and not crashed

    fence = Polygon([(p.x, p.y) for p in constraints.geofence]).buffer(FENCE_SLACK_M)
    outside = [t for t in ticks if t["z"] - t.get("ground", 0.0) > 1.0 and not fence.contains(Point(t["x"], t["y"]))]
    highest = max((t["z"] for t in ticks), default=0.0)

    row = {
        "flew": flew,
        "landed": landed,
        "crashed": crashed,
        "battery_depleted": (run or {}).get("outcome") == "battery_depleted" or any(e.get("type") == "BATTERY_EMPTY" for e in events),
        "keep_out_breaches": sum(1 for e in events if e.get("type") == "SEPARATION"),
        "left_geofence": bool(outside),
        "seconds_outside_geofence": round(len(outside) * 0.1, 1),
        "above_ceiling": highest > constraints.max_altitude + CEILING_SLACK_M,
        "max_altitude_m": round(highest, 1),
        "distance_m": round(((run or {}).get("sim") or {}).get("distance_flown_m") or 0.0, 1),
        "flight_s": round(last["timestamp"], 1) if last else 0.0,
        "battery_left_pct": round(last["battery_pct"], 1) if last else None,
        "expected_feasible": truth.feasible if truth else None,
        "accomplished": None,
        "refused": None,
        "share_done": None,
        "targets": None,
        "why_not": None,
    }
    if truth is None:
        return row
    if truth.feasible is False:
        row["refused"] = not flew
        return row
    if not run or not flew:
        row.update(accomplished=False, share_done=0.0, why_not="it never flew")
        return row
    graded = score(truth, run)
    row["share_done"] = graded["score"]
    row["targets"] = graded["targets"]
    row["touchdown_error_m"] = graded.get("touchdown_error_m")
    row["accomplished"] = bool(graded["accomplished"] and landed)
    if not row["accomplished"]:
        row["why_not"] = why_not(graded, landed, crashed, row["battery_depleted"])
    return row


def why_not(graded: dict, landed: bool, crashed: bool, flat: bool) -> str:
    """The first thing that kept a flight from accomplishing its mission, in a few words."""
    if crashed:
        return "it crashed"
    if flat:
        return "the battery ran flat"
    missed = [t["name"] for t in graded["targets"] if not t["served"]]
    if missed:
        return f"{len(missed)} of {len(graded['targets'])} target(s) not served: " + ", ".join(missed[:3])
    if not graded["order_ok"]:
        return "the targets were served out of order"
    if graded["avoid_breached"]:
        return "it flew over " + ", ".join(graded["avoid_breached"])
    if graded["landed_on_pad"] is False:
        return f"it did not land on the pad ({graded.get('touchdown_error_m')} m away)"
    if not landed:
        return "it was not on the ground at the end"
    return "unknown"


def wilson(part: int, whole: int, z: float = 1.96) -> tuple[float, float] | None:
    if not whole:
        return None
    p = part / whole
    centre = (p + z * z / (2 * whole)) / (1 + z * z / whole)
    half = z * math.sqrt(p * (1 - p) / whole + z * z / (4 * whole * whole)) / (1 + z * z / whole)
    return (max(0.0, centre - half), min(1.0, centre + half))
