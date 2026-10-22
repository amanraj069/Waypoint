"""Check the PX4 + Gazebo backend end to end, and print PASS or FAIL for each step.

    python scripts/px4_check.py              # headless, about 3 minutes
    python scripts/px4_check.py --gui        # watch it in the Gazebo window
    python scripts/px4_check.py --speed 4    # faster than real time

One scenario, flown through the unmodified closed loop with scripted planners,
so it needs PX4 and Gazebo (see README section 5c) but no API key:

  a geofence, a no-fly zone and a planned water tower; a survey of the north
  field; and, 40 m up the first leg, a crane that was not on any map.

The checks, in order:

  1. the Gazebo world PX4 flies in shows the fence, the no-fly zone and the
     water tower, where the verifier had them, as glass (no collision);
  2. PX4 and Gazebo boot;
  3. the plan is flown and lands;
  4. telemetry streams back as `UAVState`s: one per 0.1 s of simulated time,
     inside the fence, battery falling, speeds within the limit;
  5. the crane triggers a replan through the loop, the flight resumes from the
     air on the detour, and the track keeps clear of the crane;
  6. the same scenario on the custom simulator verifies and replans the same
     way, with a similar flight.

Exits 0 when every check passes. `--keep` leaves the world and PX4's logs in
~/.cache/waypoint/px4/ for a look afterwards.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from shapely.geometry import Point, Polygon

from waypoint.backends import BackendError
from waypoint.orchestrator.loop import run_mission
from waypoint.planner import (
    Action,
    DEFAULT_CONSTRAINTS,
    GeoPoint,
    Obstacle,
    Plan,
    SubTask,
    Waypoint,
)
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, Trigger
from waypoint.sim.backend import CustomSimBackend

MISSION = "survey the north field: photograph its two far corners, then come home and land"
CRANE = Obstacle(x=0.0, y=0.0, radius=6.0, height=60.0, name="mobile crane")

CONSTRAINTS = DEFAULT_CONSTRAINTS.model_copy(update={
    "no_fly_zones": [[GeoPoint(x=100, y=100), GeoPoint(x=140, y=100), GeoPoint(x=140, y=140), GeoPoint(x=100, y=140)]],
    "obstacles": [Obstacle(x=-50.0, y=120.0, radius=6.0, height=40.0, name="water tower")],
})
FAULTS = FaultSchedule.of(FaultEvent(
    type=FaultType.OBSTACLE_APPEARS, trigger=Trigger(at_waypoint=1), ahead=40.0, obstacle=CRANE, name="crane spotted",
))


def wp(x, y, z, speed, action=Action.MOVE) -> Waypoint:
    return Waypoint(x=x, y=y, z=z, speed=speed, action=action)


def planner(mission, constraints, **kwargs) -> Plan:
    return Plan(mission_text=mission, subtasks=[
        SubTask(name="takeoff", waypoints=[wp(0, 0, 25, 3)]),
        SubTask(name="survey_north_field", waypoints=[wp(0, 80, 25, 4, Action.CAPTURE), wp(40, 80, 25, 4, Action.CAPTURE)]),
        SubTask(name="return_and_land", waypoints=[wp(0, 0, 0, 2, Action.LAND)]),
    ])


def replanner(mission, previous, violations, constraints, **kwargs) -> Plan:
    """Out to the east of the crane, past it, then the rest of the survey and home."""
    crane = next(o for o in constraints.obstacles if o.name == CRANE.name)
    return Plan(mission_text=mission, subtasks=[SubTask(name="detour", waypoints=[
        wp(crane.x + 15, crane.y - 20, 25, 4), wp(crane.x + 15, crane.y + 20, 25, 4),
        wp(0, 80, 25, 4, Action.CAPTURE), wp(40, 80, 25, 4, Action.CAPTURE), wp(0, 0, 0, 2, Action.LAND),
    ])])


class Checks:
    def __init__(self):
        self.failed = 0

    def report(self, number: int, title: str, ok: bool, detail: str) -> None:
        self.failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {number}. {title}: {detail}", flush=True)


def fly(backend, runs: Path):
    return run_mission(MISSION, CONSTRAINTS, FAULTS, backend, planner=planner, replanner=replanner,
                       verbose=False, runs_dir=runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gui", action="store_true", help="Open the Gazebo window and follow the aircraft.")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed factor.")
    parser.add_argument("--keep", action="store_true", help="Keep the world and PX4's logs afterwards.")
    args = parser.parse_args(argv)

    from waypoint.px4.backend import Px4Backend  # noqa: PLC0415

    checks = Checks()
    runs = Path(tempfile.mkdtemp(prefix="px4-check-"))
    print(f'PX4 check: "{MISSION}"\n  crane appears 40 m up the first leg; runs in {runs}', flush=True)
    print("  takes about 3 minutes; Ctrl-C stops it and shuts PX4 and Gazebo down\n", flush=True)
    try:
        backend = Px4Backend(gui=args.gui, speed=args.speed, keep=True)
    except BackendError as exc:
        print(f"  [FAIL] PX4 is not usable here: {exc}")
        return 1

    booted: dict = {}
    prepare = backend.prepare

    def timed_prepare(*a, **k):
        started = time.monotonic()
        prepare(*a, **k)
        booted["seconds"] = time.monotonic() - started
        booted["world"] = backend._sitl.dir / "scene" / "world.sdf"
        print("    flying the scenario: about 95 s of flight, plus a replan when the crane appears", flush=True)

    backend.prepare = timed_prepare
    started = time.monotonic()
    try:
        log = fly(backend, runs)
    except BackendError as exc:
        backend.close()
        checks.report(2, "PX4 and Gazebo boot", False, str(exc).splitlines()[0])
        return 1
    except BaseException:
        backend.close()
        raise
    wall = time.monotonic() - started
    work = backend._sitl.dir if backend._sitl else None
    try:
        return report(args, checks, backend, log, booted, wall, runs, work)
    finally:
        backend.close()
        if not args.keep:
            shutil.rmtree(runs, ignore_errors=True)
            if work:
                shutil.rmtree(work, ignore_errors=True)


def report(args, checks: Checks, backend, log, booted: dict, wall: float, runs: Path, work: Path | None) -> int:
    """Print every check, then (with --gui) leave the window up until it is closed."""
    if not args.gui:
        backend.close()  # nothing left to look at; free PX4 and Gazebo before the custom flight

    # 1. The world matches the mission.
    sdf = booted["world"].read_text() if booted.get("world") and booted["world"].exists() else ""
    fence = sdf.count('<visual name="fence_')
    nofly = sdf.count('<visual name="nofly_')
    tower = "<pose>-50.00 120.00 20.00 0 0 0</pose>" in sdf
    colliders = re.findall(r'<collision name="([^"]+)"', sdf)
    checks.report(1, "world matches the mission", fence == 4 and nofly == 1 and tower and colliders == ["ground"],
                  f"{fence}/4 fence walls, {nofly}/1 no-fly zone, water tower {'at (-50, 120)' if tower else 'MISSING'}, "
                  f"colliders {colliders}")

    # 2. Boot.
    seconds = booted.get("seconds")
    checks.report(2, "PX4 and Gazebo boot", seconds is not None,
                  f"ready in {seconds:.0f} s" if seconds else "never got ready")

    # 3. Flown and landed.
    report = log.report
    reached = [e for e in report.events if e["type"] == "WAYPOINT"]
    checks.report(3, "plan flown and landed", log.outcome == "completed",
                  f"{log.outcome}, {len(reached)} waypoints reached, {report.duration:.0f} s simulated, "
                  f"{report.distance_flown:.0f} m, {wall:.0f} s wall")

    # 4. Telemetry.
    states = report.states
    gaps = [round(b.timestamp - a.timestamp, 3) for a, b in zip(states, states[1:-1])]
    regular = sum(abs(g - 0.1) < 0.011 for g in gaps) / max(1, len(gaps))
    fence_poly = Polygon([(p.x, p.y) for p in CONSTRAINTS.geofence])
    inside = all(fence_poly.contains(Point(s.x, s.y)) for s in states)
    falling = all(b.battery_pct <= a.battery_pct + 1e-9 for a, b in zip(states, states[1:]))
    fastest = max(s.ground_speed for s in states)
    checks.report(4, "telemetry streams back", regular > 0.98 and inside and falling and fastest <= CONSTRAINTS.max_velocity + 1,
                  f"{len(states)} states, {regular:.0%} at 0.1 s, inside fence: {inside}, battery "
                  f"{states[0].battery_pct:.1f} -> {states[-1].battery_pct:.1f}%, top speed {fastest:.1f} m/s")

    # 5. The crane triggers a replan through the loop.
    triggers = log.of("replan_triggered")
    resumed = log.of("execution_resumed")
    crane = next((o for o in report.obstacles if o["name"] == CRANE.name), None)
    closest = None
    if crane:
        after = [s for s in states if s.timestamp >= crane["appeared_at"]]
        closest = min(math.hypot(s.x - crane["x"], s.y - crane["y"]) for s in after)
    ok = bool(triggers) and bool(resumed) and closest is not None and closest > crane["radius"] + CONSTRAINTS.min_separation
    checks.report(5, "failure triggers a replan", ok,
                  f"{len(triggers)} trigger ({triggers[0]['trigger'] if triggers else '-'}), resumed {len(resumed)}x, "
                  f"closest pass {closest:.1f} m (keep-out {crane['radius'] + CONSTRAINTS.min_separation:.0f} m)"
                  if crane else "the crane never appeared")

    # 6. The custom simulator agrees.
    print("    flying the same scenario on the custom simulator", flush=True)
    custom = fly(CustomSimBackend(), runs)
    a, b = custom.report, report
    same = (custom.outcome == log.outcome
            and len(custom.of("verification")) == len(log.of("verification"))
            and len(custom.of("replan_triggered")) == len(log.of("replan_triggered")))
    close = abs(b.duration - a.duration) <= 0.15 * a.duration and abs(b.distance_flown - a.distance_flown) <= 0.10 * a.distance_flown
    checks.report(6, "custom and PX4 agree", same and close,
                  f"custom {custom.outcome} {a.duration:.0f} s {a.distance_flown:.0f} m {a.battery_remaining:.1f}%, "
                  f"px4 {log.outcome} {b.duration:.0f} s {b.distance_flown:.0f} m {b.battery_remaining:.1f}%, "
                  f"{len(log.of('verification'))} verifications and {len(triggers)} replan(s) on each" if same
                  else "they verified or replanned differently")

    if args.keep:
        print(f"\n  kept: {work} (the world and PX4's logs), {runs} (run files and mission logs)")
    print(f"\n{'ALL CHECKS PASSED' if not checks.failed else f'{checks.failed} CHECK(S) FAILED'}", flush=True)
    if args.gui:
        backend.linger()
    return 0 if not checks.failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  stopped; PX4 and Gazebo are shut down.")
        sys.exit(130)
