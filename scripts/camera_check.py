"""Fly PX4's x500_depth in Gazebo and see a crane with the depth camera alone, then go round it.

    python scripts/camera_check.py              # headless, a few minutes
    python scripts/camera_check.py --gui        # watch it

The crane scenario of `px4_check.py`, with one difference that matters: the
loop is not told about the crane. The only way it can know is the depth
camera (`GazeboCameraPerceptionSource`): no symbolic perception, and no
runtime shield (which reads the simulator's obstacle list) to cover for it.
No LLM is called; in-flight replanning is the deterministic repair.

The checks:

  1. the depth bridge delivers frames;
  2. the camera reports an obstacle that is not on the map, before the
     aircraft reaches the crane's keep-out;
  3. the obstacle it reports is where the crane really is (within 3 m);
  4. that report triggers a replan, resolved by repair without the planner;
  5. the aircraft never enters the crane's keep-out;
  6. the mission completes.

Exits 0 when every check passes.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

from waypoint.backends import BackendError
from waypoint.feedback.gazebo_camera import GazeboCameraPerceptionSource
from waypoint.orchestrator.loop import run_mission
from waypoint.orchestrator.options import system_from
from waypoint.planner import Action, DEFAULT_CONSTRAINTS, Obstacle, Plan, PlannerError, SubTask, Waypoint
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, Trigger

MISSION = "survey the north field: photograph its two far corners, then come home and land"
CRANE = Obstacle(x=0.0, y=0.0, radius=6.0, height=60.0, name="mobile crane")
FAULTS = FaultSchedule.of(FaultEvent(
    type=FaultType.OBSTACLE_APPEARS, trigger=Trigger(at_waypoint=1), ahead=40.0, obstacle=CRANE, name="crane spotted",
))


def wp(x, y, z, speed, action=Action.MOVE) -> Waypoint:
    return Waypoint(x=x, y=y, z=z, speed=speed, action=action)


def planner(mission, constraints, **kwargs) -> Plan:
    return Plan(mission_text=mission, subtasks=[
        SubTask(name="takeoff", waypoints=[wp(0, 0, 25, 3)]),
        SubTask(name="survey_north_field", waypoints=[wp(0, 80, 25, 3, Action.CAPTURE), wp(40, 80, 25, 3, Action.CAPTURE)]),
        SubTask(name="return_and_land", waypoints=[wp(0, 0, 25, 4), wp(0, 0, 0, 2, Action.LAND)]),
    ])


def replanner(*args, **kwargs) -> Plan:
    raise PlannerError("no LLM in this check: the repair has to do it")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args(argv)

    from waypoint.px4.backend import Px4Backend  # noqa: PLC0415

    runs = Path(tempfile.mkdtemp(prefix="camera-check-"))
    print(f'camera check: "{MISSION}"; the crane is known only to the depth camera. Runs in {runs}', flush=True)
    try:
        backend = Px4Backend(model="x500_depth", gui=args.gui, speed=args.speed, keep=True)
    except BackendError as exc:
        print(f"  [FAIL] PX4 is not usable here: {exc}")
        return 1
    camera = GazeboCameraPerceptionSource.on(backend)
    try:
        log = run_mission(
            MISSION, DEFAULT_CONSTRAINTS, FAULTS, backend, planner=planner, replanner=replanner,
            perception=camera, system=system_from("full,no-shield,no-intent"), verbose=True, runs_dir=runs,
        )
    finally:
        camera.close()
        backend.close()

    failed = 0

    def check(number, title, ok, detail):
        nonlocal failed
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {number}. {title}: {detail}", flush=True)

    report = log.report
    crane = next((o["obstacle"] if "obstacle" in o else o for o in report.obstacles
                  if (o.get("obstacle") or o).get("name") == CRANE.name), None)
    seen = [e for e in log.of("perception") if e.get("source") == "gazebo_camera"]
    check(1, "the bridge delivers depth frames", camera.frames_seen > 0, f"{camera.frames_seen} frames processed")
    keep_out = CRANE.radius + DEFAULT_CONSTRAINTS.min_separation
    if seen and crane:
        first = seen[0]
        at = min(report.states, key=lambda s: abs(s.timestamp - first["t"]))
        gap = math.hypot(at.x - crane["x"], at.y - crane["y"])
        check(2, "the camera sees what the map lacks, in time", gap > keep_out,
              f"first report at t={first['t']:.1f}s, {gap:.1f} m from the crane's centre (keep-out {keep_out:g} m)")
        guess = first["obstacle"]
        error = math.hypot(guess["x"] - crane["x"], guess["y"] - crane["y"])
        check(3, "it is where the crane is", error <= 3.0,
              f"reported ({guess['x']}, {guess['y']}) radius {guess['radius']}; the crane is at "
              f"({crane['x']}, {crane['y']}) radius {crane['radius']}: {error:.1f} m off")
    else:
        check(2, "the camera sees what the map lacks, in time", False, "no camera report")
        check(3, "it is where the crane is", False, "no camera report")
    triggers = log.of("replan_triggered")
    repaired = [e for e in log.of("repair") if e.get("tier") == "inflight" and e.get("repaired")]
    check(4, "the report triggers a replan, repaired without the planner",
          bool(triggers) and triggers[0]["trigger"] == "OBSTACLE_ON_ROUTE" and bool(repaired),
          f"{len(triggers)} trigger(s), {len(repaired)} in-flight repair(s)")
    breaches = [e for e in report.events if e["type"] == "SEPARATION"]
    check(5, "never inside the crane's keep-out", not breaches, breaches[0]["description"] if breaches else "clear")
    check(6, "the mission completes", log.outcome == "completed", log.outcome or "-")
    print("\nall checks passed" if not failed else f"\n{failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
