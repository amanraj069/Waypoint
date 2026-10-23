"""Running a mission for an HTTP client, one event at a time.

This is `scripts/explore.py`'s `fly()` turned inside out. The CLI prints each
stage as it happens; here each stage is appended to a list and anybody watching
is woken up, so a client can follow a mission it did not start, join one
already in flight, and see everything it missed.

A mission runs on its own thread because both halves of it block: planning is a
network call to Gemini and flying is arithmetic. Nothing here touches asyncio,
so the same code is testable without a server.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from waypoint.api.catalog import PLAN_WORLDS, load_world, plan_path
from waypoint.api.settings import FAULTS_DIR, MAX_CONCURRENT_MISSIONS, MISSION_HISTORY, RUNS_DIR
from waypoint.orchestrator.loop import MissionReport
from waypoint.orchestrator.runlog import write_run
from waypoint.scenario import constraints_for
from waypoint.planner import (
    DEFAULT_CONSTRAINTS,
    GeoPoint,
    MissionConstraints,
    Plan,
    PlannerError,
    plan as make_plan,
    replan as make_replan,
)
from waypoint.sim import Executor, FaultSchedule, SimConfig, Wind
from waypoint.verifier import DEFAULT_CONFIG, verify
from waypoint.verifier.repair import repair
from waypoint.world.landing import Site, find as find_site, survey as survey_site
from waypoint.world.models import World

# The stages a mission passes through, in order. A client draws these as a
# progress rail, so they are part of the API rather than decoration.
STAGES = ("queued", "planning", "verifying", "replanning", "flying", "done", "failed")

#: How much clearance to leave above the tallest thing in a world when the
#: ceiling has to be raised to make the place flyable at all.


class MissionError(RuntimeError):
    """The mission cannot be run as asked -- a bad world, a missing plan."""


@dataclass
class MissionRequest:
    """What to fly, and what to fly it through."""

    mission: str = ""
    plan_id: str | None = None
    world: str | None = None
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    turbulence: float = 0.0
    faults: str = "none"
    retries: int = 2
    max_altitude: float | None = None
    blind: bool = False
    #: Where the aircraft stands, in metres east/north of the world's origin.
    #: The ground under it decides the altitude, so this is only (x, y).
    home: tuple[float, float] | None = None
    #: Where it should put down at the end. Defaults to where it started.
    landing: tuple[float, float] | None = None

    def label(self) -> str:
        return self.mission or (self.plan_id or "mission")


@dataclass
class Mission:
    """One run of the pipeline, and everything anybody watching needs to see."""

    id: str
    request: MissionRequest
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    run_id: str | None = None
    error: str | None = None
    cancelled: bool = False
    _lock: threading.Condition = field(default_factory=threading.Condition, repr=False)

    # -- writing ------------------------------------------------------------

    def emit(self, kind: str, message: str = "", **payload: Any) -> None:
        event = {
            "seq": len(self.events),
            "t": round(time.time() - self.created_at, 3),
            "kind": kind,
            "message": message,
            **payload,
        }
        with self._lock:
            self.events.append(event)
            self._lock.notify_all()

    def set_status(self, status: str, message: str = "", **payload: Any) -> None:
        self.status = status
        self.emit("stage", message or status, stage=status, **payload)

    def finish(self, status: str, message: str = "", **payload: Any) -> None:
        self.status = status
        self.emit("end", message or status, stage=status, run=self.run_id, **payload)

    # -- reading ------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed")

    def snapshot(self) -> dict:
        with self._lock:
            events = list(self.events)
        return {
            "id": self.id,
            "status": self.status,
            "mission": self.request.mission,
            "plan_id": self.request.plan_id,
            "world": self.request.world,
            "run": self.run_id,
            "error": self.error,
            "created_at": self.created_at,
            "events": events,
        }

    def follow(self, start: int = 0, timeout: float = 15.0) -> Iterator[dict]:
        """Every event from `start` on, blocking until the mission ends.

        A late subscriber gets the whole history first, so opening the page
        halfway through a flight is not a different experience from starting it.
        """
        index = start
        while True:
            with self._lock:
                while index >= len(self.events) and not self.finished:
                    if not self._lock.wait(timeout):
                        yield {"kind": "heartbeat", "seq": index}
                pending = self.events[index:]
                index = len(self.events)
            for event in pending:
                yield event
            if not pending and self.finished:
                return


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def fence_around(world: World, margin: float = 20.0) -> list[GeoPoint]:
    """A geofence that contains the whole world, inset by a margin."""
    minx, miny, maxx, maxy = world.bounds
    return [
        GeoPoint(x=minx + margin, y=miny + margin),
        GeoPoint(x=maxx - margin, y=miny + margin),
        GeoPoint(x=maxx - margin, y=maxy - margin),
        GeoPoint(x=minx + margin, y=maxy - margin),
    ]


def site_for(world: World | None, point: tuple[float, float] | None, role: str) -> Site:
    """Survey a pad the user picked, and move it to usable ground if it has to.

    Everything downstream has to agree on these points: the planner writes a
    plan that starts and ends on them, the verifier measures the first leg from
    the launch pad and checks the ground under the landing, and the simulator
    puts the aircraft on them. So they are decided once, here.
    """
    x, y = point or (0.0, 0.0)
    if point is None:
        # The default pad is the origin, which every world keeps clear.
        return survey_site(world, float(x), float(y))
    site = find_site(world, float(x), float(y))
    if not site.usable:
        raise MissionError(
            f"the aircraft cannot {role} at ({x:.0f}, {y:.0f}): {site.why_not()}, "
            "and there is nowhere within 120 m that would do. Pick another spot."
        )
    return site


def home_for(world: World | None, home: tuple[float, float] | None) -> tuple[float, float, float]:
    """Backwards-compatible shorthand for the launch point."""
    return site_for(world, home, "take off").position




def load_faults(spec: str, plan: Plan) -> FaultSchedule:
    """'none', 'sample', a name in data/faults, or a schedule JSON path."""
    if not spec or spec == "none":
        return FaultSchedule()
    if spec == "sample":
        from waypoint.scenario import sample_schedule  # noqa: PLC0415 -- shared with the CLI

        return sample_schedule(plan)
    candidate = FAULTS_DIR / f"{Path(spec).name}.json"
    path = candidate if candidate.exists() else Path(spec)
    if not path.exists():
        raise MissionError(f"no fault schedule called '{spec}'")
    return FaultSchedule.model_validate_json(path.read_text())


def plan_summary(plan: Plan) -> list[dict]:
    """The decomposition, flattened the same way the run file flattens it."""
    phases, index = [], 0
    for subtask in plan.subtasks:
        waypoints = []
        for waypoint in subtask.waypoints:
            waypoints.append(
                {
                    "index": index,
                    "x": waypoint.x,
                    "y": waypoint.y,
                    "z": waypoint.z,
                    "speed": waypoint.speed,
                    "action": waypoint.action.value,
                }
            )
            index += 1
        phases.append({"name": subtask.name, "waypoints": waypoints})
    return phases


def violation_records(result) -> list[dict]:
    return [
        {
            "type": v.type.value,
            "severity": v.severity.value,
            "waypoint_index": v.waypoint_index,
            "description": v.description,
        }
        for v in result.violations
    ]


def read_intent(text, world, home, landing, constraints):
    """The mission's requirements: read by the LLM when one is configured, else by rules."""
    from waypoint.intent import extract_intent, heuristic_intent  # noqa: PLC0415
    from waypoint.planner.config import get_providers  # noqa: PLC0415

    if get_providers():
        try:
            return extract_intent(text, world, home=home, landing=landing, constraints=constraints)
        except Exception:  # noqa: BLE001 -- the mission still flies on safety alone
            pass
    return heuristic_intent(text, world, landing=landing)


def run_mission(mission: Mission) -> None:
    """Plan it, check it, fly it, write it. Every step reports as it happens."""
    request = mission.request
    world = None
    if request.world:
        try:
            world = load_world(request.world)
        except KeyError:
            raise MissionError(f"no world called '{request.world}'") from None

    constraints, notes = constraints_for(world, request.max_altitude)
    if (request.home or request.landing) and request.plan_id:
        # A bundled plan's waypoints are absolute, and its last one lands at the
        # pad it was written for. Flying it from somewhere else is not a
        # translation, it is a different mission -- so say so rather than
        # failing verification and looping through corrections.
        raise MissionError(
            "a bundled plan already starts and ends where it was written; to choose "
            "where it takes off or lands, write the mission in English instead"
        )

    launch = site_for(world, request.home, "take off")
    home = launch.position
    landing = site_for(world, request.landing, "land") if request.landing else launch
    if request.home:
        notes = [*notes, f"launching from {launch.describe()}"]
    if request.landing:
        notes = [*notes, f"landing at {landing.describe()}"]
    mission.emit(
        "world",
        f"flying in {world.name}" if world else "flying in open space",
        world=world.summary() if world else None,
        landmarks=[lm.model_dump(mode="json") for lm in world.landmarks] if world else [],
        constraints=constraints.model_dump(mode="json"),
        home=list(home),
        landing=list(landing.position),
        notes=notes,
    )
    # One launch point, shared by all three stages.
    verifier_config = DEFAULT_CONFIG.model_copy(update={"home": home})

    # -- what the mission asks for ------------------------------------------
    goal = None
    if not request.plan_id and request.mission.strip():
        goal = read_intent(request.mission, world, home, landing.position, constraints)
        mission.emit(
            "intent", goal.describe(),
            targets=[t.model_dump(mode="json") for t in goal.targets], ordered=goal.ordered,
            avoid=[a.model_dump(mode="json") for a in goal.avoid], notes=goal.notes,
        )
        from waypoint.intent import assess  # noqa: PLC0415

        reasons = assess(goal, constraints=constraints, world=world, home=home, landing=landing.position)
        if reasons:
            raise MissionError("the mission cannot be flown as asked: " + "; ".join(reasons))
    asks = {"intent": goal} if goal is not None else {}

    # -- plan ---------------------------------------------------------------
    if request.plan_id:
        path = plan_path(request.plan_id)
        if not path.exists():
            raise MissionError(f"no saved plan called '{request.plan_id}'")
        plan = Plan.model_validate_json(path.read_text())
        mission.set_status("planning", f"loaded the saved plan '{request.plan_id}'")
    else:
        if not request.mission.strip():
            raise MissionError("a mission needs either some text or a saved plan")
        mission.set_status("planning", "asking the planner to decompose the mission")
        try:
            plan = make_plan(
                request.mission, constraints, world=world, home=home,
                landing=landing.position if request.landing else None, **asks,
            )
        except PlannerError as exc:
            raise MissionError(f"planning failed: {exc}") from exc
    mission.emit("plan", f"{len(plan.subtasks)} phase(s)", phases=plan_summary(plan))

    # -- verify, and let the planner correct itself -------------------------
    mission.set_status("verifying", "checking the plan against the envelope and the world")
    result = verify(plan, constraints, config=verifier_config, world=world, intent=goal)
    mission.emit(
        "verification",
        "passed" if result.passed else "rejected",
        passed=result.passed,
        violations=violation_records(result),
        attempt=0,
    )

    for attempt in range(1, request.retries + 1):
        if result.passed or mission.cancelled:
            break
        # Geometry first: what can be fixed without asking the planner again is fixed.
        fixed = repair(plan, constraints, world=world, config=verifier_config, intent=goal, result=result)
        mission.emit("repair", "; ".join(fixed.actions) or "nothing it could fix", repaired=fixed.repaired,
                     actions=fixed.actions)
        if fixed.repaired:
            plan, result = fixed.plan, fixed.result
            mission.emit("plan", f"{len(plan.subtasks)} phase(s), repaired", phases=plan_summary(plan), attempt=attempt)
            mission.emit("verification", "passed", passed=True, violations=violation_records(result), attempt=attempt)
            break
        mission.set_status(
            "replanning",
            f"handing the violations back to the planner ({attempt}/{request.retries})",
            attempt=attempt,
        )
        try:
            plan = make_replan(
                request.mission, plan, result.by_subtask([s.name for s in plan.subtasks for _ in s.waypoints]),
                constraints, world=world, home=home,
                landing=landing.position if request.landing else None, **asks,
            )
        except PlannerError as exc:
            mission.emit("note", f"the planner could not correct it: {exc}")
            break
        mission.emit("plan", f"{len(plan.subtasks)} phase(s)", phases=plan_summary(plan), attempt=attempt)
        result = verify(plan, constraints, config=verifier_config, world=world, intent=goal)
        mission.emit(
            "verification",
            "passed" if result.passed else "still rejected",
            passed=result.passed,
            violations=violation_records(result),
            attempt=attempt,
        )

    if not result.passed:
        raise MissionError(
            "the verifier rejected the plan and the planner could not correct it; "
            "nothing was flown"
        )

    # -- fly ----------------------------------------------------------------
    wind = Wind(vector=request.wind, turbulence=request.turbulence)
    config = SimConfig(wind_compensation=0.0 if request.blind else 0.75, home=home)
    faults = load_faults(request.faults, plan)
    mission.set_status(
        "flying",
        "handing the verified plan to the simulator",
        wind=list(request.wind),
        turbulence=request.turbulence,
        faults=[event.model_dump(mode="json") for event in faults.events],
    )

    executor = Executor(
        plan, constraints=constraints, config=config, wind=wind, faults=faults, world=world, shield=True,
    )
    report = MissionReport(
        mission_text=request.mission or plan.mission_text or request.label(),
        plan=plan,
        verification=result,
        constraints=constraints,
        sim_config=config,
        wind=wind,
        faults=faults,
        world=world,
    )

    total_waypoints = len(plan.waypoints())
    next_report = 0.0
    for state in executor.run():
        if mission.cancelled:
            break
        report.states.append(state)
        # A tick is 0.1 s; reporting every one of them would swamp the client.
        # Once a simulated second is plenty to watch.
        if state.timestamp >= next_report:
            mission.emit(
                "telemetry",
                "",
                t=round(state.timestamp, 1),
                x=round(state.x, 1),
                y=round(state.y, 1),
                z=round(state.z, 1),
                speed=round(state.velocity, 2),
                battery=round(state.battery_pct, 1),
                subtask=state.subtask,
                waypoint=state.target_index,
                of=total_waypoints,
                agl=round(state.agl, 1),
                heading=round(state.heading, 0),
                faults=list(state.active_faults),
            )
            next_report = state.timestamp + 1.0

    report.events = executor.event_records()
    report.obstacles = executor.obstacle_records()
    report.distance_flown = executor.distance_flown
    report.crashed = executor.crashed
    for record in report.events:
        mission.emit("flight", record["description"], t=record["t"], type=record["type"])

    if mission.cancelled:
        raise MissionError("cancelled before the flight finished")

    run_file = write_run(report, runs_dir=RUNS_DIR)
    mission.run_id = run_file.stem
    outcome = "landed" if report.completed else "crashed" if report.crashed else "did not land cleanly"
    mission.finish(
        "done",
        f"{outcome}, {report.duration:.0f} s, {report.distance_flown:.0f} m, "
        f"{report.battery_remaining:.0f}% battery left",
        outcome=outcome,
        duration_s=round(report.duration, 1),
        distance_m=round(report.distance_flown, 1),
        battery_pct=round(report.battery_remaining, 1),
        completed=report.completed,
        crashed=report.crashed,
    )


# ---------------------------------------------------------------------------
# The register of missions this process knows about
# ---------------------------------------------------------------------------


class Missions:
    """Every mission started since the server came up, and how to start more."""

    def __init__(self, limit: int = MISSION_HISTORY) -> None:
        self._missions: dict[str, Mission] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_MISSIONS)
        self._counter = itertools.count(1)
        self.limit = limit

    def start(self, request: MissionRequest) -> Mission:
        # `plan_id` implies the world the plan was written for, unless the
        # caller insisted on another one -- which is allowed, and usually ends
        # in a rejected plan, which is a perfectly good thing to be able to see.
        if request.plan_id and not request.world:
            request.world = PLAN_WORLDS.get(request.plan_id)

        mission = Mission(id=f"m{next(self._counter)}-{uuid.uuid4().hex[:6]}", request=request)
        with self._lock:
            self._missions[mission.id] = mission
            self._order.append(mission.id)
            self._evict()

        threading.Thread(target=self._run, args=(mission,), daemon=True,
                         name=f"mission-{mission.id}").start()
        return mission

    def _run(self, mission: Mission) -> None:
        acquired = self._slots.acquire(timeout=0.0)
        if not acquired:
            mission.emit("note", "waiting for another mission to finish")
            self._slots.acquire()
        try:
            run_mission(mission)
        except MissionError as exc:
            mission.error = str(exc)
            mission.finish("failed", str(exc))
        except Exception as exc:  # a bug must not take the server with it
            mission.error = f"{type(exc).__name__}: {exc}"
            mission.emit("note", traceback.format_exc(limit=4))
            mission.finish("failed", mission.error)
        finally:
            self._slots.release()

    def get(self, mission_id: str) -> Mission:
        with self._lock:
            try:
                return self._missions[mission_id]
            except KeyError:
                raise KeyError(mission_id) from None

    def list(self) -> list[dict]:
        with self._lock:
            missions = [self._missions[i] for i in reversed(self._order)]
        return [
            {
                "id": m.id,
                "status": m.status,
                "mission": m.request.mission or m.request.plan_id,
                "world": m.request.world,
                "run": m.run_id,
                "error": m.error,
                "created_at": m.created_at,
            }
            for m in missions
        ]

    def cancel(self, mission_id: str) -> Mission:
        mission = self.get(mission_id)
        mission.cancelled = True
        mission.emit("note", "cancelling")
        return mission

    def _evict(self) -> None:
        """Forget the oldest finished missions once the history is full."""
        while len(self._order) > self.limit:
            for mission_id in list(self._order):
                if self._missions[mission_id].finished:
                    self._order.remove(mission_id)
                    del self._missions[mission_id]
                    break
            else:
                return  # nothing finished yet; let the history run long


MISSIONS = Missions()
