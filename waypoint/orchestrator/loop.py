"""The closed loop: plan, verify, fly, notice, replan, and fly on.

    PLANNING -> VERIFYING (-> REPLANNING -> RE-VERIFYING)* -> EXECUTING
        -> every tick: perception -> TelemetrySnapshot -> needs_replan(history)
        -> REPLAN TRIGGERED -> summarize_for_llm() -> replan() from here
        -> RE-VERIFYING (with its own correction loop) -> RESUMING -> ...
        -> LANDED, or a hard failure: too many replans, no safe plan, battery flat

The orchestrator touches the aircraft only through `ExecutorBackend`, so a
PX4/Gazebo backend drops in without a line of this changing. Everything it
decides goes into a `MissionLog` (JSON Lines, next to the run file); the flight
itself goes into the usual run file, which Gazebo replays.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from waypoint.backends import ExecutorBackend, make_backend
from waypoint.feedback import FeedbackConfig, ReplanTrigger, TelemetrySnapshot, TriggerKind
from waypoint.feedback import PerceptionEvent, PerceptionSource, SymbolicPerceptionSource
from waypoint.feedback import needs_replan, summarize_for_llm
from waypoint.orchestrator.mission_log import MissionLog
from waypoint.orchestrator.runlog import RUNS_DIR, run_path, write_run
from waypoint.planner import DEFAULT_CONSTRAINTS, MissionConstraints, Plan, PlannerError
from waypoint.planner import plan as default_planner
from waypoint.planner import replan as default_replanner
from waypoint.planner.models import Action, Obstacle, SubTask, Waypoint
from waypoint.sim import NO_FAULTS, FaultSchedule, SimConfig, UAVState, Wind
from waypoint.verifier import VerificationResult, verify
from waypoint.verifier.models import DEFAULT_CONFIG as VERIFIER_DEFAULTS
from waypoint.world.models import World

#: planner(mission, constraints, *, world=...) -> Plan
Planner = Callable[..., Plan]
#: replanner(mission, previous_plan, violations, constraints, *, world=, home=, landing=, situation=) -> Plan
Replanner = Callable[..., Plan]

# Sentinel so the default is resolved against the module attribute at call time
# (a plain default would bind RUNS_DIR at import and ignore reconfiguration).
_UNSET: Any = object()

# The simulator ticks at 10 Hz; printing every tick would bury the operator.
PRINT_EVERY_SECONDS = 1.0
FAILSAFE_DESCENT = 1.5


@dataclass
class MissionReport:
    """The flight behind a mission: every tick, and what the backend recorded."""

    mission_text: str
    plan: Plan
    verification: VerificationResult
    constraints: MissionConstraints | None = None
    states: list[UAVState] = field(default_factory=list)
    replans: int = 0
    run_file: Path | None = None
    events: list[dict] = field(default_factory=list)
    obstacles: list[dict] = field(default_factory=list)
    sim_config: SimConfig | None = None
    wind: Wind | None = None
    faults: FaultSchedule | None = None
    world: World | None = None
    distance_flown: float = 0.0
    crashed: bool = False
    #: Every plan that flew, in order: when it took over, from where, and why.
    plans: list[dict] = field(default_factory=list)
    backend: dict | None = None
    outcome: str | None = None

    @property
    def completed(self) -> bool:
        """Landed under control: on the ground, stopped, and not wrecked."""
        if self.crashed or not self.states:
            return False
        last = self.states[-1]
        return last.z <= last.ground + 1e-6 and last.velocity == 0.0

    @property
    def duration(self) -> float:
        return self.states[-1].timestamp if self.states else 0.0

    @property
    def battery_remaining(self) -> float:
        return self.states[-1].battery_pct if self.states else 0.0


def format_state(state: UAVState) -> str:
    line = (
        f"t={state.timestamp:6.1f}s  "
        f"pos=({state.x:7.2f}, {state.y:7.2f}, {state.z:6.2f})  "
        f"vel={state.velocity:5.2f} m/s  "
        f"hdg={state.heading:6.1f}deg  "
        f"batt={state.battery_pct:5.1f}%"
    )
    if state.active_faults:
        line += f"  [{', '.join(state.active_faults)}]"
    if state.gps_error > 0.5:
        line += f"  gps_err={state.gps_error:.1f}m"
    return line


class _Console:
    """The live account of the mission: one banner per phase, telemetry sampled."""

    def __init__(self, verbose: bool, print_every: float):
        self.verbose = verbose
        self.print_every = print_every
        self._next = 0.0

    def say(self, text: str = "") -> None:
        if self.verbose:
            print(text, flush=True)

    def phase(self, name: str, detail: str = "") -> None:
        self.say(f"\n==> {name}" + (f"  {detail}" if detail else ""))

    def plan(self, plan: Plan) -> None:
        index = 0
        for subtask in plan.subtasks:
            self.say(f"    {subtask.name}")
            for wp in subtask.waypoints:
                self.say(f"      [{index}] ({wp.x:g}, {wp.y:g}, {wp.z:g}) @ {wp.speed:g} m/s  {wp.action.value}")
                index += 1

    def event(self, event: dict) -> None:
        self.say(f"    t={event['t']:7.1f}s  {event['type']:14} {event['description']}")

    def tick(self, state: UAVState, force: bool = False) -> None:
        if force or state.timestamp >= self._next:
            self.say(f"    {format_state(state)}")
            self._next = state.timestamp + self.print_every


def run_mission(
    mission_text: str,
    constraints: MissionConstraints | None = None,
    fault_schedule: FaultSchedule | None = None,
    backend: ExecutorBackend | None = None,
    *,
    world: World | None = None,
    home: tuple[float, float, float] | None = None,
    landing: tuple[float, float, float] | None = None,
    planner: Planner | None = None,
    replanner: Replanner | None = None,
    max_replan_attempts: int = 3,
    max_total_replans: int = 3,
    check_every: int = 1,
    feedback_config: FeedbackConfig | None = None,
    perception: PerceptionSource | None = None,
    verbose: bool = True,
    print_every: float = PRINT_EVERY_SECONDS,
    runs_dir: Path | None = _UNSET,
) -> MissionLog:
    """Fly one mission through the whole closed loop, and log every decision.

    `home` is the launch pad and `landing` where it must finish (both default
    to the origin); the planner, the verifier and the backend all use them.
    `max_replan_attempts` bounds the corrections asked for after each verifier
    rejection (on the ground and in the air alike); `max_total_replans` bounds
    how often the flight may be interrupted. `check_every` is how many ticks go
    by between feedback checks. `perception` is how the aircraft sees what is
    around it; it defaults to `SymbolicPerceptionSource` over the backend.
    `planner` and `replanner` default to the LLM and are injectable, so the
    loop runs deterministically in tests.

    Returns the `MissionLog`; its `report` holds the flight, and unless
    `runs_dir` is None the run file and the JSONL log are written there.
    """
    constraints = constraints or DEFAULT_CONSTRAINTS
    faults = fault_schedule or NO_FAULTS
    backend = backend or make_backend("custom", world=world)
    planner = planner or default_planner
    replanner = replanner or default_replanner
    perception = perception or SymbolicPerceptionSource(backend)
    runs_dir = RUNS_DIR if runs_dir is _UNSET else runs_dir
    when = datetime.now()

    log_path = run_path(mission_text, when, runs_dir).with_suffix(".log.jsonl") if runs_dir else None
    mlog = MissionLog(log_path)
    out = _Console(verbose, print_every)
    ids = itertools.count()

    mlog.add(
        "mission_started",
        mission=mission_text,
        backend=backend.name,
        backend_detail=backend.describe(),
        perception=perception.name,
        world=world.name if world else None,
        home=home,
        landing=landing,
        constraints=constraints,
        faults=faults.events,
        max_replan_attempts=max_replan_attempts,
        max_total_replans=max_total_replans,
    )
    out.say(f'Mission: "{mission_text}"  (backend: {backend.name})')

    def settle(
        candidate: Plan,
        source: str,
        against: MissionConstraints,
        ask_again: Callable[[Plan, list[str]], Plan],
        *,
        t: float | None = None,
        start: tuple | None = None,
        home: tuple | None = None,
        start_label: str = "home",
    ) -> tuple[Plan, bool, VerificationResult, int]:
        """Verify, and ask for corrections until it passes or the attempts run out.

        Returns the last plan seen, whether it passed, its verdict and its id.
        """
        config = (
            VERIFIER_DEFAULTS.model_copy(update={"home": home, "start_label": start_label})
            if home else None
        )
        attempt = 0
        while True:
            plan_id = next(ids)
            mlog.add(
                "plan_proposed",
                plan_id=plan_id,
                source=source,
                attempt=attempt,
                t=t,
                start=start,
                subtasks=[s.name for s in candidate.subtasks],
                waypoints=[_waypoint_record(wp) for wp in candidate.waypoints()],
            )
            out.plan(candidate)
            out.phase("VERIFYING" if attempt == 0 else "RE-VERIFYING", f"plan {plan_id}")
            result = verify(candidate, against, config=config, world=world)
            mlog.add(
                "verification",
                plan_id=plan_id,
                attempt=attempt,
                passed=result.passed,
                violations=[v.model_dump(mode="json") for v in result.violations],
            )
            for violation in result.violations:
                out.say(f"    [{violation.severity.value}] [{violation.type.value}] {violation.description}")
            if result.passed:
                out.say("    passed")
                return candidate, True, result, plan_id
            if attempt >= max_replan_attempts:
                out.say(f"    still failing after {max_replan_attempts} correction(s)")
                return candidate, False, result, plan_id
            attempt += 1
            out.phase("REPLANNING", f"correction {attempt}/{max_replan_attempts} after the verifier's rejection")
            if source == "initial":
                source = "verification_replan"
            try:
                candidate = ask_again(candidate, result.messages())
            except PlannerError as exc:
                out.say(f"    planner failed: {exc}")
                return candidate, False, result, plan_id

    # -- 1. a verified plan, on the ground -----------------------------------

    # Only passed when chosen, so a planner that predates pads still works.
    pads = {k: v for k, v in (("home", home), ("landing", landing)) if v is not None}
    out.phase("PLANNING", f'"{mission_text}"')
    try:
        plan = planner(mission_text, constraints, world=world, **pads)
    except PlannerError as exc:
        out.say(f"    planning failed: {exc}")
        mlog.add("mission_finished", outcome="planning_failed", reason=str(exc), backend=backend.name)
        return mlog

    plan, passed, result, plan_id = settle(
        plan,
        "initial",
        constraints,
        lambda previous, violations: replanner(
            mission_text, previous, violations, constraints, world=world, **pads
        ),
        home=home,
    )
    if not passed:
        rejected = MissionReport(mission_text, plan, result, constraints, faults=faults, world=world)
        rejected.backend = backend.describe()
        rejected.outcome = "rejected"
        _finish(mlog, rejected, backend, runs_dir, when, out, reason="no plan passed verification")
        return mlog

    # -- 2. fly it, watching every tick --------------------------------------

    report = MissionReport(mission_text, plan, result, constraints, faults=faults, world=world)
    backend.prepare(constraints, faults, home=home)
    # After prepare: the settings recorded must include the pad it stood on.
    report.backend = backend.describe()
    landing = landing or _landing_of(plan, backend.state)
    report.plans.append(_segment(plan_id, "initial", plan, t=0.0, start=backend.state.position))

    mlog.add("execution_started", plan_id=plan_id, t=0.0)
    out.phase("EXECUTING", f"plan {plan_id}")
    if faults:
        out.say("    faults armed:")
        out.say(faults.describe())

    current, current_constraints = plan, constraints
    route_start = backend.state.reported_position
    feedback_config = feedback_config or FeedbackConfig()
    # Everything known about: what the plan was verified against, then every
    # obstacle perception has reported since, in the order it was first seen.
    known: list[Obstacle] = list(constraints.obstacles)
    # The current plan's recent telemetry, for trends; cleared when a plan takes over.
    history: deque[TelemetrySnapshot] = deque()
    start_battery: float | None = None
    last_battery_replan: float | None = None
    shown = 0  # backend events already printed
    interruptions = 0
    outcome: str | None = None
    reason = ""

    def fly(route: Plan, *, watch: bool) -> ReplanTrigger | None:
        nonlocal start_battery, shown
        for tick, state in enumerate(backend.run(route)):
            report.states.append(state)
            if start_battery is None:
                start_battery = state.battery_pct
            out.tick(state)
            events = backend.event_records()
            for event in events[shown:]:
                out.event(event)
            shown = len(events)
            if not watch or tick % check_every:
                continue
            seen = perception.get_events(state)
            for event in seen:
                if event.obstacle is not None and event.obstacle not in known:
                    known.append(event.obstacle)
                    mlog.add("perception", **event.as_record())
                    out.event({"t": event.t, "type": "PERCEIVED", "description": event.description})
            history.append(_snapshot(state, route, current_constraints, known, route_start, seen))
            while history[0].t < state.timestamp - feedback_config.history_seconds:
                history.popleft()
            trigger = needs_replan(history, feedback_config, last_battery_replan=last_battery_replan)
            if trigger:
                return trigger
        return None

    def failsafe(why: str) -> None:
        """No safe plan in the air: come straight down where we are."""
        here = backend.state
        # The reported position, because that is what the controller steers on:
        # aiming at the true one under a drifting GPS would fly it sideways.
        x, y, _ = here.reported_position
        mlog.add("failsafe", t=here.timestamp, reason=why, position=[x, y, here.z])
        out.phase("FAILSAFE", f"{why}; landing where it is")
        down = Plan(
            mission_text=mission_text,
            subtasks=[SubTask(name="failsafe_land", waypoints=[
                Waypoint(x=x, y=y, z=here.ground, speed=FAILSAFE_DESCENT, action=Action.LAND)
            ])],
        )
        report.plans.append(_segment(next(ids), "failsafe", down, t=here.timestamp, start=(x, y, here.z), why=why))
        fly(down, watch=False)

    while True:
        trigger = fly(current, watch=True)
        if trigger is None:
            break

        interruptions += 1
        state = backend.state
        mlog.add(
            "replan_triggered",
            # A battery trigger carries its own battery_pct, the same figure.
            **{
                "number": interruptions,
                "position": list(state.reported_position),
                "battery_pct": round(state.battery_pct, 2),
                **trigger.as_record(),
            },
        )
        out.tick(state, force=True)
        out.phase("REPLAN TRIGGERED", f"t={trigger.t:.1f}s  {trigger.reason}")

        if interruptions > max_total_replans:
            outcome, reason = "aborted", f"more than {max_total_replans} in-flight replans"
            failsafe(reason)
            break

        snapshot = history[-1]
        here = snapshot.position
        spent = (start_battery or state.battery_pct) - state.battery_pct
        budget = constraints.battery_budget_pct - spent
        if budget <= 0:
            outcome, reason = "aborted", "the mission's battery budget is spent"
            failsafe(reason)
            break
        in_air = constraints.model_copy(
            update={"obstacles": list(known), "battery_budget_pct": min(100.0, budget)}
        )
        situation = summarize_for_llm(snapshot, trigger)

        def ask(previous: Plan, violations: list[str]) -> Plan:
            return replanner(
                mission_text, previous, violations, in_air,
                world=world, home=here, landing=landing, situation=situation,
            )

        out.phase("REPLANNING", f"from ({here[0]:.1f}, {here[1]:.1f}, {here[2]:.1f}) with {state.battery_pct:.1f}% battery")
        try:
            fresh = ask(current, [])
        except PlannerError as exc:
            outcome, reason = "aborted", f"the planner failed in flight: {exc}"
            failsafe(reason)
            break
        fresh, passed, _, fresh_id = settle(
            fresh, "inflight_replan", in_air, ask,
            t=trigger.t, start=list(here), home=here, start_label="the current position",
        )
        if not passed:
            outcome, reason = "aborted", "no replacement plan passed verification"
            failsafe(reason)
            break

        if trigger.kind is TriggerKind.BATTERY_INSUFFICIENT:
            last_battery_replan = trigger.t
        report.plans.append(
            _segment(fresh_id, "inflight_replan", fresh, t=trigger.t, start=here, why=trigger.reason)
        )
        mlog.add("execution_resumed", plan_id=fresh_id, t=trigger.t)
        out.phase("RESUMING", f"plan {fresh_id}")
        current, current_constraints, route_start = fresh, in_air, here
        history.clear()

    if report.states:
        out.tick(report.states[-1], force=True)
    report.replans = interruptions
    report.events = backend.event_records()
    report.obstacles = backend.obstacle_records()
    report.distance_flown = backend.distance_flown
    report.crashed = backend.crashed
    if backend.crashed:
        outcome, reason = "crashed", reason or "the aircraft hit something"
    elif backend.battery_depleted:
        outcome, reason = "battery_depleted", reason or "the battery ran flat"
    elif outcome is None:
        outcome = "completed" if report.completed else "incomplete"
    report.outcome = outcome
    _finish(mlog, report, backend, runs_dir, when, out, reason=reason)
    return mlog


# ---------------------------------------------------------------------------


def _finish(
    mlog: MissionLog,
    report: MissionReport,
    backend: ExecutorBackend,
    runs_dir: Path | None,
    when: datetime,
    out: _Console,
    *,
    reason: str,
) -> None:
    mlog.report = report
    if runs_dir is not None:
        report.run_file = mlog.run_file = write_run(report, when=when, runs_dir=runs_dir)
    mlog.add(
        "mission_finished",
        outcome=report.outcome,
        reason=reason,
        backend=backend.name,
        t=round(report.duration, 3),
        distance_m=round(report.distance_flown, 1),
        battery_pct=round(report.battery_remaining, 2),
        inflight_replans=report.replans,
        plans_flown=len(report.plans),
        verification_attempts=len(mlog.of("verification")),
        run_file=str(report.run_file) if report.run_file else None,
    )
    headline = {
        "completed": "MISSION COMPLETE",
        "rejected": "MISSION REJECTED",
        "crashed": "MISSION ENDED IN A CRASH",
        "battery_depleted": "BATTERY DEPLETED",
        "aborted": "MISSION ABORTED",
    }.get(report.outcome or "", "MISSION INCOMPLETE")
    landed = "landed after " if report.outcome == "completed" else ""
    out.phase(
        headline,
        f"{landed}{report.duration:.1f}s, {report.distance_flown:.0f} m, {report.battery_remaining:.1f}% battery, "
        f"{report.replans} in-flight replan(s)" + (f"; {reason}" if reason and report.outcome != "completed" else ""),
    )
    if report.run_file:
        out.say(f"    run: {report.run_file}")
    if mlog.path:
        out.say(f"    log: {mlog.path}")


def _snapshot(
    state: UAVState,
    plan: Plan,
    constraints: MissionConstraints,
    known: list[Obstacle],
    route_start: tuple[float, float, float],
    perceived: list[PerceptionEvent],
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        t=state.timestamp,
        position=state.reported_position,
        velocity=state.velocity_vector,
        heading=state.heading,
        battery_pct=state.battery_pct,
        ground=state.ground,
        target_index=state.target_index or 0,
        subtask=state.subtask,
        plan_waypoints=tuple(plan.waypoints()),
        planned_obstacles=tuple(constraints.obstacles),
        known_obstacles=tuple(known),
        active_faults=state.active_faults,
        min_separation=constraints.min_separation,
        route_start=route_start,
        perception=tuple(perceived),
    )


def _landing_of(plan: Plan, pad: UAVState) -> tuple[float, float, float]:
    """Where a replacement plan must still end: the original plan's touchdown."""
    waypoints = plan.waypoints()
    if waypoints and waypoints[-1].action is Action.LAND:
        last = waypoints[-1]
        return (last.x, last.y, last.z)
    return pad.position


def _segment(plan_id: int, source: str, plan: Plan, *, t: float, start, why: str | None = None) -> dict:
    return {
        "plan_id": plan_id,
        "source": source,
        "t": round(t, 3),
        "start": [round(c, 2) for c in start],
        "reason": why,
        "waypoints": [_waypoint_record(wp) for wp in plan.waypoints()],
    }


def _waypoint_record(wp: Waypoint) -> dict:
    return {"x": wp.x, "y": wp.y, "z": wp.z, "speed": wp.speed, "action": wp.action.value}
