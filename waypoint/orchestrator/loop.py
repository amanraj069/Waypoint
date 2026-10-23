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
import math
import time
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
from waypoint.orchestrator.options import CLASSIC, SystemOptions
from waypoint.orchestrator.runlog import RUNS_DIR, run_path, write_run
from waypoint.planner import DEFAULT_CONSTRAINTS, MissionConstraints, Plan, PlannerError
from waypoint.planner import plan as default_planner
from waypoint.planner import replan as default_replanner
from waypoint.planner.models import Action, Obstacle, SubTask, Waypoint
from waypoint.sim import NO_FAULTS, FaultSchedule, SimConfig, UAVState, Wind
from waypoint.verifier import VerificationResult, verify
from waypoint.verifier.models import DEFAULT_CONFIG as VERIFIER_DEFAULTS
from waypoint.verifier.models import ViolationType
from waypoint.verifier.checks import polygon_of
from waypoint.verifier.repair import relaxed, repair, trim_for_battery
from waypoint.world.landing import find as find_site
from waypoint.world.models import World

#: planner(mission, constraints, *, world=...) -> Plan
Planner = Callable[..., Plan]
#: replanner(mission, previous_plan, violations, constraints, *, world=, home=, landing=, situation=) -> Plan
Replanner = Callable[..., Plan]
#: verifier(plan, constraints, *, config=, world=) -> VerificationResult
Verifier = Callable[..., VerificationResult]

# Sentinel so the default is resolved against the module attribute at call time
# (a plain default would bind RUNS_DIR at import and ignore reconfiguration).
_UNSET: Any = object()

# The simulator ticks at 10 Hz; printing every tick would bury the operator.
PRINT_EVERY_SECONDS = 1.0
FAILSAFE_DESCENT = 1.5
#: The most an observed battery drain trend is scaled into an in-flight plan's energy model.
PLANNING_DRAIN_CAP = 1.5
FAILSAFE_TRANSIT = 4.0


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
    #: How much of the mission's intent the flight delivered (`waypoint.intent.evaluate`), when it was read.
    intent: dict | None = None

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
    verifier: Verifier | None = None,
    adapt: bool = True,
    verbose: bool = True,
    print_every: float = PRINT_EVERY_SECONDS,
    runs_dir: Path | None = _UNSET,
    system: SystemOptions | None = None,
    intent: Any = None,
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
    loop runs deterministically in tests. `verifier` defaults to the real
    `verify`; the experiment harness swaps it only to fly an unconstrained
    baseline, and the log records which one gated the mission. `adapt=False`
    flies the plan without the in-flight monitor: nothing replans, so the
    flight is the plan, faults and all (a baseline that costs no planner calls
    once it is airborne).

    `system` switches the loop's capabilities (`orchestrator.options`):
    reading the mission's intent, deterministic repair, tiered in-flight
    replanning, hovering through replans, the runtime shield. It defaults to
    `CLASSIC`, the loop as first published; every entry point passes `FULL`.
    `intent` overrides how the intent is read: a ready `MissionIntent`, or a
    callable `(mission, world, *, home, landing, constraints) -> MissionIntent`.

    Returns the `MissionLog`; its `report` holds the flight, and unless
    `runs_dir` is None the run file and the JSONL log are written there.
    """
    system = system or CLASSIC
    constraints = constraints or DEFAULT_CONSTRAINTS
    faults = fault_schedule or NO_FAULTS
    backend = backend or make_backend("custom", world=world)
    planner = planner or default_planner
    replanner = replanner or default_replanner
    verifier = verifier or verify
    if system.gate != "preflight" and verifier is verify:
        # Nothing is refused before it flies; what the checks would have said is still recorded.
        verifier = recording
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
        verifier=getattr(verifier, "__name__", type(verifier).__name__),
        adapt=adapt,
        world=world.name if world else None,
        home=home,
        landing=landing,
        constraints=constraints,
        faults=faults.events,
        max_replan_attempts=max_replan_attempts,
        max_total_replans=max_total_replans,
        system=system.record(),
    )
    out.say(f'Mission: "{mission_text}"  (backend: {backend.name}, system: {system.label})')
    backend.shield = system.shield

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
        goal=None,
        relax: bool = False,
        aircraft=None,
    ) -> tuple[Plan, bool, VerificationResult, int]:
        """Verify, repair, and ask for corrections until it passes or the attempts run out.

        `goal` is the intent the plan is also checked against. With `relax`
        (in the air), a last candidate that is safe but falls short of the
        intent is accepted, and what the mission gave up is logged.
        `aircraft` replaces the verifier's aircraft model (in the air: the
        monitor's energy figures). Returns the last plan seen, whether it
        passed, its verdict and its id.
        """
        config = (
            (aircraft or VERIFIER_DEFAULTS).model_copy(update={"home": home, "start_label": start_label})
            if home else aircraft
        )
        extra = {"intent": goal} if goal is not None else {}
        attempt = 0
        repaired = False  # repair is tried once per plan the planner proposes
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
            result = verifier(candidate, against, config=config, world=world, **extra)
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
            if system.repair and system.gate == "preflight" and not repaired:
                repaired = True
                began = time.perf_counter()
                fixed = repair(candidate, against, world=world, config=config or VERIFIER_DEFAULTS, intent=goal, result=result)
                mlog.add(
                    "repair", plan_id=plan_id, repaired=fixed.repaired, actions=fixed.actions,
                    ms=round((time.perf_counter() - began) * 1000, 2), t=t,
                )
                if fixed.repaired:
                    out.phase("REPAIRED", f"plan {plan_id}: " + "; ".join(fixed.actions[:4]) + (" ..." if len(fixed.actions) > 4 else ""))
                    candidate, source = fixed.plan, "repair"
                    continue
            if attempt >= max_replan_attempts:
                if relax and goal is not None and relaxed(result).passed:
                    given_up = [v.description for v in result.violations if v.type is ViolationType.INTENT and v.blocking]
                    mlog.add("intent_relaxed", plan_id=plan_id, t=t, given_up=given_up)
                    out.say(f"    safe, but short of the mission: {len(given_up)} requirement(s) given up")
                    return candidate, True, relaxed(result), plan_id
                out.say(f"    still failing after {max_replan_attempts} correction(s)")
                return candidate, False, result, plan_id
            attempt += 1
            repaired = False
            out.phase("REPLANNING", f"correction {attempt}/{max_replan_attempts} after the verifier's rejection")
            if source == "initial":
                source = "verification_replan"
            try:
                phases = [s.name for s in candidate.subtasks for _ in s.waypoints]
                reasons = result.by_subtask(phases) if system.subtask_feedback else result.messages()
                candidate = ask_again(candidate, reasons)
            except PlannerError as exc:
                out.say(f"    planner failed: {exc}")
                return candidate, False, result, plan_id

    # -- 1. a verified plan, on the ground -----------------------------------

    # Only passed when chosen, so a planner that predates pads still works.
    pads = {k: v for k, v in (("home", home), ("landing", landing)) if v is not None}

    goal = None
    if system.intent or intent is not None:
        goal = _read_intent(intent, mission_text, world, home, landing, constraints, mlog, out)
        from waypoint.intent import assess  # noqa: PLC0415

        reasons = assess(goal, constraints=constraints, world=world, home=home, landing=landing or home)
        if reasons:
            mlog.add("infeasible", reasons=reasons)
            out.phase("MISSION INFEASIBLE", "; ".join(reasons))
            refused = MissionReport(
                mission_text, Plan(mission_text=mission_text), VerificationResult(passed=False), constraints,
                faults=faults, world=world,
            )
            refused.backend = backend.describe()
            refused.outcome = "infeasible"
            _finish(mlog, refused, backend, runs_dir, when, out, reason="; ".join(reasons))
            return mlog
    asks = {**pads, **({"intent": goal} if goal is not None else {})}

    out.phase("PLANNING", f'"{mission_text}"')
    try:
        plan = planner(mission_text, constraints, world=world, **asks)
    except PlannerError as exc:
        out.say(f"    planning failed: {exc}")
        mlog.add("mission_finished", outcome="planning_failed", reason=str(exc), backend=backend.name)
        return mlog

    plan, passed, result, plan_id = settle(
        plan,
        "initial",
        constraints,
        lambda previous, violations: replanner(
            mission_text, previous, violations, constraints, world=world, **asks
        ),
        home=home,
        goal=goal,
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

    if system.gate == "runtime":
        plan = clamp_altitudes(plan, constraints)
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
    battery_alarms = 0  # battery triggers so far: a drain that trips the monitor twice is not a blip
    outcome: str | None = None
    reason = ""

    def fly(route: Plan, *, watch: bool) -> ReplanTrigger | None:
        nonlocal start_battery, shown
        guarded = system.gate == "runtime" and watch
        fence = polygon_of(constraints.geofence, "geofence") if guarded else None
        for tick, state in enumerate(backend.run(route)):
            report.states.append(state)
            if start_battery is None:
                start_battery = state.battery_pct
            out.tick(state)
            events = backend.event_records()
            for event in events[shown:]:
                out.event(event)
            shown = len(events)
            if guarded:
                rejected = _guard(route, state, fence)
                if rejected:
                    history.append(_snapshot(state, route, current_constraints, known, route_start, ()))
                    return rejected
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
        """No safe plan in the air: land on the nearest clear ground, or here if it is clear."""
        here = backend.state
        # The reported position, because that is what the controller steers on:
        # aiming at the true one under a drifting GPS would fly it sideways.
        x, y, z = here.reported_position
        waypoints, where = failsafe_route(
            world, (x, y, z), here.ground, known, current_constraints.min_separation
        )
        mlog.add("failsafe", t=here.timestamp, reason=why, position=[x, y, here.z], landing=where)
        out.phase("FAILSAFE", f"{why}; {where}")
        down = Plan(mission_text=mission_text, subtasks=[SubTask(name="failsafe_land", waypoints=waypoints)])
        report.plans.append(_segment(next(ids), "failsafe", down, t=here.timestamp, start=(x, y, here.z), why=why))
        fly(down, watch=False)

    while True:
        trigger = fly(current, watch=adapt)
        if trigger is None:
            break

        interruptions += 1
        battery_alarms += trigger.kind is TriggerKind.BATTERY_INSUFFICIENT
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
            update={
                "obstacles": list(known),
                "battery_budget_pct": min(100.0, budget, state.battery_pct) if system.monitor_energy else min(100.0, budget),
            }
        )
        aircraft = (
            _monitor_aircraft(feedback_config, trigger, persistent=battery_alarms > 1) if system.monitor_energy else None
        )
        situation = summarize_for_llm(snapshot, trigger)
        left = None
        if goal is not None:
            from waypoint.intent import remaining  # noqa: PLC0415

            left = remaining(goal, flown_points(backend), world)
        inflight_asks = {"intent": left} if left is not None else {}

        def ask(previous: Plan, violations: list[str]) -> Plan:
            return replanner(
                mission_text, previous, violations, in_air,
                world=world, home=here, landing=landing, situation=situation, **inflight_asks,
            )

        out.phase("REPLANNING", f"from ({here[0]:.1f}, {here[1]:.1f}, {here[2]:.1f}) with {state.battery_pct:.1f}% battery")
        fresh = None
        if system.replan == "tiered" and system.gate == "preflight":
            # Tier 1, no LLM: repair the rest of the route around what changed,
            # or, short of battery, give up the phases that cost the most.
            config_here = (aircraft or VERIFIER_DEFAULTS).model_copy(update={"home": here, "start_label": "the current position"})
            rest = rest_of(current, state.target_index or 0)
            began = time.perf_counter()
            short = trigger.kind is TriggerKind.BATTERY_INSUFFICIENT
            tier = None
            if rest.subtasks:
                tier = (
                    trim_for_battery(rest, in_air, world=world, config=config_here, landing=landing)
                    if short else
                    repair(rest, in_air, world=world, config=config_here, intent=left)
                )
                mlog.add(
                    "repair", tier="inflight", trigger=trigger.kind.value, repaired=tier.repaired,
                    actions=tier.actions, ms=round((time.perf_counter() - began) * 1000, 2), t=trigger.t,
                )
            if tier is not None and tier.repaired:
                out.phase("REPAIRED IN FLIGHT", "; ".join(tier.actions[:4]) + (" ..." if len(tier.actions) > 4 else ""))
                given_up = [a for a in tier.actions if a.startswith(("dropped", "abandoned"))]
                if given_up:
                    mlog.add("intent_relaxed", t=trigger.t, given_up=given_up)
                fresh, passed, _, fresh_id = settle(
                    tier.plan, "inflight_repair", in_air, ask,
                    t=trigger.t, start=list(here), home=here, start_label="the current position",
                    goal=None if short else left, relax=True, aircraft=aircraft,
                )
        if fresh is None:
            began = time.perf_counter()
            try:
                fresh = ask(current, [])
            except PlannerError as exc:
                outcome, reason = "aborted", f"the planner failed in flight: {exc}"
                failsafe(reason)
                break
            fresh, passed, _, fresh_id = settle(
                fresh, "inflight_replan", in_air, ask,
                t=trigger.t, start=list(here), home=here, start_label="the current position",
                goal=left, relax=True, aircraft=aircraft,
            )
            if system.hold == "hover":
                waited = time.perf_counter() - began
                for held in backend.hold(waited):
                    report.states.append(held)
                    out.tick(held)
                if backend.crashed or backend.battery_depleted:
                    break
        if not passed:
            outcome, reason = "aborted", "no replacement plan passed verification"
            failsafe(reason)
            break

        if trigger.kind is TriggerKind.BATTERY_INSUFFICIENT:
            last_battery_replan = trigger.t
        report.plans.append(
            _segment(fresh_id, "inflight_replan", fresh, t=trigger.t, start=here, why=trigger.reason)
        )
        if system.gate == "runtime":
            fresh = clamp_altitudes(fresh, in_air)
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
    if goal is not None:
        from waypoint.intent import evaluate  # noqa: PLC0415

        track = [(s.x, s.y, s.z) for s in report.states]
        last = report.plans[-1]["waypoints"][-1] if report.plans and report.plans[-1]["waypoints"] else None
        planned = (last["x"], last["y"]) if last and last["action"] == "LAND" else None
        report.intent = evaluate(goal, flown_points(backend), track, world, planned_landing=planned)
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
        **({"intent": report.intent} if report.intent is not None else {}),
    )
    headline = {
        "completed": "MISSION COMPLETE",
        "rejected": "MISSION REJECTED",
        "crashed": "MISSION ENDED IN A CRASH",
        "battery_depleted": "BATTERY DEPLETED",
        "aborted": "MISSION ABORTED",
        "infeasible": "MISSION REFUSED",
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


def _read_intent(given, mission_text, world, home, landing, constraints, mlog, out):
    """The mission's intent: as given, from a callable, or read by the LLM (by rules when there is none)."""
    from waypoint.intent import MissionIntent, extract_intent, heuristic_intent  # noqa: PLC0415
    from waypoint.planner.config import get_providers  # noqa: PLC0415

    pad = landing or home
    began = time.perf_counter()
    source = "given"
    try:
        if isinstance(given, MissionIntent):
            goal = given
        elif callable(given):
            source = "callable"
            goal = given(mission_text, world, home=home, landing=pad, constraints=constraints)
        elif get_providers():
            source = "llm"
            goal = extract_intent(mission_text, world, home=home, landing=pad, constraints=constraints)
        else:
            source = "rules"
            goal = heuristic_intent(mission_text, world, landing=pad)
    except Exception as exc:  # noqa: BLE001 -- the mission still flies on safety alone
        mlog.add("intent_error", source=source, error=f"{type(exc).__name__}: {exc}")
        out.say(f"    intent could not be read ({exc}); reading it by rules")
        source = "rules"
        goal = heuristic_intent(mission_text, world, landing=pad)
    if goal.landing is None and pad is not None:
        goal = goal.model_copy(update={"landing": tuple(pad)})
    mlog.add(
        "intent", source=source, ms=round((time.perf_counter() - began) * 1000, 1),
        targets=[t.model_dump(mode="json") for t in goal.targets], ordered=goal.ordered,
        avoid=[a.model_dump(mode="json") for a in goal.avoid], landing=goal.landing, notes=goal.notes,
    )
    out.phase("INTENT", f"read by {source}")
    out.say(goal.describe())
    return goal


def recording(candidate: Plan, constraints, *, config=None, world=None, intent=None) -> VerificationResult:
    """A gate that refuses nothing: every check runs and is recorded, and the plan flies anyway."""
    found = verify(candidate, constraints, config=config, world=world, intent=intent)
    return VerificationResult(passed=True, violations=found.violations)


def clamp_altitudes(plan: Plan, constraints: MissionConstraints) -> Plan:
    """The runtime guard's altitude rule: an out-of-band target is clamped into the band, not refused."""
    clamped = plan.model_copy(deep=True)
    for subtask in clamped.subtasks:
        for wp in subtask.waypoints:
            if wp.action is not Action.LAND:
                wp.z = min(max(wp.z, constraints.min_altitude), constraints.max_altitude)
    return clamped


def _guard(route: Plan, state: UAVState, fence) -> ReplanTrigger | None:
    """The runtime guard's fence rule: the command now being flown is refused if it leaves the geofence."""
    from shapely.geometry import Point  # noqa: PLC0415

    waypoints = route.waypoints()
    index = state.target_index or 0
    if index >= len(waypoints):
        return None
    wp = waypoints[index]
    if fence.covers(Point(wp.x, wp.y)):
        return None
    return ReplanTrigger(
        kind=TriggerKind.COMMAND_REJECTED,
        t=state.timestamp,
        reason=(
            f"the command to fly to waypoint {index} at ({wp.x:g}, {wp.y:g}, {wp.z:g}) was refused: "
            "it is outside the geofence"
        ),
        detail={"waypoint": index},
    )


def _monitor_aircraft(feedback: FeedbackConfig, trigger: ReplanTrigger, *, persistent: bool = False):
    """The verifier's aircraft model, in the monitor's energy terms.

    The monitor costs the rest of a route as metres plus avionics time at
    cruise speed, scaled by the observed drain when the battery is failing,
    and wants a reserve left at touchdown. A plan verified on the nominal
    model could pass and then trip the monitor the moment it flew; verified on
    this one, it cannot.
    """
    ratio = float(trigger.detail.get("drain_ratio") or 1.0)
    if ratio <= feedback.battery_trend_tolerance:
        ratio = 1.0
    # A trend seen over ten seconds is trusted only so far: a cell fault that lasts
    # a few seconds would otherwise be extrapolated over the whole route and the
    # mission given up for nothing. A drain that persists trips the monitor again
    # after its cooldown, with less battery left, and is planned for then.
    if not persistent:
        ratio = min(ratio, PLANNING_DRAIN_CAP)
    per_metre = 1.0 / feedback.meters_per_battery_pct + 1.0 / (feedback.cruise_speed * feedback.seconds_per_battery_pct)
    return VERIFIER_DEFAULTS.model_copy(update={
        "meters_per_battery_pct": 1.0 / (per_metre * ratio),
        "hover_seconds_per_battery_pct": VERIFIER_DEFAULTS.hover_seconds_per_battery_pct / ratio,
        "reserve_battery_pct": feedback.battery_reserve_pct,
    })


def flown_points(backend: ExecutorBackend):
    """The waypoints the aircraft has actually reached, in order, where it reached them."""
    from waypoint.intent import RoutePoint  # noqa: PLC0415

    points = []
    for event in backend.event_records():
        if event.get("type") != "WAYPOINT":
            continue
        payload = event.get("payload") or {}
        action = payload.get("action")
        if action is None:  # an older backend: the action is in the description
            text = event.get("description", "")
            action = next((a.value for a in Action if f"({a.value})" in text), "MOVE")
        x, y, z = payload.get("position") or (0.0, 0.0, 0.0)
        points.append(RoutePoint(len(points), "", x, y, z, Action(action)))
    return points


def rest_of(plan: Plan, index: int) -> Plan:
    """The plan from waypoint `index` on, its phases kept."""
    subtasks: list[SubTask] = []
    position = 0
    for subtask in plan.subtasks:
        kept = [wp for k, wp in enumerate(subtask.waypoints, position) if k >= index]
        position += len(subtask.waypoints)
        if kept:
            subtasks.append(SubTask(name=subtask.name, waypoints=kept))
    return Plan(mission_text=plan.mission_text, subtasks=subtasks)


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


#: How far above everything in the way the failsafe crosses to its landing spot.
FAILSAFE_CLEARANCE = 10.0


def failsafe_route(
    world: World | None,
    position: tuple[float, float, float],
    ground: float,
    obstacles: list[Obstacle],
    separation: float = 0.0,
) -> tuple[list[Waypoint], str]:
    """Where an emergency landing goes, and how it gets there.

    Straight down when the ground below is flat and clear. Otherwise to the
    nearest spot that is (the same survey the launch and landing pads get),
    crossing at a height that clears every roof, hillside and known obstacle
    on the way. Coming straight down onto a roof or a crane is a crash, not a
    landing. If nothing within reach will do, it comes down where it is.
    """
    x, y, z = position
    site = find_site(world, x, y, avoid=obstacles, clearance=separation)
    if not site.usable:
        return [Waypoint(x=x, y=y, z=ground, speed=FAILSAFE_DESCENT, action=Action.LAND)], (
            f"no clear ground within reach ({site.why_not()}); landing where it is"
        )
    if not site.moved:
        return [Waypoint(x=x, y=y, z=site.elevation, speed=FAILSAFE_DESCENT, action=Action.LAND)], (
            "landing where it is"
        )
    cruise = max(z, _highest_between(world, (x, y), (site.x, site.y), obstacles) + FAILSAFE_CLEARANCE)
    route = []
    if cruise > z + 0.5:
        route.append(Waypoint(x=x, y=y, z=cruise, speed=FAILSAFE_DESCENT))  # climb clear first
    route += [
        Waypoint(x=site.x, y=site.y, z=cruise, speed=FAILSAFE_TRANSIT),
        Waypoint(x=site.x, y=site.y, z=site.elevation, speed=FAILSAFE_DESCENT, action=Action.LAND),
    ]
    return route, f"landing on clear ground {site.moved:.0f} m away at ({site.x:.0f}, {site.y:.0f})"


def _highest_between(world: World | None, a, b, obstacles: list[Obstacle], every: float = 4.0) -> float:
    """The highest roof, ground or obstacle top along the straight line a-b."""
    steps = max(1, int(math.dist(a, b) / every))
    points = [(a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps) for i in range(steps + 1)]
    highest = max(world.surface_at(px, py) for px, py in points) if world else 0.0
    for obstacle in obstacles:
        near = min(math.dist((obstacle.x, obstacle.y), p) for p in points) <= obstacle.radius + every
        if near and obstacle.height is not None:
            highest = max(highest, obstacle.height)
    return highest


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
