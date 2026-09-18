"""Mission orchestration: plan -> verify -> execute -> feedback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

from feedback import observe
from planner import DEFAULT_CONSTRAINTS, MissionConstraints, Plan
from planner import plan as default_planner
from sim import Executor, FaultSchedule, SimConfig, UAVState, Wind
from world.models import World
from orchestrator.runlog import RUNS_DIR, write_run
from verifier import VerificationResult, verify

Planner = Callable[[str, MissionConstraints], Plan]

# Sentinel so the default is resolved against the module attribute at call time
# (a plain default would bind RUNS_DIR at import and ignore reconfiguration).
_UNSET: Any = object()

# The simulator ticks at 10 Hz; printing every tick would bury the operator.
PRINT_EVERY_SECONDS = 1.0


@dataclass
class MissionReport:
    """Outcome of one run_mission call."""

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


def run_mission(
    mission_text: str,
    constraints: MissionConstraints | None = None,
    verbose: bool = True,
    planner: Planner | None = None,
    runs_dir: Path | None = _UNSET,
    *,
    wind: Wind | None = None,
    faults: FaultSchedule | None = None,
    sim_config: SimConfig | None = None,
    world: World | None = None,
    print_every: float = PRINT_EVERY_SECONDS,
) -> MissionReport:
    """Run one mission end to end through the real planner, verifier and simulator.

    `planner` is injectable so tests (and future replanning experiments) can
    drive the loop without calling the real API; it defaults to the LLM planner.
    `wind`, `faults` and `sim_config` are handed to the simulator, so a scenario
    can be flown repeatedly and deterministically.

    Every tick is collected on the report; the console shows one line per
    `print_every` seconds of flight so 10 Hz telemetry stays readable. Unless
    `runs_dir` is None the finished mission is also written to a self-contained
    JSON run file, and its path is left on `report.run_file`.
    """
    constraints = constraints or DEFAULT_CONSTRAINTS
    planner = planner or default_planner
    runs_dir = RUNS_DIR if runs_dir is _UNSET else runs_dir

    def log(message: str) -> None:
        if verbose:
            print(message)

    log(f'Mission: "{mission_text}"')

    log("\n[1/4] Planning...")
    current_plan = planner(mission_text, constraints)
    for subtask in current_plan.subtasks:
        log(f"  subtask: {subtask.name} ({len(subtask.waypoints)} waypoint(s))")
        for wp in subtask.waypoints:
            log(f"    -> ({wp.x}, {wp.y}, {wp.z}) @ {wp.speed} m/s  {wp.action.value}")

    log("\n[2/4] Verifying...")
    result = verify(current_plan, constraints, world=world)
    for warning in result.warnings:
        log(f"  warning [{warning.type.value}] {warning.description}")
    if not result.passed:
        log("  VERIFICATION FAILED:")
        for violation in result.blocking:
            log(f"    - [{violation.type.value}] {violation.description}")
        rejected = MissionReport(mission_text, current_plan, result, constraints)
        _save(rejected, runs_dir, log)
        return rejected
    log(f"  verification passed ({len(result.warnings)} warning(s))"
        if result.warnings else "  verification passed")

    report = MissionReport(
        mission_text,
        current_plan,
        result,
        constraints,
        sim_config=sim_config,
        wind=wind,
        faults=faults,
        world=world,
    )

    log("\n[3/4] Executing...")
    if faults:
        log("  fault schedule:")
        log(faults.describe())
    executor = Executor(
        current_plan,
        constraints=constraints,
        config=sim_config,
        wind=wind,
        faults=faults,
        world=world,
    )
    next_print = 0.0
    for state in executor.run():
        report.states.append(state)

        if state.timestamp >= next_print:
            log(f"  {format_state(state)}")
            next_print = state.timestamp + print_every

        signal = observe(state)
        if signal.replan_required:
            # Hook for the autonomous replanning loop; the stub never fires.
            report.replans += 1
            log(f"  replan requested: {', '.join(signal.reasons)}")

    if report.states and report.states[-1].timestamp < next_print:
        log(f"  {format_state(report.states[-1])}")  # always show touchdown

    report.events = executor.event_records()
    report.obstacles = executor.obstacle_records()
    report.distance_flown = executor.distance_flown
    report.crashed = executor.crashed

    if report.events:
        log("\n  flight log:")
        for event in report.events:
            log(f"    t={event['t']:7.1f}s  {event['type']:14} {event['description']}")

    log(
        f"\n[4/4] Mission complete -- landed. "
        f"{report.duration:.1f}s, {report.distance_flown:.0f} m flown, "
        f"{report.battery_remaining:.1f}% battery left."
        if report.completed
        else (
            "\n[4/4] Mission ended in a crash." if report.crashed
            else "\n[4/4] Mission ended without a clean landing."
        )
    )

    _save(report, runs_dir, log)
    return report


def _save(report: MissionReport, runs_dir: Path | None, log: Callable[[str], None]) -> None:
    if runs_dir is None:
        return
    report.run_file = write_run(report, runs_dir=runs_dir)
    log(f"  run written to {report.run_file}")
