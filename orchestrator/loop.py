"""Mission orchestration: plan -> verify -> execute -> feedback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

from feedback import observe
from planner import DEFAULT_CONSTRAINTS, MissionConstraints, Plan
from planner import plan as default_planner
from sim import MissionState, execute
from orchestrator.runlog import RUNS_DIR, write_run
from verifier import VerificationResult, verify

Planner = Callable[[str, MissionConstraints], Plan]

# Sentinel so the default is resolved against the module attribute at call time
# (a plain default would bind RUNS_DIR at import and ignore reconfiguration).
_UNSET: Any = object()


@dataclass
class MissionReport:
    """Outcome of one run_mission call."""

    mission_text: str
    plan: Plan
    verification: VerificationResult
    constraints: MissionConstraints | None = None
    states: list[MissionState] = field(default_factory=list)
    replans: int = 0
    run_file: Path | None = None

    @property
    def completed(self) -> bool:
        return bool(self.states) and self.states[-1].z == 0.0 and self.states[-1].velocity == 0.0


def format_state(state: MissionState) -> str:
    return (
        f"t={state.timestamp:6.1f}s  "
        f"pos=({state.x:7.2f}, {state.y:7.2f}, {state.z:6.2f})  "
        f"vel={state.velocity:5.2f} m/s  "
        f"hdg={state.heading:6.1f}deg  "
        f"batt={state.battery_pct:5.1f}%"
    )


def run_mission(
    mission_text: str,
    constraints: MissionConstraints | None = None,
    verbose: bool = True,
    planner: Planner | None = None,
    runs_dir: Path | None = _UNSET,
) -> MissionReport:
    """Run one mission end to end, printing each MissionState tick.

    `planner` is injectable so tests (and future replanning experiments) can
    drive the loop without calling the real API; it defaults to the LLM planner.

    Every tick is both printed and collected on the report. Unless `runs_dir` is
    None the finished mission is also written to a self-contained JSON run file,
    and its path is left on `report.run_file`.
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
    result = verify(current_plan)
    if not result.passed:
        log("  VERIFICATION FAILED:")
        for violation in result.violations:
            log(f"    - {violation}")
        rejected = MissionReport(mission_text, current_plan, result, constraints)
        _save(rejected, runs_dir, log)
        return rejected
    log("  verification passed")

    report = MissionReport(mission_text, current_plan, result, constraints)

    log("\n[3/4] Executing...")
    for state in execute(current_plan):
        report.states.append(state)
        log(f"  {format_state(state)}")

        signal = observe(state)
        if signal.replan_required:
            # Hook for the autonomous replanning loop; the stub never fires.
            report.replans += 1
            log(f"  replan requested: {', '.join(signal.reasons)}")

    log("\n[4/4] Mission complete -- landed." if report.completed
        else "\n[4/4] Mission ended without a clean landing.")

    _save(report, runs_dir, log)
    return report


def _save(report: MissionReport, runs_dir: Path | None, log: Callable[[str], None]) -> None:
    if runs_dir is None:
        return
    report.run_file = write_run(report, runs_dir=runs_dir)
    log(f"  run written to {report.run_file}")
