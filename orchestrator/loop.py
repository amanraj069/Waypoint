"""Mission orchestration: plan -> verify -> execute -> feedback."""

from __future__ import annotations

from dataclasses import dataclass, field

from feedback import observe
from planner import Plan, plan as make_plan
from sim import MissionState, execute
from verifier import VerificationResult, verify


@dataclass
class MissionReport:
    """Outcome of one run_mission call."""

    mission_text: str
    plan: Plan
    verification: VerificationResult
    states: list[MissionState] = field(default_factory=list)
    replans: int = 0

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


def run_mission(mission_text: str, verbose: bool = True) -> MissionReport:
    """Run one mission end to end, printing each MissionState tick."""

    def log(message: str) -> None:
        if verbose:
            print(message)

    log(f'Mission: "{mission_text}"')

    log("\n[1/4] Planning...")
    current_plan = make_plan(mission_text)
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
        return MissionReport(mission_text, current_plan, result)
    log("  verification passed")

    report = MissionReport(mission_text, current_plan, result)

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
    return report
