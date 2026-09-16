"""Minimal programmatic run of the mission loop."""

from orchestrator import run_mission

if __name__ == "__main__":
    report = run_mission("survey the north field")
    print(f"\nticks={len(report.states)} replans={report.replans} completed={report.completed}")
