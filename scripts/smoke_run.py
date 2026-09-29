"""Minimal programmatic run of the closed loop (plans with Gemini, so it needs a key)."""

from waypoint.orchestrator import run_mission

if __name__ == "__main__":
    log = run_mission("survey the north field")
    report = log.report
    print(f"\noutcome={log.outcome} ticks={len(report.states) if report else 0} "
          f"replans={report.replans if report else 0} log={log.path}")
