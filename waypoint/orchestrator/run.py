"""`python -m waypoint.orchestrator.run` is the closed-loop CLI, `run_full`, by its older name."""

from waypoint.orchestrator.run_full import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
