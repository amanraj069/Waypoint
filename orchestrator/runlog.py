"""Persist a finished mission to a self-contained JSON run file.

A run file carries everything needed to replay, plot or animate the mission
without the code that produced it: the mission text, the constraints, the
verification result, the planned waypoints (each tagged with its subtask), the
simulator settings, the fault schedule, the obstacles that existed or appeared,
the flight event timeline and every telemetry tick.

Schema 2 added the simulator sections; `gazebo/scene.py` reads this file
directly.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")
SCHEMA_VERSION = 2
_SLUG_MAX = 48


def slugify(text: str) -> str:
    """Turn mission text into a filesystem-safe stem.

    Falls back to 'mission' when the text has no usable characters, so a run is
    never written to a file named for nothing.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:_SLUG_MAX].rstrip("-") or "mission"


def run_path(mission_text: str, when: datetime | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    stamp = (when or datetime.now()).strftime("%Y%m%dT%H%M%S")
    return runs_dir / f"{slugify(mission_text)}_{stamp}.json"


def build_run(report: Any, when: datetime | None = None) -> dict:
    """Render a MissionReport as the plain dict that gets written to disk."""
    return {
        "schema_version": SCHEMA_VERSION,
        "mission_text": report.mission_text,
        "created_at": (when or datetime.now()).isoformat(timespec="seconds"),
        "constraints": report.constraints.model_dump() if report.constraints else None,
        "verification": {
            "passed": report.verification.passed,
            "violations": [
                v.model_dump(mode="json") for v in report.verification.violations
            ],
        },
        "replans": report.replans,
        "completed": report.completed,
        "sim": _sim_section(report),
        "world": report.world.model_dump(mode="json") if report.world else None,
        "faults": [event.model_dump(mode="json") for event in (report.faults.events if report.faults else [])],
        "events": list(report.events),
        "obstacles": list(report.obstacles),
        # Flattened, but each waypoint keeps its subtask so the hierarchy is
        # recoverable by grouping on it.
        "waypoints": [
            {
                "subtask": subtask.name,
                "index": index,
                "x": wp.x,
                "y": wp.y,
                "z": wp.z,
                "speed": wp.speed,
                "action": wp.action.value,
            }
            for index, (subtask, wp) in enumerate(
                (s, w) for s in report.plan.subtasks for w in s.waypoints
            )
        ],
        "ticks": [state.as_record() for state in report.states],
    }


def _sim_section(report: Any) -> dict:
    """Everything needed to re-fly this run exactly as it was flown."""
    from sim import DEFAULT_CONFIG, NO_WIND

    config = report.sim_config or DEFAULT_CONFIG
    wind = report.wind or NO_WIND
    return {
        "config": config.model_dump(mode="json"),
        "wind": wind.model_dump(mode="json"),
        "distance_flown_m": round(report.distance_flown, 2),
        "crashed": bool(getattr(report, "crashed", False)),
        "duration_s": round(report.duration, 2),
        "battery_remaining_pct": round(report.battery_remaining, 2),
    }


def _unique(path: Path) -> Path:
    """Avoid clobbering an existing run.

    The timestamp only resolves to the second, so two missions flown in the same
    second would otherwise overwrite each other.
    """
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Too many runs already written for {path.stem}")


def write_run(report: Any, when: datetime | None = None, runs_dir: Path = RUNS_DIR) -> Path:
    """Write the run to runs/<mission_slug>_<timestamp>.json and return the path."""
    when = when or datetime.now()
    path = run_path(report.mission_text, when, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _unique(path)
    path.write_text(json.dumps(build_run(report, when), indent=2) + "\n")
    return path


def read_run(path: Path | str) -> dict:
    """Load a run file, failing clearly if it is not one."""
    path = Path(path)
    try:
        run = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"No run file at {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(run, dict) or "ticks" not in run:
        raise ValueError(f"{path} does not look like a mission run file (no 'ticks' key).")
    return run
