"""`MissionLog`: every decision the loop made, in order, as JSON Lines.

One line per entry, appended and flushed as it happens, so a mission that dies
halfway still leaves a log up to the moment it died, and `jq` or pandas can
read it without loading anything else. The kinds, in the order a mission
produces them:

| kind                | when |
| ------------------- | ---- |
| `mission_started`   | first: the mission, the backend, the constraints, the faults |
| `plan_proposed`     | every plan the planner returns: `source` is `initial`, `verification_replan` or `inflight_replan` |
| `verification`      | every verifier verdict, pass or fail, with its violations |
| `execution_started` | the first verified plan starts flying |
| `perception`        | the perception source reported an obstacle not known before: `type`, `position`, `severity`, `description`, `obstacle` |
| `replan_triggered`  | the feedback monitor stopped the flight: `trigger` (what), `reason`, `t`, `position` |
| `execution_resumed` | a new verified plan takes over mid-flight |
| `failsafe`          | no safe plan could be found in the air, so it comes down: here if the ground is clear, else on the nearest clear ground (`landing` says which) |
| `mission_finished`  | last: the outcome, and the numbers to compare runs by |

Every entry has `seq` (its position), `kind`, and `wall` (wall-clock time);
entries about the flight also carry `t`, seconds of flight.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

KINDS = (
    "mission_started",
    "plan_proposed",
    "verification",
    "execution_started",
    "perception",
    "replan_triggered",
    "execution_resumed",
    "failsafe",
    "mission_finished",
)

RESERVED = frozenset({"seq", "kind", "wall"})


class MissionLog:
    """An append-only record of one mission, mirrored to a JSONL file if given a path."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.entries: list[dict] = []
        #: Set when the mission is flown; the flight data behind the run file.
        self.report: Any = None
        self.run_file: Path | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def add(self, kind: str, /, **data: Any) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown mission log entry kind {kind!r}")
        clash = RESERVED & data.keys()
        if clash:
            raise ValueError(f"{', '.join(sorted(clash))} is set by the log itself")
        line = json.dumps(
            {"seq": len(self.entries), "kind": kind,
             "wall": datetime.now().isoformat(timespec="milliseconds"), **data},
            default=_jsonable,
        )
        # Kept as the JSON it was written as, so the log in memory and the log
        # on disk are the same thing, whatever objects it was handed.
        entry = json.loads(line)
        self.entries.append(entry)
        if self.path is not None:
            with self.path.open("a") as handle:
                handle.write(line + "\n")
        return entry

    # -- reading it back ------------------------------------------------------

    @property
    def kinds(self) -> list[str]:
        return [entry["kind"] for entry in self.entries]

    def of(self, kind: str) -> list[dict]:
        return [entry for entry in self.entries if entry["kind"] == kind]

    @property
    def outcome(self) -> str | None:
        finished = self.of("mission_finished")
        return finished[-1]["outcome"] if finished else None

    @property
    def backend(self) -> str | None:
        started = self.of("mission_started")
        return started[0]["backend"] if started else None

    @staticmethod
    def read(path: Path) -> list[dict]:
        """The entries of a log written earlier, in order."""
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot write {type(value).__name__} to the mission log")
