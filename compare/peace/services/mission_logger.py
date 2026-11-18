"""PEACE's mission logging service (peace/services/mission_logger_service.py).

One CSV row per mission, with the released columns, so a log written here
reads like one written by the release: the query, the plan's reasoning, every
executed step with its result, the final state, the robot state at start and
end, and the world model observed when the query arrived.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from typing import Optional

from compare.peace.schemas import RobotState, WorldModel

COLUMNS = [
    "mission_id",
    "query",
    "metadata_json",
    "world_model_json",
    "robot_state_json",
    "final_robot_state_json",
    "plan_reasoning",
    "steps_json",
    "final_state",
    "started_at",
    "duration_s",
    "replan_count",
    "model",
]


class MissionLoggerService:
    """Thread-safe CSV logger for mission data collection."""

    def __init__(self, output_dir: str, enabled: bool = True, filename: str = "agent_log.csv") -> None:
        self._enabled = enabled
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self.path: str | None = None
        #: Every row written, in memory too, so a caller need not read the file back.
        self.rows: list[dict] = []

        if not enabled:
            return

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, filename)
        self.path = log_path
        write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

        self._file = open(log_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log_mission(
        self,
        mission_id: str,
        query: str,
        plan_reasoning: str,
        executed_steps: list[dict],
        final_state: str,
        ego_state: Optional[RobotState],
        world_model: Optional[WorldModel],
        metadata: Optional[dict[str, str]] = None,
        final_ego_state: Optional[RobotState] = None,
        started_at: float = 0.0,
        duration_s: float = 0.0,
        replan_count: int = 0,
        model: str = "",
    ) -> None:
        """Log the complete outcome of a mission as a single row."""
        row = {
            "mission_id": mission_id,
            "query": query,
            "metadata_json": json.dumps(metadata or {}),
            "world_model_json": world_model.model_dump_json() if world_model else "",
            "robot_state_json": ego_state.model_dump_json() if ego_state else "",
            "final_robot_state_json": final_ego_state.model_dump_json() if final_ego_state else "",
            "plan_reasoning": plan_reasoning,
            "steps_json": json.dumps(executed_steps),
            "final_state": final_state,
            "started_at": f"{started_at:.6f}" if started_at else "",
            "duration_s": f"{duration_s:.3f}" if duration_s else "",
            "replan_count": replan_count,
            "model": model,
        }
        self.rows.append(row)
        if not self._enabled or self._writer is None:
            return
        with self._lock:
            if self._writer is not None:
                self._writer.writerow(row)
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is None:
                return
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
