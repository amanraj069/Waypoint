"""PEACE's world model aggregation service (peace/services/world_model_service.py)."""

from __future__ import annotations

import math
import threading
from collections import deque

import numpy as np

from compare.peace.schemas import WorldModel


class WorldModelAggregatorService:
    """Thread-safe aggregator for environment observations (detections, obstacles)."""

    # Sector definitions: name -> (angle_min_deg, angle_max_deg)
    # 0 deg = forward, positive counter-clockwise when viewed from above
    LIDAR_SECTORS: dict[str, tuple[float, float]] = {
        "front": (-22.5, 22.5),
        "front_right": (-67.5, -22.5),
        "right": (-112.5, -67.5),
        "back_right": (-157.5, -112.5),
        "back": (157.5, -157.5),  # wraps: start=157.5, end=-157.5 across +-180
        "back_left": (112.5, 157.5),
        "left": (67.5, 112.5),
        "front_left": (22.5, 67.5),
    }

    def __init__(self, context_window: int = 10, max_detections: int = 10) -> None:
        self._max_detections = max_detections
        self._world_model = WorldModel()
        self._history: deque[WorldModel] = deque(maxlen=context_window)
        self._world_model_lock = threading.Lock()

    def update_lidar(
        self, ranges: list[float], angle_min: float, angle_increment: float, range_min: float, range_max: float
    ) -> None:
        """Parse a LiDAR scan into named sectors and store the minimum distance in each."""
        arr = np.array(ranges, dtype=np.float32)
        valid = np.isfinite(arr) & (arr >= range_min) & (arr <= range_max)
        if not np.any(valid):
            return

        num = len(arr)
        angles = angle_min + np.arange(num) * angle_increment

        new_ranges: dict[str, float] = {}
        for name, (a_start_deg, a_end_deg) in self.LIDAR_SECTORS.items():
            a_start = math.radians(a_start_deg)
            a_end = math.radians(a_end_deg)
            if name == "back":  # sector wraps around +-pi: start > end
                mask = valid & ((angles >= a_start) | (angles <= a_end))
            else:
                mask = valid & (angles >= a_start) & (angles < a_end)
            new_ranges[name] = float(np.min(arr[mask])) if np.any(mask) else float("inf")

        new_ranges["min_range"] = float(np.min(arr[valid]))

        with self._world_model_lock:
            self._world_model.lidar_ranges = new_ranges

    def update_detections(self, detections: list[dict]) -> None:
        """Store the latest object detections (capped at max_detections) and append a snapshot to history."""
        with self._world_model_lock:
            self._world_model.detections = detections[: self._max_detections]
            self._history.append(self._world_model.model_copy())

    def get_world_model(self) -> WorldModel:
        with self._world_model_lock:
            return self._world_model.model_copy()

    def get_world_model_history(self, n: int) -> list[WorldModel]:
        with self._world_model_lock:
            history = list(self._history)
        return history[-n:] if n < len(history) else history
