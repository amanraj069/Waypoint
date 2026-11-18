"""PEACE's robot state aggregation service (peace/services/robot_state_service.py)."""

from __future__ import annotations

import math
import threading
from collections import deque

from compare.peace.schemas import RobotState


class RobotStateAggregatorService:
    """Thread-safe aggregator for robot state.

    The flight stack calls update_*() as MAVROS's callbacks did; the agent
    calls get_ego_state() for a complete snapshot for the LLM.
    """

    def __init__(self, context_window: int = 10) -> None:
        self._ego_state = RobotState()
        self._history: deque[RobotState] = deque(maxlen=context_window)
        self._ego_state_lock = threading.Lock()

    def update_odom(
        self, x: float, y: float, z: float, vx: float, vy: float, vz: float, qx: float, qy: float, qz: float, qw: float
    ) -> None:
        """Update position, velocity, and orientation from a single odometry message (ZYX Euler)."""
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        with self._ego_state_lock:
            self._ego_state.position_x = x
            self._ego_state.position_y = y
            self._ego_state.position_z = z
            self._ego_state.velocity_x = vx
            self._ego_state.velocity_y = vy
            self._ego_state.velocity_z = vz
            self._ego_state.orientation_roll = math.degrees(roll)
            self._ego_state.orientation_pitch = math.degrees(pitch)
            self._ego_state.orientation_yaw = math.degrees(yaw)
            self._history.append(self._ego_state.model_copy())

    def update_flight_mode(self, mode: str, armed: bool, connected: bool) -> None:
        with self._ego_state_lock:
            self._ego_state.flight_mode = mode
            self._ego_state.armed = armed
            self._ego_state.connected = connected

    def get_ego_state(self) -> RobotState:
        """Return a complete snapshot of the current robot state."""
        with self._ego_state_lock:
            return self._ego_state.model_copy()

    def get_ego_state_history(self, n: int) -> list[RobotState]:
        """Return the last n odometry-stamped robot state snapshots, oldest first."""
        with self._ego_state_lock:
            history = list(self._history)
        return history[-n:] if n < len(history) else history
