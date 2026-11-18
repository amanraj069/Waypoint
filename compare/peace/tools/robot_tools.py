"""PEACE's generic tool set (peace/platforms/robot_tools.py).

The fixed action space the planner may call: forward, backward, left, right,
rotate, waypoint, navigate_to_object, localize_object. The geometry, the
tolerances, the step sizes and every result string are the released ones; a
result that starts with "ERROR" or contains "failed" is what the executor
treats as a tool failure.

The only change is the actuation boundary: where the release sleeps on the
wall clock while MAVROS flies, this asks the flight stack to advance simulated
time (`_sleep`).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional

log = logging.getLogger("compare.peace")

# Default timing/step constants for the generic primitives.
_SCAN_SETTLE_TIME: float = 1.5  # seconds to settle at each yaw step
_LOCALIZE_STEPS: int = 36  # 360 / 10 degrees per step: fine scan for object localisation
_ROTATE_STEP_DEG: float = 30.0  # max yaw delta per intermediate setpoint during rotate()

CONTROL_DISABLED = "Control disabled (ACTION_ENABLE_CONTROL=False)"


def tool(fn: Callable) -> Callable:
    """Mark a method as LLM-callable; `RobotTools.__init_subclass__` collects them."""
    fn._is_tool = True  # type: ignore[attr-defined]
    return fn


def _collect_tools(cls: type) -> dict[str, Callable]:
    """Walk the MRO and collect every method tagged with `@tool`; a subclass overrides its parent."""
    merged: dict[str, Callable] = {}
    for base in reversed(cls.__mro__):
        for name, attr in vars(base).items():
            if callable(attr) and getattr(attr, "_is_tool", False):
                merged[name] = attr
    return merged


class RobotTools(ABC):
    """Abstract base for platform-specific tool sets exposed to the LLM planner."""

    _tools: ClassVar[dict[str, Callable]] = {}

    SCAN_SETTLE_TIME: ClassVar[float] = _SCAN_SETTLE_TIME
    LOCALIZE_STEPS: ClassVar[int] = _LOCALIZE_STEPS
    ROTATE_STEP_DEG: ClassVar[float] = _ROTATE_STEP_DEG

    def __init__(self, ego_state_svc, world_model_svc, constraint_svc, settings) -> None:
        self._ego_state_svc = ego_state_svc
        self._world_model_svc = world_model_svc
        self._constraint_svc = constraint_svc
        self._settings = settings

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._tools = _collect_tools(cls)

    # -- dispatch ---------------------------------------------------------------

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch an LLM ToolCall by name with the given keyword arguments."""
        fn = self._tools.get(tool_name)
        if fn is None:
            return f"ERROR: Unknown tool '{tool_name}'"
        try:
            return fn(self, **arguments)
        except TypeError as e:
            return f"ERROR: Bad arguments for '{tool_name}': {e}"
        except Exception as e:  # noqa: BLE001 -- a tool that raises is a failed step, as released
            return f"ERROR: '{tool_name}' raised {type(e).__name__}: {e}"

    @classmethod
    def tool_names(cls) -> list[str]:
        return list(cls._tools.keys())

    # -- generic motion primitives ----------------------------------------------

    @tool
    def forward(self, dist: float = 0.5) -> str:
        """Move forward dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = math.cos(yaw_rad) * dist
        dy = math.sin(yaw_rad) * dist
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3, label=f"Moved forward {dist}m")

    @tool
    def backward(self, dist: float = 0.5) -> str:
        """Move backward dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = math.cos(yaw_rad) * (-dist)
        dy = math.sin(yaw_rad) * (-dist)
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3, label=f"Moved backward {dist}m")

    @tool
    def left(self, dist: float = 0.5) -> str:
        """Move left dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = -math.sin(yaw_rad) * dist
        dy = math.cos(yaw_rad) * dist
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3, label=f"Moved left {dist}m")

    @tool
    def right(self, dist: float = 0.5) -> str:
        """Move right dist metres relative to current body heading."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        dx = math.sin(yaw_rad) * dist
        dy = -math.cos(yaw_rad) * dist
        tx, ty, tz = ego.position_x + dx, ego.position_y + dy, ego.position_z
        return self._navigate_to_waypoint(tx, ty, tz, yaw_rad, tolerance=0.3, label=f"Moved right {dist}m")

    @tool
    def rotate(self, degrees: float) -> str:
        """Rotate by degrees relative to current heading (positive = counter-clockwise in ENU)."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        start_yaw_deg = ego.orientation_yaw
        pos_x, pos_y, pos_z = ego.position_x, ego.position_y, ego.position_z

        # Slice the rotation into intermediate setpoints no larger than ROTATE_STEP_DEG
        # so the low-level controller tracks smoothly instead of slewing in one jump.
        self._hold_position(pos_x, pos_y, pos_z, math.radians(start_yaw_deg))
        num_steps = max(1, math.ceil(abs(degrees) / self.ROTATE_STEP_DEG))
        step_deg = degrees / num_steps
        for i in range(1, num_steps + 1):
            yaw_deg = start_yaw_deg + i * step_deg
            self._hold_position(pos_x, pos_y, pos_z, math.radians(yaw_deg))
            self._sleep(self.SCAN_SETTLE_TIME)

        ego_after = self._ego_state_svc.get_ego_state()
        return f"Rotated {degrees:+.0f}°. Yaw={ego_after.orientation_yaw:.0f}°"

    @tool
    def waypoint(self, x: float, y: float, z: float, yaw: Optional[float] = None, tolerance: float = 0.5) -> str:
        """Fly/drive to an absolute ENU position."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(yaw) if yaw is not None else math.radians(ego.orientation_yaw)
        return self._navigate_to_waypoint(
            x, y, z, yaw_rad, tolerance=tolerance, label=f"Moved to waypoint ({x:.1f},{y:.1f},{z:.1f})m"
        )

    @tool
    def navigate_to_object(
        self, label: str, stop_distance: float = 2.0, z: Optional[float] = None, tolerance: float = 0.5
    ) -> str:
        """Move toward a detected object and stop stop_distance metres short."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        detections = self._world_model_svc.get_world_model().detections
        target = next((d for d in detections if d["label"] == label and d["depth_valid"]), None)
        if target is None:
            return f"ERROR: Object '{label}' not found or depth invalid"

        obj_x, obj_y = target["world_x"], target["world_y"]
        alt = z if z is not None else ego.position_z
        dx, dy = obj_x - ego.position_x, obj_y - ego.position_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= stop_distance:
            log.info("Already within %sm of '%s', skipping approach", stop_distance, label)
            return f"Already within {stop_distance}m of '{label}'"

        scale = (dist - stop_distance) / dist
        ap_x = ego.position_x + dx * scale
        ap_y = ego.position_y + dy * scale
        approach_yaw = math.atan2(dy, dx)

        log.info("Approaching '%s': stopping %sm short at (%.1f,%.1f)", label, stop_distance, ap_x, ap_y)
        result = self._navigate_to_waypoint(
            ap_x, ap_y, alt, approach_yaw, tolerance=tolerance, label=f"Approached '{label}' to {stop_distance}m"
        )

        detections_after = self._world_model_svc.get_world_model().detections
        still_visible = any(d["label"] == label for d in detections_after)
        if still_visible:
            log.info("'%s' confirmed visible at approach position", label)
        else:
            log.warning("'%s' no longer visible after approach", label)
        return result

    @tool
    def localize_object(
        self, label: str, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None
    ) -> str:
        """Rotate in fine yaw steps until the labelled object appears in detections."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        hold_x = x if x is not None else ego.position_x
        hold_y = y if y is not None else ego.position_y
        hold_z = z if z is not None else ego.position_z
        start_yaw_deg = ego.orientation_yaw
        step_deg = 360.0 / self.LOCALIZE_STEPS

        # Prime the stream at start_yaw so OFFBOARD engages before the first
        # rotation step; the loop then yaws away from start_yaw on every tick.
        self._hold_position(hold_x, hold_y, hold_z, math.radians(start_yaw_deg))

        for i in range(1, self.LOCALIZE_STEPS + 1):
            yaw_deg = start_yaw_deg + i * step_deg
            yaw_rad = math.radians(yaw_deg)
            self._hold_position(hold_x, hold_y, hold_z, yaw_rad)
            self._sleep(self.SCAN_SETTLE_TIME)

            detections = self._world_model_svc.get_world_model().detections
            found = next((d for d in detections if d["label"] == label), None)
            if found:
                bearing_rad = math.atan2(found["world_y"] - hold_y, found["world_x"] - hold_x)
                bearing_deg = math.degrees(bearing_rad)
                self._hold_position(hold_x, hold_y, hold_z, bearing_rad)
                self._sleep(self.SCAN_SETTLE_TIME)
                return f"Found '{label}' at yaw={bearing_deg:.1f}°"

        log.warning("'%s' not found after full 360 degree scan", label)
        return f"ERROR: '{label}' not found after full 360° scan"

    # -- the actuation backend; a subclass implements ---------------------------

    @abstractmethod
    def _navigate_to_waypoint(self, x: float, y: float, z: float, yaw_rad: float, tolerance: float, label: str) -> str:
        """Drive the platform to (x, y, z, yaw) and block until arrival or timeout."""

    @abstractmethod
    def _hold_position(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        """Command the platform to maintain (x, y, z, yaw). Idempotent."""

    @abstractmethod
    def _sleep(self, seconds: float) -> None:
        """Let `seconds` pass while the platform keeps flying."""

    @abstractmethod
    def stop_safely(self) -> None:
        """Bring the platform to a safe halt (loiter, brake, etc.) and stop control output."""


# RobotTools itself registers its own decorated methods; __init_subclass__ only fires for subclasses.
RobotTools._tools = _collect_tools(RobotTools)
