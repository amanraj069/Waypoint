"""PEACE's UAV tool set (peace/platforms/uav_tools.py), on the flight stack.

Adds the UAV-only tools (up, down, arm, disarm, takeoff, land) and the
actuation backend under every motion tool. The released sequence is kept step
for step: stream a setpoint, wait 0.3 s, switch to OFFBOARD, arm, poll for
arrival every 0.2 s until within tolerance or the command time-out.

`_navigate_to_waypoint` is where the paper's constraint enforcement sits:
the altitude is clamped into the band, then the position is validated against
the horizontal geofence, and a breach returns
"ERROR: Safety violation - ..." without commanding anything. Note what the
release leaves outside it, kept here as it is: `takeoff` clamps its height with
the same limits but never checks the fence, and a `land(x, y)` whose approach
is refused still lands where the aircraft is.

MAVROS's publisher and service clients are replaced by the flight stack's
`set_target`, `set_mode` and `arm`; wall-clock sleeps by simulated time.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from compare.common.flight import FlightStack
from compare.peace.exceptions import RobotError, SafetyViolationError
from compare.peace.tools.robot_tools import CONTROL_DISABLED, RobotTools, tool

log = logging.getLogger("compare.peace")

_OFFBOARD_PRESTREAM_DELAY: float = 0.3  # seconds (PX4 requirement before mode switch)
_POLL: float = 0.2  # seconds between arrival checks
#: PX4's default LAND descent rate, which the release budgets a landing's time-out with.
_LAND_DESCENT_RATE: float = 0.7


class UAVTools(RobotTools):
    """UAV tool set: the flight-specific tools on top of the generic primitives.

    Each tool returns a descriptive string (success or failure reason). With
    ACTION_ENABLE_CONTROL false every control method returns at once and
    commands nothing.
    """

    def __init__(self, stack: FlightStack, ego_state_svc, world_model_svc, constraint_svc, settings) -> None:
        super().__init__(ego_state_svc, world_model_svc, constraint_svc, settings)
        self._stack = stack

    # -- UAV-specific tools -----------------------------------------------------

    @tool
    def up(self, dist: float = 0.5) -> str:
        """Move up dist metres."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        tz = ego.position_z + dist
        return self._navigate_to_waypoint(
            ego.position_x, ego.position_y, tz, yaw_rad, tolerance=0.3, label=f"Moved up {dist}m"
        )

    @tool
    def down(self, dist: float = 0.5) -> str:
        """Move down dist metres."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        yaw_rad = math.radians(ego.orientation_yaw)
        tz = max(self._settings.action_min_altitude, ego.position_z - dist)
        return self._navigate_to_waypoint(
            ego.position_x, ego.position_y, tz, yaw_rad, tolerance=0.3, label=f"Moved down {dist}m"
        )

    @tool
    def arm(self) -> str:
        """Arm the drone motors."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        if not self._set_arm(True):
            return "ERROR: Arming failed"
        return "Armed successfully"

    @tool
    def disarm(self) -> str:
        """Disarm the drone motors."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        if not self._set_arm(False):
            return "ERROR: Disarming failed"
        return "Disarmed successfully"

    @tool
    def takeoff(self, height: float = 3.0) -> str:
        """Arm the drone and climb to height metres."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()

        target_z = max(self._settings.action_min_altitude, min(self._settings.action_max_altitude, height))

        # PX4 sequence: stream setpoints first, then OFFBOARD, then arm
        self._start_position_stream(ego.position_x, ego.position_y, target_z, math.radians(ego.orientation_yaw))
        self._sleep(_OFFBOARD_PRESTREAM_DELAY)

        if not self._set_mode("OFFBOARD"):
            return "ERROR: Failed to switch to OFFBOARD mode for takeoff"

        # Re-read armed state: PX4 auto-disarms on-ground drones after ~10s,
        # so the armed flag sampled before the OFFBOARD switch may be stale.
        ego_pre_arm = self._ego_state_svc.get_ego_state()
        if not ego_pre_arm.armed:
            if not self._set_arm(True):
                return "ERROR: Arming failed before takeoff"

        arrived = self._wait_for_arrival(
            ego.position_x, ego.position_y, target_z, tolerance=0.5, timeout=self._settings.action_command_timeout
        )
        ego_after = self._ego_state_svc.get_ego_state()
        alt = ego_after.position_z
        if arrived:
            return f"Takeoff complete. Altitude={alt:.1f}m"
        return f"Takeoff timeout - current altitude={alt:.1f}m (target={target_z:.1f}m)"

    @tool
    def land(self, x: Optional[float] = None, y: Optional[float] = None) -> str:
        """Land at the given position (current position if omitted) and wait for touchdown."""
        if not self._settings.action_enable_control:
            return CONTROL_DISABLED
        ego = self._ego_state_svc.get_ego_state()
        land_x = x if x is not None else ego.position_x
        land_y = y if y is not None else ego.position_y

        # A horizontal tolerance decides whether a re-position is needed.
        horiz_err = math.hypot(land_x - ego.position_x, land_y - ego.position_y)
        if horiz_err > 0.5:
            self._navigate_to_waypoint(
                land_x,
                land_y,
                ego.position_z,
                math.radians(ego.orientation_yaw),
                tolerance=0.5,
                label="Navigating to landing position",
            )

        # Stop the OFFBOARD setpoint stream BEFORE switching to AUTO.LAND so PX4
        # does not see a fresh setpoint arrive mid mode-switch and re-engage OFFBOARD.
        self._stop_stream()

        if not self._set_mode("AUTO.LAND"):
            return "ERROR: Failed to set AUTO.LAND mode"

        # Touchdown timeout scales with altitude, with slack on top.
        ego_at_land = self._ego_state_svc.get_ego_state()
        descent_budget = max(
            self._settings.action_command_timeout, ego_at_land.position_z / _LAND_DESCENT_RATE + 10.0
        )
        deadline = self._stack.now + descent_budget

        # Wait for PX4's auto-disarm after touchdown.
        while self._stack.now < deadline:
            ego_now = self._ego_state_svc.get_ego_state()
            if not ego_now.armed:
                self._landed(land_x, land_y)
                return f"Landed at ({land_x:.1f},{land_y:.1f}), altitude={ego_now.position_z:.2f}m"
            self._sleep(_POLL)

        ego_final = self._ego_state_svc.get_ego_state()
        return (
            f"Land timeout - alt={ego_final.position_z:.2f}m, armed={ego_final.armed} "
            f"at ({land_x:.1f},{land_y:.1f})"
        )

    # -- the actuation backend ---------------------------------------------------

    def _navigate_to_waypoint(self, x: float, y: float, z: float, yaw_rad: float, tolerance: float, label: str) -> str:
        """Stream setpoints, engage OFFBOARD, wait for arrival. Returns the result string."""
        # Constraint enforcement: clamp altitude, reject geofence violations
        z = self._constraint_svc.clamp_altitude(z)
        try:
            self._constraint_svc.validate_position(x, y, z)
        except SafetyViolationError as e:
            self._rejected(x, y, z, str(e))
            return f"ERROR: Safety violation - {e}"

        self._start_position_stream(x, y, z, yaw_rad)
        # PX4 requires setpoints to be streaming for >= 0.3s before switching to OFFBOARD
        self._sleep(_OFFBOARD_PRESTREAM_DELAY)
        self._ensure_offboard_and_armed()
        arrived = self._wait_for_arrival(x, y, z, tolerance=tolerance, timeout=self._settings.action_command_timeout)
        ego = self._ego_state_svc.get_ego_state()
        pos_str = (
            f"Position: ({ego.position_x:.1f},{ego.position_y:.1f},{ego.position_z:.1f})m, "
            f"yaw={ego.orientation_yaw:.0f}°, alt={ego.position_z:.1f}m"
        )
        if arrived:
            self._arrived(label)
            return f"{label}. {pos_str}"
        return f"{label} (arrival timeout). {pos_str}"

    def _hold_position(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        """Command a hold setpoint. Idempotently engages OFFBOARD on first call."""
        self._start_position_stream(x, y, z, yaw_rad)
        if self._stack.mode != "OFFBOARD":
            self._sleep(_OFFBOARD_PRESTREAM_DELAY)
            self._set_mode("OFFBOARD")

    def _sleep(self, seconds: float) -> None:
        self._stack.advance(seconds)

    def stop_safely(self) -> None:
        """Switch to AUTO.LOITER if in OFFBOARD, then stop the setpoint stream."""
        if self._stack.mode == "OFFBOARD":
            self._set_mode("AUTO.LOITER")
            self._sleep(0.5)
        self._stop_stream()

    # -- internal navigation helpers ----------------------------------------------

    def _ensure_offboard_and_armed(self) -> None:
        """Switch to OFFBOARD mode if not already there (it does not arm, as released)."""
        if self._stack.mode != "OFFBOARD":
            if not self._set_mode("OFFBOARD"):
                raise RobotError("Failed to switch to OFFBOARD mode")

    def _wait_for_arrival(self, x: float, y: float, z: float, tolerance: float, timeout: float) -> bool:
        """Poll the robot state until within tolerance or timeout. True on arrival."""
        deadline = self._stack.now + timeout
        while self._stack.now < deadline:
            ego = self._ego_state_svc.get_ego_state()
            dist = math.sqrt((ego.position_x - x) ** 2 + (ego.position_y - y) ** 2 + (ego.position_z - z) ** 2)
            if dist <= tolerance:
                return True
            self._sleep(_POLL)
        return False

    def _start_position_stream(self, x: float, y: float, z: float, yaw: float) -> None:
        self._stack.set_target(x, y, z, yaw)

    def _stop_stream(self) -> None:
        self._stack.stop_stream()

    def _set_mode(self, mode: str) -> bool:
        ok = self._stack.set_mode(mode)
        if not ok:
            log.error("Failed to set mode to %s", mode)
        return ok

    def _set_arm(self, arm: bool) -> bool:
        ok = self._stack.arm(arm)
        if not ok:
            log.error("Arming command rejected")
        return ok

    # -- the flight record (not part of PEACE; lets a flight be replayed and scored) -----

    def _arrived(self, label: str) -> None:
        from waypoint.planner.models import Action  # noqa: PLC0415

        self._stack.mark_waypoint(Action.MOVE, label)

    def _landed(self, x: float, y: float) -> None:
        self._stack.mark("TOUCHDOWN", f"landed where the plan asked, at ({x:.1f}, {y:.1f})", {"local": [x, y]})

    def _rejected(self, x: float, y: float, z: float, why: str) -> None:
        self._stack.mark("CONSTRAINT", f"command to ({x:.1f}, {y:.1f}, {z:.1f}) refused: {why}", {"target": [x, y, z]})
