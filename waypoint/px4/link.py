"""The MAVLink side of the PX4 backend: telemetry in, setpoints and commands out.

Two threads. One reads everything PX4 sends and keeps the newest of each
message, waking whoever waits for a new position. The other repeats the
current setpoint at 50 Hz, because PX4 leaves offboard mode when setpoints
stop arriving; changing what it sends is just replacing the setpoint.

Frames: PX4 speaks local NED (x north, y east, z down). Conversion to the
project's ENU happens in the backend, never here.
"""

from __future__ import annotations

import math
import os
import struct
import threading
import time
from dataclasses import dataclass

from waypoint.backends import BackendError

os.environ.setdefault("MAVLINK20", "1")
try:
    from pymavlink import mavutil  # after choosing MAVLink 2
except ImportError:  # the px4 extra: pip install -e ".[px4]"
    mavutil = None
mavlink = mavutil.mavlink if mavutil else None

#: PX4's onboard link for the first SITL instance sends here.
ADDRESS = "udpin:127.0.0.1:14540"
STREAM_HZ = 50.0

# set_position_target_local_ned type masks: which fields PX4 should ignore.
_IGNORE_POSITION = 0b0000_0000_0111
_IGNORE_VELOCITY = 0b0000_0011_1000
_IGNORE_ACCEL = 0b0001_1100_0000
_IGNORE_YAW_RATE = 0b1000_0000_0000
VELOCITY_MASK = _IGNORE_POSITION | _IGNORE_ACCEL | _IGNORE_YAW_RATE
POSITION_MASK = _IGNORE_VELOCITY | _IGNORE_ACCEL | _IGNORE_YAW_RATE

# PX4's custom main modes, as MAV_CMD_DO_SET_MODE takes them.
OFFBOARD = 6
#: MAV_CMD_COMPONENT_ARM_DISARM's magic number for "disarm, even in the air".
FORCE = 21196


@dataclass(frozen=True)
class Setpoint:
    """What PX4 should do now: hold a NED position, or fly a NED velocity; yaw in radians."""

    mask: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0

    @classmethod
    def hold(cls, position, yaw: float) -> "Setpoint":
        return cls(POSITION_MASK, position=tuple(position), yaw=yaw)

    @classmethod
    def fly(cls, velocity, yaw: float) -> "Setpoint":
        return cls(VELOCITY_MASK, velocity=tuple(velocity), yaw=yaw)


class Link:
    """A MAVLink connection to one PX4 instance."""

    def __init__(self, address: str = ADDRESS):
        if mavutil is None:
            raise BackendError('the px4 backend needs pymavlink: pip install -e ".[px4]"')
        self.conn = mavutil.mavlink_connection(address, source_system=245, source_component=190)
        self.latest: dict[str, object] = {}
        self._before = None  # the LOCAL_POSITION_NED before the latest one
        self.statustext: list[str] = []
        self._acks: dict[int, object] = {}
        self._params: dict[str, float] = {}
        self._new = threading.Condition()
        self._send = threading.Lock()
        self._setpoint: Setpoint | None = None
        self._closed = False
        self._reader = threading.Thread(target=self._read, name="mavlink-read", daemon=True)
        self._streamer = threading.Thread(target=self._stream, name="mavlink-setpoints", daemon=True)

    # -- connection -------------------------------------------------------------

    def connect(self, timeout: float = 30.0) -> None:
        heartbeat = self.conn.wait_heartbeat(timeout=timeout)
        if heartbeat is None:
            raise BackendError(f"no heartbeat from PX4 on {ADDRESS} within {timeout:g} s")
        self._reader.start()
        self._streamer.start()
        # The position at 50 Hz, so a 10 Hz tick always has a fresh one.
        self.interval(mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 50.0)
        self.interval(mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 10.0)
        self.interval(mavlink.MAVLINK_MSG_ID_HOME_POSITION, 1.0)

    def close(self) -> None:
        self._closed = True
        with self._new:
            self._new.notify_all()
        for thread in (self._reader, self._streamer):
            if thread.is_alive():
                thread.join(timeout=2)
        try:
            self.conn.close()
        except OSError:
            pass

    # -- reading ------------------------------------------------------------------

    def _read(self) -> None:
        while not self._closed:
            try:
                message = self.conn.recv_match(blocking=True, timeout=0.5)
            except (OSError, ValueError):
                continue
            if message is None:
                continue
            kind = message.get_type()
            if kind == "BAD_DATA":
                continue
            with self._new:
                if kind == "LOCAL_POSITION_NED":
                    self._before = self.latest.get(kind)
                self.latest[kind] = message
                if kind == "COMMAND_ACK":
                    self._acks[message.command] = message
                elif kind == "PARAM_VALUE":
                    self._params[message.param_id] = message.param_value
                elif kind == "STATUSTEXT":
                    self.statustext.append(message.text)
                    del self.statustext[:-50]
                self._new.notify_all()

    def wait_position(self, after_ms: int, timeout: float):
        """The first LOCAL_POSITION_NED stamped at or after `after_ms` of PX4 time."""
        deadline = time.monotonic() + timeout
        with self._new:
            while True:
                message = self.latest.get("LOCAL_POSITION_NED")
                if message is not None and message.time_boot_ms >= after_ms:
                    return message
                left = deadline - time.monotonic()
                if left <= 0 or self._closed:
                    return None
                self._new.wait(left)

    def wait_sample(self, due_ms: int, timeout: float):
        """The first LOCAL_POSITION_NED at or after `due_ms`, with the one before it: (before, after)."""
        deadline = time.monotonic() + timeout
        with self._new:
            while True:
                message = self.latest.get("LOCAL_POSITION_NED")
                if message is not None and message.time_boot_ms >= due_ms:
                    return self._before, message
                left = deadline - time.monotonic()
                if left <= 0 or self._closed:
                    return None
                self._new.wait(left)

    def wait_for(self, kind: str, timeout: float, check=lambda message: True):
        deadline = time.monotonic() + timeout
        with self._new:
            while True:
                message = self.latest.get(kind)
                if message is not None and check(message):
                    return message
                left = deadline - time.monotonic()
                if left <= 0 or self._closed:
                    return None
                self._new.wait(left)

    @property
    def armed(self) -> bool:
        heartbeat = self.latest.get("HEARTBEAT")
        return bool(heartbeat and heartbeat.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    @property
    def on_ground(self) -> bool:
        state = self.latest.get("EXTENDED_SYS_STATE")
        return bool(state and state.landed_state == mavlink.MAV_LANDED_STATE_ON_GROUND)

    def last_words(self, count: int = 4) -> str:
        """PX4's most recent status messages, for an error that needs a reason."""
        return "; ".join(self.statustext[-count:])

    # -- writing ------------------------------------------------------------------

    @property
    def setpoint(self) -> Setpoint | None:
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value: Setpoint | None) -> None:
        self._setpoint = value

    def _stream(self) -> None:
        period = 1.0 / STREAM_HZ
        beat = 0.0
        while not self._closed:
            if time.monotonic() - beat >= 1.0:
                # Announce a ground station, so PX4's "GCS connected" check is met honestly.
                beat = time.monotonic()
                with self._send:
                    self.conn.mav.heartbeat_send(
                        mavlink.MAV_TYPE_GCS, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, mavlink.MAV_STATE_ACTIVE
                    )
            point = self._setpoint
            if point is not None:
                with self._send:
                    self.conn.mav.set_position_target_local_ned_send(
                        0, self.conn.target_system, self.conn.target_component,
                        mavlink.MAV_FRAME_LOCAL_NED, point.mask,
                        *point.position, *point.velocity, 0.0, 0.0, 0.0, point.yaw, 0.0,
                    )
            time.sleep(period)

    def command(self, command: int, *params: float, timeout: float = 3.0) -> int | None:
        """Send a COMMAND_LONG and return PX4's MAV_RESULT, or None if it never answered."""
        params = (list(params) + [0.0] * 7)[:7]
        with self._new:
            self._acks.pop(command, None)
        with self._send:
            self.conn.mav.command_long_send(
                self.conn.target_system, self.conn.target_component, command, 0, *params
            )
        ack = self.wait_for("COMMAND_ACK", timeout, lambda m: self._acks.get(command) is not None)
        return None if ack is None else self._acks[command].result

    def interval(self, message_id: int, hz: float) -> None:
        self.command(mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, message_id, 1e6 / hz)

    def set_param(self, name: str, value: float, *, integer: bool = False, timeout: float = 3.0) -> bool:
        """Set a parameter and wait for PX4 to echo it back; True if it now holds `value`."""
        kind = mavlink.MAV_PARAM_TYPE_INT32 if integer else mavlink.MAV_PARAM_TYPE_REAL32
        # PX4 carries an integer parameter's own four bytes in the float field.
        wire = _int_as_float(int(value)) if integer else float(value)
        with self._new:
            self._params.pop(name, None)
        with self._send:
            self.conn.mav.param_set_send(
                self.conn.target_system, self.conn.target_component, name.encode(), wire, kind
            )
        echo = self.wait_for("PARAM_VALUE", timeout, lambda m: name in self._params)
        if echo is None:
            return False
        held = _float_as_int(self._params[name]) if integer else self._params[name]
        return math.isclose(held, value, abs_tol=1e-4)

    def set_offboard(self) -> int | None:
        return self.command(mavlink.MAV_CMD_DO_SET_MODE, mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, OFFBOARD)

    def arm(self) -> int | None:
        return self.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)

    def kill(self) -> int | None:
        """Stop the motors wherever it is: what hitting a building does to a quadrotor."""
        return self.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, FORCE)

    def land(self) -> int | None:
        """PX4's own landing, straight down from where it is."""
        nan = float("nan")
        return self.command(mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, nan, nan, nan, nan)


def _int_as_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<i", value))[0]


def _float_as_int(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", value))[0]
