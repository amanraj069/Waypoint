"""The aircraft as DroneServer's tools see it: what MavSDK offers, over the flight stack.

Upstream's tools call MavSDK (`drone.action.goto_location(lat, lon, amsl, yaw)`,
`drone.telemetry.position()`, `drone.mission_raw...`, `drone.param...`), which
talks MAVLink to ArduPilot or PX4. `Vehicle` offers the same operations in the
same units and frames, on `compare.common.flight.FlightStack`, with the
behaviour of ArduCopter, the firmware of the paper's core campaign:

* positions are WGS84 latitude and longitude, altitudes are metres above mean
  sea level (absolute) or above home (relative), headings are compass degrees;
* home is where the aircraft last armed, so the relative altitude's datum
  moves with every arming; the session's launch point does not;
* flight modes are named as MavSDK names ArduCopter's: GUIDED is OFFBOARD,
  LOITER is HOLD, AUTO is MISSION, RTL is RETURN_TO_LAUNCH;
* a mission is a list of raw MAVLink items. Item 0 is the HOME slot and is
  never flown (ArduPilot reserves it, as the release's own notes say), a
  NAV_TAKEOFF item climbs, a NAV_WAYPOINT item is flown to and held for its
  hold time, a NAV_RETURN_TO_LAUNCH item returns, lands and disarms. A mission
  started on the ground does not lift off: "ArduCopter will not lift off in
  AUTO without RC throttle input" (release, docs/tool_groups.md);
* MAV_CMD_DO_CHANGE_SPEED is refused unless the aircraft is flying in GUIDED or
  AUTO (the release's transcripts show it refused on the ground, and the tool
  falling back to WPNAV_SPEED);
* take-off is refused in the air; DO_ORBIT is UNSUPPORTED (release firmware notes);
* parameters are a name to number table with ArduCopter's names and units, and
  WPNAV_SPEED (cm/s) really is the cruise speed;
* there is no camera unless one is fitted: "plain SITL has no camera backend"
  (release, list_cameras), which is what the paper's simulations flew.

The local frame is a flat-earth tangent plane at the origin, with the constants
upstream's own offset arithmetic uses (111 320 m per degree of latitude).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from compare.common.flight import LAND, LOITER, OFFBOARD, RTL, TAKEOFF, FlightStack
from waypoint.planner.models import Action

M_PER_DEG_LAT = 111320.0
#: Where ArduPilot's simulator starts (the CMAC field near Canberra), used when the world has no origin.
DEFAULT_ORIGIN = (-35.363262, 149.165237, 584.0)

#: ArduCopter parameters a client is likely to read or write, with ArduCopter's defaults.
DEFAULT_PARAMETERS: dict[str, float] = {
    "WPNAV_SPEED": 1000.0,  # cm/s
    "WPNAV_SPEED_UP": 250.0,
    "WPNAV_SPEED_DN": 150.0,
    "WPNAV_RADIUS": 200.0,  # cm
    "WPNAV_ACCEL": 250.0,
    "RTL_ALT": 1500.0,  # cm
    "RTL_SPEED": 0.0,
    "LAND_SPEED": 50.0,
    "FENCE_ENABLE": 0.0,
    "FENCE_TYPE": 7.0,
    "FENCE_ALT_MAX": 100.0,
    "FENCE_RADIUS": 300.0,
    "FENCE_ACTION": 1.0,
    "ARMING_CHECK": 1.0,
    "BATT_CAPACITY": 3300.0,
    "BATT_LOW_VOLT": 10.5,
    "FS_THR_ENABLE": 1.0,
    "FS_GCS_ENABLE": 0.0,
    "ANGLE_MAX": 4500.0,
    "PILOT_SPEED_UP": 250.0,
    "SYSID_THISMAV": 1.0,
    "DISARM_DELAY": 10.0,
}
INTEGER_PARAMETERS = frozenset(
    {"FENCE_ENABLE", "FENCE_TYPE", "FENCE_ACTION", "ARMING_CHECK", "FS_THR_ENABLE", "FS_GCS_ENABLE", "ANGLE_MAX",
     "SYSID_THISMAV", "BATT_CAPACITY", "DISARM_DELAY"}
)

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21


class ActionError(RuntimeError):
    """The autopilot refused a command (MavSDK's ActionError: "RESULT: 'Text'; origin: ...")."""


def action_error(result: str, origin: str, params: tuple = ()) -> ActionError:
    return ActionError(f"{result}: '{result.replace('_', ' ').title().replace(' ', '')}'; origin: {origin}; params: {params}")


@dataclass
class RawMission:
    """A raw MAVLink mission, as ArduCopter holds it."""

    items: list[dict]
    #: The item being executed (MISSION_CURRENT); item 0 is home and is skipped.
    current: int = 1
    active: bool = False
    paused: bool = False
    finished: bool = False
    #: Seconds left holding at the current waypoint once it is reached.
    holding: float | None = None
    phase: str = ""
    rtl_after: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.items)


class Vehicle:
    """One simulated ArduCopter behind a MavSDK-shaped interface."""

    firmware = "ArduCopter (simulated on Waypoint's simulator)"

    def __init__(
        self,
        stack: FlightStack,
        *,
        origin: tuple[float, float, float] | None = None,
        parameters: dict[str, float] | None = None,
        camera: bool = False,
    ) -> None:
        self.stack = stack
        world = stack.world
        if origin is None and world is not None and world.origin_lat is not None and world.origin_lon is not None:
            origin = (world.origin_lat, world.origin_lon, DEFAULT_ORIGIN[2])
        #: Latitude, longitude and AMSL elevation of the world frame's origin.
        self.origin = origin or DEFAULT_ORIGIN
        self.parameters = dict(DEFAULT_PARAMETERS)
        self.parameters.update(parameters or {})
        self.camera = camera
        self.mission: RawMission | None = None
        self.mode_name = "HOLD"
        # The connector's per-flight latches (upstream: MAVLinkConnector).
        self.pending_destination: dict | None = None
        self.last_movement: dict | None = None
        self.landing_in_progress = False
        self.was_airborne = False
        self.logs_erased = False
        self.photos: list[dict] = []
        self.firmware_fence: dict | None = None
        self.takeoff_altitude = 2.5  # MavSDK's default, until set_takeoff_altitude
        self.rtl_after_mission = False
        #: Set to make every telemetry read fail, for the unknown-state failsafe.
        self.telemetry_lost = False
        self._home_world = stack.state.position
        self._apply_speed()
        #: Where this session started: recorded once, before anything armed.
        position = self.position()
        self.session_launch = {**position, "source": "position at link-up, before any arming"}
        self._listeners: list = []
        stack.on_odom(self._on_tick)

    # -- frames ------------------------------------------------------------------

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        """World east, north in metres to latitude, longitude."""
        lat0, lon0, _ = self.origin
        return (
            lat0 + y / M_PER_DEG_LAT,
            lon0 + x / (M_PER_DEG_LAT * max(math.cos(math.radians(lat0)), 1e-6)),
        )

    def to_world(self, latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
        lat0, lon0, _ = self.origin
        return (
            (longitude_deg - lon0) * M_PER_DEG_LAT * max(math.cos(math.radians(lat0)), 1e-6),
            (latitude_deg - lat0) * M_PER_DEG_LAT,
        )

    def amsl(self, world_z: float) -> float:
        return self.origin[2] + world_z

    # -- telemetry ---------------------------------------------------------------

    def position(self) -> dict:
        x, y, z = self.stack.state.reported_position
        lat, lon = self.to_latlon(x, y)
        return {
            "latitude_deg": lat,
            "longitude_deg": lon,
            "absolute_altitude_m": self.amsl(z),
            "relative_altitude_m": z - self._home_world[2],
        }

    def home(self) -> dict:
        lat, lon = self.to_latlon(self._home_world[0], self._home_world[1])
        return {"latitude_deg": lat, "longitude_deg": lon, "absolute_altitude_m": self.amsl(self._home_world[2])}

    def velocity_ned(self) -> tuple[float, float, float]:
        vx, vy, vz = self.stack.velocity
        return (vy, vx, -vz)

    @property
    def armed(self) -> bool:
        return self.stack.armed

    @property
    def in_air(self) -> bool:
        return not self.stack.on_ground

    @property
    def landed_state(self) -> str:
        if self.stack.on_ground:
            return "ON_GROUND"
        if self.stack.mode == TAKEOFF:
            return "TAKING_OFF"
        if self.stack.mode in (LAND,) or (self.stack.mode == RTL and self.stack._phase.get("step") == "land"):
            return "LANDING"
        return "IN_AIR"

    @property
    def flight_mode(self) -> str:
        """MavSDK's name for the current ArduCopter mode."""
        if self.stack.mode == LAND:
            return "LAND"
        if self.stack.mode == RTL:
            return "MISSION" if self.mission is not None and self.mission.phase == "rtl" else "RETURN_TO_LAUNCH"
        return self.mode_name

    @property
    def heading_deg(self) -> float:
        """Compass heading: 0 north, 90 east."""
        return (90.0 - self.stack.yaw_deg) % 360.0

    def battery(self) -> dict:
        """MavSDK's battery: voltage, and remaining as a fraction (0 to 1)."""
        fraction = self.stack.battery_pct / 100.0
        return {"voltage_v": round(12.6 - (1.0 - fraction) * 2.1, 2), "remaining_fraction": fraction}

    def distance_to(self, latitude_deg: float, longitude_deg: float) -> float:
        here = self.position()
        tx, ty = self.to_world(latitude_deg, longitude_deg)
        hx, hy = self.to_world(here["latitude_deg"], here["longitude_deg"])
        return math.hypot(tx - hx, ty - hy)

    # -- actions -----------------------------------------------------------------

    def arm(self, force: bool = False) -> None:
        if self.stack.ended:
            raise action_error("COMMAND_DENIED", "arm()")
        if not self.stack.armed and self.stack.on_ground and self.stack.mode in (LAND, RTL):
            # MavSDK's arm() on ArduPilot first leaves a mode that cannot arm (LAND, RTL, AUTO) for
            # HOLD, then arms: how a model re-arms after an auto-landing in the authors' runs.
            self._stop_mission()
            self.stack.stop_stream()
            self.stack.set_mode(LOITER)
            self.mode_name = "HOLD"
        if not self.stack.armed:
            if not self.stack.arm(True):
                raise action_error("COMMAND_DENIED", "arm()")
            # ArduPilot moves home to wherever the aircraft arms.
            self._home_world = self.stack.state.position

    def disarm(self) -> None:
        if not self.stack.arm(False):
            raise action_error("COMMAND_DENIED", "disarm()")

    def set_takeoff_altitude(self, altitude_m: float) -> None:
        self.takeoff_altitude = float(altitude_m)

    def takeoff(self) -> None:
        """ArduCopter's GUIDED take-off: armed, on the ground, climbs to the take-off altitude and holds."""
        if not self.stack.armed or self.in_air:
            raise action_error("COMMAND_DENIED", "takeoff()")
        self._stop_mission()
        self.stack.takeoff_altitude = self.takeoff_altitude
        if not self.stack.set_mode(TAKEOFF):
            raise action_error("FAILED", "takeoff()")
        self.mode_name = "OFFBOARD"

    def land(self) -> None:
        self._stop_mission()
        self.stack.stop_stream()
        if not self.stack.set_mode(LAND):
            raise action_error("FAILED", "land()")

    def return_to_launch(self, from_mission: bool = False) -> None:
        if not from_mission:
            self._stop_mission()
        self.stack.rtl_altitude = max(self.parameters.get("RTL_ALT", 1500.0) / 100.0, 0.0)
        self.stack.rtl_home = self._home_world
        if not self.stack.set_mode(RTL):
            raise action_error("FAILED", "return_to_launch()")

    def hold(self) -> None:
        """MavSDK's action.hold(): LOITER on ArduCopter."""
        self._stop_mission(pause=True)
        x, y, z = self.stack.state.reported_position
        self._seek_world(x, y, z, math.nan)
        self.mode_name = "HOLD"

    def goto(self, latitude_deg: float, longitude_deg: float, absolute_altitude_m: float, yaw_deg: float = math.nan) -> None:
        """MAV_CMD_DO_REPOSITION in GUIDED. ArduCopter accepts it only when armed."""
        if not self.stack.armed:
            raise action_error("COMMAND_DENIED", "goto_location()")
        self._stop_mission()
        self._seek_world(*self.to_world(latitude_deg, longitude_deg), absolute_altitude_m - self.origin[2], yaw_deg)
        self.mode_name = "OFFBOARD"

    def set_current_speed(self, speed_m_s: float) -> None:
        """MAV_CMD_DO_CHANGE_SPEED: ArduCopter acts on it only while flying in GUIDED or AUTO."""
        if not (self.in_air and self.armed and self.flight_mode in ("OFFBOARD", "MISSION")):
            raise action_error("FAILED", "set_current_speed()", (float(speed_m_s),))
        self.stack.cruise_speed = max(0.2, float(speed_m_s))

    def do_orbit(self, *args: Any) -> None:
        raise action_error("UNSUPPORTED", "do_orbit()", tuple(args))

    def kill(self) -> None:
        self._stop_mission()
        self.stack.kill()

    def _seek_world(self, x: float, y: float, z: float, heading_deg: float) -> None:
        local = self.stack.local_of((x, y, z))
        yaw = None if math.isnan(heading_deg) else math.radians(90.0 - heading_deg)
        self.stack.set_target(local[0], local[1], local[2], yaw)
        if not self.stack.set_mode(OFFBOARD):
            raise action_error("FAILED", "goto_location()")

    # -- parameters --------------------------------------------------------------

    def get_param(self, name: str) -> float:
        if name not in self.parameters:
            raise KeyError(name)
        return self.parameters[name]

    def set_param(self, name: str, value: float) -> float:
        if name not in self.parameters:
            raise KeyError(name)
        old = self.parameters[name]
        self.parameters[name] = float(value)
        self._apply_speed()
        return old

    def _apply_speed(self) -> None:
        self.stack.cruise_speed = max(0.2, self.parameters.get("WPNAV_SPEED", 1000.0) / 100.0)

    # -- raw missions --------------------------------------------------------------

    def upload_raw(self, items: list[dict]) -> None:
        """MissionRaw.upload_mission: replaces the mission on the autopilot."""
        if not items:
            raise action_error("INVALID_ARGUMENT", "upload_mission()")
        self._stop_mission()
        self.mission = RawMission(items=[dict(item) for item in items], current=1 if len(items) > 1 else 0)

    def clear_mission(self) -> None:
        self._stop_mission()
        self.mission = None

    def start_mission(self) -> None:
        """Switch to AUTO and run the mission from its current item."""
        mission = self.mission
        if mission is None or mission.total <= 1:
            raise action_error("NO_MISSION_AVAILABLE", "start_mission()")
        mission.active, mission.paused = True, False
        if mission.finished:
            mission.finished, mission.current = False, 1
        mission.current = max(1, mission.current)
        self.mode_name = "MISSION"
        if self.in_air and self.armed:
            self._begin_item()
        # On the ground ArduCopter sits in AUTO at 0 m: no lift-off without RC throttle.

    def pause_mission(self) -> None:
        """MavSDK mission pause on ArduCopter: LOITER where it is, the mission keeps its place."""
        if self.mission is not None and self.mission.active:
            self.mission.paused = True
        x, y, z = self.stack.state.reported_position
        self._seek_world(x, y, z, math.nan)
        self.mode_name = "HOLD"

    def set_current_item(self, index: int) -> None:
        if self.mission is None or not 0 <= index < self.mission.total:
            raise action_error("INVALID_ARGUMENT", "set_current_mission_item()", (index,))
        self.mission.current = max(1, index)
        self.mission.finished = False
        if self.mission.active and not self.mission.paused and self.in_air:
            self._begin_item()

    def mission_progress(self) -> tuple[int, int] | None:
        """MISSION_CURRENT as MavSDK reports it: (current, total), or None with no mission."""
        if self.mission is None:
            return None
        return (self.mission.total if self.mission.finished else self.mission.current, self.mission.total)

    def mission_finished(self) -> bool:
        return self.mission is not None and self.mission.finished

    def _stop_mission(self, pause: bool = False) -> None:
        if self.mission is not None and self.mission.active:
            if pause:
                self.mission.paused = True
            else:
                self.mission.active = False

    def _item_target(self, item: dict) -> tuple[float, float, float]:
        x, y = self.to_world(item["x"] / 1e7, item["y"] / 1e7)
        return (x, y, self._home_world[2] + float(item["z"]))

    def _begin_item(self) -> None:
        mission = self.mission
        while mission.current < mission.total:
            item = mission.items[mission.current]
            command = int(item.get("command", MAV_CMD_NAV_WAYPOINT))
            mission.holding = None
            if command == MAV_CMD_NAV_TAKEOFF:
                altitude = self._home_world[2] + float(item["z"])
                if self.stack.state.z >= altitude - 1.0:
                    mission.current += 1  # already up there: the take-off item completes at once
                    continue
                here = self.stack.state.reported_position
                mission.phase = "takeoff"
                self._seek_world(here[0], here[1], altitude, math.nan)
                self.mode_name = "MISSION"
                return
            if command == MAV_CMD_NAV_RETURN_TO_LAUNCH:
                mission.phase = "rtl"
                self.return_to_launch(from_mission=True)
                return
            if command == MAV_CMD_NAV_LAND:
                mission.phase = "land"
                self.stack.stop_stream()
                self.stack.set_mode(LAND)
                return
            if command == MAV_CMD_NAV_WAYPOINT:
                mission.phase = "waypoint"
                self._seek_world(*self._item_target(item), math.nan)
                self.mode_name = "MISSION"
                return
            mission.current += 1  # a command this vehicle does not model: skipped
        self._end_mission()

    def _end_mission(self) -> None:
        mission = self.mission
        mission.finished, mission.active = True, False
        mission.current = mission.total
        if mission.rtl_after and self.in_air:
            self.return_to_launch(from_mission=True)
        # Otherwise ArduCopter holds over the last waypoint, still in AUTO.

    def _on_tick(self, *_odom: Any) -> None:
        mission = self.mission
        if mission is None or not mission.active or mission.paused or self.stack.ended:
            return
        if not self.in_air:
            if mission.phase in ("rtl", "land") and not self.stack.armed:
                mission.current = mission.total
                mission.finished, mission.active = True, False
            return
        if mission.phase == "" or (self.stack.mode == OFFBOARD and not self.stack.streaming):
            self._begin_item()
            return
        if mission.phase in ("rtl", "land"):
            return
        item = mission.items[mission.current]
        target = self._item_target(item) if mission.phase == "waypoint" else None
        if mission.phase == "takeoff":
            if self.stack.state.z < self._home_world[2] + float(item["z"]) - 1.0:
                return
        else:
            radius = float(item.get("param2") or 0.0) or self.parameters.get("WPNAV_RADIUS", 200.0) / 100.0
            if math.dist(self.stack.state.reported_position, target) > max(radius, 0.5):
                return
            if mission.holding is None:
                mission.holding = float(item.get("param1") or 0.0)
                self.stack.mark_waypoint(Action.MOVE, f"mission item {mission.current}/{mission.total}")
            if mission.holding > 0:
                mission.holding -= self.stack.dt
                return
        mission.current += 1
        self._begin_item()

    # -- camera ------------------------------------------------------------------

    def take_photo(self) -> dict:
        if not self.camera:
            raise action_error("NO_SYSTEM", "take_photo()", (100,))
        here = self.position()
        photo = {"index": len(self.photos), **here, "time_s": round(self.stack.state.timestamp, 1)}
        self.photos.append(photo)
        self.stack.mark_waypoint(Action.CAPTURE, f"photo {photo['index']}")
        return photo

    # -- time --------------------------------------------------------------------

    def sleep(self, seconds: float) -> None:
        """Let the aircraft fly for `seconds`; a tool's `asyncio.sleep` while it waits on the vehicle."""
        self.stack.advance(seconds)

    @property
    def now(self) -> float:
        return self.stack.now
