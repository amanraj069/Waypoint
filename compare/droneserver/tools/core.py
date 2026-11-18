"""DroneServer's tools, on the simulated vehicle.

Upstream registers 98 tools over 223 MavSDK methods. All 98 are registered
here under their released names, with their released parameters and their
released descriptions (`data/tool_catalog.json`, extracted from the upstream
source), so the catalogue a model is handed at `tools/list` is the released
one and every tool passes the released tier table.

Fifty-five of them are implemented, each ported from its release body
(commit `06b7d1a`, `src/droneserver/tools/*.py`): the same MavSDK calls, in
the same order, on `compare.droneserver.vehicle.Vehicle`, which answers them as
ArduCopter does; the same checks, the same result keys and the same wording.
That covers the flight actions, `monitor_flight`, telemetry, raw and managed
missions (`managed.py` is the release's server-side mission runner),
parameters, the geofence, the emergency stop, flight logs, the camera, and
system information. The other 43 (offboard setpoints, gimbal, calibration,
MAVLink FTP and shell, follow-me, winch, transponder, RTK, ...) have no
counterpart on a point-mass simulator; they answer `status: "failed"` with a
plain reason, after passing the same guard as any other call.

Where a release tool waits on the aircraft (`asyncio.sleep` while it climbs,
`monitor_flight`'s thirty seconds, a landing), the wait is spent in flight time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from compare.droneserver.geo import haversine_distance
from compare.droneserver.managed import MissionRunner, Phase, ground_evidence
from compare.droneserver.vehicle import (
    INTEGER_PARAMETERS,
    MAV_CMD_NAV_WAYPOINT,
    ActionError,
    Vehicle,
    action_error,
)

CATALOG: dict[str, dict] = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "tool_catalog.json").read_text(encoding="utf-8")
)
#: The release's constants (tools/action.py, tools/telemetry.py).
COMPLETION_RADIUS_M = 20.0
HOME_DIVERGENCE_M = 20.0
LANDING_GATE_M = 20.0
MONITOR_WINDOW_S = 30.0
ARM_CONFIRM_TIMEOUT_S = 10.0
ARM_POLL_INTERVAL_S = 0.5
SETTLED_RATE_M_S = 1.0
#: MavSDK's command latency: a command and the telemetry read after it are not instantaneous.
COMMAND_S = 0.2
CONN_ERROR = {"status": "failed", "error": "Drone connection timeout. Please wait and try again."}


def _ok(**payload: Any) -> dict:
    return {"status": "success", **payload}


def _fail(error: str, **extra: Any) -> dict:
    return {"status": "failed", "error": error, **extra}


def implemented(vehicle: Vehicle, fence: Callable | None = None) -> dict[str, Callable]:
    """Every tool implemented on the simulated vehicle, by its released name.

    `fence` returns the safety layer's geofence; the managed-mission runner
    checks a mission against it before uploading, as the release's does.
    """
    tools: dict[str, Callable] = {}
    runner = MissionRunner(vehicle, fence or (lambda: None))

    def tool(fn: Callable) -> Callable:
        tools[fn.__name__] = fn
        return fn

    # -- the connector's per-flight latches and observables (release: tools/action.py) ---------

    def reset_flight_latches() -> None:
        vehicle.landing_in_progress = False
        vehicle.was_airborne = False
        vehicle.pending_destination = None
        vehicle.last_movement = None

    def note_movement(name: str, target: dict | None) -> None:
        vehicle.last_movement = {"tool": name, "target": target, "commanded_at": vehicle.stack.now}

    def target_of() -> dict | None:
        if vehicle.pending_destination:
            return vehicle.pending_destination
        return (vehicle.last_movement or {}).get("target")

    def telemetry_now() -> dict:
        position = vehicle.position()
        north, east, down = vehicle.velocity_ned()
        return {
            "landed_state": vehicle.landed_state,
            "in_air": vehicle.in_air,
            **position,
            "ground_speed_m_s": math.hypot(north, east),
            "vertical_speed_m_s": -down,
            "flight_mode": vehicle.flight_mode,
        }

    def observables(reading: dict) -> dict:
        lat, lon = reading["latitude_deg"], reading["longitude_deg"]
        launch = vehicle.session_launch
        fields: dict = {
            "position": {"latitude_deg": lat, "longitude_deg": lon},
            "altitude_m": round(reading["relative_altitude_m"], 1),
            "height_above_launch_m": round(reading["absolute_altitude_m"] - launch["absolute_altitude_m"], 1),
            "absolute_altitude_m": round(reading["absolute_altitude_m"], 1),
            "vertical_speed_m_s": round(reading["vertical_speed_m_s"], 1),
            "ground_speed_m_s": round(reading["ground_speed_m_s"], 1),
            "flight_mode": reading["flight_mode"],
            "target": None,
            "distance_to_target_m": None,
            "distance_from_launch_point_m": round(haversine_distance(lat, lon, launch["latitude_deg"], launch["longitude_deg"]), 1),
        }
        target = target_of()
        if target:
            fields["target"] = {
                "latitude_deg": target["latitude"], "longitude_deg": target["longitude"],
                "label": target.get("label", "the commanded destination"),
            }
            fields["distance_to_target_m"] = round(haversine_distance(lat, lon, target["latitude"], target["longitude"]), 1)
        return fields

    def where_text(fields: dict) -> str:
        parts = []
        if fields["distance_to_target_m"] is not None:
            parts.append(f"{fields['distance_to_target_m']:.0f}m from {(fields['target'] or {}).get('label', 'target')}")
        if fields["distance_from_launch_point_m"] is not None:
            parts.append(f"{fields['distance_from_launch_point_m']:.0f}m from launch point")
        if fields["vertical_speed_m_s"] is not None:
            parts.append(f"{fields['vertical_speed_m_s']:+.1f}m/s vertical")
        if fields["position"]:
            parts.append(f"at {fields['position']['latitude_deg']:.6f},{fields['position']['longitude_deg']:.6f}")
        return " | ".join(parts)

    def suffix(where: str) -> str:
        return f" | {where}" if where else ""

    def vertical_verb(fields: dict) -> str:
        rate = fields.get("vertical_speed_m_s")
        if rate is None:
            return "Landing in progress"
        if rate < -0.3:
            return "Descending"
        if rate > 0.3:
            return "CLIMBING (not descending)"
        return "Holding altitude"

    def settled_on_ground(reading: dict) -> bool:
        return (reading["landed_state"] == "ON_GROUND" and not reading["in_air"]
                and abs(reading["vertical_speed_m_s"]) <= SETTLED_RATE_M_S)

    def wait_until_armed() -> tuple[bool, float]:
        waited = 0.0
        while True:
            if vehicle.armed:
                return True, waited
            if waited >= ARM_CONFIRM_TIMEOUT_S:
                return False, waited
            vehicle.sleep(ARM_POLL_INTERVAL_S)
            waited += ARM_POLL_INTERVAL_S

    # -- action ------------------------------------------------------------------

    @tool
    async def arm_drone(force: bool = False) -> dict:
        reset_flight_latches()
        try:
            vehicle.arm(force=force)
        except ActionError as e:
            return {"status": "failed", "armed": False, "error": f"Arming failed: {e}"}
        vehicle.sleep(COMMAND_S)
        armed, waited = wait_until_armed()
        label = "FORCE-armed (prearm checks bypassed)" if force else "armed"
        if armed:
            return {
                "status": "success", "message": f"Drone {label}", "armed": True,
                "evidence": f"the autopilot reported the vehicle armed {waited:.1f}s after the command",
            }
        return {
            "status": "failed",
            "armed": False,
            "error": (
                f"the autopilot accepted the {'force-' if force else ''}arm command but the vehicle is still "
                f"DISARMED {waited:.0f}s later. It is not armed, and nothing that needs an armed vehicle - "
                "takeoff, any navigation command - will work."
            ),
            "remedy": (
                "Read get_health / print_status_text for what the autopilot is refusing on (a prearm check, a "
                "safety switch, a failsafe, or a mode that will not arm), fix that, and arm again. Do not "
                "command a takeoff: the aircraft is on the ground and disarmed."
                + ("" if force else " force=True bypasses PREARM checks only; it cannot make the autopilot arm.")
            ),
        }

    @tool
    async def disarm_drone() -> dict:
        try:
            vehicle.disarm()
            return {"status": "success", "message": "Drone disarmed - motors stopped"}
        except ActionError as e:
            return {"status": "failed", "error": f"Disarm failed: {e}"}

    @tool
    async def takeoff(takeoff_altitude: float = 3.0, wait_for_altitude: bool = True) -> dict:
        # As released, outside any try: a refused take-off surfaces as a tool error.
        vehicle.set_takeoff_altitude(takeoff_altitude)
        vehicle.takeoff()
        if not wait_for_altitude:
            return {
                "status": "success",
                "message": f"Takeoff initiated to {takeoff_altitude}m AGL (relative)",
                "warning": "⚠️ Takeoff in progress - do NOT send navigation commands until altitude is reached!",
            }
        altitude_threshold, max_wait_time, check_interval = 0.5, 60, 1.0
        elapsed = 0.0
        while elapsed < max_wait_time:
            current = vehicle.position()["relative_altitude_m"]
            if current >= takeoff_altitude - altitude_threshold:
                return {
                    "status": "success",
                    "message": f"Takeoff complete - drone at {current:.1f}m AGL",
                    "altitude_reached_m": round(current, 1),
                    "target_altitude_m": takeoff_altitude,
                    "safe_to_navigate": True,
                }
            vehicle.sleep(check_interval)
            elapsed += check_interval
        current = vehicle.position()["relative_altitude_m"]
        return {
            "status": "warning",
            "message": f"Takeoff timeout - drone at {current:.1f}m (target was {takeoff_altitude}m)",
            "altitude_reached_m": round(current, 1),
            "target_altitude_m": takeoff_altitude,
            "safe_to_navigate": current >= (takeoff_altitude - altitude_threshold),
        }

    @tool
    async def land(force: bool = False) -> dict:
        if not vehicle.armed and not vehicle.in_air:
            return {
                "status": "no_action",
                "message": "No landing was commanded: the aircraft is already on the ground with its motors disarmed.",
                "armed": False,
                "in_air": False,
                "next_step": (
                    "Nothing is flying. If you expected the aircraft to be somewhere else, read get_position "
                    "and compare it with the coordinate you meant to reach - do not treat this call as a landing."
                ),
            }
        if vehicle.pending_destination and not force:
            dest = vehicle.pending_destination
            position = vehicle.position()
            distance = haversine_distance(position["latitude_deg"], position["longitude_deg"], dest["latitude"], dest["longitude"])
            if distance > LANDING_GATE_M:
                return {
                    "status": "blocked",
                    "message": f"Cannot land - drone is {distance:.0f}m from destination!",
                    "distance_to_destination_m": round(distance, 1),
                    "current_position": {"latitude": position["latitude_deg"], "longitude": position["longitude_deg"]},
                    "destination": {"latitude": dest["latitude"], "longitude": dest["longitude"]},
                    "recommendation": "Call monitor_flight() to check progress, or use land(force=True) for emergency landing",
                    "safe_to_land": False,
                }
            vehicle.pending_destination = None
        vehicle.pending_destination = None
        vehicle.landing_in_progress = True
        note_movement("land", target_of())
        vehicle.land()
        return {"status": "success", "message": "Landing initiated", "next_step": "Call monitor_flight() until mission_complete is true"}

    @tool
    async def return_to_launch() -> dict:
        try:
            vehicle.return_to_launch()
        except ActionError as e:
            return {"status": "failed", "error": f"Return to Launch failed: {e}"}
        result: dict = {"status": "success", "message": "Return to Launch initiated - the aircraft is flying to the autopilot's home position"}
        home = vehicle.home()
        target = {"latitude": home["latitude_deg"], "longitude": home["longitude_deg"], "label": "the autopilot's home"}
        result["destination"] = {**home, "note": "this is where RTL will fly - the autopilot's home, which moves to wherever it last armed"}
        position = vehicle.position()
        vehicle.pending_destination = {
            **target,
            "altitude_msl": home["absolute_altitude_m"],
            "initial_distance": haversine_distance(position["latitude_deg"], position["longitude_deg"], home["latitude_deg"], home["longitude_deg"]),
            "start_time": vehicle.stack.now,
            "source": "return_to_launch",
        }
        note_movement("return_to_launch", target)
        launch = vehicle.session_launch
        drift = haversine_distance(home["latitude_deg"], home["longitude_deg"], launch["latitude_deg"], launch["longitude_deg"])
        result["distance_from_session_launch_point_m"] = round(drift, 1)
        if drift > COMPLETION_RADIUS_M:
            result["warning"] = (
                f"the autopilot's home is {drift:.0f} m from where this session started "
                f"({launch['latitude_deg']:.6f},{launch['longitude_deg']:.6f}) - this return will fly to "
                "the home coordinate above, NOT to the launch point. If you want the launch point, "
                "fly there with go_to_location instead."
            )
        return result

    @tool
    async def kill_motors() -> dict:
        try:
            vehicle.kill()
        except ActionError as e:
            return {"status": "failed", "error": f"Motor kill failed: {e}"}
        return {"status": "success", "message": "EMERGENCY: Motors killed - drone will fall!", "warning": "This is an emergency action. Drone may be damaged."}

    def goto_here(yaw_deg: float = float("nan")) -> dict:
        """The release's hold: goto_location to where the aircraft is now (GUIDED, not LOITER)."""
        position = vehicle.position()
        vehicle.goto(position["latitude_deg"], position["longitude_deg"], position["absolute_altitude_m"], yaw_deg)
        return position

    @tool
    async def hold_position() -> dict:
        try:
            position = goto_here()
        except ActionError as e:
            return {"status": "failed", "error": f"Hold position failed: {e}"}
        return {
            "status": "success",
            "message": "Drone holding position in GUIDED mode",
            "position": {
                "latitude_deg": position["latitude_deg"], "longitude_deg": position["longitude_deg"],
                "altitude_m": position["relative_altitude_m"], "altitude_rel": position["relative_altitude_m"],
            },
            "note": "Using GUIDED mode instead of LOITER to prevent altitude drift",
        }

    @tool
    async def go_to_location(latitude_deg: float, longitude_deg: float, absolute_altitude_m: float, yaw_deg: float = float("nan")) -> dict:
        if not (-90 <= latitude_deg <= 90):
            return {"status": "failed", "error": f"Invalid latitude: {latitude_deg}. Must be between -90 and 90."}
        if not (-180 <= longitude_deg <= 180):
            return {"status": "failed", "error": f"Invalid longitude: {longitude_deg}. Must be between -180 and 180."}
        try:
            position = vehicle.position()
            home_alt = position["absolute_altitude_m"] - position["relative_altitude_m"]
            relative_alt = absolute_altitude_m - home_alt
            initial_distance = haversine_distance(position["latitude_deg"], position["longitude_deg"], latitude_deg, longitude_deg)
            north, east, _ = vehicle.velocity_ned()
            eta_seconds = initial_distance / max(math.hypot(north, east), 10.0)
            vehicle.goto(latitude_deg, longitude_deg, absolute_altitude_m, yaw_deg)
            vehicle.pending_destination = {
                "latitude": latitude_deg, "longitude": longitude_deg, "label": "the commanded destination",
                "altitude_msl": absolute_altitude_m, "initial_distance": initial_distance,
                "start_time": vehicle.stack.now, "source": "go_to_location",
            }
            note_movement("go_to_location", {"latitude": latitude_deg, "longitude": longitude_deg, "label": "the commanded destination"})
            return {
                "status": "success",
                "message": "Navigation started. Call monitor_flight() to track progress.",
                "initial_distance_m": round(initial_distance, 1),
                "estimated_flight_time_seconds": round(eta_seconds, 0),
                "target": {
                    "latitude": latitude_deg, "longitude": longitude_deg, "altitude_msl": absolute_altitude_m,
                    "altitude_agl": round(relative_alt, 1), "yaw": yaw_deg if not math.isnan(yaw_deg) else "maintain current",
                },
                "next_step": "Call monitor_flight() repeatedly until mission_complete is true",
            }
        except ActionError as e:
            return {"status": "failed", "error": f"Navigation failed: {e}"}

    @tool
    async def move_to_relative(north_m: float, east_m: float, down_m: float, yaw_deg: float = 0.0) -> dict:
        try:
            position = vehicle.position()
            target_alt = position["absolute_altitude_m"] - down_m
            target_lat = position["latitude_deg"] + north_m / 111320.0
            target_lon = position["longitude_deg"] + east_m / (111320.0 * math.cos(math.radians(position["latitude_deg"])))
            vehicle.goto(target_lat, target_lon, target_alt, yaw_deg)
            note_movement("move_to_relative", {"latitude": target_lat, "longitude": target_lon, "label": "the offset target"})
            return {
                "status": "success",
                "message": f"Moving: north={north_m}m, east={east_m}m, altitude_change={-down_m}m",
                "target_position": {"latitude_deg": target_lat, "longitude_deg": target_lon, "altitude_m": target_alt},
            }
        except ActionError as e:
            return {"status": "failed", "error": f"Movement failed: {e}"}

    @tool
    async def reposition(latitude_deg: float, longitude_deg: float, altitude_m: float) -> dict:
        if not (-90 <= latitude_deg <= 90):
            return {"status": "failed", "error": f"Invalid latitude: {latitude_deg}. Must be between -90 and 90."}
        if not (-180 <= longitude_deg <= 180):
            return {"status": "failed", "error": f"Invalid longitude: {longitude_deg}. Must be between -180 and 180."}
        try:
            vehicle.goto(latitude_deg, longitude_deg, altitude_m, float("nan"))
        except ActionError as e:
            return {"status": "failed", "error": f"Reposition failed: {e}"}
        note_movement("reposition", {"latitude": latitude_deg, "longitude": longitude_deg, "label": "the repositioning target"})
        return {
            "status": "success",
            "message": "Repositioning to new location",
            "target": {"latitude": latitude_deg, "longitude": longitude_deg, "altitude_msl": altitude_m},
            "note": "Drone will fly to location and loiter (hover) there",
        }

    @tool
    async def check_arrival(latitude_deg: float, longitude_deg: float, threshold_m: float = 10.0) -> dict:
        if not (-90 <= latitude_deg <= 90):
            return {"status": "failed", "error": f"Invalid latitude: {latitude_deg}. Must be between -90 and 90."}
        if not (-180 <= longitude_deg <= 180):
            return {"status": "failed", "error": f"Invalid longitude: {longitude_deg}. Must be between -180 and 180."}
        position = vehicle.position()
        distance = haversine_distance(position["latitude_deg"], position["longitude_deg"], latitude_deg, longitude_deg)
        body = {
            "distance_m": round(distance, 1),
            "current_position": {"latitude": position["latitude_deg"], "longitude": position["longitude_deg"], "altitude_m": position["relative_altitude_m"]},
            "target": {"latitude": latitude_deg, "longitude": longitude_deg},
        }
        if distance <= threshold_m:
            return {"status": "arrived", "message": f"Drone has arrived at target location! Distance: {distance:.1f}m", **body}
        return {"status": "in_progress", "message": f"Still {distance:.1f}m from target. Call check_arrival again in a few seconds.", **body}

    @tool
    async def monitor_flight(arrival_threshold_m: float = 20.0, auto_land: bool = True) -> dict:
        reading = telemetry_now()
        current_alt = reading["relative_altitude_m"]
        live = observables(reading)
        where = where_text(live)
        grounded = ground_evidence(reading["landed_state"], reading["in_air"])
        if grounded is False:
            vehicle.was_airborne = True
        on_the_ground = grounded is True
        if on_the_ground and not vehicle.was_airborne:
            headline = "no flight is under way (the previous one has ended)" if vehicle.last_movement is not None else "the drone has not flown yet"
            return {
                "DISPLAY_TO_USER": f"⚠️ ON THE GROUND | Alt: {current_alt:.1f}m | {headline}" + suffix(where),
                "status": "not_started",
                **live,
                "action_required": (
                    "The drone is on the ground and nothing is flying, so there is nothing to monitor. "
                    "This is NOT a completed mission. If the aircraft still has somewhere to be, arm and "
                    "take off (or start the mission), then call monitor_flight() again."
                ),
                "mission_complete": False,
            }
        if on_the_ground:
            target = target_of()
            missed_by = live["distance_to_target_m"]
            vehicle.landing_in_progress = False
            if target and missed_by is not None and missed_by > max(arrival_threshold_m, COMPLETION_RADIUS_M):
                label = live["target"]["label"]
                return {
                    "DISPLAY_TO_USER": f"⚠️ ON THE GROUND, NOT AT THE TARGET | {missed_by:.0f}m from {label}" + suffix(where),
                    "status": "landed_away_from_target",
                    **live,
                    "landed_away_from_target": {"distance_m": missed_by, "target": live["target"], "commanded_by": (vehicle.last_movement or {}).get("tool")},
                    "action_required": (
                        f"The aircraft is on the ground {missed_by:.0f} m from {label}, so the flight you "
                        "commanded did not happen or did not finish. Do NOT report it as completed. Check "
                        "get_position and get_armed, then fly the remaining leg (arm, takeoff, "
                        "go_to_location) if you still want to reach it."
                    ),
                    "mission_complete": False,
                }
            vehicle.was_airborne = False
            vehicle.pending_destination = None
            return {
                "DISPLAY_TO_USER": "✅ LANDED | this monitored flight has ended" + suffix(where),
                "status": "landed",
                **live,
                "action_required": (
                    "This flight is over. It is not necessarily your whole task: if the task has another "
                    "leg, fly it. Check the position above against where the task asked the aircraft to end up."
                ),
                "mission_complete": True,
            }
        if reading["landed_state"] == "LANDING" or vehicle.landing_in_progress:
            return {
                "DISPLAY_TO_USER": f"🛬 LANDING PHASE | Alt: {current_alt:.1f}m | {vertical_verb(live)}" + suffix(where),
                "status": "landing",
                **live,
                "action_required": "SHOW the DISPLAY_TO_USER to user, then CALL monitor_flight() AGAIN",
                "mission_complete": False,
            }
        if not vehicle.pending_destination:
            return {
                "DISPLAY_TO_USER": f"🚁 HOVERING | Alt: {current_alt:.1f}m | No destination set" + suffix(where),
                "status": "hovering",
                **live,
                "action_required": "Call go_to_location() to set destination, or land() to land here",
                "mission_complete": False,
            }
        dest = vehicle.pending_destination
        initial_distance = dest["initial_distance"]
        start_time = dest.get("start_time", vehicle.stack.now)
        elapsed = 0.0
        distance = ground_speed = progress = 0.0
        eta_seconds = None
        while elapsed < MONITOR_WINDOW_S:
            position = vehicle.position()
            current_alt = position["relative_altitude_m"]
            distance = haversine_distance(position["latitude_deg"], position["longitude_deg"], dest["latitude"], dest["longitude"])
            if initial_distance > 0:
                progress = max(0.0, min(100.0, (initial_distance - distance) / initial_distance * 100))
            else:
                progress = 100.0 if distance <= arrival_threshold_m else 0.0
            north, east, _ = vehicle.velocity_ned()
            ground_speed = math.hypot(north, east)
            eta_seconds = distance / ground_speed if ground_speed > 0.5 else None
            if distance <= arrival_threshold_m:
                vehicle.pending_destination = None
                if auto_land:
                    vehicle.landing_in_progress = True
                    vehicle.land()
                    landing_start = vehicle.stack.now
                    while vehicle.stack.now - landing_start < 120:
                        touchdown = telemetry_now()
                        current_alt = touchdown["relative_altitude_m"]
                        if settled_on_ground(touchdown):
                            vehicle.sleep(3.0)
                            if settled_on_ground(telemetry_now()):
                                vehicle.landing_in_progress = False
                                vehicle.was_airborne = False
                                total_flight_time = vehicle.stack.now - start_time
                                live = observables(telemetry_now())
                                return {
                                    "DISPLAY_TO_USER": f"✅ LANDED AT THE COMMANDED DESTINATION | Flight time: {total_flight_time:.0f}s" + suffix(where_text(live)),
                                    "status": "landed",
                                    **live,
                                    "flight_time_seconds": round(total_flight_time, 0),
                                    "action_required": (
                                        "This leg is finished. If your task also asks the aircraft to go "
                                        "somewhere else afterwards, that has NOT happened yet - the aircraft "
                                        "is at the position above, disarming or disarmed."
                                    ),
                                    "mission_complete": True,
                                }
                        if vehicle.stack.ended:
                            break
                        vehicle.sleep(2.0)
                    live = observables(telemetry_now())
                    return {
                        "DISPLAY_TO_USER": f"⚠️ LANDING TIMEOUT | Alt: {current_alt:.1f}m | Check drone status" + suffix(where_text(live)),
                        "status": "landing_timeout",
                        **live,
                        "mission_complete": False,
                    }
                live = observables(telemetry_now())
                return {
                    "DISPLAY_TO_USER": f"✅ ARRIVED | Distance: {distance:.1f}m | Alt: {current_alt:.1f}m | Call land() to land" + suffix(where_text(live)),
                    "status": "arrived",
                    "distance_m": round(distance, 1),
                    **live,
                    "mission_complete": False,
                }
            if vehicle.stack.ended:
                break
            vehicle.sleep(1.0)
            elapsed += 1.0
        if eta_seconds:
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"
        else:
            eta_str = "calculating..."
        live = observables(telemetry_now())
        return {
            "DISPLAY_TO_USER": f"🚁 FLYING | Dist: {distance:.0f}m | Alt: {current_alt:.1f}m | Speed: {ground_speed:.1f}m/s | ETA: {eta_str} | {progress:.0f}%" + suffix(where_text(live)),
            "status": "in_progress",
            "distance_m": round(distance, 1),
            "progress_percent": round(progress, 0),
            **live,
            "action_required": "call monitor_flight again",
            "mission_complete": False,
        }

    @tool
    async def set_max_speed(speed_m_s: float) -> dict:
        if speed_m_s <= 0:
            return {"status": "failed", "error": f"Invalid speed: {speed_m_s}. Must be positive."}
        if speed_m_s > 30:
            return {"status": "failed", "error": f"Speed too high: {speed_m_s} m/s. Maximum is 30 m/s for safety."}
        try:
            vehicle.set_current_speed(float(speed_m_s))
            return {"status": "success", "message": f"Current target speed set to {speed_m_s} m/s (DO_CHANGE_SPEED)", "speed_kmh": round(speed_m_s * 3.6, 1)}
        except ActionError as speed_error:
            try:
                vehicle.set_param("WPNAV_SPEED", float(speed_m_s) * 100.0)
            except KeyError as e:
                return {"status": "failed", "error": f"Set max speed failed: {e} (speed command: {speed_error})"}
            return {
                "status": "success",
                "message": f"Waypoint speed limit set to {speed_m_s} m/s via WPNAV_SPEED (DO_CHANGE_SPEED was rejected: {speed_error})",
                "speed_kmh": round(speed_m_s * 3.6, 1),
            }

    @tool
    async def set_yaw(yaw_deg: float, yaw_rate_deg_s: float = 30.0) -> dict:
        yaw_normalized = yaw_deg % 360
        if yaw_rate_deg_s <= 0:
            return {"status": "failed", "error": f"Invalid yaw rate: {yaw_rate_deg_s}. Must be positive."}
        try:
            goto_here(yaw_normalized)
        except ActionError as e:
            return {"status": "failed", "error": f"Yaw control failed: {e}"}
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return {
            "status": "success",
            "message": f"Rotating to heading {yaw_normalized}°",
            "yaw_degrees": yaw_normalized,
            "cardinal_direction": directions[int((yaw_normalized + 22.5) / 45) % 8],
            "yaw_rate_deg_s": yaw_rate_deg_s,
        }

    @tool
    async def set_flight_mode(mode: str) -> dict:
        mode_upper = mode.upper().strip()
        supported = {"HOLD": "hold", "LOITER": "hold", "RTL": "return_to_launch", "RETURN_TO_LAUNCH": "return_to_launch", "LAND": "land", "GUIDED": "guided"}
        if mode_upper not in supported:
            return {
                "status": "failed",
                "error": f"Unsupported mode: {mode}. Supported modes: HOLD, RTL, LAND, GUIDED",
                "hint": "For AUTO mode, use initiate_mission or resume_mission instead.",
            }
        try:
            action = supported[mode_upper]
            if action == "hold":
                vehicle.hold()
                result_mode = "HOLD/LOITER"
            elif action == "return_to_launch":
                vehicle.return_to_launch()
                result_mode = "RTL"
            elif action == "land":
                vehicle.land()
                result_mode = "LAND"
            else:
                goto_here()
                result_mode = "GUIDED"
            vehicle.sleep(0.5)
            return {"status": "success", "message": f"Flight mode changed to {result_mode}", "requested_mode": mode_upper, "actual_mode": vehicle.flight_mode}
        except ActionError as e:
            return {"status": "failed", "error": f"Failed to set flight mode: {e}"}

    @tool
    async def vehicle_power(action: str, confirm: bool = False) -> dict:
        action = str(action).lower()
        if action not in ("reboot", "shutdown", "terminate"):
            return {"status": "failed", "error": f'action must be "reboot", "shutdown" or "terminate", got {action!r}'}
        if not confirm:
            return {"status": "failed", "error": f"{action} refused: pass confirm=true to execute this critical action"}
        if action == "shutdown":  # release firmware notes: UNSUPPORTED on ArduPilot
            return {"status": "failed", "error": f"{action} failed: {action_error('UNSUPPORTED', 'shutdown()')}"}
        vehicle.stack.mark("POWER", f"vehicle_power({action})", {"action": action})
        if action == "terminate" or vehicle.in_air:
            vehicle.kill()  # an autopilot rebooted in flight stops flying
        else:
            vehicle.stack.arm(False)
        return {"status": "success", "message": f"{action} command sent"}

    @tool
    async def flight_altitudes(action: str, altitude_m: float = 0.0) -> dict:
        action = str(action).lower()
        if action not in ("get_takeoff", "set_takeoff", "get_rtl", "set_rtl"):
            return {"status": "failed", "error": f'action must be "get_takeoff", "set_takeoff", "get_rtl" or "set_rtl", got {action!r}'}
        if action.startswith("set_") and not 1.0 <= float(altitude_m) <= 500.0:
            return {"status": "failed", "error": f"altitude_m must be between 1 and 500, got {altitude_m}"}
        if action == "get_takeoff":
            return {"status": "success", "altitude_m": vehicle.takeoff_altitude}
        if action == "set_takeoff":
            vehicle.set_takeoff_altitude(float(altitude_m))
            return {"status": "success", "message": f"default takeoff altitude set to {altitude_m} m"}
        # The RTL altitude actions use PX4's parameter; ArduPilot answers PARAMETER_ERROR (release firmware notes).
        return {
            "status": "failed",
            "error": f"{action} failed: {action_error('PARAMETER_ERROR', 'get_return_to_launch_altitude()')} "
                     "(on ArduPilot use set_parameter('RTL_ALT', cm) for RTL altitude)",
        }

    # -- telemetry (release: tools/telemetry.py) -----------------------------------------

    @tool
    async def get_position() -> dict:
        return {"status": "success", "position": vehicle.position()}

    @tool
    async def get_flight_mode() -> dict:
        return {"status": "success", "flight_mode": vehicle.flight_mode}

    @tool
    async def get_battery() -> dict:
        battery = vehicle.battery()
        fraction = battery["remaining_fraction"]
        data: dict = {"voltage_v": round(battery["voltage_v"], 2), "remaining_percent": round(fraction * 100, 1)}
        if fraction < 0.20:
            data["warning"] = "⚠️  LOW BATTERY - Land soon!"
        elif fraction < 0.30:
            data["warning"] = "Battery getting low - consider landing"
        return {"status": "success", "battery": data}

    @tool
    async def get_health() -> dict:
        ok = not vehicle.stack.ended
        data = {
            "is_gyrometer_calibrated": True, "is_accelerometer_calibrated": True, "is_magnetometer_calibrated": True,
            "is_local_position_ok": ok, "is_global_position_ok": ok, "is_home_position_ok": ok, "is_armable": ok,
        }
        data["overall_status"] = "HEALTHY" if ok else "ISSUES DETECTED"
        if not ok:
            data["warnings"] = ["⚠️  No GPS lock - cannot fly safely!", "⚠️  Drone is not armable - check for errors"]
        return {"status": "success", "health": data}

    @tool
    async def get_health_all_ok() -> dict:
        ok = not vehicle.stack.ended
        return {
            "status": "success",
            "health_all_ok": ok,
            "status_text": "ALL SYSTEMS GO ✓" if ok else "SYSTEMS NOT READY ✗",
            "recommendation": "Ready for flight" if ok else "Run get_health() for details on what's not ready",
        }

    @tool
    async def get_home_position() -> dict:
        home = vehicle.home()
        result: dict = {
            "status": "success",
            "home": {**home, "meaning": "where RTL will fly to; the autopilot moves this to wherever the aircraft last armed"},
        }
        launch = vehicle.session_launch
        separation = haversine_distance(home["latitude_deg"], home["longitude_deg"], launch["latitude_deg"], launch["longitude_deg"])
        result["session_launch_point"] = {k: launch[k] for k in ("latitude_deg", "longitude_deg", "absolute_altitude_m", "source")}
        result["distance_between_m"] = round(separation, 1)
        matches = separation <= HOME_DIVERGENCE_M
        result["home_matches_session_launch"] = matches
        if not matches:
            result["warning"] = (
                f"the autopilot's home is {separation:.0f} m from where this session started. RTL will fly to "
                "the 'home' coordinate, NOT to the launch point. If you want the launch point, use "
                "session_launch_point with go_to_location."
            )
        return result

    @tool
    async def get_speed() -> dict:
        north, east, down = vehicle.velocity_ned()
        ground = math.hypot(north, east)
        return {
            "status": "success",
            "velocity": {
                "north_m_s": north, "east_m_s": east, "down_m_s": down,
                "ground_speed_m_s": round(ground, 2), "ground_speed_kmh": round(ground * 3.6, 2),
            },
        }

    @tool
    async def get_attitude() -> dict:
        yaw = vehicle.heading_deg
        return {"status": "success", "attitude": {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": round(yaw if yaw <= 180 else yaw - 360, 2)}}

    @tool
    async def get_gps_info() -> dict:
        return {"status": "success", "gps": {"num_satellites": 10, "fix_type": "FIX_3D", "quality": "Excellent"}}

    @tool
    async def get_in_air() -> dict:
        return {"status": "success", "in_air": vehicle.in_air, "status_text": "IN AIR (flying)" if vehicle.in_air else "ON GROUND"}

    @tool
    async def get_armed() -> dict:
        return {"status": "success", "armed": vehicle.armed, "status_text": "ARMED (motors ready)" if vehicle.armed else "DISARMED (motors off)"}

    @tool
    async def get_landed_state() -> dict:
        name = vehicle.landed_state
        descriptions = {
            "UNKNOWN": "State cannot be determined", "ON_GROUND": "Drone is on the ground, not moving",
            "IN_AIR": "Drone is flying/airborne", "TAKING_OFF": "Drone is in the process of taking off",
            "LANDING": "Drone is in the process of landing",
        }
        return {
            "status": "success", "landed_state": name, "description": descriptions.get(name, name),
            "is_on_ground": name == "ON_GROUND", "is_in_air": name == "IN_AIR", "is_transitioning": name in ("TAKING_OFF", "LANDING"),
        }

    @tool
    async def get_heading() -> dict:
        heading = vehicle.heading_deg % 360
        compass = [(22.5, "N", "North"), (67.5, "NE", "Northeast"), (112.5, "E", "East"), (157.5, "SE", "Southeast"),
                   (202.5, "S", "South"), (247.5, "SW", "Southwest"), (292.5, "W", "West"), (337.5, "NW", "Northwest"), (360.1, "N", "North")]
        cardinal, direction = next((c, d) for limit, c, d in compass if heading < limit)
        return {"status": "success", "heading_deg": round(heading, 1), "cardinal_direction": cardinal, "direction_name": direction}

    # -- missions (release: tools/mission.py) ----------------------------------------------

    def progress_or_unknown() -> tuple[int, int]:
        return vehicle.mission_progress() or (0, 0)

    @tool
    async def upload_mission(waypoints: list) -> dict:
        if not waypoints:
            return {
                "status": "failed", "error": "No waypoints provided",
                "example": [{"latitude_deg": 33.645, "longitude_deg": -117.842, "relative_altitude_m": 10},
                            {"latitude_deg": 33.646, "longitude_deg": -117.843, "relative_altitude_m": 15}],
            }
        if not isinstance(waypoints, list):
            return {
                "status": "failed", "error": f"Waypoints must be a list, got {type(waypoints).__name__}",
                "hint": "Pass waypoints as: [{'latitude_deg': 33.645, 'longitude_deg': -117.842, 'relative_altitude_m': 10}]",
            }
        items = []
        try:
            for i, wp in enumerate(waypoints):
                if not isinstance(wp, dict):
                    return {"status": "failed", "error": f"Waypoint {i} must be a dictionary, got {type(wp).__name__}",
                            "hint": "Each waypoint needs: latitude_deg, longitude_deg, relative_altitude_m"}
                required = ["latitude_deg", "longitude_deg", "relative_altitude_m"]
                missing = [f for f in required if f not in wp]
                if missing:
                    return {"status": "failed", "error": f"Waypoint {i} missing required fields: {', '.join(missing)}",
                            "received": list(wp.keys()), "required": required}
                if not (-90 <= wp["latitude_deg"] <= 90):
                    return {"status": "failed", "error": f"Waypoint {i}: invalid latitude {wp['latitude_deg']} (must be -90 to 90)"}
                if not (-180 <= wp["longitude_deg"] <= 180):
                    return {"status": "failed", "error": f"Waypoint {i}: invalid longitude {wp['longitude_deg']} (must be -180 to 180)"}
                if wp["relative_altitude_m"] < 0:
                    return {"status": "failed", "error": f"Waypoint {i}: altitude cannot be negative"}
                # As released: seq 0 carries the first waypoint, which ArduPilot keeps as its home slot.
                items.append({"seq": i, "frame": 3, "command": MAV_CMD_NAV_WAYPOINT, "current": 1 if i == 0 else 0,
                              "autocontinue": 1, "param1": 0, "param2": 2.0, "x": int(wp["latitude_deg"] * 1e7),
                              "y": int(wp["longitude_deg"] * 1e7), "z": float(wp["relative_altitude_m"])})
            vehicle.upload_raw(items)
        except (ActionError, TypeError) as e:
            return {"status": "failed", "error": f"Mission upload failed: {e}",
                    "troubleshooting": "Ensure waypoints are formatted correctly as list of dictionaries"}
        return {
            "status": "success",
            "message": f"Mission uploaded with {len(waypoints)} waypoints",
            "waypoint_count": len(waypoints),
            "waypoints_summary": [f"WP{i}: ({wp['latitude_deg']:.5f}, {wp['longitude_deg']:.5f}) @ {wp['relative_altitude_m']}m" for i, wp in enumerate(waypoints)],
            "note": "Mission uploaded but NOT started. Use initiate_mission to start.",
        }

    @tool
    async def initiate_mission(mission_points: list, return_to_launch: bool = True) -> dict:
        items = []
        for i, point in enumerate(mission_points):
            try:
                if not (-90 <= point["latitude_deg"] <= 90):
                    return {"status": "failed", "error": f"Invalid latitude_deg: {point['latitude_deg']}. Must be between -90 and 90."}
                if not (-180 <= point["longitude_deg"] <= 180):
                    return {"status": "failed", "error": f"Invalid longitude_deg: {point['longitude_deg']}. Must be between -180 and 180."}
                items.append({"seq": i, "frame": 3, "command": MAV_CMD_NAV_WAYPOINT, "current": 1 if i == 0 else 0, "autocontinue": 1,
                              "param1": point.get("loiter_time_s", 0), "param2": point.get("acceptance_radius_m", 2.0),
                              "param4": point.get("yaw_deg", float("nan")), "x": int(point["latitude_deg"] * 1e7),
                              "y": int(point["longitude_deg"] * 1e7), "z": float(point["relative_altitude_m"])})
            except KeyError as e:
                return {"status": "failed", "error": f"Missing required field in mission point: {e}"}
        # As released: the RTL flag is MavSDK's mission-plugin setting, which a raw upload does not read;
        # the raw upload and the start are outside any try, so a refusal surfaces as a tool error.
        vehicle.rtl_after_mission = bool(return_to_launch)
        vehicle.upload_raw(items)
        vehicle.start_mission()
        return {"status": "success", "message": f"Mission started with {len(items)} waypoints",
                "note": "Flight mode automatically changed to AUTO for mission execution"}

    @tool
    async def resume_mission() -> dict:
        try:
            current, total = progress_or_unknown()
            vehicle.start_mission()
            vehicle.sleep(0.5)
            mode = vehicle.flight_mode
            return {
                "status": "success", "message": f"Mission resumed from waypoint {current}/{total}",
                "current_waypoint": current, "total_waypoints": total, "flight_mode": mode,
                "mode_transition_ok": "AUTO" in mode or "MISSION" in mode,
                "note": "Flight mode should have changed to AUTO/MISSION for mission execution",
            }
        except ActionError as e:
            return {"status": "failed", "error": f"Mission resume failed: {e}"}

    @tool
    async def pause_mission(mode: str = "guided_hold") -> dict:
        mode = str(mode).lower()
        if mode not in ("guided_hold", "native_hold"):
            return {"status": "failed", "error": f'mode must be "guided_hold" or "native_hold", got {mode!r}'}
        if mode == "native_hold":
            try:
                vehicle.pause_mission()
            except ActionError as e:
                return {"status": "failed", "error": f"Mission pause failed: {e}"}
            return {
                "status": "success", "message": "Mission paused (firmware Hold/Loiter mode)",
                "warning": "⚠️ On ArduPilot, LOITER altitude tracks RC throttle - without an RC at mid-throttle the drone can descend. Prefer mode=guided_hold without RC.",
                "note": "Use resume_mission to continue.",
            }
        try:
            current, total = progress_or_unknown()
            position = goto_here()
        except ActionError as e:
            return {"status": "failed", "error": f"Mission pause failed: {e}"}
        return {
            "status": "success",
            "message": f"Mission paused - holding position in GUIDED mode (was at waypoint {current}/{total})",
            "was_at_waypoint": current, "total_waypoints": total,
            "position": {"latitude": position["latitude_deg"], "longitude": position["longitude_deg"], "altitude": position["absolute_altitude_m"]},
            "flight_mode": "GUIDED",
            "note": "Use set_current_waypoint + resume_mission to continue the mission.",
        }

    @tool
    async def hold_mission_position() -> dict:
        try:
            current, total = progress_or_unknown()
            position = goto_here()
        except ActionError as e:
            return {"status": "failed", "error": f"Hold mission position failed: {e}"}
        return {
            "status": "success",
            "message": f"Holding position in GUIDED mode - mission interrupted at waypoint {current}/{total}",
            "was_at_waypoint": current, "total_waypoints": total,
            "position": {"latitude": position["latitude_deg"], "longitude": position["longitude_deg"], "altitude": position["absolute_altitude_m"]},
            "flight_mode": "GUIDED",
            "note": "Mission stopped. To continue: use set_current_waypoint() then resume_mission(), or start a new mission.",
        }

    @tool
    async def clear_mission() -> dict:
        vehicle.clear_mission()
        return {"status": "success", "message": "Mission cleared - all waypoints removed"}

    @tool
    async def download_mission() -> dict:
        items = vehicle.mission.items if vehicle.mission is not None else []
        waypoints = [
            {"seq": item["seq"], "latitude_deg": item["x"] / 1e7, "longitude_deg": item["y"] / 1e7,
             "relative_altitude_m": item["z"], "frame": item["frame"], "command": item["command"]}
            for item in items if int(item.get("command", MAV_CMD_NAV_WAYPOINT)) == MAV_CMD_NAV_WAYPOINT
        ]
        return {"status": "success", "waypoint_count": len(waypoints), "waypoints": waypoints, "note": None}

    @tool
    async def print_mission_progress() -> dict:
        progress = vehicle.mission_progress()
        if progress is None:
            vehicle.sleep(10.0)
            return {"status": "failed", "error": "No mission progress received within 10s (progress is emitted on waypoint transitions; there may be no active mission)"}
        return {"status": "success", "current": progress[0], "total": progress[1]}

    @tool
    async def is_mission_finished() -> dict:
        finished = vehicle.mission_finished()
        current, total = progress_or_unknown()
        return {
            "status": "success", "mission_finished": finished, "status_text": "FINISHED" if finished else "IN PROGRESS",
            "current_waypoint": current, "total_waypoints": total, "flight_mode": vehicle.flight_mode,
            "progress_percentage": round((current / total * 100) if total > 0 else 0, 1),
        }

    @tool
    async def set_current_waypoint(waypoint_index: int) -> dict:
        if waypoint_index < 0:
            return {"status": "failed", "error": f"Invalid waypoint index: {waypoint_index}. Must be 0 or greater."}
        try:
            vehicle.set_current_item(int(waypoint_index))
        except ActionError as e:
            return {"status": "failed", "error": f"Set waypoint failed: {e}"}
        return {"status": "success", "message": f"Current waypoint set to index {waypoint_index}", "waypoint_index": waypoint_index,
                "note": "Mission will continue from this waypoint"}

    @tool
    async def rtl_after_mission(action: str, enabled: bool = True) -> dict:
        action = str(action).lower()
        if action not in ("get", "set"):
            return {"status": "failed", "error": f'action must be "get" or "set", got {action!r}'}
        if action == "get":
            return {"status": "success", "enabled": vehicle.rtl_after_mission}
        vehicle.rtl_after_mission = bool(enabled)
        return {"status": "success", "message": f"RTL-after-mission set to {bool(enabled)}"}

    # -- managed missions (release: tools/managed_mission.py, missions/runner.py) -----------------

    @tool
    async def start_managed_mission(waypoints: list, takeoff_altitude_m: float = 20.0, return_to_launch: bool = True) -> dict:
        if not waypoints or not isinstance(waypoints, list):
            return _fail("waypoints must be a non-empty list")
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, dict):
                return _fail(f"waypoints[{i}] must be an object with latitude_deg/longitude_deg/altitude_m")
            if wp.get("latitude_deg", wp.get("lat")) is None or wp.get("longitude_deg", wp.get("lon")) is None:
                return _fail(f"waypoints[{i}] is missing latitude_deg/longitude_deg")
        if runner.record is not None and runner.record.active:
            return _fail(f"mission {runner.record.mission_id} is still {runner.record.phase}",
                         remedy="Wait for it to finish, or abort it with control_managed_mission(action='abort').")
        record = runner.start(waypoints, float(takeoff_altitude_m), bool(return_to_launch))
        return {
            "status": "success", "mission_id": record.mission_id, "phase": record.phase, "waypoint_count": record.waypoint_count,
            "message": "Mission accepted. The server is flying and monitoring it.",
            "next_step": "Poll get_mission_status when you want an update. You may disconnect and reconnect at any time - the mission continues without you.",
        }

    @tool
    async def get_mission_status(include_events: bool = True, event_limit: int = 20) -> dict:
        return runner.status_payload(bool(include_events), int(event_limit))

    @tool
    async def control_managed_mission(action: str) -> dict:
        action = str(action).lower()
        if action not in ("pause", "resume", "abort", "clear"):
            return _fail(f'action must be "pause", "resume", "abort" or "clear", got {action!r}')
        record = runner.record
        if record is None:
            return _fail("no mission has been submitted")
        if action == "clear":
            if record.active:
                return _fail(f"mission {record.mission_id} is still {record.phase}", remedy="Abort it first with control_managed_mission(action='abort').")
            runner.record = None
            return {"status": "success", "message": f"cleared finished mission {record.mission_id}"}
        if not record.active:
            return _fail(f"mission {record.mission_id} already finished ({record.phase})")
        if action == "pause":
            if record.phase_enum is not Phase.RUNNING:
                return _fail(f"can only pause a running mission (currently {record.phase})")
            runner.request_pause()
        elif action == "resume":
            if record.phase_enum is not Phase.PAUSED:
                return _fail(f"can only resume a paused mission (currently {record.phase})")
            runner.request_resume()
        else:
            runner.request_abort()
        return {
            "status": "success", "mission_id": record.mission_id, "requested": action, "phase": record.phase,
            "note": "The request is applied by the mission monitor within a poll interval; call get_mission_status to confirm.",
        }

    # -- parameters (release: tools/param.py) ------------------------------------------------

    def read_param(name: str, kind: str) -> float | int:
        value = vehicle.get_param(name)
        is_int = name in INTEGER_PARAMETERS
        if kind == "int" and not is_int or kind == "float" and is_int:
            raise ActionError("WRONG_TYPE")
        return int(value) if is_int else float(value)

    @tool
    async def get_parameter(name: str, param_type: str = "auto") -> dict:
        try:
            if param_type in ("int", "float"):
                return {"status": "success", "name": name, "value": read_param(name, param_type), "type": param_type}
            if param_type == "custom":
                raise ActionError("UNSUPPORTED: ArduPilot does not implement PARAM_EXT")
            try:
                return {"status": "success", "name": name, "value": read_param(name, "float"), "type": "float"}
            except ActionError:
                return {"status": "success", "name": name, "value": read_param(name, "int"), "type": "int"}
        except (ActionError, KeyError) as e:
            return {"status": "failed", "error": f"Parameter '{name}' not found or inaccessible: {e}",
                    "suggestion": "Check parameter name spelling. Use list_parameters to see available parameters."}

    @tool
    async def set_parameter(name: str, value: float = 0.0, param_type: str = "auto", custom_value: str = "") -> dict:
        try:
            if param_type == "custom":
                if not custom_value:
                    return {"status": "failed", "error": 'param_type="custom" requires custom_value'}
                raise ActionError("UNSUPPORTED: ArduPilot does not implement PARAM_EXT")
            old_value = None
            if param_type == "auto":
                try:
                    old_value, final = read_param(name, "float"), "float"
                except (ActionError, KeyError):
                    try:
                        old_value, final = read_param(name, "int"), "int"
                    except (ActionError, KeyError):
                        final = "float"
            else:
                final = param_type
                try:
                    old_value = read_param(name, param_type)
                except (ActionError, KeyError):
                    old_value = None
            if (final == "int") != (name in INTEGER_PARAMETERS) and name in vehicle.parameters:
                raise ActionError("WRONG_TYPE")
            vehicle.set_param(name, int(value) if final == "int" else float(value))
            return {
                "status": "success", "name": name, "old_value": old_value,
                "new_value": int(value) if final == "int" else float(value), "type": final,
                "message": f"Parameter '{name}' set to {value}", "warning": "Some parameters may require a reboot to take effect.",
            }
        except (ActionError, KeyError) as e:
            return {"status": "failed", "error": f"Failed to set parameter '{name}': {e}",
                    "suggestion": "Verify parameter name and value are valid for this drone."}

    @tool
    async def list_parameters(filter_prefix: str = "") -> dict:
        listed = [{"name": n, "value": int(v) if n in INTEGER_PARAMETERS else float(v), "type": "int" if n in INTEGER_PARAMETERS else "float"}
                  for n, v in sorted(vehicle.parameters.items()) if n.upper().startswith(filter_prefix.upper())]
        if filter_prefix:
            return {"status": "success", "filter": filter_prefix, "count": len(listed), "parameters": listed}
        return {"status": "success", "count": len(listed), "parameters": listed,
                "warning": f"This is a large list ({len(listed)} parameters). Consider using filter_prefix to narrow results."}

    # -- geofence, emergency, logs, camera, system -------------------------------------------

    @tool
    async def upload_geofence(polygons: list | None = None, circles: list | None = None) -> dict:
        polygons, circles = polygons or [], circles or []
        if not polygons and not circles:
            return {"status": "failed", "error": "provide at least one polygon or circle"}
        for i, polygon in enumerate(polygons):
            if len((polygon or {}).get("points") or []) < 3:
                return {"status": "failed", "error": f"polygons[{i}] needs at least 3 points"}
        vehicle.firmware_fence = {"polygons": polygons, "circles": circles}
        return {
            "status": "success",
            "message": f"Geofence uploaded ({len(polygons)} polygon(s), {len(circles)} circle(s))",
            "note": "On ArduPilot, enforcement also requires FENCE_ENABLE=1 (see set_parameter).",
        }

    @tool
    async def clear_geofence() -> dict:
        vehicle.firmware_fence = None
        return {"status": "success", "message": "Geofence cleared - no fence containment is active"}

    @tool
    async def emergency_stop(mode: str = "land") -> dict:
        mode = str(mode).lower()
        if mode not in ("land", "rtl", "kill"):
            return {"status": "failed", "error": f'mode must be "land", "rtl" or "kill", got {mode!r}'}
        actions: list[str] = []
        try:
            if mode == "land":
                vehicle.land()
                actions.append("landing at current position")
            elif mode == "rtl":
                vehicle.return_to_launch()
                actions.append("returning to launch")
            else:
                vehicle.kill()
                actions.append("MOTORS KILLED - vehicle is falling")
        except ActionError as e:
            return {"status": "failed", "error": f"emergency stop ({mode}) failed: {e}", "actions_taken": actions,
                    "escalate": "Use the out-of-band chain NOW: RC mode switch / GCS / kill switch. See docs/estop.md."}
        return {
            "status": "success", "mode": mode, "actions_taken": actions,
            "next_step": "Monitor with get_landed_state / get_armed until the drone is down and disarmed.",
            "out_of_band": "Human override remains available at all times - see docs/estop.md.",
        }

    @tool
    async def flight_logs(action: str, log_id: int = -1) -> dict:
        action = str(action).lower()
        if action not in ("list", "download", "erase_all"):
            return _fail(f'action must be "list", "download" or "erase_all", got {action!r}')
        if action == "download" and log_id < 0:
            return _fail('download requires log_id (use action="list" first)')
        if action == "list":
            entries = [] if vehicle.logs_erased else [{"id": 1, "date": "simulated", "size_bytes": 1024 * max(1, len(vehicle.stack.states))}]
            return _ok(entries=entries)
        if action == "erase_all":
            vehicle.logs_erased = True
            vehicle.stack.mark("LOGS_ERASED", "all onboard flight logs erased")
            return _ok(message="all onboard logs erased")
        return _fail("flight logs download failed: log transfer is not simulated")

    @tool
    async def list_cameras() -> dict:
        if not vehicle.camera:
            vehicle.sleep(8.0)
            return _ok(cameras=[], note="No camera announced itself within 8s")
        return _ok(cameras=[{"component_id": 100, "vendor_name": "simulated", "model_name": "fixed forward camera",
                             "resolution_horizontal": 1920, "resolution_vertical": 1080}])

    @tool
    async def camera_capture(component_id: int, action: str, interval_s: float = 1.0, stream_id: int = 0) -> dict:
        action = str(action).lower()
        actions = ("take_photo", "start_photo_interval", "stop_photo_interval", "start_video", "stop_video",
                   "start_video_streaming", "stop_video_streaming", "video_stream_info", "last_capture_info")
        if action not in actions:
            return _fail(f"action must be one of {actions}, got {action!r}")
        if action == "start_photo_interval" and not 0.1 <= float(interval_s) <= 3600:
            return _fail(f"interval_s must be between 0.1 and 3600, got {interval_s}")
        try:
            if not vehicle.camera or int(component_id) != 100:
                raise action_error("NO_SYSTEM", f"{action}()", (component_id,))
            if action == "take_photo":
                vehicle.take_photo()
                return _ok(message=f"{action} ok")
            if action == "last_capture_info":
                if not vehicle.photos:
                    return _fail("No capture info received within 8s")
                return _ok(capture_info=vehicle.photos[-1])
            raise action_error("UNSUPPORTED", f"{action}()", (component_id,))
        except ActionError as e:
            return _fail(f"camera {action} failed: {e}")

    @tool
    async def send_status_text(text: str, severity: str = "info") -> dict:
        levels = ("emergency", "alert", "critical", "error", "warning", "notice", "info", "debug")
        if str(severity).lower() not in levels:
            return _fail(f"severity must be one of {sorted(levels)}, got {severity!r}")
        if not text:
            return _fail("text must not be empty")
        vehicle.stack.mark("STATUSTEXT", f"[{severity}] {text}", {"severity": severity})
        return _ok(message="status text sent")

    @tool
    async def system_info(topic: str = "version") -> dict:
        topic = str(topic).lower()
        topics = ("version", "identification", "product", "flight_information", "speed_factor")
        if topic not in topics:
            return _fail(f"topic must be one of {topics}, got {topic!r}")
        if topic in ("flight_information", "speed_factor"):  # release: ArduPilot answers INFORMATION_NOT_RECEIVED_YET
            return _fail(f"system {topic} unavailable: INFORMATION_NOT_RECEIVED_YET")
        return _ok(topic=topic, info={"autopilot": vehicle.firmware, "vehicle": "quadcopter"})

    tools["_runner"] = runner  # type: ignore[assignment]
    return tools


def unsupported(name: str) -> Callable:
    """A released tool with no counterpart on the simulated vehicle: registered, guarded, and honest."""
    spec = CATALOG[name]

    async def stub(**_arguments) -> dict:
        return {
            "status": "failed",
            "error": f"{name} is not available on this vehicle: the simulated aircraft has no {spec['module']} subsystem.",
        }

    stub.__name__ = name
    return stub
