"""Server-executed missions: the release's `missions/runner.py`, on the simulated vehicle.

The model submits a mission with `start_managed_mission` and the SERVER flies
and monitors it; the model polls `get_mission_status` and steers with
`control_managed_mission`. This is the path the paper's T4 ("upload, execute
and monitor a mission plan") is flown on.

Ported step for step from the release (commit `06b7d1a`):

    SUBMITTED -> VALIDATING -> UPLOADING -> ARMING -> RUNNING -> COMPLETED
                                                       |  \\-> RETURNING / LANDING -> COMPLETED
                                                       \\---> PAUSED -> RUNNING
    (any phase) -> FAILED | ABORTED

* the whole mission is checked against the server's geofence before anything
  is uploaded, and a radius fence with no known home fails the mission;
* the raw items are laid out as ArduPilot wants them: seq 0 a HOME
  placeholder, seq 1 a NAV_TAKEOFF, the waypoints (frame relative to home,
  `hold_s` as the hold time), and NAV_RETURN_TO_LAUNCH when asked;
* the server arms (retrying for 90 s), takes off in GUIDED to the take-off
  altitude, and only then starts AUTO, because ArduCopter will not lift off in
  AUTO without RC throttle;
* a start counts only with evidence (MISSION flight mode, or progress); if
  none comes in 30 s the mission is re-uploaded without the placeholder, and
  if still none, it FAILS and the aircraft is brought down;
* the monitor polls every 2 s: waypoint and mode events, battery marks, the
  battery and geofence auto-actions, and the end of the flight. COMPLETED needs
  the mission seen executing, real progress, and the aircraft landed and
  disarmed; anything else is FAILED, with the evidence;
* once every item is flown the server commands RTL itself, and a mission with
  no progress for 240 s airborne is brought down and failed.

What is not ported: the JSON checkpoint and restart recovery (the server here
never restarts mid-flight), and link-loss handling (the simulated link does not
drop). Waits that the release spends in `asyncio.sleep` are spent in flight
time: the runner is a generator advanced by the flight stack's clock.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from compare.droneserver.geo import haversine_distance
from compare.droneserver.safety.geofence import Geofence, check_mission, check_position
from compare.droneserver.vehicle import (
    MAV_CMD_NAV_RETURN_TO_LAUNCH,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    ActionError,
    Vehicle,
)

#: Simulated wall-clock origin for event timestamps.
EPOCH = datetime(2026, 8, 7, 12, 0, tzinfo=UTC).timestamp()
GROUND_HEIGHT_M = 2.0
SETTLED_RATE_M_S = 1.0
IN_THE_AIR_STATES = ("IN_AIR", "TAKING_OFF", "LANDING")


@dataclass
class MissionSettings:
    """The release's MISSION_* defaults."""

    poll_interval_s: float = 2.0
    max_events: int = 500
    auto_actions_enabled: bool = True
    low_battery_action: str = "rtl"
    low_battery_threshold: float = 0.25
    critical_battery_threshold: float = 0.10
    geofence_breach_action: str = "rtl"
    mission_complete_action: str = "rtl"
    link_loss_action: str = "none"
    arm_timeout_s: float = 90.0
    start_confirm_timeout_s: float = 30.0
    progress_distance_m: float = 15.0
    no_progress_timeout_s: float = 240.0


class Phase(str, Enum):
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    UPLOADING = "uploading"
    ARMING = "arming"
    RUNNING = "running"
    PAUSED = "paused"
    RETURNING = "returning"
    LANDING = "landing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


TERMINAL = frozenset({Phase.COMPLETED, Phase.FAILED, Phase.ABORTED})
ALLOWED: dict[Phase, frozenset] = {
    Phase.SUBMITTED: frozenset({Phase.VALIDATING, Phase.FAILED, Phase.ABORTED}),
    Phase.VALIDATING: frozenset({Phase.UPLOADING, Phase.FAILED, Phase.ABORTED}),
    Phase.UPLOADING: frozenset({Phase.ARMING, Phase.FAILED, Phase.ABORTED}),
    Phase.ARMING: frozenset({Phase.RUNNING, Phase.FAILED, Phase.ABORTED}),
    Phase.RUNNING: frozenset({Phase.PAUSED, Phase.RETURNING, Phase.LANDING, Phase.COMPLETED, Phase.FAILED, Phase.ABORTED}),
    Phase.PAUSED: frozenset({Phase.RUNNING, Phase.RETURNING, Phase.LANDING, Phase.ABORTED, Phase.FAILED}),
    Phase.RETURNING: frozenset({Phase.LANDING, Phase.COMPLETED, Phase.FAILED, Phase.ABORTED}),
    Phase.LANDING: frozenset({Phase.COMPLETED, Phase.FAILED, Phase.ABORTED}),
    Phase.COMPLETED: frozenset(),
    Phase.FAILED: frozenset(),
    Phase.ABORTED: frozenset(),
}


def ground_evidence(landed_state: str | None, in_air: bool | None) -> bool | None:
    if landed_state == "ON_GROUND":
        return True
    if landed_state in IN_THE_AIR_STATES:
        return False
    if in_air is not None:
        return not in_air
    return None


def height_above_launch_m(launch_amsl_m, absolute_altitude_m, relative_altitude_m) -> float | None:
    if launch_amsl_m is not None and absolute_altitude_m is not None:
        return absolute_altitude_m - launch_amsl_m
    return relative_altitude_m


@dataclass
class MissionRecord:
    mission_id: str
    phase: str = Phase.SUBMITTED.value
    created_at: str = ""
    updated_at: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    waypoint_count: int = 0
    current_item: int = 0
    total_items: int = 0
    takeoff_altitude_m: float = 20.0
    return_to_launch: bool = True
    last_position: dict | None = None
    last_battery: dict | None = None
    last_flight_mode: str | None = None
    last_armed: bool = True
    last_landed_state: str | None = None
    last_in_air: bool | None = None
    last_vertical_speed_m_s: float | None = None
    launch_amsl_m: float | None = None
    mission_mode_confirmed: bool = False
    baseline_item: int = 0
    items_reached: int = 0
    start_position: dict | None = None
    max_distance_from_start_m: float = 0.0
    error: str | None = None
    auto_actions_fired: list = field(default_factory=list)
    events: list = field(default_factory=list)
    resumed_after_restart: bool = False

    @property
    def phase_enum(self) -> Phase:
        return Phase(self.phase)

    @property
    def active(self) -> bool:
        return self.phase_enum not in TERMINAL

    def progress_percent(self) -> float:
        if not self.total_items:
            return 0.0
        return round(min(self.current_item / self.total_items, 1.0) * 100, 1)

    def progressed(self, distance_threshold_m: float) -> bool:
        if self.items_reached > self.baseline_item:
            return True
        return distance_threshold_m > 0 and self.max_distance_from_start_m >= distance_threshold_m

    def progress_evidence(self) -> dict:
        return {
            "mission_mode_confirmed": self.mission_mode_confirmed,
            "baseline_item": self.baseline_item,
            "items_reached": self.items_reached,
            "total_items": self.total_items,
            "max_distance_from_start_m": round(self.max_distance_from_start_m, 1),
        }


def mission_items(waypoints: list, takeoff_altitude_m: float, rtl: bool, home_placeholder: bool = True) -> list[dict]:
    """The release's `_mission_items`: raw items in the layout ArduPilot expects."""
    first = waypoints[0]
    first_lat = float(first.get("latitude_deg", first.get("lat", 0.0)))
    first_lon = float(first.get("longitude_deg", first.get("lon", 0.0)))
    items: list[dict] = []
    if home_placeholder:
        items.append({"seq": 0, "frame": 5, "command": MAV_CMD_NAV_WAYPOINT, "current": 0, "autocontinue": 1,
                      "x": int(round(first_lat * 1e7)), "y": int(round(first_lon * 1e7)), "z": 0.0})
    items.append({"seq": len(items), "frame": 3, "command": MAV_CMD_NAV_TAKEOFF, "current": 1, "autocontinue": 1,
                  "x": int(round(first_lat * 1e7)), "y": int(round(first_lon * 1e7)), "z": float(takeoff_altitude_m)})
    for wp in waypoints:
        lat = float(wp.get("latitude_deg", wp.get("lat", 0.0)))
        lon = float(wp.get("longitude_deg", wp.get("lon", 0.0)))
        alt = float(wp.get("altitude_m", wp.get("relative_altitude_m", wp.get("alt", takeoff_altitude_m))))
        items.append({"seq": len(items), "frame": 3, "command": MAV_CMD_NAV_WAYPOINT, "current": 0, "autocontinue": 1,
                      "param1": float(wp.get("hold_s", 0.0)), "x": int(round(lat * 1e7)), "y": int(round(lon * 1e7)), "z": alt})
    if rtl:
        items.append({"seq": len(items), "frame": 2, "command": MAV_CMD_NAV_RETURN_TO_LAUNCH, "current": 0, "autocontinue": 1,
                      "x": 0, "y": 0, "z": 0.0})
    return items


class MissionRunner:
    """Owns the current managed mission and flies it on the vehicle's clock (the release's RUNNER)."""

    def __init__(self, vehicle: Vehicle, fence: Callable[[], Geofence], settings: MissionSettings | None = None) -> None:
        self.vehicle = vehicle
        self.fence = fence
        self.s = settings or MissionSettings()
        self.record: MissionRecord | None = None
        self._task: Generator[float, None, None] | None = None
        self._wait = 0.0
        self._pause_requested = self._abort_requested = self._resume_requested = False
        self._descent_commanded = False
        self._stepping = False
        vehicle.stack.on_odom(self._tick)

    # -- time and events ------------------------------------------------------------

    def _ts(self) -> str:
        return datetime.fromtimestamp(EPOCH + self.vehicle.stack.now, tz=UTC).isoformat()

    def _tick(self, *_odom) -> None:
        # A command the runner sends publishes telemetry, which lands here again: one step at a time.
        if self._task is None or self._stepping:
            return
        self._wait -= self.vehicle.stack.dt
        self._stepping = True
        try:
            while self._task is not None and self._wait <= 1e-9:
                try:
                    self._wait += next(self._task)
                except StopIteration:
                    self._task = None
                except ActionError as e:  # the release's _run(): an unhandled error fails the mission
                    self._task = None
                    if self.record is not None:
                        self.record.error = str(e)
                        self.set_phase(Phase.FAILED, reason=f"unhandled error: {e}")
        finally:
            self._stepping = False

    def emit(self, kind: str, message: str, **data) -> None:
        if self.record is None:
            return
        self.record.events.append({"ts": self._ts(), "kind": kind, "message": message, "data": data})
        del self.record.events[: max(0, len(self.record.events) - self.s.max_events)]
        self.record.updated_at = self._ts()

    def set_phase(self, target: Phase, reason: str = "") -> bool:
        if self.record is None:
            return False
        current = self.record.phase_enum
        if current is target:
            return True
        if target not in ALLOWED.get(current, frozenset()):
            return False
        self.record.phase = target.value
        if target is Phase.RUNNING and self.record.started_at is None:
            self.record.started_at = self.vehicle.stack.now
        if target in TERMINAL:
            self.record.finished_at = self.vehicle.stack.now
        self.emit("phase_change", f"{current.value} -> {target.value}{f' ({reason})' if reason else ''}",
                  from_phase=current.value, to_phase=target.value, reason=reason)
        return True

    def elapsed_s(self) -> float | None:
        record = self.record
        if record is None or record.started_at is None:
            return None
        end = record.finished_at if record.finished_at is not None else self.vehicle.stack.now
        return round(end - record.started_at, 1)

    # -- control ------------------------------------------------------------------

    def start(self, waypoints: list, takeoff_altitude_m: float, return_to_launch: bool) -> MissionRecord:
        record = MissionRecord(
            mission_id=f"m_{uuid.uuid4().hex[:12]}", waypoint_count=len(waypoints),
            takeoff_altitude_m=takeoff_altitude_m, return_to_launch=return_to_launch,
            created_at=self._ts(), updated_at=self._ts(),
        )
        self.record = record
        self._pause_requested = self._abort_requested = self._resume_requested = False
        self._descent_commanded = False
        self.emit("info", f"mission submitted with {len(waypoints)} waypoint(s)", waypoints=len(waypoints),
                  takeoff_altitude_m=takeoff_altitude_m)
        self._task = self._execute(list(waypoints))
        self._wait = 0.0  # the release's task starts on the event loop's next turn
        return record

    def request_pause(self) -> None:
        self._pause_requested = True

    def request_resume(self) -> None:
        self._resume_requested = True

    def request_abort(self) -> None:
        self._abort_requested = True

    # -- the mission ------------------------------------------------------------------

    def _execute(self, waypoints: list) -> Generator[float, None, None]:
        record, s, vehicle = self.record, self.s, self.vehicle
        self.set_phase(Phase.VALIDATING)
        fence = self.fence()
        if fence.max_radius_m > 0 and fence.home is None:
            record.error = ("a radius geofence is configured but the drone's home position is not known, "
                            "so the mission cannot be checked against it")
            self.emit("error", record.error, rule="geofence.home_unknown")
            self.set_phase(Phase.FAILED, reason="geofence.home_unknown")
            return
        violations = check_mission(fence, waypoints)
        if violations:
            idx, first = violations[0]
            record.error = f"mission item {idx} violates the geofence: {first.detail}"
            self.emit("error", record.error, item=idx, rule=first.rule)
            self.set_phase(Phase.FAILED, reason="geofence")
            return
        if record.launch_amsl_m is None:
            record.launch_amsl_m = round(vehicle.position()["absolute_altitude_m"], 2)
        self.set_phase(Phase.UPLOADING)
        items = mission_items(waypoints, record.takeoff_altitude_m, record.return_to_launch)
        if not self._upload(items):
            return
        self.emit("info", f"uploaded {len(items)} mission items", items=len(items))
        self.set_phase(Phase.ARMING)
        waited, armed = 0.0, False
        while waited < s.arm_timeout_s and not self._abort_requested:
            try:
                vehicle.arm()
                armed = True
                break
            except ActionError:
                yield 3.0
                waited += 3.0
        if self._abort_requested:
            self.set_phase(Phase.ABORTED, reason="aborted before arming")
            return
        if not armed:
            record.error = "could not arm within the timeout (prearm checks failing?)"
            self.emit("error", record.error)
            self.set_phase(Phase.FAILED, reason="arming")
            return
        self.emit("info", "armed")
        try:
            vehicle.set_takeoff_altitude(record.takeoff_altitude_m)
            vehicle.takeoff()
        except ActionError as e:
            record.error = f"takeoff failed: {e}"
            self.emit("error", record.error)
            self.set_phase(Phase.FAILED, reason="takeoff")
            return
        self.emit("info", f"climbing to {record.takeoff_altitude_m:.0f} m before starting the mission")
        climbed, reached = 0.0, 0.0
        while climbed < 90 and not self._abort_requested:
            yield 2.0
            climbed += 2.0
            reached = vehicle.position()["relative_altitude_m"]
            if reached >= record.takeoff_altitude_m - 1.5:
                break
        self.emit("info", f"reached {reached:.1f} m", altitude_m=round(reached, 1))
        if self._abort_requested:
            self._do_action("land", reason="operator abort")
            self.set_phase(Phase.ABORTED, reason="aborted during takeoff")
            return
        self._baseline()
        if not self._start_mission():
            record.error = "mission start was refused by the autopilot"
            self.emit("error", record.error)
            yield from self._descend_and_fail("start")
            return
        self.set_phase(Phase.RUNNING)
        confirmed = yield from self._confirm_running()
        if not confirmed:
            self.emit("info", "the autopilot did not enter mission execution - retrying without the "
                      "seq-0 home placeholder (PX4 refuses that layout in flight)", **record.progress_evidence())
            retry = mission_items(waypoints, record.takeoff_altitude_m, record.return_to_launch, home_placeholder=False)
            if self._upload(retry):
                self._baseline()
                if self._start_mission():
                    confirmed = yield from self._confirm_running()
            if not confirmed:
                record.error = ("the autopilot accepted the start command but never entered mission "
                                "execution: the flight mode never became MISSION and no mission item "
                                "was reached. The mission did NOT fly.")
                self.emit("error", record.error, rule="mission.start_unconfirmed", **record.progress_evidence())
                yield from self._descend_and_fail("start_unconfirmed")
                return
        yield from self._monitor()

    def _upload(self, items: list[dict]) -> bool:
        try:
            self.vehicle.upload_raw(items)
        except ActionError as e:
            self.record.error = f"mission upload failed: {e}"
            self.emit("error", self.record.error)
            if self.record.phase_enum is Phase.UPLOADING:
                self.set_phase(Phase.FAILED, reason="upload")
            return False
        self.record.total_items = len(items)
        return True

    def _start_mission(self) -> bool:
        for _ in range(3):
            try:
                self.vehicle.start_mission()
                return True
            except ActionError:
                continue
        return False

    def _baseline(self) -> None:
        record = self.record
        progress = self.vehicle.mission_progress()
        record.baseline_item = int(progress[0]) if progress and progress[1] > 0 else 0
        record.items_reached = record.baseline_item
        record.max_distance_from_start_m = 0.0
        record.mission_mode_confirmed = False
        here = self.vehicle.position()
        record.start_position = {
            "latitude_deg": round(here["latitude_deg"], 7), "longitude_deg": round(here["longitude_deg"], 7),
            "relative_altitude_m": round(here["relative_altitude_m"], 1),
            "absolute_altitude_m": round(here["absolute_altitude_m"], 1),
        }

    def _confirm_running(self) -> Generator[float, None, bool]:
        record, waited = self.record, 0.0
        while waited < self.s.start_confirm_timeout_s and not self._abort_requested:
            self._sample()
            if record.mission_mode_confirmed or record.progressed(self.s.progress_distance_m):
                self.emit("info", "mission execution confirmed", **record.progress_evidence())
                return True
            yield self.s.poll_interval_s
            waited += self.s.poll_interval_s
        return False

    def _ground_state(self) -> bool | None:
        record = self.record
        grounded = ground_evidence(record.last_landed_state, record.last_in_air)
        rate = record.last_vertical_speed_m_s
        if grounded is True and rate is not None and abs(rate) > SETTLED_RATE_M_S:
            return False
        if grounded is not None:
            return grounded
        position = record.last_position or {}
        height = height_above_launch_m(record.launch_amsl_m, position.get("absolute_altitude_m"), position.get("relative_altitude_m"))
        return None if height is None else height <= GROUND_HEIGHT_M

    def _descend_and_fail(self, reason: str) -> Generator[float, None, None]:
        if self.record.last_position is None:
            self._sample()
        if self._ground_state() is not True:
            self._do_action(self.s.mission_complete_action, reason=reason)
        self.set_phase(Phase.FAILED, reason=reason)
        return
        yield  # a generator

    def _monitor(self) -> Generator[float, None, None]:
        record, s, vehicle = self.record, self.s, self.vehicle
        last_item, last_mode = record.current_item, record.last_flight_mode
        battery_marks: set[int] = set()
        was_airborne, airborne_since, stalled = False, None, False
        while record.active:
            yield s.poll_interval_s
            if self._abort_requested:
                self._do_action("land", reason="operator abort")
                self.set_phase(Phase.ABORTED, reason="operator abort")
                return
            if self._pause_requested and record.phase_enum is Phase.RUNNING:
                self._pause_requested = False
                vehicle.pause_mission()
                self.set_phase(Phase.PAUSED, reason="operator pause")
            if self._resume_requested and record.phase_enum is Phase.PAUSED:
                self._resume_requested = False
                try:
                    vehicle.start_mission()
                except ActionError:
                    pass
                self.set_phase(Phase.RUNNING, reason="operator resume")
            self._sample()
            if record.current_item != last_item:
                self.emit("waypoint", f"reached item {record.current_item}/{record.total_items}",
                          current=record.current_item, total=record.total_items)
                last_item = record.current_item
            if record.last_flight_mode != last_mode:
                self.emit("mode", f"flight mode -> {record.last_flight_mode}", mode=record.last_flight_mode)
                last_mode = record.last_flight_mode
            fraction = (record.last_battery or {}).get("remaining_fraction")
            if fraction is not None and fraction > 0:
                for mark in (50, 30, 25, 20, 15, 10):
                    if fraction * 100 <= mark and mark not in battery_marks:
                        battery_marks.add(mark)
                        self.emit("battery", f"battery at {fraction * 100:.0f}%", percent=round(fraction * 100, 1))
                if s.auto_actions_enabled:
                    if fraction <= s.critical_battery_threshold:
                        self._auto_action("land", f"critical battery {fraction * 100:.0f}%", "critical_battery")
                    elif fraction <= s.low_battery_threshold and s.low_battery_action != "none":
                        self._auto_action(s.low_battery_action, f"low battery {fraction * 100:.0f}%", "low_battery")
            if s.auto_actions_enabled and s.geofence_breach_action != "none" and record.last_position:
                position = record.last_position
                violation = check_position(
                    self.fence(), position.get("latitude_deg"), position.get("longitude_deg"),
                    height_above_launch_m(record.launch_amsl_m, position.get("absolute_altitude_m"), position.get("relative_altitude_m")),
                )
                if violation is not None:
                    self._auto_action(s.geofence_breach_action, f"geofence breach: {violation.detail}", "geofence_breach")
            grounded = self._ground_state()
            if record.last_armed and grounded is False:
                was_airborne = True
                airborne_since = airborne_since if airborne_since is not None else vehicle.stack.now
            if was_airborne and not record.last_armed and grounded is True:
                if stalled or not record.mission_mode_confirmed or not record.progressed(s.progress_distance_m):
                    record.error = record.error or (
                        "the aircraft landed without flying the mission: no mission item past the one it started "
                        "on was reached and it never left the point it started from")
                    self.emit("error", record.error, rule="mission.no_progress", **record.progress_evidence())
                    self.set_phase(Phase.FAILED, reason="landed without flying the mission")
                else:
                    self.emit("info", "mission flown", **record.progress_evidence())
                    self.set_phase(Phase.COMPLETED, reason="landed and disarmed")
                return
            if vehicle.stack.ended:
                record.error = record.error or f"the flight ended in the air ({vehicle.stack.ended})"
                self.emit("error", record.error)
                self.set_phase(Phase.FAILED, reason=vehicle.stack.ended)
                return
            if was_airborne and record.phase_enum is Phase.RUNNING and not self._descent_commanded and self._items_done():
                self._descent_commanded = True
                self.emit("info", "mission items complete - commanding return-to-launch", **record.progress_evidence())
                self._do_action(s.mission_complete_action, reason="mission items complete")
            if (s.no_progress_timeout_s > 0 and was_airborne and not stalled and not self._descent_commanded
                    and record.phase_enum is Phase.RUNNING and not record.progressed(s.progress_distance_m)
                    and airborne_since is not None and vehicle.stack.now - airborne_since > s.no_progress_timeout_s):
                stalled = self._descent_commanded = True
                record.error = (f"the mission made no progress in {s.no_progress_timeout_s:.0f} s: no mission item "
                                "past the one it started on was reached and the aircraft did not leave the point it started from")
                self.emit("error", record.error, rule="mission.stalled", **record.progress_evidence())
                self._do_action(s.mission_complete_action, reason="mission made no progress")

    def _sample(self) -> None:
        record, vehicle = self.record, self.vehicle
        here = vehicle.position()
        record.last_position = {
            "latitude_deg": round(here["latitude_deg"], 7), "longitude_deg": round(here["longitude_deg"], 7),
            "relative_altitude_m": round(here["relative_altitude_m"], 1),
            "absolute_altitude_m": round(here["absolute_altitude_m"], 1),
        }
        height = height_above_launch_m(record.launch_amsl_m, record.last_position["absolute_altitude_m"],
                                       record.last_position["relative_altitude_m"])
        record.last_position["height_above_launch_m"] = None if height is None else round(height, 1)
        record.last_landed_state = vehicle.landed_state
        record.last_in_air = vehicle.in_air
        record.last_vertical_speed_m_s = round(-vehicle.velocity_ned()[2], 2)
        battery = vehicle.battery()
        record.last_battery = {"voltage_v": battery["voltage_v"], "remaining_fraction": round(battery["remaining_fraction"], 3)}
        record.last_flight_mode = vehicle.flight_mode
        if record.last_flight_mode == "MISSION":
            record.mission_mode_confirmed = True
        record.last_armed = vehicle.armed
        progress = vehicle.mission_progress()
        if progress and progress[1] > 0:
            record.current_item = progress[0]
            record.items_reached = max(record.items_reached, int(progress[0]))
            record.total_items = max(record.total_items, progress[1])
        if record.start_position:
            moved = haversine_distance(record.start_position["latitude_deg"], record.start_position["longitude_deg"],
                                       record.last_position["latitude_deg"], record.last_position["longitude_deg"])
            record.max_distance_from_start_m = max(record.max_distance_from_start_m, moved)

    def _items_done(self) -> bool:
        record = self.record
        if not record.mission_mode_confirmed or not record.progressed(self.s.progress_distance_m):
            return False
        if record.total_items > 0 and record.items_reached >= record.total_items:
            return True
        return (record.last_flight_mode or "").upper() == "HOLD"

    def _auto_action(self, action: str, reason: str, trigger: str) -> None:
        if any(a.get("trigger") == trigger for a in self.record.auto_actions_fired):
            return
        self.record.auto_actions_fired.append({"action": action, "trigger": trigger, "reason": reason, "ts": EPOCH + self.vehicle.stack.now})
        self.emit("auto_action", f"auto-action '{action}' fired: {reason}", action=action, reason=reason)
        self._do_action(action, reason)

    def _do_action(self, action: str, reason: str) -> None:
        try:
            if action == "rtl":
                self.vehicle.return_to_launch()
                self.set_phase(Phase.RETURNING, reason=reason)
            elif action == "land":
                self.vehicle.land()
                self.set_phase(Phase.LANDING, reason=reason)
            elif action == "hold":
                self.vehicle.pause_mission()
                self.set_phase(Phase.PAUSED, reason=reason)
        except ActionError as e:
            self.emit("error", f"auto-action '{action}' failed: {e}", action=action)

    # -- what get_mission_status returns ------------------------------------------------

    def status_payload(self, include_events: bool, event_limit: int) -> dict:
        record = self.record
        if record is None:
            return {"status": "success", "mission": None, "message": "no mission has been submitted"}
        s = self.s
        payload = {
            "status": "success",
            "mission": {
                "mission_id": record.mission_id, "phase": record.phase, "active": record.active,
                "progress_percent": record.progress_percent(), "current_item": record.current_item,
                "total_items": record.total_items, "elapsed_s": self.elapsed_s(), "created_at": record.created_at,
                "updated_at": record.updated_at, "position": record.last_position, "battery": record.last_battery,
                "flight_mode": record.last_flight_mode, "armed": record.last_armed, "error": record.error,
                "progress_evidence": record.progress_evidence(), "auto_actions_fired": record.auto_actions_fired,
                "resumed_after_restart": record.resumed_after_restart, "event_count": len(record.events),
            },
            "auto_actions_configured": {
                "enabled": s.auto_actions_enabled,
                "low_battery": f"{s.low_battery_action} below {s.low_battery_threshold * 100:.0f}%",
                "critical_battery": f"land below {s.critical_battery_threshold * 100:.0f}%",
                "geofence_breach": s.geofence_breach_action,
                "link_loss": s.link_loss_action,
            },
        }
        if include_events:
            limit = max(1, min(int(event_limit), 200))
            payload["events"] = record.events[-limit:]
        return payload

