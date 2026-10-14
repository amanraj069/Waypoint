"""PX4 SITL with Gazebo as an `ExecutorBackend`: the real autopilot flies the plan.

The custom simulator is a point mass; here PX4's controllers fly an x500
quadrotor through Gazebo's physics, in a Gazebo world written from the same
`World` the planner and verifier saw. The backend steers PX4 in offboard mode
leg by leg, the same way the simulator's controller does: toward the next
waypoint at its commanded speed, slowing to arrive, on the position the
aircraft *believes* it has. It hands the last few metres of a landing to
PX4's own landing mode.

What stays the same as the custom backend, so their runs compare:

* **frame and telemetry**: local ENU metres from the world origin, one
  `UAVState` every `dt` of simulated time, the same run file;
* **battery**: the same model (metres and seconds per percent, plus drain
  spikes), not PX4's simulated battery, so both backends spend the budget the
  verifier checked against;
* **crashes**: judged against the `World` the same way. Buildings are drawn in
  Gazebo but are not physics bodies, so hitting one is detected here, and the
  motors are cut;
* **faults**: one schedule drives both. An obstacle appears in the obstacle
  list (and in Gazebo); a battery spike drains the battery model; a gust is a
  drag force on the airframe that PX4 has to fight; GPS drift offsets the
  position the backend steers on, so the aircraft flies off its line believing
  it is on it, exactly as in the simulator. (PX4's own estimate is left
  alone; corrupting its GPS would test PX4's estimator, not the planner.)
* **pausing**: while the orchestrator replans, the aircraft holds position and,
  by default, Gazebo is paused, which stops PX4's clock too (they run in
  lockstep). Replanning then costs no flight time, as in the simulator; set
  `hold="hover"` to let the clock run and hover while the planner thinks.

Settings come from the environment when not passed: `WAYPOINT_PX4_DIR` (the
PX4-Autopilot checkout), `WAYPOINT_PX4_GUI=1` (open the Gazebo window),
`WAYPOINT_PX4_SPEED` (simulation speed factor) and `WAYPOINT_PX4_HOLD`.
"""

from __future__ import annotations

import atexit
import math
import os
import time
from collections.abc import Iterator
from dataclasses import replace

from waypoint.backends import BackendError, ExecutorBackend
from waypoint.gazebo.scene import REROUTE, _obstacle, _route, _sdf_document
from waypoint.px4.link import Setpoint
from waypoint.planner.models import Action, MissionConstraints, Obstacle, Plan, Waypoint
from waypoint.sim.config import DEFAULT_CONFIG, NO_WIND, SimConfig, Vector, Wind
from waypoint.sim.dynamics import add, desired_velocity, distance, heading_of
from waypoint.sim.executor import LiveObstacle, RunEvent, place
from waypoint.sim.faults import FaultEvent, FaultSchedule, FaultType, NO_FAULTS
from waypoint.sim.state import UAVState
from waypoint.world.models import World

#: Metres above the landing spot where PX4's own landing takes over.
LAND_HANDOFF = 3.0
#: Drag-equivalent force per (m/s)^2 of wind: half the air density times an
#: x500's frontal area and drag coefficient, near enough.
WIND_DRAG = 0.06
#: Wall-clock seconds without a new position before PX4 is declared lost.
TICK_TIMEOUT = 20.0
ARM_TIMEOUT = 90.0
#: The aircraft has flown once it has been this high above the ground.
AIRBORNE_AGL = 2.0

HOLD_MODES = ("freeze", "hover")

#: PX4 parameters for an unattended SITL flight driven from a script:
#: (value, is it an integer).
PARAMS: dict[str, tuple[float, bool]] = {
    "NAV_DLL_ACT": (0, True),         # no ground station on the link, and none needed
    "COM_RC_IN_MODE": (4, True),      # no RC transmitter either
    "COM_RCL_EXCEPT": (4, True),      # offboard never waits on RC
    "MPC_XY_VEL_MAX": (15.0, False),  # the simulator's airframe ceiling
    "MPC_Z_VEL_MAX_UP": (4.0, False),  # and its climb rate, both ways
    "MPC_Z_VEL_MAX_DN": (4.0, False),
    "MPC_LAND_SPEED": (1.5, False),   # the simulator's landing speed
}

#: How each fault reaches PX4, recorded with every run.
FAULT_MODELS = {
    FaultType.GPS_DRIFT.value: "offsets the position the backend steers on; PX4's estimate is untouched",
    FaultType.OBSTACLE_APPEARS.value: "added to the obstacle list and drawn in Gazebo",
    FaultType.WIND_GUST.value: f"a drag force of {WIND_DRAG:g} N per (m/s)^2 on the airframe, which PX4 fights",
    FaultType.BATTERY_DRAIN_SPIKE.value: "extra drain in the shared battery model",
}


class Px4Backend(ExecutorBackend):
    """One flight of PX4's x500 in Gazebo, steered leg by leg in offboard mode."""

    name = "px4"

    def __init__(
        self,
        *,
        world: World | None = None,
        wind: Wind | None = None,
        config: SimConfig | None = None,
        gui: bool | None = None,
        speed: float | None = None,
        hold: str | None = None,
        install=None,
        keep: bool = False,
    ):
        from waypoint.px4.sitl import Px4Install  # noqa: PLC0415

        self.world = world
        self.wind = wind or NO_WIND
        self.config = config or DEFAULT_CONFIG
        self.gui = _flag(os.environ.get("WAYPOINT_PX4_GUI")) if gui is None else gui
        self.speed = float(os.environ.get("WAYPOINT_PX4_SPEED") or 1.0) if speed is None else speed
        self.hold = (hold or os.environ.get("WAYPOINT_PX4_HOLD") or "freeze").lower()
        if self.hold not in HOLD_MODES:
            raise BackendError(f"hold must be one of {', '.join(HOLD_MODES)}, not {self.hold!r}")
        if self.speed <= 0:
            raise BackendError("the simulation speed factor must be positive")
        # Fails now, at make_backend, when PX4 is not installed: not twenty flights later.
        self.install = install or Px4Install.find()
        self.keep = keep
        self._version: str | None = None
        self._sitl = None
        self._link = None
        self._state: UAVState | None = None
        self._closed_at_exit = False

    # -- the flight -------------------------------------------------------------

    def prepare(
        self,
        constraints: MissionConstraints,
        faults: FaultSchedule,
        home: tuple[float, float, float] | None = None,
    ) -> None:
        from waypoint.px4.link import Link  # noqa: PLC0415
        from waypoint.px4.sitl import Sitl  # noqa: PLC0415

        self.close()  # one backend is one flight; preparing again starts afresh
        config = self.config
        if home is not None:
            config = config.model_copy(update={"home": tuple(float(v) for v in home)})
        self.config = config
        x, y, z = config.home
        ground = self.world.elevation_at(x, y) if self.world else 0.0
        self._pad = (x, y, max(z, ground))

        self._sitl = Sitl(self.world, self._pad, gui=self.gui, speed=self.speed, install=self.install, keep=self.keep)
        self._sitl.start()
        try:
            self._link = Link()
            self._link.connect()
            refused = [name for name, (value, integer) in PARAMS.items()
                       if not self._link.set_param(name, value, integer=integer)]
            if refused:
                raise BackendError(f"PX4 did not take the parameters {', '.join(refused)}")
            if self._link.wait_for("HOME_POSITION", 60.0) is None:
                raise BackendError(f"PX4 never set its home position (no GPS fix?): {self._link.last_words()}")
        except BaseException:
            self.close()
            raise
        if not self._closed_at_exit:
            atexit.register(self.close)
            self._closed_at_exit = True

        self.constraints = constraints
        self.faults = faults or NO_FAULTS
        self._state = UAVState(
            x=self._pad[0], y=self._pad[1], z=self._pad[2],
            battery_pct=config.start_battery_pct, ground=ground,
        )
        self.events: list[RunEvent] = []
        self._obstacles = [LiveObstacle(obstacle=o) for o in constraints.obstacles]
        self._fired: set[int] = set()
        self._active: list[_Active] = []
        self._breached: set[int] = set()
        self._gps_offset: Vector = (0.0, 0.0, 0.0)
        self._force: Vector = (0.0, 0.0, 0.0)
        self._pushed_at = -math.inf
        self._target_index = 0
        self._subtask = ""
        self._route: list[tuple[str, Waypoint]] = []
        self._distance = 0.0
        self._crashed = self._out_of_battery = self._landing = False
        self._airborne = self._done = self._paused = False
        self._armed_once = False
        self._origin = None  # PX4's local NED position on the pad
        self._t0_ms = self._last_ms = 0
        self._yaw = math.pi / 2  # spawned facing east, which is NED yaw 90 degrees
        self._plans = 0

    def run(self, plan: Plan) -> Iterator[UAVState]:
        self._require()
        if self._paused:
            self._sitl.pause(False)
            self._paused = False
        self._show_route(plan)
        return self._fly(plan, first=self._origin is None)

    def _fly(self, plan: Plan, *, first: bool) -> Iterator[UAVState]:
        finished = False
        try:
            if first:
                self._take_off_ready()
                yield self._state  # t = 0, on the pad, as the simulator starts
            yield from self._fly_plan(plan)
            finished = True
        finally:
            if not finished and not self._done:
                self._hold_here()

    def _fly_plan(self, plan: Plan) -> Iterator[UAVState]:
        self._route = [(s.name, wp) for s in plan.subtasks for wp in s.waypoints]
        if self._done:
            return
        self._ensure_offboard()
        landing_index = None
        for index, (subtask, waypoint) in enumerate(self._route):
            if self._stopped:
                break
            self._target_index, self._subtask = index, subtask
            if waypoint.action is Action.LAND:
                # PX4 flies the last few metres down; whatever follows a LAND is not flown.
                yield from self._fly_to(index, waypoint, handoff=True)
                landing_index = index
                break
            yield from self._fly_to(index, waypoint)
            if self._stopped:
                break
            yield from self._dwell(waypoint)
        if not self._crashed:
            yield from self._land(landing_index)

    # -- phases -------------------------------------------------------------------

    def _take_off_ready(self) -> None:
        """Offboard and armed on the pad, with the pad's local position as the frame origin."""
        link = self._link
        here = link.wait_position(0, TICK_TIMEOUT)
        if here is None:
            raise BackendError(f"PX4 is not reporting a local position: {link.last_words()}")
        attitude = link.latest.get("ATTITUDE")
        if attitude is not None:
            self._yaw = attitude.yaw
        link.setpoint = Setpoint.hold((here.x, here.y, here.z), self._yaw)
        time.sleep(0.3)  # PX4 wants setpoints streaming before it accepts offboard
        deadline = time.monotonic() + ARM_TIMEOUT
        while not link.armed:
            if time.monotonic() > deadline:
                raise BackendError(f"PX4 would not arm within {ARM_TIMEOUT:g} s: {link.last_words()}")
            link.set_offboard()
            link.arm()
            if not link.armed:
                time.sleep(1.0)
        self._armed_once = True
        here = link.wait_position(0, TICK_TIMEOUT)
        self._origin = (here.x, here.y, here.z)
        self._t0_ms = self._last_ms = here.time_boot_ms

    def _ensure_offboard(self) -> None:
        """Take control back from PX4's landing mode, or re-arm, before flying a new plan."""
        link = self._link
        if not self._landing and link.armed:
            return
        self._landing = False
        here = link.latest.get("LOCAL_POSITION_NED")
        link.setpoint = Setpoint.hold((here.x, here.y, here.z), self._yaw)
        deadline = time.monotonic() + ARM_TIMEOUT
        while True:
            link.set_offboard()
            if not link.armed:
                link.arm()
            heartbeat = link.latest.get("HEARTBEAT")
            if link.armed and heartbeat is not None and (heartbeat.custom_mode >> 16) & 0xFF == 6:
                return
            if time.monotonic() > deadline:
                raise BackendError(f"PX4 would not return to offboard control: {link.last_words()}")
            time.sleep(0.5)

    def _fly_to(self, index: int, waypoint: Waypoint, *, handoff: bool = False) -> Iterator[UAVState]:
        config = self.config
        self._landing = waypoint.action is Action.LAND
        target = (waypoint.x, waypoint.y, waypoint.z)
        if handoff:
            target = (waypoint.x, waypoint.y, waypoint.z + LAND_HANDOFF)
        cruise = min(waypoint.speed, config.max_velocity)
        started = self._state.timestamp
        while distance(self._state.reported_position, target) > config.arrival_radius:
            if self._state.timestamp - started >= config.max_leg_seconds:
                self._record(
                    "LEG_TIMEOUT",
                    f"gave up on waypoint {index} after {config.max_leg_seconds:g}s "
                    f"({distance(self._state.reported_position, target):.1f} m short)",
                    {"waypoint_index": index},
                )
                return
            self._steer(desired_velocity(self._state.reported_position, target, cruise, config))
            yield self._tick()
            if self._stopped:
                return
        if not handoff:
            self._reached(index, waypoint)

    def _dwell(self, waypoint: Waypoint) -> Iterator[UAVState]:
        seconds = {
            Action.HOVER: self.config.hover_seconds,
            Action.CAPTURE: self.config.capture_seconds,
        }.get(waypoint.action, 0.0)
        if seconds <= 0:
            return
        self._hold_position()
        until = self._state.timestamp + seconds
        while self._state.timestamp < until - 1e-6:
            yield self._tick()
            if self._stopped:
                return

    def _land(self, landing_index: int | None) -> Iterator[UAVState]:
        """PX4's landing, straight down from where it is, then settle on the ground."""
        link = self._link
        self._landing = True
        for _ in range(3):
            if link.land() == 0:
                break
        started = self._state.timestamp
        while not link.on_ground and self._state.timestamp - started < self.config.max_leg_seconds:
            yield self._tick()
            if self._crashed:
                return
        state = self._state
        self._state = replace(state, z=state.ground, vx=0.0, vy=0.0, vz=0.0, active_faults=())
        if landing_index is not None:
            self._reached(landing_index, self._route[landing_index][1])
        self._record(
            "LANDED",
            f"touched down at ({state.x:.1f}, {state.y:.1f}) with "
            f"{state.battery_pct:.1f}% battery after {state.timestamp:.1f}s",
            {"battery_pct": round(state.battery_pct, 2)},
        )
        self._done = True
        yield self._state

    # -- one tick -----------------------------------------------------------------

    def _tick(self) -> UAVState:
        """Wait for PX4's position `dt` of simulated time on, and account for that interval."""
        config = self.config
        previous = self._state
        message = self._link.wait_position(self._last_ms + round(config.dt * 1000), TICK_TIMEOUT)
        if message is None:
            alive = self._sitl.running if self._sitl else False
            raise BackendError(
                f"PX4 stopped reporting its position{'' if alive else ' (the simulation exited)'}: "
                f"{self._link.last_words()}"
            )
        self._last_ms = message.time_boot_ms
        t = round((message.time_boot_ms - self._t0_ms) / 1000.0, 3)
        dt = max(0.0, t - previous.timestamp)

        self._apply_faults(t)
        wind = add(self.wind.at(t), self._gust())
        self._blow(wind)

        x, y, z = self._enu(message.x, message.y, message.z)
        velocity = (message.vy, message.vx, -message.vz)
        ground = self.world.elevation_at(x, y) if self.world else 0.0
        travelled = distance(previous.position, (x, y, z))
        self._distance += travelled
        battery = self._drain(travelled, dt, t)
        self._gps_offset = add(self._gps_offset, self._drift(dt))

        self._state = UAVState(
            x=x, y=y, z=max(ground, z),
            vx=velocity[0], vy=velocity[1], vz=velocity[2],
            heading=heading_of(velocity, previous.heading),
            battery_pct=battery,
            timestamp=t,
            gps_offset=self._gps_offset,
            wind=wind,
            ground=ground,
            target_index=self._target_index,
            subtask=self._subtask,
            active_faults=tuple(active.event.label for active in self._active),
        )
        if self._state.agl > AIRBORNE_AGL:
            self._airborne = True
        self._check_separation()
        self._check_impact()
        return self._state

    def _steer(self, velocity: Vector) -> None:
        """Fly an ENU velocity, nose along the track while moving."""
        vx, vy, vz = velocity
        if math.hypot(vx, vy) > 0.5:
            self._yaw = math.atan2(vx, vy)  # NED yaw: from north, toward east
        self._link.setpoint = Setpoint.fly((vy, vx, -vz), self._yaw)

    def _hold_position(self) -> None:
        here = self._link.latest.get("LOCAL_POSITION_NED")
        if here is not None:
            self._link.setpoint = Setpoint.hold((here.x, here.y, here.z), self._yaw)

    def _hold_here(self) -> None:
        """The orchestrator stopped listening: hold position, and freeze the clock unless hovering."""
        if self._link is None:
            return
        if not self._landing:
            self._hold_position()
        if self.hold == "freeze" and self._sitl is not None:
            try:
                self._sitl.pause(True)
                self._paused = True
            except BackendError:
                pass  # still hovering in place, which is the other mode anyway

    # -- the shared models: battery, faults, separation, impact -----------------

    def _drain(self, travelled: float, dt: float, t: float) -> float:
        config = self.config
        used = (
            travelled / config.meters_per_battery_pct
            + dt / config.seconds_per_battery_pct
            + self._spike() * dt
        )
        battery = max(0.0, self._state.battery_pct - used)
        if battery <= 0.0 and not self._out_of_battery:
            self._out_of_battery = True
            self._record("BATTERY_EMPTY", f"battery reached 0% at t={t:.1f}s -- abandoning the plan and landing")
        return battery

    def _apply_faults(self, t: float) -> None:
        for active in list(self._active):
            if active.expires is not None and t >= active.expires:
                self._active.remove(active)
                self._record("FAULT_END", f"{active.event.label} ended",
                             {"fault": active.event.type.value, "index": active.index})
        state = self._state
        due = self.faults.due(
            fired=self._fired,
            t=t,
            battery_pct=state.battery_pct,
            waypoint_index=self._target_index,
            distance_from_home=math.dist(state.position, self._pad),
            altitude=state.z,
        )
        for index, event in due:
            self._fired.add(index)
            self._record("FAULT", event.describe(),
                         {"fault": event.type.value, "index": index, "magnitude": event.magnitude})
            if event.type is FaultType.OBSTACLE_APPEARS and event.obstacle:
                ahead = [(wp.x, wp.y) for _, wp in self._route[self._target_index:]]
                obstacle = place(event, (state.x, state.y), ahead)
                self._obstacles.append(LiveObstacle(obstacle=obstacle, appeared_at=t))
                self.events[-1].payload["obstacle"] = obstacle.model_dump()
                self._show_obstacle(obstacle, len(self._obstacles))
                continue
            expires = None if event.duration is None else t + event.duration
            self._active.append(_Active(index=index, event=event, expires=expires))

    def _gust(self) -> Vector:
        gust = (0.0, 0.0, 0.0)
        for active in self._active:
            if active.event.type is FaultType.WIND_GUST:
                gust = add(gust, active.event.vector)
        return gust

    def _drift(self, dt: float) -> Vector:
        drift = (0.0, 0.0, 0.0)
        for active in self._active:
            if active.event.type is FaultType.GPS_DRIFT:
                drift = add(drift, tuple(c * dt for c in active.event.vector))  # type: ignore[arg-type]
        return drift

    def _spike(self) -> float:
        return sum(a.event.magnitude for a in self._active if a.event.type is FaultType.BATTERY_DRAIN_SPIKE)

    def _blow(self, wind: Vector) -> None:
        """Push the airframe with the drag the wind would put on it, when that changes."""
        speed = math.sqrt(sum(c * c for c in wind))
        force = tuple(round(WIND_DRAG * speed * c, 2) for c in wind)
        changed = math.dist(force, self._force) > 0.25 or (any(self._force) and not any(force))
        # At most once a simulated second, so turbulence cannot flood Gazebo with commands.
        if changed and self._state.timestamp - self._pushed_at >= 1.0:
            self._force, self._pushed_at = force, self._state.timestamp
            self._sitl.push(force)

    def _check_separation(self) -> None:
        separation = self.constraints.min_separation if self.constraints else 0.0
        state = self._state
        for index, live in enumerate(self._obstacles):
            if index in self._breached:
                continue
            obstacle = live.obstacle
            clearance = obstacle.radius + separation
            if clearance <= 0 or (obstacle.height is not None and state.z > obstacle.height):
                continue
            gap = math.dist((state.x, state.y), (obstacle.x, obstacle.y))
            if gap < clearance:
                self._breached.add(index)
                self._record(
                    "SEPARATION",
                    f"came within {gap:.1f} m of '{obstacle.name}' (keep-out {clearance:.1f} m)",
                    {"obstacle": obstacle.name, "gap": round(gap, 2)},
                )

    def _check_impact(self) -> None:
        """Into a building, or into the ground when not landing: cut the motors, as an impact would."""
        if self._crashed or not self.config.collisions:
            return
        state = self._state
        building = self.world.collides(state.x, state.y, state.z) if self.world else None
        if building is not None:
            what = (f"struck {building.label()} at ({state.x:.1f}, {state.y:.1f}, {state.z:.1f}) "
                    f"doing {state.velocity:.1f} m/s -- its roof is at {building.roof:.0f} m")
            payload = {"building": building.name, "speed": round(state.velocity, 2)}
        elif self._airborne and not self._landing and state.agl <= 0.3:
            what = (f"flew into the ground at ({state.x:.1f}, {state.y:.1f}), elevation "
                    f"{state.ground:.1f} m, doing {state.velocity:.1f} m/s")
            payload = {"speed": round(state.velocity, 2)}
        else:
            return
        self._crashed = self._done = True
        self._record("COLLISION", what, payload)
        self._link.setpoint = None
        self._link.kill()
        self._state = replace(state, vx=0.0, vy=0.0, vz=0.0)

    def _reached(self, index: int, waypoint: Waypoint) -> None:
        state = self._state
        self._record(
            "WAYPOINT",
            f"reached waypoint {index} ({waypoint.action.value}) at ({state.x:.1f}, {state.y:.1f}, {state.z:.1f})",
            {"waypoint_index": index, "position": [round(c, 2) for c in state.position],
             "gps_error": round(state.gps_error, 2)},
        )

    def _record(self, type: str, description: str, payload: dict | None = None) -> None:
        self.events.append(RunEvent(t=self._state.timestamp, type=type, description=description, payload=payload or {}))

    # -- frames and drawing -----------------------------------------------------------

    def _enu(self, north: float, east: float, down: float) -> Vector:
        """PX4's local NED, relative to where it stood on the pad, as world ENU."""
        n0, e0, d0 = self._origin
        x0, y0, z0 = self._pad
        return (x0 + east - e0, y0 + north - n0, z0 - (down - d0))

    def _show_route(self, plan: Plan) -> None:
        self._plans += 1
        state = self._state
        waypoints = [{"x": w.x, "y": w.y, "z": w.z, "action": w.action.value} for w in plan.waypoints()]
        world = self.world or World()
        if self._plans == 1:
            model = _route(world, state.position, waypoints, name="route")
        else:
            model = _route(world, state.position, waypoints, name=f"reroute_{self._plans - 1}", colour=REROUTE)
        if model:
            self._sitl.show(_sdf_document(model))

    def _show_obstacle(self, obstacle: Obstacle, number: int) -> None:
        constraints = self.constraints.model_dump(mode="json") if self.constraints else {}
        model = _obstacle(self.world or World(), obstacle.model_dump(), constraints, f"appeared_{number}")
        self._sitl.show(_sdf_document(model))

    @property
    def _stopped(self) -> bool:
        return self._crashed or self._out_of_battery

    # -- what the orchestrator reads -------------------------------------------------

    @property
    def state(self) -> UAVState:
        self._require()
        return self._state

    @property
    def obstacles(self) -> list[Obstacle]:
        self._require()
        return [live.obstacle for live in self._obstacles]

    def event_records(self) -> list[dict]:
        return [event.as_record() for event in getattr(self, "events", [])]

    def obstacle_records(self) -> list[dict]:
        return [live.as_record() for live in getattr(self, "_obstacles", [])]

    @property
    def distance_flown(self) -> float:
        return getattr(self, "_distance", 0.0)

    @property
    def crashed(self) -> bool:
        return getattr(self, "_crashed", False)

    @property
    def battery_depleted(self) -> bool:
        return getattr(self, "_out_of_battery", False)

    def describe(self) -> dict:
        if self._version is None:
            self._version = self.install.version() or "unknown"
        return {
            "name": self.name,
            "px4": self._version,
            "airframe": "gz_x500",
            "speed_factor": self.speed,
            "while_replanning": self.hold,
            "config": self.config.model_dump(mode="json"),
            "wind": self.wind.model_dump(mode="json"),
            "faults": FAULT_MODELS,
        }

    def close(self) -> None:
        """Stop PX4 and Gazebo; safe to call twice, and called at exit as a last resort."""
        link, sitl = self._link, self._sitl
        self._link = self._sitl = None
        if link is not None:
            link.setpoint = None
            link.close()
        if sitl is not None:
            sitl.stop()

    def _require(self) -> None:
        if self._state is None:
            raise RuntimeError("call prepare(constraints, faults) before flying")


class _Active:
    """A fault that has fired and not yet run out."""

    __slots__ = ("index", "event", "expires")

    def __init__(self, *, index: int, event: FaultEvent, expires: float | None):
        self.index, self.event, self.expires = index, event, expires


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}
