"""PX4 as an offboard client sees it, flown by Waypoint's `ExecutorBackend`.

PEACE drives PX4 through MAVROS and DroneServer through MAVSDK: both stream
position setpoints, switch flight modes, arm, and read odometry back. Neither
ROS 2 nor MAVSDK belongs in this repository, and Waypoint already has the
aircraft: its simulator (and PX4 SITL behind the same interface). `FlightStack`
is the thin layer between the two. It offers the calls an offboard client
makes, with PX4's semantics, and turns them into flight on the backend:

    stack = make_stack(world=world)                 # on the pad, disarmed, in AUTO.LOITER
    stack.set_target(0, 0, 5, yaw=0.0)              # start streaming a setpoint
    stack.set_mode("OFFBOARD"); stack.arm(True)
    stack.advance(10.0)                             # ten seconds of flight
    stack.position, stack.yaw_deg, stack.armed, stack.mode

What is kept from PX4:

* OFFBOARD needs a setpoint stream before it is accepted, and falls back to
  AUTO.LOITER one second after the stream stops (COM_OF_LOSS_T).
* AUTO.LAND descends where it is and disarms on touchdown; AUTO.RTL climbs to
  the return altitude, flies home and lands; AUTO.TAKEOFF climbs to the
  take-off altitude and loiters.
* An armed aircraft left on the ground disarms itself after ten seconds
  (COM_DISARM_PRFLT). Disarming in the air is refused.
* Yaw is commanded with the setpoint and slews at 45 deg/s (MPC_YAWRAUTO_MAX).

The frame is the client's: local ENU metres from the take-off point, yaw 0 at
east and counter-clockwise positive, as MAVROS reports it. The aircraft itself
(acceleration, climb rate, battery, wind, faults, collisions) is Waypoint's
simulator untouched, so a rebuilt system and Waypoint fly the same airframe.

Time is simulated time. `advance(seconds)` is the only thing that moves it, so
a rebuilt system's sleeps and time-outs are deterministic and a flight that
took a minute in the paper takes milliseconds here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any

from waypoint.backends import ExecutorBackend
from waypoint.planner.models import (
    DEFAULT_CONSTRAINTS,
    Action,
    MissionConstraints,
    Plan,
    SubTask,
    Waypoint,
)
from waypoint.sim.config import SimConfig, Wind
from waypoint.sim.faults import FaultSchedule
from waypoint.sim.state import UAVState
from waypoint.world.models import World

Vector = tuple[float, float, float]

#: The simulator's own arrival radius. Kept below `hover_radius`, so the
#: simulator never decides by itself that a setpoint was a waypoint and lands.
SIM_ARRIVAL_RADIUS = 0.02
OFFBOARD = "OFFBOARD"
LOITER = "AUTO.LOITER"
LAND = "AUTO.LAND"
RTL = "AUTO.RTL"
TAKEOFF = "AUTO.TAKEOFF"
MODES = (OFFBOARD, LOITER, LAND, RTL, TAKEOFF)


def sim_config(**changes: Any) -> SimConfig:
    """Waypoint's airframe, told never to end a leg by itself."""
    return SimConfig(arrival_radius=SIM_ARRIVAL_RADIUS, max_leg_seconds=1.0e9, **changes)


def make_stack(
    *,
    world: World | None = None,
    wind: Wind | None = None,
    constraints: MissionConstraints | None = None,
    faults: FaultSchedule | None = None,
    home: Vector | None = None,
    backend: ExecutorBackend | None = None,
    **options: Any,
) -> "FlightStack":
    """A flight stack on Waypoint's built-in simulator, standing on the pad."""
    if backend is None:
        from waypoint.sim.backend import CustomSimBackend  # noqa: PLC0415

        backend = CustomSimBackend(world=world, wind=wind, config=sim_config())
    return FlightStack(backend, constraints or DEFAULT_CONSTRAINTS, faults, home=home, world=world, **options)


class FlightStack:
    """One aircraft, commanded the way an offboard client commands PX4."""

    #: Seconds an armed aircraft may sit on the ground before PX4 disarms it.
    disarm_preflight_s = 10.0
    #: Seconds without a setpoint before PX4 leaves OFFBOARD.
    offboard_loss_s = 1.0

    def __init__(
        self,
        backend: ExecutorBackend,
        constraints: MissionConstraints,
        faults: FaultSchedule | None = None,
        *,
        home: Vector | None = None,
        world: World | None = None,
        cruise_speed: float | None = None,
        yaw_rate_deg: float = 45.0,
        hover_radius: float = 0.15,
        takeoff_altitude: float = 2.5,
        rtl_altitude: float = 30.0,
        origin: Vector | None = None,
        count_legs: bool = False,
    ):
        from waypoint.sim.faults import NO_FAULTS  # noqa: PLC0415

        self.backend = backend
        self.constraints = constraints
        self.world = world
        self.faults = faults or NO_FAULTS
        backend.prepare(constraints, self.faults, home=home)
        pad = backend.state
        #: The take-off point in the world's frame.
        self.home: Vector = (pad.x, pad.y, pad.z)
        #: The origin of the client's local frame, in the world's frame: the take-off point unless a
        #: surveyed datum is given (an autopilot's local origin need not be where it took off).
        self.origin: Vector = tuple(origin) if origin is not None else self.home
        #: Number each distinct setpoint as a waypoint for the simulator, so a fault scheduled
        #: "at waypoint k" fires on the k-th place the client sends the aircraft, as it does on
        #: the k-th waypoint of one of Waypoint's own plans.
        self.count_legs = count_legs
        self.legs = 0
        self.dt = float(getattr(getattr(backend, "sim_config", None), "dt", 0.1))
        self.cruise_speed = float(cruise_speed if cruise_speed is not None else constraints.max_velocity)
        self.yaw_rate = math.radians(yaw_rate_deg)
        self.hover_radius = hover_radius
        self.takeoff_altitude = takeoff_altitude
        self.rtl_altitude = rtl_altitude

        self.mode = LOITER
        self.armed = False
        self.connected = True
        self.now = 0.0
        self._yaw = 0.0
        self._yaw_target: float | None = None
        self._target: Vector | None = None  # local frame
        self._stream_gap = 0.0
        self._ground_idle = 0.0
        self._phase: dict[str, Any] = {}
        self._key: tuple | None = None
        self._gen: Iterator[UAVState] | None = None
        self._held: tuple | None = None
        self._leg_index = 0
        self._leg_spot: tuple | None = None
        #: How far a holding aircraft may be pushed off its setpoint before it flies back to it.
        self.release_radius = max(3.0 * hover_radius, 0.45)
        self._ended: str | None = None
        #: Where AUTO.RTL returns to, in the world frame, when it is not the take-off point (an autopilot whose home moves).
        self.rtl_home: Vector | None = None
        #: Whether the motors were cut while it was flying (a kill switch, not a crash into something).
        self.killed_in_air = False

        self.states: list[UAVState] = [pad]
        #: What the client did and what it reached, in flight time, beside the backend's own events.
        self.marks: list[dict] = []
        self._state_listeners: list[Callable[[str, bool, bool], None]] = []
        self._odom_listeners: list[Callable[..., None]] = []

    # -- what the client reads ------------------------------------------------

    @property
    def state(self) -> UAVState:
        return self.backend.state

    @property
    def position(self) -> Vector:
        """Where the aircraft believes it is, in the local frame (GPS drift included, as odometry has it)."""
        return self.local_of(self.backend.state.reported_position)

    @property
    def true_position(self) -> Vector:
        return self.local_of(self.backend.state.position)

    @property
    def velocity(self) -> Vector:
        return self.backend.state.velocity_vector

    @property
    def yaw(self) -> float:
        """Heading in radians: 0 is east (+x), counter-clockwise positive."""
        return self._yaw

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self._yaw)

    @property
    def battery_pct(self) -> float:
        return self.backend.state.battery_pct

    @property
    def on_ground(self) -> bool:
        state = self.backend.state
        return state.z <= state.ground + 0.05

    @property
    def relative_altitude(self) -> float:
        """Height above the take-off point."""
        return self.backend.state.z - self.home[2]

    @property
    def ended(self) -> str | None:
        """Why the flight is over, if it is: "crashed", "battery_depleted" or "killed"."""
        return self._ended

    def local_of(self, world_point: Vector) -> Vector:
        return (world_point[0] - self.origin[0], world_point[1] - self.origin[1], world_point[2] - self.origin[2])

    def world_of(self, local_point: Vector) -> Vector:
        return (local_point[0] + self.origin[0], local_point[1] + self.origin[1], local_point[2] + self.origin[2])

    def on_state(self, listener: Callable[[str, bool, bool], None]) -> None:
        """Call `listener(mode, armed, connected)` after every tick, as MAVROS publishes /state."""
        self._state_listeners.append(listener)
        listener(self.mode, self.armed, self.connected)

    def on_odom(self, listener: Callable[..., None]) -> None:
        """Call `listener(x, y, z, vx, vy, vz, yaw)` after every tick, as MAVROS publishes odometry."""
        self._odom_listeners.append(listener)
        self._publish()

    # -- what the client commands ---------------------------------------------

    def set_target(self, x: float, y: float, z: float, yaw: float | None = None) -> None:
        """The position setpoint being streamed, in the local frame; `yaw` in radians."""
        self._target = (float(x), float(y), float(z))
        self._stream_gap = 0.0
        if yaw is not None:
            self._yaw_target = float(yaw)

    def stop_stream(self) -> None:
        self._target = None

    @property
    def streaming(self) -> bool:
        return self._target is not None

    def set_mode(self, mode: str) -> bool:
        if mode not in MODES or self._ended:
            return False
        if mode == OFFBOARD and self._target is None:
            return False  # PX4 refuses OFFBOARD until setpoints are arriving
        if mode != self.mode:
            self._enter(mode)
        return True

    def arm(self, value: bool = True) -> bool:
        if self._ended:
            return False
        if value:
            if not self.armed:
                self.armed = True
                self._ground_idle = 0.0
                self._note("ARMED", "armed")
        else:
            if self.armed and not self.on_ground:
                return False  # PX4 refuses to disarm in flight
            if self.armed:
                self.armed = False
                self._note("DISARMED", "disarmed")
        self._publish()
        return True

    def kill(self) -> None:
        """Cut the motors now. In the air that loses the aircraft: the flight ends here."""
        if self._ended:
            return
        airborne = not self.on_ground
        self._note(
            "KILL",
            "motors cut" + (f" at {self.relative_altitude:.1f} m above the pad: the aircraft falls" if airborne else " on the ground"),
            {"airborne": airborne, "height": round(self.relative_altitude, 2)},
        )
        self.killed_in_air = self.killed_in_air or airborne
        if airborne:
            self._end("killed")
        else:
            self.armed = False
            self._key = self._gen = None
        self._publish()

    def advance(self, seconds: float) -> None:
        """Let `seconds` of simulated time pass."""
        for _ in range(max(0, round(seconds / self.dt))):
            self._tick()

    def advance_until(self, done: Callable[[], bool], timeout: float, *, poll: float | None = None) -> bool:
        """Advance until `done()` is true; False when `timeout` seconds pass first."""
        poll = poll or self.dt
        elapsed = 0.0
        while True:
            if done():
                return True
            if elapsed >= timeout:
                return False
            self.advance(poll)
            elapsed += poll

    def mark(self, type: str, description: str, payload: dict | None = None) -> None:
        """Record something the client did, at the current flight time."""
        self._note(type, description, payload)

    def mark_waypoint(self, action: Action = Action.MOVE, label: str = "") -> None:
        """Record that the aircraft reached a place it was sent to, the way the simulator records a waypoint.

        Waypoint's mission scoring reads these events, so a rebuilt system's
        flight is scored exactly as one of Waypoint's own.
        """
        state = self.backend.state
        index = sum(1 for m in self.marks if m["type"] == "WAYPOINT")
        self._note(
            "WAYPOINT",
            f"reached waypoint {index} ({action.value}) at ({state.x:.1f}, {state.y:.1f}, {state.z:.1f})"
            + (f": {label}" if label else ""),
            {
                "waypoint_index": index,
                "action": action.value,
                "position": [round(c, 2) for c in state.position],
                "gps_error": round(state.gps_error, 2),
                "label": label,
            },
        )

    # -- the record -----------------------------------------------------------

    def events(self) -> list[dict]:
        """The backend's timeline and the client's marks, in time order."""
        records = []
        for event in self.backend.event_records():
            if event.get("type") == "HOLD":
                event = {**event, "description": "holding position"}
            elif event.get("type") == "WAYPOINT":
                continue  # only the client decides what counted as reaching a waypoint
            records.append(event)
        records.extend(self.marks)
        return sorted(records, key=lambda e: e.get("t", 0.0))

    def close(self) -> None:
        self.backend.close()

    # -- one tick -------------------------------------------------------------

    def _tick(self) -> None:
        self.now = round(self.now + self.dt, 6)
        if self._ended:
            self._publish()
            return
        if self.backend.battery_depleted and not self.on_ground:
            self._descend()
        elif not self.armed:
            pass
        elif self.mode == OFFBOARD:
            if self._target is None:
                self._stream_gap += self.dt
                if self._stream_gap >= self.offboard_loss_s:
                    self._enter(LOITER, "the setpoint stream stopped")
                self._hover()
            else:
                self._seek(self.world_of(self._target))
        elif self.mode == LAND:
            self._descend()
        elif self.mode == RTL:
            self._return_home()
        elif self.mode == TAKEOFF:
            self._climb_out()
        else:
            self._hover()
        self._slew_yaw()
        self._settle()
        self._publish()

    def _seek(self, target: Vector) -> bool:
        """Fly toward `target` (world frame); True once it is holding there."""
        here = self.backend.state.reported_position
        gap = math.dist(here, target)
        spot = (round(target[0], 3), round(target[1], 3), round(target[2], 3))
        # Once it is holding at a setpoint it stays in the hold until it is clearly off it,
        # so the last few centimetres of an approach do not flip it between the two.
        holding = self._key == ("hold",) and self._held == spot
        if gap <= self.hover_radius or (holding and gap <= self.release_radius):
            self._hover()
            self._held = spot
            return True
        speed = max(0.1, self.cruise_speed)
        key = ("goto", *spot, speed)
        if spot != self._leg_spot:  # a new place, not a return to the same setpoint after a hold
            self._leg_spot = spot
            self._leg_index = self.legs
            self.legs += 1
        index = min(self._leg_index, 40) if self.count_legs else 0
        self._pull(key, lambda: self.backend.run(_leg(target, speed, index, self.backend.state.reported_position)))
        return False

    def _hover(self) -> None:
        if self.on_ground:
            self._key = self._gen = None
            return
        self._pull(("hold",), lambda: self.backend.hold(3600.0))

    def _descend(self) -> None:
        if self.on_ground and self._key != ("land",):
            self._touchdown()
            return
        state = self._pull(("land",), lambda: self.backend.run(Plan()))
        # The simulator's landing ends with one settled state on the ground, at rest: that is touchdown.
        if state is None or (self.on_ground and state.velocity == 0.0):
            self._touchdown()

    def _return_home(self) -> None:
        phase = self._phase
        home = self.rtl_home or self.home
        home_z = home[2]
        if phase.get("step") is None:
            here = self.backend.state.reported_position
            phase.update(step="climb", altitude=max(here[2], home_z + self.rtl_altitude), start=(here[0], here[1]))
        if phase["step"] == "climb":
            if self._seek((phase["start"][0], phase["start"][1], phase["altitude"])):
                phase["step"] = "transit"
        elif phase["step"] == "transit":
            if self._seek((home[0], home[1], phase["altitude"])):
                phase["step"] = "land"
        else:
            self._descend()

    def _climb_out(self) -> None:
        phase = self._phase
        if phase.get("target") is None:
            here = self.backend.state.reported_position
            phase["target"] = (here[0], here[1], self.home[2] + self.takeoff_altitude)
        if self._seek(phase["target"]):
            self._enter(LOITER, "take-off altitude reached")

    def _pull(self, key: tuple, factory: Callable[[], Any]) -> UAVState | None:
        if key != self._key:
            self._key, self._gen = key, iter(factory())
        state = next(self._gen, None) if self._gen is not None else None
        if state is None:
            self._key = self._gen = None
            return None
        if state is not self.states[-1]:
            self.states.append(state)
        return state

    def _slew_yaw(self) -> None:
        if self._yaw_target is None or not self.armed:
            return
        gap = (self._yaw_target - self._yaw + math.pi) % (2.0 * math.pi) - math.pi
        step = max(-self.yaw_rate * self.dt, min(self.yaw_rate * self.dt, gap))
        self._yaw += step

    def _settle(self) -> None:
        """What PX4 decides by itself: a crash, a flat battery, disarming on the ground."""
        if self.backend.crashed:
            self._end("crashed")
            return
        if self.backend.battery_depleted and self.on_ground:
            self._end("battery_depleted")
            return
        if self.armed and self.on_ground and self.mode != LAND:
            climbing = self.mode == OFFBOARD and self._target is not None and self._target[2] > 0.05
            climbing = climbing or self.mode in (TAKEOFF,)
            self._ground_idle = 0.0 if climbing else self._ground_idle + self.dt
            if self._ground_idle >= self.disarm_preflight_s:
                self.armed = False
                self._note("DISARMED", "disarmed by itself after sitting armed on the ground")
        else:
            self._ground_idle = 0.0

    def _touchdown(self) -> None:
        if self.armed:
            self.armed = False
            self._note("DISARMED", "disarmed on touchdown")
        self._key = self._gen = None

    def _enter(self, mode: str, why: str = "") -> None:
        previous, self.mode = self.mode, mode
        self._phase = {}
        self._stream_gap = 0.0
        if mode != OFFBOARD:
            self._key = self._gen = None
        self._note("MODE", f"flight mode {previous} -> {mode}" + (f" ({why})" if why else ""), {"mode": mode})
        self._publish()

    def _end(self, why: str) -> None:
        self._ended = why
        self.armed = False
        self._key = self._gen = None

    def _note(self, type: str, description: str, payload: dict | None = None) -> None:
        self.marks.append(
            {"t": self.backend.state.timestamp, "type": type, "description": description, "payload": payload or {}}
        )

    def _publish(self) -> None:
        for listener in self._state_listeners:
            listener(self.mode, self.armed, self.connected)
        if self._odom_listeners:
            x, y, z = self.position
            vx, vy, vz = self.velocity
            for listener in self._odom_listeners:
                listener(x, y, z, vx, vy, vz, self._yaw)


def _leg(target: Vector, speed: float, index: int = 0, here: Vector | None = None) -> Plan:
    """One setpoint as a plan. `index` waypoints at the aircraft's own position come first, so the
    simulator counts the setpoint as waypoint `index`; they are reached at once and cost no time."""
    passed = [
        Waypoint(x=here[0], y=here[1], z=here[2], speed=speed, action=Action.MOVE) for _ in range(index if here else 0)
    ]
    goal = Waypoint(x=target[0], y=target[1], z=target[2], speed=speed, action=Action.MOVE)
    return Plan(subtasks=[SubTask(name="setpoint", waypoints=[*passed, goal])])
