"""The simulator: fly a verified plan tick by tick and report what happened.

`Executor` walks the plan's waypoints with the point-mass model in
`sim.dynamics`, disturbed by wind and by whatever the `FaultSchedule` injects,
and yields one `UAVState` per `dt`. It is a generator, so the orchestrator sees
telemetry as it happens rather than after the flight.

Three things make it a simulator rather than an interpolator:

* the aircraft accelerates -- it leaves and arrives at a waypoint at a speed the
  airframe could actually manage, and wind blows it off the straight line;
* it flies on its *reported* position, so GPS drift takes it somewhere other
  than where it believes it is going;
* it can run out of battery, in which case it stops flying the plan and lands.

Every run also collects a list of `RunEvent`s -- faults firing and expiring,
waypoints reached, separation lost, touchdown -- which is what the run log
keeps and Gazebo prints as the flight plays.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from waypoint.planner.models import Action, MissionConstraints, Obstacle, Plan, Waypoint
from waypoint.sim.config import DEFAULT_CONFIG, NO_WIND, SimConfig, Vector, Wind
from waypoint.sim.dynamics import add, desired_velocity, distance, heading_of, step
from waypoint.sim.faults import FaultEvent, FaultSchedule, FaultType, NO_FAULTS
from waypoint.sim.state import UAVState
from waypoint.world.models import World

LANDING_SPEED = 1.5


@dataclass
class RunEvent:
    """Something worth marking on the timeline."""

    t: float
    type: str
    description: str
    payload: dict = field(default_factory=dict)

    def as_record(self) -> dict:
        return {
            "t": round(self.t, 3),
            "type": self.type,
            "description": self.description,
            **({"payload": self.payload} if self.payload else {}),
        }


@dataclass
class LiveObstacle:
    """An obstacle the aircraft knows about, and when it learned of it."""

    obstacle: Obstacle
    appeared_at: float = 0.0

    def as_record(self) -> dict:
        return {**self.obstacle.model_dump(), "appeared_at": round(self.appeared_at, 3)}


@dataclass
class _Active:
    """A fault that has fired and not yet expired."""

    index: int
    event: FaultEvent
    started: float
    expires: float | None

    def alive_at(self, t: float) -> bool:
        return self.expires is None or t < self.expires


def initial_state(battery_pct: float = 100.0, config: SimConfig | None = None) -> UAVState:
    """State at the launch pad: on the ground, stationary, full battery."""
    config = config or DEFAULT_CONFIG
    return UAVState(
        x=config.home[0],
        y=config.home[1],
        z=config.home[2],
        battery_pct=battery_pct,
        timestamp=0.0,
    )


class Executor:
    """Flies one plan, once."""

    def __init__(
        self,
        plan: Plan,
        *,
        constraints: MissionConstraints | None = None,
        config: SimConfig | None = None,
        wind: Wind | None = None,
        faults: FaultSchedule | None = None,
        start: UAVState | None = None,
        world: World | None = None,
    ):
        self.plan = plan
        self.constraints = constraints
        self.config = config or DEFAULT_CONFIG
        self.wind = wind or NO_WIND
        self.faults = faults or NO_FAULTS
        self.world = world

        self.state = start or initial_state(self.config.start_battery_pct, self.config)
        if world is not None and start is None:
            # Stand the aircraft on the ground rather than inside the hill.
            pad = world.elevation_at(self.state.x, self.state.y)
            self.state.z = max(self.state.z, pad)
            self.state.ground = pad
        self.events: list[RunEvent] = []
        self.obstacles: list[LiveObstacle] = [
            LiveObstacle(obstacle=obstacle) for obstacle in (constraints.obstacles if constraints else [])
        ]

        self._fired: set[int] = set()
        self._active: list[_Active] = []
        self._breached: set[int] = set()
        self._gps_offset: Vector = (0.0, 0.0, 0.0)
        self._target_index = 0
        self._subtask = ""
        self._out_of_battery = False
        self._crashed = False
        self._landing = False
        self._distance_flown = 0.0

    # -- public API ---------------------------------------------------------

    @property
    def distance_flown(self) -> float:
        return self._distance_flown

    @property
    def crashed(self) -> bool:
        return self._crashed

    @property
    def out_of_battery(self) -> bool:
        return self._out_of_battery

    def run(self) -> Iterator[UAVState]:
        """Yield one state per tick, from the pad to touchdown."""
        yield self.state  # t = 0, on the pad
        yield from self._fly_plan()

    def resume(self, plan: Plan) -> Iterator[UAVState]:
        """Fly a new plan from wherever the aircraft is now.

        For replanning mid-flight: stop iterating `run()` (or an earlier
        `resume`) and call this. The clock, the battery, the GPS offset, the
        obstacles and the faults already fired all carry on; only the route
        changes. Waypoint indices, and so `at_waypoint` triggers, count from
        the start of the new plan.
        """
        self.plan = plan
        self._landing = False
        yield from self._fly_plan()

    def _fly_plan(self) -> Iterator[UAVState]:
        for index, (subtask, waypoint) in enumerate(self._flat_plan()):
            if self._out_of_battery or self._crashed:
                break
            self._target_index = index
            self._subtask = subtask
            yield from self._fly_to(index, waypoint)
            if self._out_of_battery or self._crashed:
                break
            yield from self._dwell(waypoint)

        if not self._crashed:
            yield from self._land()

    def run_to_list(self) -> list[UAVState]:
        return list(self.run())

    def event_records(self) -> list[dict]:
        return [event.as_record() for event in self.events]

    def obstacle_records(self) -> list[dict]:
        return [live.as_record() for live in self.obstacles]

    # -- flight phases ------------------------------------------------------

    def _flat_plan(self) -> list[tuple[str, Waypoint]]:
        return [(s.name, wp) for s in self.plan.subtasks for wp in s.waypoints]

    def _fly_to(self, index: int, waypoint: Waypoint) -> Iterator[UAVState]:
        self._landing = waypoint.action is Action.LAND
        target = (waypoint.x, waypoint.y, waypoint.z)
        cruise = min(waypoint.speed, self.config.max_velocity)
        started = self.state.timestamp

        while distance(self.state.reported_position, target) > self.config.arrival_radius:
            if self.state.timestamp - started >= self.config.max_leg_seconds:
                self._record(
                    "LEG_TIMEOUT",
                    f"gave up on waypoint {index} after {self.config.max_leg_seconds:g}s "
                    f"({distance(self.state.reported_position, target):.1f} m short)",
                    {"waypoint_index": index},
                )
                return
            yield self._tick(target, cruise)
            if self._out_of_battery or self._crashed:
                return

        self._record(
            "WAYPOINT",
            f"reached waypoint {index} ({waypoint.action.value}) at "
            f"({self.state.x:.1f}, {self.state.y:.1f}, {self.state.z:.1f})",
            {
                "waypoint_index": index,
                "position": [round(c, 2) for c in self.state.position],
                "gps_error": round(self.state.gps_error, 2),
            },
        )

    def _dwell(self, waypoint: Waypoint) -> Iterator[UAVState]:
        """Hold station at a HOVER or CAPTURE waypoint."""
        seconds = {
            Action.HOVER: self.config.hover_seconds,
            Action.CAPTURE: self.config.capture_seconds,
        }.get(waypoint.action, 0.0)
        target = (waypoint.x, waypoint.y, waypoint.z)
        for _ in range(round(seconds / self.config.dt)):
            yield self._tick(target, 0.0, hold=True)
            if self._out_of_battery or self._crashed:
                return

    def _land(self) -> Iterator[UAVState]:
        """Descend to the ground wherever we are and settle."""
        self._landing = True
        target = (self.state.x, self.state.y, 0.0)
        started = self.state.timestamp
        ground = self.world.elevation_at(self.state.x, self.state.y) if self.world else 0.0
        target = (self.state.x, self.state.y, ground)
        while (
            self.state.z > ground + 0.05
            and self.state.timestamp - started < self.config.max_leg_seconds
        ):
            yield self._tick(target, LANDING_SPEED)
            if self._crashed:
                return

        self.state = UAVState(
            x=self.state.x,
            y=self.state.y,
            z=ground,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            heading=self.state.heading,
            battery_pct=self.state.battery_pct,
            timestamp=round(self.state.timestamp + self.config.dt, 6),
            gps_offset=self._gps_offset,
            wind=self.state.wind,
            ground=ground,
            target_index=self._target_index,
            subtask=self._subtask,
        )
        self._record(
            "LANDED",
            f"touched down at ({self.state.x:.1f}, {self.state.y:.1f}) with "
            f"{self.state.battery_pct:.1f}% battery after {self.state.timestamp:.1f}s",
            {"battery_pct": round(self.state.battery_pct, 2)},
        )
        yield self.state

    # -- one tick -----------------------------------------------------------

    def _tick(self, target: Vector, cruise: float, hold: bool = False) -> UAVState:
        config = self.config
        t = round(self.state.timestamp + config.dt, 6)

        self._apply_faults(t)
        wind = add(self.wind.at(t), self._gust())

        desired = (
            (0.0, 0.0, 0.0)
            if hold
            else desired_velocity(self.state.reported_position, target, cruise, config)
        )
        position, velocity = step(
            self.state.position, self.state.velocity_vector, desired, wind, config
        )

        travelled = distance(self.state.position, position)
        self._distance_flown += travelled
        battery = self._drain(travelled, t)

        self._gps_offset = add(self._gps_offset, self._drift_step())

        ground = self.world.elevation_at(position[0], position[1]) if self.world else 0.0
        self.state = UAVState(
            x=position[0],
            y=position[1],
            z=max(ground, position[2]),
            vx=velocity[0],
            vy=velocity[1],
            vz=velocity[2],
            heading=heading_of(velocity, self.state.heading),
            battery_pct=battery,
            timestamp=t,
            gps_offset=self._gps_offset,
            wind=wind,
            ground=ground,
            target_index=self._target_index,
            subtask=self._subtask,
            active_faults=tuple(active.event.label for active in self._active),
        )
        self._check_separation()
        self._check_impact()
        return self.state

    def _drain(self, travelled: float, t: float) -> float:
        config = self.config
        used = (
            travelled / config.meters_per_battery_pct
            + config.dt / config.seconds_per_battery_pct
            + self._battery_spike() * config.dt
        )
        battery = max(0.0, self.state.battery_pct - used)
        if battery <= 0.0 and not self._out_of_battery:
            self._out_of_battery = True
            self._record(
                "BATTERY_EMPTY",
                f"battery reached 0% at t={t:.1f}s -- abandoning the plan and landing",
            )
        return battery

    # -- faults -------------------------------------------------------------

    def _apply_faults(self, t: float) -> None:
        for active in list(self._active):
            if not active.alive_at(t):
                self._active.remove(active)
                self._record(
                    "FAULT_END",
                    f"{active.event.label} ended",
                    {"fault": active.event.type.value, "index": active.index},
                )

        home = self.config.home
        due = self.faults.due(
            fired=self._fired,
            t=t,
            battery_pct=self.state.battery_pct,
            waypoint_index=self._target_index,
            distance_from_home=math.dist(self.state.position, home),
            altitude=self.state.z,
        )
        for index, event in due:
            self._fired.add(index)
            self._record(
                "FAULT",
                f"{event.describe()}",
                {"fault": event.type.value, "index": index, "magnitude": event.magnitude},
            )
            if event.type is FaultType.OBSTACLE_APPEARS and event.obstacle:
                obstacle = self._place(event)
                self.obstacles.append(LiveObstacle(obstacle=obstacle, appeared_at=t))
                self.events[-1].payload["obstacle"] = obstacle.model_dump()
                continue
            self._active.append(
                _Active(
                    index=index,
                    event=event,
                    started=t,
                    expires=None if event.duration is None else t + event.duration,
                )
            )

    def _place(self, event: FaultEvent) -> Obstacle:
        """Where an appearing obstacle goes: its own x/y, or `ahead` metres up the route."""
        obstacle = event.obstacle
        if not event.ahead:
            return obstacle
        route = [(self.state.x, self.state.y)] + [
            (wp.x, wp.y) for _, wp in self._flat_plan()[self._target_index:]
        ]
        left = event.ahead
        x, y = route[0]
        for (ax, ay), (bx, by) in zip(route, route[1:]):
            length = math.hypot(bx - ax, by - ay)
            if length >= left:
                x, y = ax + (bx - ax) * left / length, ay + (by - ay) * left / length
                break
            left -= length
            x, y = bx, by
        return obstacle.model_copy(update={"x": round(x, 2), "y": round(y, 2)})

    def _gust(self) -> Vector:
        gust = (0.0, 0.0, 0.0)
        for active in self._active:
            if active.event.type is FaultType.WIND_GUST:
                gust = add(gust, active.event.vector)
        return gust

    def _drift_step(self) -> Vector:
        drift = (0.0, 0.0, 0.0)
        for active in self._active:
            if active.event.type is FaultType.GPS_DRIFT:
                vector = active.event.vector
                drift = add(drift, tuple(c * self.config.dt for c in vector))  # type: ignore[arg-type]
        return drift

    def _battery_spike(self) -> float:
        return sum(
            active.event.magnitude
            for active in self._active
            if active.event.type is FaultType.BATTERY_DRAIN_SPIKE
        )

    # -- monitoring ---------------------------------------------------------

    def _check_separation(self) -> None:
        """Note the first time the aircraft busts an obstacle's keep-out radius."""
        separation = self.constraints.min_separation if self.constraints else 0.0
        for index, live in enumerate(self.obstacles):
            if index in self._breached:
                continue
            obstacle = live.obstacle
            clearance = obstacle.radius + separation
            if clearance <= 0:
                continue
            if obstacle.height is not None and self.state.z > obstacle.height:
                continue
            gap = math.dist((self.state.x, self.state.y), (obstacle.x, obstacle.y))
            if gap < clearance:
                self._breached.add(index)
                self._record(
                    "SEPARATION",
                    f"came within {gap:.1f} m of '{obstacle.name}' "
                    f"(keep-out {clearance:.1f} m)",
                    {"obstacle": obstacle.name, "gap": round(gap, 2)},
                )

    def _check_impact(self) -> None:
        """Did we just fly into the ground or into a building?

        This is the point of having a world: a plan the verifier passed can
        still end badly once wind pushes the aircraft sideways or GPS drift
        sends it somewhere it did not mean to go.
        """
        if self._crashed or self.world is None or not self.config.collisions:
            return

        state = self.state
        building = self.world.collides(state.x, state.y, state.z)
        if building is not None:
            self._crashed = True
            self._record(
                "COLLISION",
                f"struck {building.label()} at ({state.x:.1f}, {state.y:.1f}, {state.z:.1f}) "
                f"doing {state.velocity:.1f} m/s -- its roof is at {building.roof:.0f} m",
                {"building": building.name, "speed": round(state.velocity, 2)},
            )
        elif (
            not self._landing
            and state.z <= state.ground + 1e-6
            and state.timestamp > self.config.dt * 2
        ):
            self._crashed = True
            self._record(
                "COLLISION",
                f"flew into the ground at ({state.x:.1f}, {state.y:.1f}), elevation "
                f"{state.ground:.1f} m, doing {state.velocity:.1f} m/s",
                {"speed": round(state.velocity, 2)},
            )
        if self._crashed:
            self.state = replace(state, vx=0.0, vy=0.0, vz=0.0)

    def _record(self, type: str, description: str, payload: dict | None = None) -> None:
        self.events.append(
            RunEvent(t=self.state.timestamp, type=type, description=description, payload=payload or {})
        )


def execute(
    plan: Plan,
    start: UAVState | None = None,
    *,
    constraints: MissionConstraints | None = None,
    config: SimConfig | None = None,
    wind: Wind | None = None,
    faults: FaultSchedule | None = None,
    world: World | None = None,
) -> Iterator[UAVState]:
    """Fly `plan` and yield a `UAVState` per tick.

    Convenience wrapper around `Executor` for callers that only want telemetry.
    Use `Executor` directly when you also want its events and obstacle list.
    """
    executor = Executor(
        plan,
        constraints=constraints,
        config=config,
        wind=wind,
        faults=faults,
        start=start,
        world=world,
    )
    yield from executor.run()
