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
from waypoint.sim.shield import DisturbanceObserver, GpsDriftObserver, SafetyShield
from waypoint.sim.state import UAVState
from waypoint.world.models import World

LANDING_SPEED = 1.5
#: How fast position hold may steer back to its point when the wind pushes it off.
HOLD_SPEED = 2.0


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


#: How much room an appearing obstacle leaves beyond its keep-out, so a sensor
#: that sees it has room to turn: it never appears with the aircraft inside it.
SEEN_MARGIN = 5.0


def place(
    event: FaultEvent,
    here: tuple[float, float],
    route: list[tuple[float, float]],
    separation: float = 0.0,
) -> Obstacle:
    """Where an appearing obstacle goes: its own x/y, or `ahead` metres up the route still to fly.

    `route` is the x/y of the waypoints not yet reached, starting with the one
    the aircraft is flying toward. Shared by every backend, so a schedule puts
    its obstacle in the same place whichever one flies it.

    Measured along a route that doubles back (a survey's parallel passes), the
    point `ahead` metres on can lie right beside the aircraft. A sensor would
    have seen that obstacle long before it was that close, and no plan could
    start outside its keep-out, so the obstacle moves further along the route
    until the aircraft is at least its radius, the `separation` and
    `SEEN_MARGIN` away from it.
    """
    obstacle = event.obstacle
    if not event.ahead:
        return obstacle
    points = [here, *route]
    clear = obstacle.radius + separation + SEEN_MARGIN
    distance = event.ahead
    x, y = best = _along(points, distance)
    total = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    while math.dist((x, y), here) < clear and distance < total:
        distance += 1.0
        x, y = _along(points, distance)
        if math.dist((x, y), here) > math.dist(best, here):
            best = (x, y)
    if math.dist((x, y), here) < clear:
        x, y = best  # nowhere on the route is clear: the farthest point there is
    return obstacle.model_copy(update={"x": round(x, 2), "y": round(y, 2)})


def _along(points: list[tuple[float, float]], distance: float) -> tuple[float, float]:
    """The point `distance` metres along a polyline (its end, past the end)."""
    left = distance
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        length = math.hypot(bx - ax, by - ay)
        if length >= left:
            return ax + (bx - ax) * left / length, ay + (by - ay) * left / length
        left -= length
    return points[-1]


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
        shield: bool = False,
    ):
        self.plan = plan
        self.constraints = constraints
        self.config = config or DEFAULT_CONFIG
        self.wind = wind or NO_WIND
        self.faults = faults or NO_FAULTS
        self.world = world
        #: The runtime safety filter under the controller, when it is on (`sim.shield`).
        self.shield = (
            SafetyShield(constraints, world, lambda: [live.obstacle for live in self.obstacles]) if shield else None
        )
        self.shield_ticks = 0
        self._shielding: tuple[str, ...] = ()
        self._observer = DisturbanceObserver()
        self._gps = GpsDriftObserver()
        self._gps_flagged = False

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

    def hold(self, seconds: float) -> Iterator[UAVState]:
        """Hover where it is for `seconds`, as it does while the planner is thinking.

        Position hold, not a frozen clock: the wind still blows, faults still
        fire and the battery still drains, so a slow replan costs what it
        would cost a real aircraft. Nothing happens on the ground.
        """
        if self.state.z <= self.state.ground + 0.5 or seconds <= 0:
            return
        target = self.state.reported_position
        ticks = round(seconds / self.config.dt)
        if ticks <= 0:
            return
        self._record("HOLD", f"hovering {seconds:.1f}s while the planner replans", {"seconds": round(seconds, 2)})
        for _ in range(ticks):
            yield self._tick(target, HOLD_SPEED)
            if self._out_of_battery or self._crashed:
                return

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
                "action": waypoint.action.value,
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
        started = self.state.timestamp
        ground = self.world.elevation_at(self.state.x, self.state.y) if self.world else 0.0
        # Straight down from where it believes it is. The controller steers on
        # the reported position, so aiming at the true one under a GPS offset
        # would slide the aircraft sideways by the whole offset on the way down.
        rx, ry, _ = self.state.reported_position
        target = (rx, ry, ground + self._gps_offset[2])
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
        if self.shield is not None:
            desired = self._shielded(desired)
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

    def _shielded(self, desired: Vector) -> Vector:
        """The controller's velocity through the safety filter; a SHIELD event when what it avoids changes."""
        disturbance = self._observer.update(self.state.velocity_vector)
        where = self._gps.update(self.state.reported_position, self.state.velocity_vector, self.config.dt)
        if self._gps.error > 3.0 and not self._gps_flagged:
            self._gps_flagged = True
            self._record(
                "GPS_INTEGRITY",
                f"GPS disagrees with the airframe's own motion: the fix has drifted about {self._gps.error:.1f} m; "
                "the safety filter guards the corrected position",
                {"error": round(self._gps.error, 2)},
            )
        filtered, hazards = self.shield.filter(
            where, desired, velocity=self.state.velocity_vector,
            disturbance=disturbance, landing=self._landing,
        )
        self._observer.commanded(filtered)
        hazards = tuple(sorted(set(hazards)))
        if hazards:
            self.shield_ticks += 1
            if hazards != self._shielding:
                change = distance(filtered, desired)
                self._record(
                    "SHIELD",
                    f"safety filter steering clear of {', '.join(hazards)} (changed the command by {change:.1f} m/s)",
                    {"hazards": list(hazards), "change": round(change, 2)},
                )
        self._shielding = hazards
        return filtered

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
        route = [(wp.x, wp.y) for _, wp in self._flat_plan()[self._target_index:]]
        separation = self.constraints.min_separation if self.constraints else 0.0
        return place(event, (self.state.x, self.state.y), route, separation)

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
