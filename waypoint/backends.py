"""What a flight backend promises the orchestrator, and nothing more.

The orchestrator only ever talks to an `ExecutorBackend`, so swapping the
simulator for PX4 SITL + Gazebo is a change of `--backend`, not of the loop:

    backend = make_backend("custom", world=world, wind=wind)
    backend.prepare(constraints, faults)          # on the pad, faults armed
    for state in backend.run(plan):               # fly; stop iterating to pause
        ...
    for state in backend.run(new_plan):           # resume from where it is
        ...

One backend is one flight. `run` always continues from the aircraft's current
state, so pausing is simply to stop iterating, and resuming is `run` again with
the new plan. `close()` releases whatever the backend started (PX4 and Gazebo
for `px4`); whoever made the backend closes it. Every backend shares the frame (local ENU metres from the world
origin), the obstacle model (`planner.models.Obstacle`) and the telemetry
(`sim.state.UAVState`), so their runs are comparable and write the same file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from waypoint.planner.models import MissionConstraints, Obstacle, Plan
from waypoint.sim.faults import FaultSchedule
from waypoint.sim.state import UAVState


class BackendError(RuntimeError):
    pass


class ExecutorBackend(ABC):
    """A flight in progress: the aircraft, and whatever is flying it."""

    #: Written into the mission log and the run file, so a run says what flew it.
    name: str = "backend"

    @abstractmethod
    def prepare(
        self,
        constraints: MissionConstraints,
        faults: FaultSchedule,
        home: tuple[float, float, float] | None = None,
    ) -> None:
        """Put the aircraft on the pad at `home` (the origin if None), faults armed, ready to fly."""

    @abstractmethod
    def run(self, plan: Plan) -> Iterator[UAVState]:
        """Fly `plan` from the aircraft's current state, one state per tick.

        The first call starts on the pad. Stopping iteration pauses the flight
        where it is; calling `run` again flies a new plan from there.
        """

    @property
    @abstractmethod
    def state(self) -> UAVState:
        """The latest tick."""

    @property
    @abstractmethod
    def obstacles(self) -> list[Obstacle]:
        """Every obstacle the aircraft knows of now: the planned ones and any that appeared."""

    @abstractmethod
    def event_records(self) -> list[dict]:
        """The flight's timeline so far: faults, waypoints reached, touchdown."""

    @abstractmethod
    def obstacle_records(self) -> list[dict]:
        """Obstacles as written to the run file, with when each appeared."""

    @property
    @abstractmethod
    def distance_flown(self) -> float:
        """Metres actually travelled, wind and drift included."""

    @property
    @abstractmethod
    def crashed(self) -> bool:
        """Whether the aircraft hit something."""

    @property
    @abstractmethod
    def battery_depleted(self) -> bool:
        """Whether the battery ran flat, which ends the plan wherever it is."""

    def describe(self) -> dict:
        """What to record about this backend, beyond its name."""
        return {"name": self.name}

    def close(self) -> None:
        """Stop anything the flight started. Nothing to do for an in-process backend."""


BACKENDS = ("custom", "px4")


def make_backend(name: str, **scenario) -> ExecutorBackend:
    """A backend by the name the CLI uses.

    `custom` is the built-in simulator. `px4` flies PX4 SITL in Gazebo (see
    `waypoint.px4`); it raises BackendError here, before anything flies, when
    PX4, Gazebo or pymavlink is missing.
    """
    if name == "custom":
        from waypoint.sim.backend import CustomSimBackend  # noqa: PLC0415

        return CustomSimBackend(**scenario)
    if name == "px4":
        from waypoint.px4.link import mavutil  # noqa: PLC0415

        if mavutil is None:
            raise BackendError('the px4 backend needs pymavlink: pip install -e ".[px4]"')
        from waypoint.px4.backend import Px4Backend  # noqa: PLC0415

        return Px4Backend(**scenario)
    raise BackendError(f"unknown backend {name!r}; choose one of {', '.join(BACKENDS)}")
