"""The built-in simulator as an `ExecutorBackend`."""

from __future__ import annotations

from collections.abc import Iterator

from waypoint.backends import ExecutorBackend
from waypoint.planner.models import MissionConstraints, Obstacle, Plan
from waypoint.sim.config import SimConfig, Wind
from waypoint.sim.executor import Executor
from waypoint.sim.faults import FaultSchedule
from waypoint.sim.state import UAVState
from waypoint.world.models import World


class CustomSimBackend(ExecutorBackend):
    """Point-mass physics, wind and fault injection, in process, needing nothing installed.

    The scenario (world, wind, airframe) is fixed when it is built; the
    constraints and faults arrive with `prepare`, as they do for every backend.
    """

    name = "custom"

    def __init__(
        self,
        *,
        world: World | None = None,
        wind: Wind | None = None,
        config: SimConfig | None = None,
    ):
        self.world = world
        self.wind = wind
        self.config = config
        self._executor: Executor | None = None
        self._started = False

    def prepare(
        self,
        constraints: MissionConstraints,
        faults: FaultSchedule,
        home: tuple[float, float, float] | None = None,
    ) -> None:
        config = self.config or SimConfig()
        if home is not None:
            config = config.model_copy(update={"home": tuple(float(v) for v in home)})
        # The plan is not known yet; the executor gets each one as it is flown.
        self._executor = Executor(
            Plan(),
            constraints=constraints,
            config=config,
            wind=self.wind,
            faults=faults,
            world=self.world,
            shield=self.shield,
        )
        self._started = False

    def hold(self, seconds: float) -> Iterator[UAVState]:
        return self._require().hold(seconds)

    def run(self, plan: Plan) -> Iterator[UAVState]:
        executor = self._require()
        if not self._started:
            self._started = True
            executor.plan = plan
            return executor.run()
        return executor.resume(plan)

    @property
    def state(self) -> UAVState:
        return self._require().state

    @property
    def obstacles(self) -> list[Obstacle]:
        return [live.obstacle for live in self._require().obstacles]

    def event_records(self) -> list[dict]:
        return self._require().event_records()

    def obstacle_records(self) -> list[dict]:
        return self._require().obstacle_records()

    @property
    def distance_flown(self) -> float:
        return self._require().distance_flown

    @property
    def crashed(self) -> bool:
        return self._require().crashed

    @property
    def battery_depleted(self) -> bool:
        return self._require().out_of_battery

    @property
    def sim_config(self) -> SimConfig:
        return self._require().config

    def describe(self) -> dict:
        """The name, plus the airframe and wind, so the run file can re-fly it exactly."""
        from waypoint.sim.config import DEFAULT_CONFIG, NO_WIND  # noqa: PLC0415

        config = self._executor.config if self._executor else (self.config or DEFAULT_CONFIG)
        return {
            "name": self.name,
            "config": config.model_dump(mode="json"),
            "wind": (self.wind or NO_WIND).model_dump(mode="json"),
        }

    def _require(self) -> Executor:
        if self._executor is None:
            raise RuntimeError("call prepare(constraints, faults) before flying")
        return self._executor
