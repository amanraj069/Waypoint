"""What a batch is: missions, the conditions they are flown under, and their faults.

A batch file is JSON:

    {"description": "...",
     "missions": [{"id": "lodhi_bara_gumbad",
                   "mission": "photograph Bara Gumbad from two sides, then come home",
                   "world": "lodhi",                 # optional; open space if absent
                   "start": "the Lodhi Road gate",   # optional; a landmark or x,y
                   "land": "the east lawn",          # optional; default back at the start
                   "faults": "random",               # none | crane_ahead | storm | ... | random | a path
                   "wind": "3,0,0",                  # optional, m/s
                   "expect": {...}}]}                # optional: the ground truth the flight is scored on

`expect` is hidden from every planner: it is the mission's intent written by
hand, in the same shape the intent extractor returns (targets with landmark
names, views, sides, heights; ordered; avoid), plus `feasible` (false when the
mission cannot be flown as asked, null when that is not known). The metrics
ground it in the world and score each flight against it, so intent success
never depends on the system's own reading of the mission.

`random` draws a schedule from a seed made of the batch seed, the mission id
and the repeat number, so every condition (and the unconstrained baseline)
flies the same faults, and rerunning a batch with the same seed reproduces
them exactly.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from waypoint.orchestrator.options import CLASSIC, FULL, PRESETS, SystemOptions, system_from
from waypoint.planner import MissionConstraints, Obstacle, Plan, PlannerOptions
from waypoint.scenario import constraints_for, load_faults, load_world, pad_at, parse_wind, resolve_point
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, Trigger, Wind
from waypoint.world.landing import survey
from waypoint.world.models import World

RANDOM = "random"


class ExpectedTarget(BaseModel):
    """A target as the ground truth names it: a landmark, or a point relative to the world's origin."""

    landmark: str = ""
    label: str = ""
    x: float = 0.0
    y: float = 0.0
    action: str = "CAPTURE"
    views: int = 1
    sides: list[str] = Field(default_factory=list)
    overhead: bool = False
    altitude_agl: float = 0.0
    max_altitude_agl: float = 0.0


class ExpectedAvoid(BaseModel):
    landmark: str = ""
    label: str = ""
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0


class Expectation(BaseModel):
    """What a mission asks for, written by hand: the ground truth a flight is scored against."""

    targets: list[ExpectedTarget] = Field(default_factory=list)
    ordered: bool = False
    avoid: list[ExpectedAvoid] = Field(default_factory=list)
    feasible: bool | None = True

    def payload(self) -> dict:
        """In the extractor's shape, for `waypoint.intent.ground`."""
        return {
            "targets": [{**t.model_dump(), "label": t.label or t.landmark} for t in self.targets],
            "ordered": self.ordered,
            "avoid": [{**a.model_dump(), "label": a.label or a.landmark} for a in self.avoid],
            "notes": "",
        }


class MissionSpec(BaseModel):
    """One mission of a batch, as written in the batch file."""

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    mission: str = Field(min_length=1)
    world: str | None = None
    start: str | None = None
    land: str | None = None
    faults: str = "none"
    wind: str | None = None
    max_altitude: float | None = Field(default=None, gt=0)
    expect: Expectation | None = None

    @field_validator("faults")
    @classmethod
    def _not_sample(cls, value: str) -> str:
        if value == "sample":
            raise ValueError(
                "'sample' aims its faults at a known plan, and a batch plans from scratch; "
                "use 'random' or a named schedule"
            )
        return value


class Batch(BaseModel):
    description: str = ""
    missions: list[MissionSpec] = Field(min_length=1)

    @field_validator("missions")
    @classmethod
    def _unique_ids(cls, missions: list[MissionSpec]) -> list[MissionSpec]:
        seen: set[str] = set()
        for spec in missions:
            if spec.id in seen:
                raise ValueError(f"mission id {spec.id!r} appears twice")
            seen.add(spec.id)
        return missions

    @classmethod
    def load(cls, path: Path) -> "Batch":
        return cls.model_validate(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

#: Every ablation, in a fixed order: the planner as designed comes first, and
#: plots colour each condition by its place here, never by its rank in a run.
CONDITIONS: tuple[PlannerOptions, ...] = (
    PlannerOptions(hierarchical=True, violation_feedback=True),
    PlannerOptions(hierarchical=False, violation_feedback=True),
    PlannerOptions(hierarchical=True, violation_feedback=False),
    PlannerOptions(hierarchical=False, violation_feedback=False),
)
CONDITION_LABELS = tuple(c.label for c in CONDITIONS)
#: The systems in their fixed order, for the same reason.
SYSTEM_LABELS = tuple(PRESETS)


@dataclass(frozen=True)
class Condition:
    """One way of flying the batch: how the planner is asked, and what the loop around it does."""

    planner: PlannerOptions
    system: SystemOptions = FULL
    #: Batches written before systems existed name a condition by its planner alone (and fly classic).
    legacy: bool = False

    @property
    def label(self) -> str:
        return self.planner.label if self.legacy else f"{self.planner.label}|{self.system.label}"

    @classmethod
    def of(cls, value: "Condition | PlannerOptions") -> "Condition":
        """A condition, from a condition or from bare planner options (the classic loop, labelled as before)."""
        return value if isinstance(value, Condition) else cls(value, CLASSIC, legacy=True)


def condition_from_label(label: str) -> PlannerOptions:
    """The planner options of a label ("hierarchical+feedback", with or without "|system")."""
    planner = label.split("|", 1)[0]
    for condition in CONDITIONS:
        if condition.label == planner:
            return condition
    raise ValueError(f"unknown condition {label!r}; one of {', '.join(CONDITION_LABELS)}")


def split_label(label: str) -> tuple[str, str]:
    """(planner part, system part) of a condition label; labels from before systems are classic."""
    planner, _, system = label.partition("|")
    return planner, system or CLASSIC.label


def system_of(label: str) -> SystemOptions | None:
    """The system a condition label names, if it names a known one."""
    try:
        return system_from(split_label(label)[1])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# One mission, ready to fly
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """Everything `run_mission` needs for one mission, built from its spec."""

    spec: MissionSpec
    world: World | None
    constraints: MissionConstraints
    home: tuple[float, float, float]
    landing: tuple[float, float, float]
    faults: FaultSchedule
    wind: Wind
    fault_seed: int | None = None


def mission_seed(batch_seed: int, mission_id: str, repeat: int) -> int:
    """A stable seed per mission and repeat (Python's hash() changes per process)."""
    digest = hashlib.sha256(f"{batch_seed}:{mission_id}:{repeat}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def prepare(spec: MissionSpec, *, seed: int = 0, repeat: int = 0) -> Scenario:
    """Resolve the world, envelope, pads, wind and faults for one mission.

    Raises ValueError, saying why, when a pad cannot be placed.
    """
    world = load_world(spec.world)
    constraints, _ = constraints_for(world, spec.max_altitude)
    start = _pad(world, spec.start, "take off") if spec.start else survey(world, 0.0, 0.0)
    landing = _pad(world, spec.land, "land") if spec.land else start

    fault_seed = None
    if spec.faults == RANDOM:
        fault_seed = mission_seed(seed, spec.id, repeat)
        faults = random_schedule(random.Random(fault_seed))
    else:
        faults = load_faults(spec.faults, Plan())

    return Scenario(
        spec=spec,
        world=world,
        constraints=constraints,
        home=start.position,
        landing=landing.position,
        faults=faults,
        wind=parse_wind(spec.wind),
        fault_seed=fault_seed,
    )


def _pad(world: World | None, where: str, role: str):
    x, y, _ = resolve_point(world, where)
    return pad_at(world, x, y, role)


# ---------------------------------------------------------------------------
# Randomized faults
# ---------------------------------------------------------------------------


def random_schedule(rng: random.Random, *, most: int = 3) -> FaultSchedule:
    """One to `most` faults of different kinds, with plausible sizes and timings.

    Obstacles are placed `ahead` on whatever route is flown, so they land on
    the route rather than in empty space; the rest fire on the clock, inside
    the first 40 s, where every mission in a batch is still flying.
    """
    kinds = rng.sample(list(FaultType), rng.randint(1, min(most, len(FaultType))))
    events = [_random_fault(kind, rng) for kind in kinds]
    return FaultSchedule(events=events)


def _random_fault(kind: FaultType, rng: random.Random) -> FaultEvent:
    r = lambda lo, hi: round(rng.uniform(lo, hi), 1)  # noqa: E731
    if kind is FaultType.OBSTACLE_APPEARS:
        return FaultEvent(
            type=kind,
            name="obstacle spotted",
            trigger=Trigger(at_waypoint=rng.randint(1, 3)),
            ahead=r(25, 50),
            obstacle=Obstacle(x=0.0, y=0.0, radius=r(5, 12), height=r(30, 70), name="unmapped obstacle"),
        )
    if kind is FaultType.WIND_GUST:
        return FaultEvent(
            type=kind, name="gust", trigger=Trigger(time=r(8, 40)),
            magnitude=r(4, 10), direction=r(0, 360), duration=r(3, 8),
        )
    if kind is FaultType.BATTERY_DRAIN_SPIKE:
        return FaultEvent(
            type=kind, name="cell fault", trigger=Trigger(time=r(8, 40)),
            magnitude=r(0.2, 1.0), duration=r(5, 20),
        )
    return FaultEvent(
        type=FaultType.GPS_DRIFT, name="gps slip", trigger=Trigger(time=r(5, 40)),
        magnitude=r(0.3, 1.2), direction=r(0, 360), duration=r(10, 30),
    )
