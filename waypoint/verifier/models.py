"""Types returned and consumed by the deterministic safety verifier.

Nothing here is learned or sampled: a `Violation` is a statement of fact about
a plan, and its `description` is written to be pasted straight back to the
planner as the reason the plan was rejected.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ViolationType(str, Enum):
    """Every way a plan can fail the safety envelope."""

    GEOFENCE = "GEOFENCE"
    NO_FLY_ZONE = "NO_FLY_ZONE"
    ALTITUDE = "ALTITUDE"
    MAX_VELOCITY = "MAX_VELOCITY"
    TURN_FEASIBILITY = "TURN_FEASIBILITY"
    BATTERY = "BATTERY"
    MIN_SEPARATION = "MIN_SEPARATION"
    BUILDING = "BUILDING"
    TERRAIN = "TERRAIN"
    LANDING = "LANDING"


class Severity(str, Enum):
    """How badly a violation should be taken.

    WARNING is advisory: the plan is flyable but marginal, so it does not fail
    verification. ERROR and CRITICAL are blocking -- ERROR breaks the operating
    envelope, CRITICAL risks hitting something.
    """

    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


BLOCKING_SEVERITIES = frozenset({Severity.ERROR, Severity.CRITICAL})


class Violation(BaseModel):
    """One concrete, human-readable reason a plan is unsafe."""

    type: ViolationType
    waypoint_index: int | None = Field(
        default=None,
        description=(
            "0-based index into the flattened waypoint sequence, or None for a "
            "violation about the plan as a whole (e.g. battery)."
        ),
    )
    description: str = Field(
        description="Feedback for the planner: what is wrong, by how much, and how to fix it."
    )
    severity: Severity = Severity.ERROR

    @property
    def blocking(self) -> bool:
        return self.severity in BLOCKING_SEVERITIES

    def __str__(self) -> str:
        return self.description


class VerificationResult(BaseModel):
    """The verdict on one plan.

    `passed` is False when any blocking violation was found. Warnings are still
    reported -- they are worth telling the planner about -- but they do not on
    their own reject a plan.
    """

    passed: bool
    violations: list[Violation] = Field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.blocking]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if not v.blocking]

    def messages(self) -> list[str]:
        """The violation descriptions, ready for `planner.replan()`."""
        return [v.description for v in self.violations]

    def of_type(self, violation_type: ViolationType) -> list[Violation]:
        return [v for v in self.violations if v.type is violation_type]


class VerifierConfig(BaseModel):
    """Aircraft model the verifier reasons with.

    These describe the vehicle, not the mission, which is why they live here and
    not on `MissionConstraints`. Every value is a plain constant -- no fitting,
    no estimation from data.
    """

    # Energy: a linear model, metres flown per 1% of battery plus a hover cost.
    meters_per_battery_pct: float = Field(
        default=40.0, gt=0, description="Metres of level flight per 1% of battery."
    )
    hover_seconds_per_battery_pct: float = Field(
        default=30.0, gt=0, description="Seconds of hovering per 1% of battery."
    )
    hover_seconds_per_waypoint: float = Field(
        default=2.0, ge=0, description="Dwell time charged to each HOVER waypoint."
    )
    reserve_battery_pct: float = Field(
        default=0.0, ge=0, description="Battery that must still be unspent at touchdown."
    )

    # Turn feasibility: the aircraft can turn `max_turn_angle_deg` between legs
    # at or below `turn_reference_speed`, and proportionally less above it.
    max_turn_angle_deg: float = Field(default=150.0, gt=0, le=180)
    turn_reference_speed: float = Field(default=5.0, gt=0)

    # Optional soft margin inside the geofence; 0 disables the warning.
    geofence_margin: float = Field(default=0.0, ge=0)

    # Flying over a world with relief and buildings in it.
    min_ground_clearance: float = Field(
        default=5.0, ge=0, description="Metres the route must stay above the ground."
    )
    roof_clearance: float = Field(
        default=3.0, ge=0, description="Metres to clear a roof by when crossing above one."
    )
    terrain_sample_step: float = Field(
        default=5.0, gt=0, description="Spacing when walking a leg to check ground clearance."
    )
    landing_tolerance: float = Field(
        default=1.5,
        gt=0,
        description="How far above the ground a LAND waypoint may sit before it is wrong.",
    )

    # Where the aircraft starts and ends, used for the first and last legs.
    home: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # What to call that point in a violation. A plan made in the air starts
    # from wherever the aircraft is, and calling that "home" would mislead
    # the planner the violation is read back to.
    start_label: str = "home"

    def max_turn_at(self, speed: float) -> float:
        """Largest flyable heading change, in degrees, at the given speed."""
        if speed <= self.turn_reference_speed:
            return self.max_turn_angle_deg
        return self.max_turn_angle_deg * self.turn_reference_speed / speed

    def speed_for_turn(self, angle_deg: float) -> float | None:
        """Fastest speed at which `angle_deg` is flyable, or None if it never is."""
        if angle_deg > self.max_turn_angle_deg:
            return None
        return self.max_turn_angle_deg * self.turn_reference_speed / angle_deg


DEFAULT_CONFIG = VerifierConfig()
