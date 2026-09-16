"""Plan data model produced by the planner and consumed by verifier + sim."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Action(str, Enum):
    MOVE = "MOVE"
    HOVER = "HOVER"
    CAPTURE = "CAPTURE"
    LAND = "LAND"


class Waypoint(BaseModel):
    x: float
    y: float
    z: float
    speed: float = Field(gt=0, description="Commanded ground speed in m/s.")
    action: Action = Action.MOVE


class SubTask(BaseModel):
    name: str
    waypoints: list[Waypoint] = Field(min_length=1)


class Plan(BaseModel):
    """A hierarchical plan: a mission decomposed into ordered subtasks."""

    mission_text: str = ""
    subtasks: list[SubTask] = Field(default_factory=list)

    def waypoints(self) -> list[Waypoint]:
        """Flatten the hierarchy into the executable waypoint sequence."""
        return [wp for subtask in self.subtasks for wp in subtask.waypoints]


class GeoPoint(BaseModel):
    """A vertex of the geofence polygon, in local ENU metres."""

    x: float
    y: float


class MissionConstraints(BaseModel):
    """The safety envelope handed to the planner and enforced by the verifier."""

    geofence: list[GeoPoint] = Field(
        min_length=3, description="Polygon the UAV must stay inside, in order."
    )
    max_altitude: float = Field(gt=0)
    min_altitude: float = Field(ge=0)
    max_velocity: float = Field(gt=0)
    battery_budget_pct: float = Field(gt=0, le=100)

    @model_validator(mode="after")
    def _check_envelope(self) -> "MissionConstraints":
        if self.min_altitude > self.max_altitude:
            raise ValueError("min_altitude must not exceed max_altitude")
        return self

    def describe(self) -> str:
        """Render the envelope as prompt-ready text."""
        fence = ", ".join(f"({p.x:g}, {p.y:g})" for p in self.geofence)
        return (
            f"- Geofence polygon (must stay strictly inside): {fence}\n"
            f"- Altitude AGL: at least {self.min_altitude:g} m, at most {self.max_altitude:g} m\n"
            f"- Speed: at most {self.max_velocity:g} m/s on every waypoint\n"
            f"- Battery budget for the whole mission: {self.battery_budget_pct:g}%"
        )


DEFAULT_CONSTRAINTS = MissionConstraints(
    geofence=[
        GeoPoint(x=-100.0, y=-100.0),
        GeoPoint(x=200.0, y=-100.0),
        GeoPoint(x=200.0, y=200.0),
        GeoPoint(x=-100.0, y=200.0),
    ],
    max_altitude=120.0,
    min_altitude=0.0,
    max_velocity=12.0,
    battery_budget_pct=80.0,
)
