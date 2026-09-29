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
    """A vertex of a polygon (geofence or no-fly zone), in local ENU metres."""

    x: float
    y: float


class Obstacle(BaseModel):
    """A known hazard the route must stand off from.

    Modelled as a vertical cylinder: a circle of `radius` at (x, y) rising from
    the ground to `height` (None means it reaches any altitude, which is the
    conservative default).
    """

    x: float
    y: float
    radius: float = Field(default=0.0, ge=0, description="Physical radius in metres.")
    height: float | None = Field(
        default=None, ge=0, description="Top of the obstacle in metres AGL; None = unbounded."
    )
    name: str = "obstacle"


class MissionConstraints(BaseModel):
    """The safety envelope handed to the planner and enforced by the verifier."""

    geofence: list[GeoPoint] = Field(
        min_length=3, description="Polygon the UAV must stay inside, in order."
    )
    max_altitude: float = Field(gt=0)
    min_altitude: float = Field(ge=0)
    max_velocity: float = Field(gt=0)
    battery_budget_pct: float = Field(gt=0, le=100)
    no_fly_zones: list[list[GeoPoint]] = Field(
        default_factory=list,
        description="Polygons the route must not enter or cross, each as ordered vertices.",
    )
    obstacles: list[Obstacle] = Field(
        default_factory=list, description="Known hazards the route must stand off from."
    )
    min_separation: float = Field(
        default=0.0,
        ge=0,
        description="Clearance in metres to keep beyond every obstacle's own radius.",
    )

    @model_validator(mode="after")
    def _check_envelope(self) -> "MissionConstraints":
        if self.min_altitude > self.max_altitude:
            raise ValueError("min_altitude must not exceed max_altitude")
        for index, zone in enumerate(self.no_fly_zones):
            if len(zone) < 3:
                raise ValueError(f"no-fly zone {index} needs at least 3 vertices")
        return self

    def describe(self) -> str:
        """Render the envelope as prompt-ready text."""
        lines = [
            f"- Geofence polygon (must stay strictly inside): {render_polygon(self.geofence)}",
            f"- Altitude AGL: at least {self.min_altitude:g} m, at most {self.max_altitude:g} m",
            f"- Speed: at most {self.max_velocity:g} m/s on every waypoint",
            f"- Battery budget for the whole mission: {self.battery_budget_pct:g}%",
        ]
        for index, zone in enumerate(self.no_fly_zones, 1):
            lines.append(
                f"- No-fly zone {index} (never enter it, and no leg may cross it): "
                f"{render_polygon(zone)}"
            )
        for obstacle in self.obstacles:
            top = "any altitude" if obstacle.height is None else f"up to {obstacle.height:g} m"
            lines.append(
                f"- Obstacle '{obstacle.name}' at ({obstacle.x:g}, {obstacle.y:g}), "
                f"radius {obstacle.radius:g} m, {top}: keep at least "
                f"{obstacle.radius + self.min_separation:g} m from its centre"
            )
        if self.min_separation and not self.obstacles:
            lines.append(f"- Minimum separation from any obstacle: {self.min_separation:g} m")
        return "\n".join(lines)


def render_polygon(points: list[GeoPoint]) -> str:
    """Render polygon vertices as prompt-ready text."""
    return ", ".join(f"({p.x:g}, {p.y:g})" for p in points)


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
