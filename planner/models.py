"""Plan data model produced by the planner and consumed by verifier + sim."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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
