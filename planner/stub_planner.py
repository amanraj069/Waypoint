"""Stub planner.

Stands in for the LLM hierarchical planner. Returns the same hardcoded plan for
any mission text so the orchestration control flow can be exercised end to end.
"""

from __future__ import annotations

from planner.models import Action, Plan, SubTask, Waypoint


def plan(mission_text: str) -> Plan:
    """Return a hardcoded 3-waypoint plan regardless of the mission text."""
    return Plan(
        mission_text=mission_text,
        subtasks=[
            SubTask(
                name="transit_to_survey_area",
                waypoints=[Waypoint(x=50.0, y=0.0, z=30.0, speed=5.0, action=Action.MOVE)],
            ),
            SubTask(
                name="survey",
                waypoints=[Waypoint(x=50.0, y=50.0, z=30.0, speed=3.0, action=Action.CAPTURE)],
            ),
            SubTask(
                name="return_and_land",
                waypoints=[Waypoint(x=0.0, y=0.0, z=0.0, speed=2.0, action=Action.LAND)],
            ),
        ],
    )
