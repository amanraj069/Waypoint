import pytest

from planner import Action, Plan, Waypoint, plan


def test_plan_returns_a_plan_with_three_waypoints():
    result = plan("survey the north field")
    assert isinstance(result, Plan)
    assert len(result.waypoints()) == 3


def test_plan_is_hierarchical_and_records_the_mission_text():
    result = plan("survey the north field")
    assert [s.name for s in result.subtasks] == [
        "transit_to_survey_area",
        "survey",
        "return_and_land",
    ]
    assert result.mission_text == "survey the north field"


def test_plan_is_hardcoded_regardless_of_input():
    assert plan("map the ridge").waypoints() == plan("inspect the dam").waypoints()


def test_plan_ends_with_a_landing():
    assert plan("anything").waypoints()[-1].action is Action.LAND


def test_waypoint_rejects_non_positive_speed():
    with pytest.raises(ValueError):
        Waypoint(x=0, y=0, z=0, speed=0, action=Action.MOVE)


def test_waypoint_defaults_to_move():
    assert Waypoint(x=1, y=2, z=3, speed=4).action is Action.MOVE
