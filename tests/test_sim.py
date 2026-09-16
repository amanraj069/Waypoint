import math
from collections.abc import Iterator

from planner import Action, Plan, SubTask, Waypoint, plan
from sim import MissionState, execute, initial_state


def test_execute_returns_a_generator():
    assert isinstance(execute(plan("x")), Iterator)


def test_execute_yields_mission_states():
    states = list(execute(plan("survey the north field")))
    assert states
    assert all(isinstance(s, MissionState) for s in states)


def test_execution_ends_landed_and_stationary():
    final = list(execute(plan("survey the north field")))[-1]
    assert final.z == 0.0
    assert final.velocity == 0.0


def test_timestamps_increase_monotonically():
    states = list(execute(plan("x")))
    assert all(b.timestamp > a.timestamp for a, b in zip(states, states[1:]))


def test_battery_drains_monotonically_and_stays_in_range():
    states = list(execute(plan("x")))
    assert all(b.battery_pct <= a.battery_pct for a, b in zip(states, states[1:]))
    assert all(0.0 <= s.battery_pct <= 100.0 for s in states)


def test_linear_interpolation_hits_each_waypoint():
    single_leg = Plan(
        subtasks=[
            SubTask(
                name="leg",
                waypoints=[Waypoint(x=10.0, y=0.0, z=0.0, speed=5.0, action=Action.MOVE)],
            )
        ]
    )
    states = list(execute(single_leg, start=initial_state()))
    # 10 m at 5 m/s over 1 s ticks -> 2 motion ticks, then a touchdown tick.
    assert [round(s.x, 6) for s in states[:2]] == [5.0, 10.0]
    assert states[-1].position == (10.0, 0.0, 0.0)


def test_heading_points_at_the_target():
    north = Plan(
        subtasks=[
            SubTask(
                name="leg",
                waypoints=[Waypoint(x=0.0, y=10.0, z=0.0, speed=10.0, action=Action.MOVE)],
            )
        ]
    )
    assert math.isclose(next(execute(north)).heading, 90.0)


def test_hover_holds_position():
    hover = Plan(
        subtasks=[
            SubTask(
                name="hold",
                waypoints=[Waypoint(x=0.0, y=0.0, z=0.0, speed=1.0, action=Action.HOVER)],
            )
        ]
    )
    assert all(s.position == (0.0, 0.0, 0.0) for s in execute(hover))


def test_initial_state_is_on_the_pad():
    state = initial_state()
    assert state.position == (0.0, 0.0, 0.0)
    assert state.battery_pct == 100.0
    assert state.timestamp == 0.0
