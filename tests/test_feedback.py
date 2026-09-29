"""The in-flight monitor: when it asks for a replan, and what it tells the planner."""

from __future__ import annotations

import pytest

from waypoint.feedback import (
    FeedbackConfig,
    TelemetrySnapshot,
    TriggerKind,
    estimated_battery_need,
    needs_replan,
    summarize_for_llm,
)
from waypoint.planner import Obstacle
from waypoint.planner.models import Action, Waypoint

ROUTE = (
    Waypoint(x=0, y=0, z=25, speed=3),
    Waypoint(x=0, y=80, z=25, speed=4, action=Action.CAPTURE),
    Waypoint(x=40, y=80, z=25, speed=4, action=Action.CAPTURE),
    Waypoint(x=0, y=0, z=0, speed=2, action=Action.LAND),
)


def snapshot(*, known=(), planned=(), battery=90.0, position=(0.0, 10.0, 25.0), target=1, **extra):
    return TelemetrySnapshot(
        t=12.0,
        position=position,
        velocity=(0.0, 4.0, 0.0),
        heading=90.0,
        battery_pct=battery,
        ground=0.0,
        target_index=target,
        subtask="survey",
        plan_waypoints=ROUTE,
        planned_obstacles=tuple(planned),
        known_obstacles=tuple(known),
        **extra,
    )


def test_a_quiet_flight_needs_nothing():
    assert needs_replan(snapshot()) is None


def test_an_obstacle_that_appears_on_the_route_triggers_a_replan():
    crane = Obstacle(x=2, y=40, radius=6, height=60, name="crane")
    trigger = needs_replan(snapshot(known=[crane]))
    assert trigger.kind is TriggerKind.OBSTACLE_ON_ROUTE
    assert "crane" in trigger.reason and trigger.t == 12.0
    assert trigger.detail["gap"] == pytest.approx(2.0)


def test_an_obstacle_off_the_route_is_left_alone():
    assert needs_replan(snapshot(known=[Obstacle(x=90, y=20, radius=6, name="silo")])) is None


def test_an_obstacle_the_route_flies_over_is_left_alone():
    assert needs_replan(snapshot(known=[Obstacle(x=0, y=40, radius=6, height=20, name="shed")])) is None


def test_an_obstacle_the_plan_already_allowed_for_is_left_alone():
    crane = Obstacle(x=2, y=40, radius=6, name="crane")
    assert needs_replan(snapshot(known=[crane], planned=[crane])) is None


def test_separation_widens_the_keep_out():
    near = Obstacle(x=9, y=40, radius=6, name="mast")
    assert needs_replan(snapshot(known=[near])) is None
    assert needs_replan(snapshot(known=[near], min_separation=5.0)).kind is TriggerKind.OBSTACLE_ON_ROUTE


def test_only_the_route_still_ahead_counts():
    """Past the first leg, something on it is behind the aircraft."""
    behind = Obstacle(x=0, y=40, radius=4, name="behind")
    assert needs_replan(snapshot(known=[behind], position=(20.0, 80.0, 25.0), target=2)) is None


def test_too_little_battery_for_the_rest_triggers_a_replan():
    trigger = needs_replan(snapshot(battery=12.0))
    assert trigger.kind is TriggerKind.BATTERY_INSUFFICIENT
    assert trigger.detail["estimated_need_pct"] == pytest.approx(estimated_battery_need(snapshot()), abs=0.01)


def test_a_battery_replan_gets_time_to_work():
    config = FeedbackConfig(battery_cooldown_s=20.0)
    assert needs_replan(snapshot(battery=12.0), config, last_battery_replan=5.0) is None
    assert needs_replan(snapshot(battery=12.0), config, last_battery_replan=-10.0) is not None


def test_the_summary_says_where_it_is_what_is_done_and_why():
    crane = Obstacle(x=2, y=40, radius=6, height=60, name="crane")
    snap = snapshot(known=[crane], active_faults=("gps slip",))
    text = summarize_for_llm(snap, needs_replan(snap))

    assert "Current position: (0.0, 10.0, 25.0)" in text
    assert "Battery: 90.0%" in text
    assert "crane" in text and "keep-out" in text
    assert "already reached" in text and "(0, 0, 25)" in text
    assert "not yet flown" in text and "(40, 80, 25) CAPTURE" in text
    assert "gps slip" in text
