"""The in-flight monitor: when it asks for a replan, and what it tells the planner."""

from __future__ import annotations

import pytest

from waypoint.feedback import (
    FeedbackConfig,
    PerceptionEvent,
    PerceptionKind,
    PerceptionSeverity,
    PerceptionSource,
    SymbolicPerceptionSource,
    TelemetrySnapshot,
    TriggerKind,
    battery_drain_ratio,
    estimated_battery_need,
    needs_replan,
    path_deviation,
    summarize_for_llm,
)
from waypoint.planner import Obstacle
from waypoint.planner.models import Action, Waypoint
from waypoint.sim import UAVState

ROUTE = (
    Waypoint(x=0, y=0, z=25, speed=3),
    Waypoint(x=0, y=80, z=25, speed=4, action=Action.CAPTURE),
    Waypoint(x=40, y=80, z=25, speed=4, action=Action.CAPTURE),
    Waypoint(x=0, y=0, z=0, speed=2, action=Action.LAND),
)


def snapshot(*, known=(), planned=(), battery=90.0, position=(0.0, 10.0, 25.0), target=1, t=12.0, **extra):
    return TelemetrySnapshot(
        t=t,
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


# ---------------------------------------------------------------------------
# History: battery trend and sustained deviation
# ---------------------------------------------------------------------------


def flown(seconds, *, drain_per_s, speed=4.0, offset=0.0, start_battery=90.0, dt=0.5):
    """Snapshots up the first northbound leg at `speed`, `offset` metres east of it."""
    history = []
    for i in range(round(seconds / dt) + 1):
        t = i * dt
        history.append(snapshot(
            position=(offset, 10.0 + speed * t, 25.0),
            battery=start_battery - drain_per_s * t,
            route_start=(0.0, 0.0, 25.0),
            t=12.0 + t,
        ))
    return history


def modelled_drain(speed=4.0, config=FeedbackConfig()):
    return speed / config.meters_per_battery_pct + 1 / config.seconds_per_battery_pct


def test_a_battery_draining_as_modelled_has_a_ratio_of_one():
    assert battery_drain_ratio(flown(8, drain_per_s=modelled_drain())) == pytest.approx(1.0, abs=0.01)


def test_too_little_history_has_no_trend():
    assert battery_drain_ratio(flown(2, drain_per_s=1.0)) is None


def test_a_battery_draining_too_fast_triggers_a_replan_before_the_model_would():
    history = flown(8, drain_per_s=modelled_drain() * 6, start_battery=40.0)
    assert needs_replan(history[-1]) is None  # the model alone thinks there is plenty
    trigger = needs_replan(history)
    assert trigger.kind is TriggerKind.BATTERY_INSUFFICIENT
    assert trigger.detail["drain_ratio"] == pytest.approx(6.0, rel=0.02)
    assert "faster than expected" in trigger.reason


def test_a_little_noise_in_the_drain_is_not_a_trend():
    assert needs_replan(flown(8, drain_per_s=modelled_drain() * 1.2, start_battery=20.0)) is None


def test_deviation_is_cross_track_not_distance_to_the_waypoint():
    snap = snapshot(position=(6.0, 40.0, 25.0), target=1)  # halfway up the first leg, 6 m east
    assert snap.deviation == pytest.approx(6.0)
    assert path_deviation(snap, 5.0) and not path_deviation(snap, 10.0)


def test_the_first_leg_starts_where_the_plan_took_over():
    route = (Waypoint(x=0, y=80, z=25, speed=4),)
    on_line = snapshot(position=(50.0, 40.0, 25.0), target=0, route_start=(100.0, 0.0, 25.0))
    assert TelemetrySnapshot(**{**on_line.__dict__, "plan_waypoints": route}).deviation == pytest.approx(0.0)


def test_sustained_deviation_triggers_a_replan():
    trigger = needs_replan(flown(4, drain_per_s=modelled_drain(), offset=20.0))
    assert trigger.kind is TriggerKind.OFF_ROUTE
    assert trigger.detail["deviation"] == pytest.approx(20.0)


def test_a_moment_off_route_is_not_a_reason():
    history = flown(4, drain_per_s=modelled_drain())
    history[-1] = TelemetrySnapshot(**{**history[-1].__dict__, "position": (20.0, 26.0, 25.0)})
    assert needs_replan(history) is None  # off now, but not for the hold time
    assert needs_replan(history[-1]) is None  # one snapshot can never be sustained


def test_an_empty_history_needs_nothing():
    assert needs_replan([]) is None


def test_the_summary_reports_deviation_and_what_perception_sees_nearby():
    crane = Obstacle(x=2, y=40, radius=6, height=60, name="crane")
    event = PerceptionEvent(
        kind=PerceptionKind.OBSTACLE, t=12.0, position=(2, 40, 0), severity=PerceptionSeverity.WARNING,
        description="'crane' at (2, 40), 24 m from the aircraft", source="test", distance=24.0, obstacle=crane,
    )
    far = PerceptionEvent(
        kind=PerceptionKind.OBSTACLE, t=12.0, position=(900, 0, 0), severity=PerceptionSeverity.INFO,
        description="'pylon' far away", source="test", distance=880.0,
    )
    snap = snapshot(known=[crane], position=(8.0, 10.0, 25.0), perception=(event, far))
    text = summarize_for_llm(snap, needs_replan(snap))
    assert "Off the planned leg by 8.0 m" in text
    assert "[WARNING] 'crane' at (2, 40), 24 m from the aircraft" in text
    assert "pylon" not in text


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


class Holding:
    """Just enough of a backend for perception: the obstacles it knows."""

    def __init__(self, *obstacles):
        self.obstacles = list(obstacles)


def test_symbolic_perception_reports_what_the_backend_knows():
    crane = Obstacle(x=0, y=30, radius=5, height=60, name="crane")
    events = SymbolicPerceptionSource(Holding(crane)).get_events(UAVState(x=0, y=0, z=25, timestamp=3.0))
    (event,) = events
    assert event.kind is PerceptionKind.OBSTACLE and event.obstacle == crane
    assert event.distance == pytest.approx(25.0) and event.t == 3.0
    assert event.severity is PerceptionSeverity.WARNING
    assert event.as_record()["type"] == "OBSTACLE"


def test_severity_grows_as_the_aircraft_closes():
    source = SymbolicPerceptionSource(Holding(Obstacle(x=0, y=0, radius=2, name="mast")))
    severity = lambda y: source.get_events(UAVState(x=0, y=y, z=10))[0].severity  # noqa: E731
    assert [severity(100), severity(30), severity(5)] == [
        PerceptionSeverity.INFO, PerceptionSeverity.WARNING, PerceptionSeverity.CRITICAL,
    ]


def test_a_sensor_range_hides_what_is_too_far_to_see():
    source = SymbolicPerceptionSource(Holding(Obstacle(x=0, y=100, radius=5, name="silo")), sensor_range=50)
    assert source.get_events(UAVState(x=0, y=0, z=25)) == []
    assert len(source.get_events(UAVState(x=0, y=50, z=25))) == 1


def test_any_perception_source_satisfies_the_interface():
    class Fixed(PerceptionSource):
        name = "fixed"

        def get_events(self, current_state):
            return []

    assert Fixed().get_events(UAVState(x=0, y=0, z=0)) == []
    with pytest.raises(TypeError):
        PerceptionSource()  # abstract
