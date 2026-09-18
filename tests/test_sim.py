"""Simulator tests: kinematics, battery, wind and fault injection.

The simulator is deterministic, so every assertion here is exact rather than
statistical -- including the turbulence, which is seeded.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from planner import Action, MissionConstraints, Obstacle, Plan, SubTask, Waypoint
from sim import (
    Executor,
    FaultEvent,
    FaultSchedule,
    FaultType,
    SimConfig,
    Trigger,
    UAVState,
    Wind,
    execute,
    initial_state,
)
from sim.dynamics import desired_velocity, distance

CONFIG = SimConfig()


def wp(x, y, z, speed=5.0, action=Action.MOVE) -> Waypoint:
    return Waypoint(x=x, y=y, z=z, speed=speed, action=action)


def plan_of(*waypoints: Waypoint, name: str = "leg") -> Plan:
    return Plan(mission_text="test", subtasks=[SubTask(name=name, waypoints=list(waypoints))])


@pytest.fixture
def out_and_back() -> Plan:
    """Climb, fly 60 m east, come home, land."""
    return plan_of(
        wp(0, 0, 20, 3),
        wp(60, 0, 20, 6),
        wp(0, 0, 20, 6),
        wp(0, 0, 0, 2, Action.LAND),
    )


def flown(plan: Plan, **kwargs) -> list[UAVState]:
    return Executor(plan, **kwargs).run_to_list()


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def test_execute_still_yields_a_generator_of_states(sample_plan):
    stream = execute(sample_plan)
    assert isinstance(stream, Iterator)
    assert all(isinstance(state, UAVState) for state in stream)


def test_the_flight_starts_on_the_pad(out_and_back):
    first = flown(out_and_back)[0]
    assert first.position == (0.0, 0.0, 0.0)
    assert first.velocity == 0.0
    assert first.timestamp == 0.0


def test_every_waypoint_is_reached(out_and_back):
    """Convergence: the aircraft passes within the arrival radius of each target."""
    states = flown(out_and_back)
    for waypoint in out_and_back.waypoints():
        target = (waypoint.x, waypoint.y, waypoint.z)
        closest = min(distance(s.position, target) for s in states)
        assert closest <= CONFIG.arrival_radius, f"never reached {target}"


def test_each_waypoint_is_reached_in_order(out_and_back):
    executor = Executor(out_and_back)
    executor.run_to_list()
    reached = [e.payload["waypoint_index"] for e in executor.events if e.type == "WAYPOINT"]
    assert reached == [0, 1, 2, 3]


def test_the_flight_ends_landed_and_stationary(out_and_back):
    final = flown(out_and_back)[-1]
    assert final.z == 0.0
    assert final.velocity == 0.0


def test_time_advances_by_exactly_one_tick(out_and_back):
    states = flown(out_and_back)
    steps = {round(b.timestamp - a.timestamp, 6) for a, b in zip(states, states[1:])}
    assert steps == {CONFIG.dt}


def test_nothing_teleports(out_and_back):
    """No tick moves further than the speed limit allows."""
    states = flown(out_and_back)
    ceiling = CONFIG.max_velocity * CONFIG.dt + 1e-9
    for a, b in zip(states, states[1:]):
        assert distance(a.position, b.position) <= ceiling


def test_acceleration_is_limited(out_and_back):
    """Velocity changes by at most max_acceleration * dt per tick.

    The touchdown tick is excluded: the ground stops the aircraft, so the last
    fraction of a metre per second goes to zero at once by design.
    """
    states = flown(out_and_back)[:-1]
    ceiling = CONFIG.max_acceleration * CONFIG.dt + 1e-9
    for a, b in zip(states, states[1:]):
        change = math.dist(a.velocity_vector, b.velocity_vector)
        assert change <= ceiling, f"{change} m/s in one tick at t={b.timestamp}"


def test_the_aircraft_builds_up_speed_rather_than_starting_at_cruise(out_and_back):
    states = flown(out_and_back)
    assert states[1].velocity < 0.5  # one tick of acceleration, not a jump to cruise
    assert max(s.velocity for s in states) > 3.0  # but it does get going


def test_commanded_speed_is_respected():
    """A waypoint asking for 2 m/s is flown at 2 m/s, not at the airframe limit."""
    states = flown(plan_of(wp(0, 0, 10, 2), wp(80, 0, 10, 2)))
    assert max(s.velocity for s in states) <= 2.0 + 1e-6


def test_the_airframe_speed_ceiling_is_respected():
    """Even when a leg asks for more than the airframe can do."""
    config = SimConfig(max_velocity=6.0)
    states = flown(plan_of(wp(0, 0, 10, 3), wp(200, 0, 10, 20)), config=config)
    assert max(s.velocity for s in states) <= 6.0 + 1e-6


def test_the_climb_rate_is_capped():
    config = SimConfig(max_climb_rate=2.0)
    states = flown(plan_of(wp(0, 0, 60, 10)), config=config)
    assert max(abs(s.vz) for s in states) <= 2.0 + 1e-6


def test_heading_follows_the_direction_of_travel():
    """Flying north is a heading of 90 degrees in this ENU frame."""
    states = flown(plan_of(wp(0, 0, 0, 5), wp(0, 60, 0, 5)))
    cruising = [s for s in states if s.ground_speed > 2.0]
    assert cruising
    assert all(math.isclose(s.heading, 90.0, abs_tol=1.0) for s in cruising)


def test_hover_holds_station_for_the_configured_dwell():
    config = SimConfig(hover_seconds=3.0)
    states = flown(plan_of(wp(0, 0, 10, 3), wp(20, 0, 10, 4, Action.HOVER)), config=config)
    reached = next(i for i, s in enumerate(states) if distance(s.position, (20, 0, 10)) <= 1.5)
    holding = states[reached : reached + round(3.0 / config.dt)]
    spread = max(distance(s.position, holding[0].position) for s in holding)
    assert spread < 2.0  # it stays put rather than flying on


def test_desired_velocity_slows_down_on_approach():
    """The controller asks for less speed the closer it gets, so it can stop."""
    far = desired_velocity((0, 0, 0), (100, 0, 0), 10.0, CONFIG)
    near = desired_velocity((99, 0, 0), (100, 0, 0), 10.0, CONFIG)
    assert far[0] > near[0]
    assert near[0] >= CONFIG.min_approach_speed


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def test_battery_drains_monotonically_and_stays_in_range(out_and_back):
    states = flown(out_and_back)
    assert all(b.battery_pct <= a.battery_pct for a, b in zip(states, states[1:]))
    assert all(0.0 <= s.battery_pct <= 100.0 for s in states)
    assert states[-1].battery_pct < states[0].battery_pct


def test_battery_drain_tracks_distance_flown():
    """Twice the route, about twice the energy (plus the time-based baseline)."""
    short = Executor(plan_of(wp(0, 0, 10, 5), wp(50, 0, 10, 5)))
    long = Executor(plan_of(wp(0, 0, 10, 5), wp(100, 0, 10, 5)))
    short_states, long_states = short.run_to_list(), long.run_to_list()

    short_used = 100.0 - short_states[-1].battery_pct
    long_used = 100.0 - long_states[-1].battery_pct
    assert long_used > short_used
    assert long.distance_flown > short.distance_flown * 1.5


def test_a_flat_battery_abandons_the_plan_and_lands():
    config = SimConfig(start_battery_pct=1.0)
    executor = Executor(plan_of(wp(0, 0, 30, 5), wp(400, 0, 30, 5), wp(0, 0, 0, 2, Action.LAND)), config=config)
    states = executor.run_to_list()

    assert states[-1].battery_pct == 0.0
    assert states[-1].z == 0.0  # it still comes down
    assert any(e.type == "BATTERY_EMPTY" for e in executor.events)


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------


def crosswind_config() -> SimConfig:
    """A UAV flying blind: it cannot sense the wind, so the wind moves it."""
    return SimConfig(wind_compensation=0.0)


def test_wind_visibly_perturbs_the_trajectory(out_and_back):
    calm = flown(out_and_back, config=crosswind_config())
    blown = flown(
        out_and_back, config=crosswind_config(), wind=Wind(vector=(0.0, 2.0, 0.0))
    )

    overlap = min(len(calm), len(blown))
    sideways = max(abs(a.y - b.y) for a, b in zip(calm[:overlap], blown[:overlap]))
    assert sideways > 1.0, "a 2 m/s crosswind should push the aircraft off the line"
    assert all(abs(s.y) < 1e-9 for s in calm), "the calm run flies dead straight"


def test_the_aircraft_still_reaches_its_waypoints_in_wind(out_and_back):
    blown = flown(out_and_back, config=crosswind_config(), wind=Wind(vector=(0.0, 2.0, 0.0)))
    for waypoint in out_and_back.waypoints():
        target = (waypoint.x, waypoint.y, waypoint.z)
        assert min(distance(s.position, target) for s in blown) <= CONFIG.arrival_radius


def test_wind_compensation_reduces_the_deviation(out_and_back):
    wind = Wind(vector=(0.0, 2.0, 0.0))
    blind = flown(out_and_back, config=SimConfig(wind_compensation=0.0), wind=wind)
    aware = flown(out_and_back, config=SimConfig(wind_compensation=0.9), wind=wind)

    assert max(abs(s.y) for s in aware) < max(abs(s.y) for s in blind)


def test_a_slowly_varying_wind_changes_over_time():
    wind = Wind(vector=(2.0, 0.0, 0.0), variation=(2.0, 0.0, 0.0), period=20.0)
    assert wind.at(0.0)[0] == pytest.approx(2.0)
    assert wind.at(5.0)[0] == pytest.approx(4.0)  # quarter cycle: full amplitude
    assert wind.at(15.0)[0] == pytest.approx(0.0)  # three quarters: full negative


def test_turbulence_is_seeded_so_runs_replay_identically(out_and_back):
    wind = Wind(vector=(1.0, 0.0, 0.0), turbulence=0.5, seed=42)
    first = flown(out_and_back, wind=wind)
    second = flown(out_and_back, wind=wind)

    assert [s.position for s in first] == [s.position for s in second]
    assert wind.at(3.0) != Wind(vector=(1.0, 0.0, 0.0), turbulence=0.5, seed=7).at(3.0)


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


def test_gps_drift_separates_true_position_from_reported(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.GPS_DRIFT,
            trigger=Trigger(time=5.0),
            magnitude=0.5,
            direction=90.0,
            duration=10.0,
            name="gps slip",
        )
    )
    states = flown(out_and_back, faults=schedule)

    assert all(s.gps_error == 0.0 for s in states if s.timestamp < 5.0)
    assert max(s.gps_error for s in states) == pytest.approx(5.0, abs=0.2)  # 0.5 m/s for 10 s
    # Flying on a wrong position estimate puts the aircraft somewhere else.
    clean = flown(out_and_back)
    assert abs(states[-1].y - clean[-1].y) > 1.0


def test_an_obstacle_appears_mid_flight_and_is_reported(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.OBSTACLE_APPEARS,
            trigger=Trigger(time=6.0),
            obstacle=Obstacle(x=30, y=0, radius=10, name="crane"),
            name="crane spotted",
        )
    )
    constraints = MissionConstraints(
        geofence=[{"x": -200, "y": -200}, {"x": 200, "y": -200}, {"x": 200, "y": 200}],
        max_altitude=100,
        min_altitude=0,
        max_velocity=12,
        battery_budget_pct=80,
        min_separation=5.0,
    )
    executor = Executor(out_and_back, constraints=constraints, faults=schedule)
    executor.run_to_list()

    assert [live.obstacle.name for live in executor.obstacles] == ["crane"]
    assert executor.obstacles[0].appeared_at == pytest.approx(6.0, abs=0.11)
    # The route flies straight through it, so the breach is noticed too.
    assert any(e.type == "SEPARATION" and "crane" in e.description for e in executor.events)


def test_an_obstacle_the_route_misses_raises_no_separation_event(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.OBSTACLE_APPEARS,
            trigger=Trigger(time=6.0),
            obstacle=Obstacle(x=30, y=200, radius=10, name="far crane"),
        )
    )
    executor = Executor(out_and_back, faults=schedule)
    executor.run_to_list()
    assert not [e for e in executor.events if e.type == "SEPARATION"]


def test_a_wind_gust_is_temporary(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.WIND_GUST,
            trigger=Trigger(time=4.0),
            magnitude=8.0,
            direction=90.0,
            duration=3.0,
            name="squall",
        )
    )
    executor = Executor(out_and_back, config=crosswind_config(), faults=schedule)
    states = executor.run_to_list()

    # The gust window is [4.0, 7.0): it expires on the tick that reaches 7 s.
    during = [s for s in states if 4.0 <= s.timestamp < 7.0]
    assert all(s.active_faults == ("squall",) for s in during)
    assert any(abs(s.wind[1] - 8.0) < 1e-6 for s in during)
    assert all(not s.active_faults for s in states if s.timestamp > 7.2)
    assert [e.type for e in executor.events if e.type.startswith("FAULT")] == ["FAULT", "FAULT_END"]
    assert max(abs(s.y) for s in states) > 1.0  # it got blown sideways


def test_a_battery_spike_drains_faster_for_a_while(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.BATTERY_DRAIN_SPIKE,
            trigger=Trigger(time=3.0),
            magnitude=1.0,  # an extra 1% per second
            duration=10.0,
            name="cell fault",
        )
    )
    clean = flown(out_and_back)
    spiked = flown(out_and_back, faults=schedule)

    extra = clean[-1].battery_pct - spiked[-1].battery_pct
    assert extra == pytest.approx(10.0, abs=0.5)  # 1%/s for 10 s


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


def test_a_fault_can_trigger_on_battery_level(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.GPS_DRIFT,
            trigger=Trigger(battery_below=99.5),
            magnitude=0.2,
            name="late drift",
        )
    )
    executor = Executor(out_and_back, faults=schedule)
    states = executor.run_to_list()

    fired = next(e for e in executor.events if e.type == "FAULT")
    at_trigger = next(s for s in states if s.timestamp >= fired.t)
    assert at_trigger.battery_pct < 99.5
    assert states[0].battery_pct == 100.0


def test_a_fault_can_trigger_on_reaching_a_waypoint(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.WIND_GUST,
            trigger=Trigger(at_waypoint=2),
            magnitude=5.0,
            direction=90.0,
            name="gust at the turn",
        )
    )
    executor = Executor(out_and_back, faults=schedule)
    executor.run_to_list()

    reached_1 = next(e.t for e in executor.events if e.type == "WAYPOINT" and e.payload["waypoint_index"] == 1)
    fired = next(e.t for e in executor.events if e.type == "FAULT")
    assert fired >= reached_1


def test_a_fault_can_trigger_on_distance_from_home(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(
            type=FaultType.BATTERY_DRAIN_SPIKE,
            trigger=Trigger(beyond_distance=40.0),
            magnitude=0.5,
            name="far from home",
        )
    )
    executor = Executor(out_and_back, faults=schedule)
    states = executor.run_to_list()

    fired = next(e.t for e in executor.events if e.type == "FAULT")
    at_trigger = next(s for s in states if s.timestamp >= fired)
    assert math.dist(at_trigger.position, (0, 0, 0)) >= 40.0


def test_an_empty_trigger_never_fires(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(type=FaultType.GPS_DRIFT, trigger=Trigger(), magnitude=5.0)
    )
    executor = Executor(out_and_back, faults=schedule)
    states = executor.run_to_list()

    assert not [e for e in executor.events if e.type == "FAULT"]
    assert all(s.gps_error == 0.0 for s in states)


def test_conditions_combine_with_and(out_and_back):
    trigger = Trigger(time=5.0, above_altitude=10.0)
    assert trigger.fires(t=6.0, battery_pct=100, waypoint_index=0, distance_from_home=0, altitude=20)
    assert not trigger.fires(t=4.0, battery_pct=100, waypoint_index=0, distance_from_home=0, altitude=20)
    assert not trigger.fires(t=6.0, battery_pct=100, waypoint_index=0, distance_from_home=0, altitude=5)


def test_a_fault_fires_only_once(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(type=FaultType.WIND_GUST, trigger=Trigger(time=2.0), magnitude=3.0, duration=1.0)
    )
    executor = Executor(out_and_back, faults=schedule)
    executor.run_to_list()
    assert len([e for e in executor.events if e.type == "FAULT"]) == 1


def test_an_obstacle_fault_needs_an_obstacle():
    with pytest.raises(ValueError, match="needs an `obstacle`"):
        FaultEvent(type=FaultType.OBSTACLE_APPEARS, trigger=Trigger(time=1.0))


def test_a_schedule_round_trips_through_json():
    schedule = FaultSchedule.of(
        FaultEvent(type=FaultType.WIND_GUST, trigger=Trigger(time=2.0), magnitude=3.0, duration=1.0),
        FaultEvent(
            type=FaultType.OBSTACLE_APPEARS,
            trigger=Trigger(at_waypoint=1),
            obstacle=Obstacle(x=1, y=2, radius=3, name="pylon"),
        ),
    )
    assert FaultSchedule.model_validate_json(schedule.model_dump_json()) == schedule


# ---------------------------------------------------------------------------
# Determinism and the event log
# ---------------------------------------------------------------------------


def test_two_runs_of_the_same_scenario_are_identical(out_and_back):
    schedule = FaultSchedule.of(
        FaultEvent(type=FaultType.WIND_GUST, trigger=Trigger(time=4.0), magnitude=6.0, direction=45.0, duration=3.0)
    )
    wind = Wind(vector=(1.0, 0.5, 0.0), variation=(1.0, 0, 0), period=12.0, turbulence=0.3, seed=11)

    first = flown(out_and_back, wind=wind, faults=schedule)
    second = flown(out_and_back, wind=wind, faults=schedule)
    assert [s.as_record() for s in first] == [s.as_record() for s in second]


def test_the_event_log_ends_with_touchdown(out_and_back):
    executor = Executor(out_and_back)
    executor.run_to_list()
    assert executor.events[-1].type == "LANDED"
    assert "battery" in executor.events[-1].description


def test_initial_state_is_on_the_pad():
    state = initial_state()
    assert state.position == (0.0, 0.0, 0.0)
    assert state.battery_pct == 100.0
    assert state.timestamp == 0.0
    assert state.velocity == 0.0
