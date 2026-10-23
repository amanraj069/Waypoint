"""The runtime shield: the commanded velocity never carries the aircraft across a barrier."""

from __future__ import annotations

import math

import pytest

from waypoint.planner import DEFAULT_CONSTRAINTS, Obstacle, Plan
from waypoint.planner.models import GeoPoint
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, SimConfig, Trigger, Wind
from waypoint.sim.executor import Executor
from waypoint.sim.shield import SafetyShield, _project
from waypoint.world.models import Building, World


def straight(*points) -> Plan:
    return Plan.model_validate({"subtasks": [{"name": "leg", "waypoints": [
        {"x": x, "y": y, "z": z, "speed": 8, "action": a} for x, y, z, a in points
    ]}]})


def test_far_from_everything_nothing_changes():
    shield = SafetyShield(DEFAULT_CONSTRAINTS, None)
    assert shield.filter((0, 0, 30), (5.0, 0.0, 0.0)) == ((5.0, 0.0, 0.0), [])


def test_heading_into_an_obstacle_the_closing_speed_is_cut_and_the_sideways_kept():
    crane = Obstacle(x=20, y=0, radius=5, height=None, name="crane")
    shield = SafetyShield(DEFAULT_CONSTRAINTS.model_copy(update={"obstacles": [crane]}), None)
    v, hazards = shield.filter((12, 0, 30), (6.0, 3.0, 0.0), velocity=(6.0, 3.0, 0.0))
    assert hazards == ["'crane'"]
    assert v[0] < 1.0  # barely closing, or backing off
    assert v[1] == pytest.approx(3.0, abs=0.5)  # sliding past is fine


def test_dykstra_finds_the_nearest_point_in_two_halfspaces():
    # x >= 1 and y >= 1, from (0, 0): the corner.
    assert _project((0.0, 0.0, 0.0), [((1, 0, 0), 1.0), ((0, 1, 0), 1.0)]) == pytest.approx((1.0, 1.0, 0.0), abs=1e-4)


def test_a_plan_straight_through_a_crane_stops_short_of_it():
    """No verifier and no replanner here: only the shield stands between the aircraft and the crane."""
    crane = Obstacle(x=0, y=60, radius=8, height=None, name="crane")
    constraints = DEFAULT_CONSTRAINTS.model_copy(update={"obstacles": [crane], "min_separation": 2})
    plan = straight((0, 0, 25, "MOVE"), (0, 120, 25, "MOVE"))
    config = SimConfig(max_leg_seconds=40)
    unshielded = Executor(plan, constraints=constraints, config=config)
    list(unshielded.run())
    shielded = Executor(plan, constraints=constraints, config=config, shield=True)
    states = list(shielded.run())
    assert any(e.type == "SEPARATION" for e in unshielded.events)
    assert not any(e.type == "SEPARATION" for e in shielded.events)
    closest = min(math.dist((s.x, s.y), (0, 60)) for s in states)
    assert closest >= 10.0 - 0.3
    assert any(e.type == "SHIELD" for e in shielded.events)


def test_a_gust_does_not_blow_it_into_a_wall():
    wall = Building(footprint=[GeoPoint(x=8, y=0), GeoPoint(x=20, y=0), GeoPoint(x=20, y=200), GeoPoint(x=8, y=200)],
                    height=60, name="the wall")
    world = World(buildings=[wall])
    gust = FaultSchedule.of(FaultEvent(type=FaultType.WIND_GUST, trigger=Trigger(time=8.0), magnitude=9.0,
                                       direction=0.0, duration=8.0, name="gust"))
    plan = straight((0, 0, 25, "MOVE"), (0, 150, 25, "MOVE"))
    hit = Executor(plan, constraints=DEFAULT_CONSTRAINTS, world=world, faults=gust)
    list(hit.run())
    saved = Executor(plan, constraints=DEFAULT_CONSTRAINTS, world=world, faults=gust, shield=True)
    list(saved.run())
    assert hit.crashed
    assert not saved.crashed


def test_a_drifting_gps_does_not_fly_it_into_a_wall():
    """The fix drifts east, so the controller steers west of where it should: into a wall it cannot see.

    The shield compares the fix with the airframe's own measured motion, sees the
    drift, and guards where the aircraft really is.
    """
    wall = Building(footprint=[GeoPoint(x=-20, y=0), GeoPoint(x=-8, y=0), GeoPoint(x=-8, y=200), GeoPoint(x=-20, y=200)],
                    height=60, name="the wall")
    world = World(buildings=[wall])
    slip = FaultSchedule.of(FaultEvent(type=FaultType.GPS_DRIFT, trigger=Trigger(time=6.0), magnitude=1.2,
                                       direction=0.0, duration=25.0, name="gps slip"))
    plan = straight((0, 0, 25, "MOVE"), (0, 150, 25, "MOVE"))
    blind = Executor(plan, constraints=DEFAULT_CONSTRAINTS, world=world, faults=slip)
    list(blind.run())
    guarded = Executor(plan, constraints=DEFAULT_CONSTRAINTS, world=world, faults=slip, shield=True)
    list(guarded.run())
    assert blind.crashed
    assert not guarded.crashed
    assert any(e.type == "GPS_INTEGRITY" for e in guarded.events)


def test_the_geofence_holds_against_a_plan_that_leaves_it():
    fence = [GeoPoint(x=-50, y=-50), GeoPoint(x=50, y=-50), GeoPoint(x=50, y=50), GeoPoint(x=-50, y=50)]
    constraints = DEFAULT_CONSTRAINTS.model_copy(update={"geofence": fence})
    plan = straight((0, 0, 25, "MOVE"), (0, 90, 25, "MOVE"))
    ex = Executor(plan, constraints=constraints, config=SimConfig(max_leg_seconds=30), shield=True)
    states = list(ex.run())
    assert max(s.y for s in states) <= 50.0


def test_the_ceiling_holds():
    plan = straight((0, 0, 140, "MOVE"))
    ex = Executor(plan, constraints=DEFAULT_CONSTRAINTS, config=SimConfig(max_leg_seconds=40), shield=True)
    states = list(ex.run())
    assert max(s.z for s in states) <= DEFAULT_CONSTRAINTS.max_altitude + 0.5


def test_landing_is_not_blocked_by_the_ground():
    plan = straight((0, 0, 20, "MOVE"), (0, 0, 0, "LAND"))
    ex = Executor(plan, constraints=DEFAULT_CONSTRAINTS, world=World(), shield=True)
    states = list(ex.run())
    assert states[-1].z == pytest.approx(0.0)
    assert any(e.type == "LANDED" for e in ex.events)


def test_wind_alone_does_not_make_the_shield_twitch_in_open_sky():
    plan = straight((0, 0, 25, "MOVE"), (0, 100, 25, "MOVE"))
    ex = Executor(plan, constraints=DEFAULT_CONSTRAINTS, wind=Wind(vector=(3, 0, 0)), shield=True)
    list(ex.run())
    assert ex.shield_ticks == 0
