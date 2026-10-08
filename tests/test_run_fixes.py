"""Regressions for what skewed the first full experiment runs, one test per cause.

Each failure here was seen in both the gpt-5-mini and the gpt-5-nano batches,
under every condition, so it was the system and not the planner:

* landing under a GPS offset slid the aircraft sideways by the whole offset;
* the failsafe came straight down onto whatever was below it, roofs included;
* a straight-up climb from the current position was judged at ~10^6 m/s;
* a landing pad below the datum could satisfy LANDING or ALTITUDE, never both;
* the planner was told to land at z = 0, wherever the pad was;
* an empty phase in an otherwise good plan failed the whole replan.
"""

from __future__ import annotations

import math

import pytest

from conftest import VALID_PAYLOAD, json_response, SequenceClient
from waypoint.orchestrator.loop import failsafe_route
from waypoint.planner import Action, DEFAULT_CONSTRAINTS, GeoPoint, Obstacle, Plan, SubTask, Waypoint, plan
from waypoint.planner.prompts import PLAN_SCHEMA, SYSTEM_PROMPT
from waypoint.sim import Executor, FaultEvent, FaultSchedule, FaultType, Trigger
from waypoint.verifier import ViolationType, verify
from waypoint.world import Building, Terrain, World


def box(x0, y0, x1, y1) -> list[GeoPoint]:
    return [GeoPoint(x=x0, y=y0), GeoPoint(x=x1, y=y0), GeoPoint(x=x1, y=y1), GeoPoint(x=x0, y=y1)]


def wp(x, y, z, speed=4.0, action=Action.MOVE) -> Waypoint:
    return Waypoint(x=x, y=y, z=z, speed=speed, action=action)


def plan_of(*waypoints) -> Plan:
    return Plan(mission_text="t", subtasks=[SubTask(name="route", waypoints=list(waypoints))])


# ---------------------------------------------------------------------------
# Landing under a GPS offset
# ---------------------------------------------------------------------------


def test_landing_under_a_gps_offset_comes_straight_down():
    """The drift ends long before landing but leaves its offset: the descent must not chase it."""
    drift = FaultSchedule.of(FaultEvent(
        type=FaultType.GPS_DRIFT, trigger=Trigger(time=3.0), magnitude=1.0, direction=45.0, duration=10.0,
    ))
    states = Executor(
        plan_of(wp(0, 0, 20), wp(40, 0, 20), wp(0, 0, 20), wp(0, 0, 0, 2, Action.LAND)), faults=drift
    ).run_to_list()

    assert states[-1].gps_error == pytest.approx(10.0, abs=0.3)
    descent = [s for s in states if s.target_index == 3 and s.z < 19.0]
    sideways = math.dist((descent[0].x, descent[0].y), (states[-1].x, states[-1].y))
    assert sideways < 1.0  # it used to slide the full 10 m


# ---------------------------------------------------------------------------
# The failsafe
# ---------------------------------------------------------------------------


@pytest.fixture
def market() -> World:
    """A 15 m roof under (0, 0), open ground to the east."""
    return World(name="market", buildings=[Building(footprint=box(-20, -20, 20, 20), height=15, name="the market")])


def test_the_failsafe_lands_in_place_when_the_ground_is_clear():
    route, where = failsafe_route(None, (5.0, 5.0, 30.0), 0.0, [])
    assert [w.action for w in route] == [Action.LAND] and (route[0].x, route[0].y) == (5.0, 5.0)
    assert where == "landing where it is"


def test_the_failsafe_never_comes_down_on_a_roof(market):
    route, where = failsafe_route(market, (0.0, 0.0, 18.0), 0.0, [])
    land = route[-1]
    assert land.action is Action.LAND
    assert market.collides(land.x, land.y, 0.5) is None  # clear ground, not the market
    assert all(w.z > 15.0 for w in route[:-1])  # crosses above the roof
    assert route[0].z == pytest.approx(25.0)  # climbs to roof + clearance before moving
    assert "clear ground" in where


def test_the_failsafe_route_never_hits_anything(market):
    route, _ = failsafe_route(market, (0.0, 0.0, 18.0), 0.0, [])
    states = Executor(plan_of(*route), world=market, config=_at((0.0, 0.0, 18.0))).run_to_list()
    assert states[-1].z == pytest.approx(0.0, abs=0.1)
    assert not any(e.type == "COLLISION" for e in _events(market, route))


def test_the_failsafe_keeps_clear_of_obstacles_that_appeared_in_flight():
    crane = Obstacle(x=0.0, y=0.0, radius=8.0, height=60.0, name="crane")
    route, _ = failsafe_route(None, (0.0, 0.0, 20.0), 0.0, [crane], separation=5.0)
    land = route[-1]
    assert math.hypot(land.x, land.y) >= 13.0
    assert all(w.z > 60.0 for w in route[:-1])  # over the crane, not through it


def _at(position):
    from waypoint.sim import SimConfig

    return SimConfig(home=position)


def _events(world, route):
    executor = Executor(plan_of(*route), world=world, config=_at((0.0, 0.0, 18.0)))
    executor.run_to_list()
    return executor.events


# ---------------------------------------------------------------------------
# The verifier
# ---------------------------------------------------------------------------


def test_a_vertical_climb_from_the_current_position_is_not_a_speeding_leg():
    """A replan echoing the current position, rounded, makes a leg of a few micrometres sideways."""
    from waypoint.verifier.models import DEFAULT_CONFIG

    here = (-19.306, 6.6e-13, 0.06)
    config = DEFAULT_CONFIG.model_copy(update={"home": here, "start_label": "the current position"})
    climb = plan_of(wp(-19.306004, 0, 5, 1.0), wp(-19.306, 40, 5), wp(-19.306, 40, 0, 1.5, Action.LAND))
    result = verify(climb, DEFAULT_CONSTRAINTS, config=config)
    assert not result.of_type(ViolationType.MAX_VELOCITY)


def test_a_steep_slanted_climb_at_a_legal_speed_is_not_refused():
    """The dam replans: 50 m up over 20 m across at 5 m/s, flown at the 4 m/s climb rate."""
    steep = plan_of(wp(0, 0, 20, 3), wp(0, 20, 70, 5), wp(0, 0, 70, 3), wp(0, 0, 0, 1.5, Action.LAND))
    assert not verify(steep, DEFAULT_CONSTRAINTS).of_type(ViolationType.MAX_VELOCITY)


def test_an_over_limit_speed_is_still_refused():
    fast = plan_of(wp(0, 0, 20, 3), wp(0, 80, 20, 15), wp(0, 0, 0, 1.5, Action.LAND))
    assert verify(fast, DEFAULT_CONSTRAINTS).of_type(ViolationType.MAX_VELOCITY)


@pytest.fixture
def sunken_lawn() -> World:
    """Flat ground 2.7 m below the datum everywhere east of x = 100."""
    xs = [-200.0 + 20 * i for i in range(31)]
    heights = [[0.0 if x < 100 else -2.7 for x in xs] for _ in xs]
    return World(name="lawn", terrain=Terrain(origin_x=-200.0, origin_y=-200.0, spacing=20.0, heights=heights))


def test_a_landing_pad_below_the_datum_can_be_landed_on(sunken_lawn):
    to_lawn = plan_of(wp(0, 0, 20), wp(250, 0, 20), wp(250, 0, -2.7, 1.5, Action.LAND))
    result = verify(to_lawn, DEFAULT_CONSTRAINTS.model_copy(update={"geofence": box(-200, -200, 400, 400)}),
                    world=sunken_lawn)
    blocking = {v.type for v in result.blocking}
    assert ViolationType.ALTITUDE not in blocking and ViolationType.LANDING not in blocking


def test_without_a_world_a_negative_altitude_is_still_refused():
    buried = plan_of(wp(0, 0, 20), wp(0, 0, -3, 1.5, Action.LAND))
    assert verify(buried, DEFAULT_CONSTRAINTS).of_type(ViolationType.ALTITUDE)


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


def test_the_planner_is_told_to_land_at_the_pads_height_not_zero():
    z = PLAN_SCHEMA["properties"]["subtasks"]["items"]["properties"]["waypoints"]["items"]["properties"]["z"]
    assert "except 0 for a waypoint on the ground" not in z["description"]
    assert "ground height of its spot" in z["description"]
    assert "LAND at the takeoff point, at z = 0" not in SYSTEM_PROMPT
    assert "z included" in SYSTEM_PROMPT


def test_an_empty_phase_does_not_sink_an_otherwise_good_plan():
    payload = {"subtasks": [*VALID_PAYLOAD["subtasks"], {"name": "land", "waypoints": []}]}
    result = plan("survey", DEFAULT_CONSTRAINTS, client=SequenceClient(json_response(payload)))
    assert [s.name for s in result.subtasks] == ["takeoff", "survey_north_field", "return_and_land"]


def test_a_plan_with_only_empty_phases_still_fails():
    from waypoint.planner import PlanValidationError

    payload = {"subtasks": [{"name": "land", "waypoints": []}]}
    with pytest.raises(PlanValidationError):
        plan("survey", DEFAULT_CONSTRAINTS, client=SequenceClient(json_response(payload)))
