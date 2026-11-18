"""Deterministic repair: what it fixes, how, and that it never hands back a plan the verifier refuses."""

from __future__ import annotations

import math
import random

import pytest
from shapely.geometry import LineString, Point, box

from waypoint.planner import DEFAULT_CONSTRAINTS, Obstacle, Plan
from waypoint.planner.models import GeoPoint
from waypoint.scenario import constraints_for, load_world
from waypoint.verifier import verify
from waypoint.verifier.models import DEFAULT_CONFIG
from waypoint.verifier.repair import relaxed, repair, route_length, shortest_path, trim_for_battery
from waypoint.world.landing import survey


def plan(*points) -> Plan:
    return Plan.model_validate({"subtasks": [{"name": "route", "waypoints": [
        {"x": x, "y": y, "z": z, "speed": s, "action": a} for x, y, z, s, a in points
    ]}]})


def test_shortest_path_goes_round_the_corner_of_a_block():
    block = box(-10, 40, 10, 60)
    corners = shortest_path((0, 0), (0, 100), [block.buffer(2, join_style=2)], box(-200, -200, 200, 200))
    assert corners and len(corners) == 2
    route = LineString([(0, 0), *corners, (0, 100)])
    assert not route.crosses(block) and not route.within(block) and route.distance(block) >= 1.9
    assert shortest_path((0, 0), (0, 100), [], None) == []


def test_shortest_path_says_when_there_is_no_way_through():
    wall = box(-300, 40, 300, 60)
    assert shortest_path((0, 0), (0, 100), [wall], box(-200, -200, 200, 200)) is None


def test_a_leg_through_a_no_fly_zone_goes_around_it():
    zone = [GeoPoint(x=-10, y=30), GeoPoint(x=10, y=30), GeoPoint(x=10, y=50), GeoPoint(x=-10, y=50)]
    constraints = DEFAULT_CONSTRAINTS.model_copy(update={"no_fly_zones": [zone]})
    bad = plan((0, 0, 25, 3, "MOVE"), (0, 80, 25, 4, "CAPTURE"), (0, 0, 0, 2, "LAND"))
    fixed = repair(bad, constraints)
    assert fixed.repaired
    assert verify(fixed.plan, constraints).passed
    assert any("no-fly" in a for a in fixed.actions)


def test_a_crane_is_flown_over_when_that_is_shorter():
    crane = Obstacle(x=0, y=40, radius=30, height=30, name="crane")
    constraints = DEFAULT_CONSTRAINTS.model_copy(update={"obstacles": [crane], "min_separation": 5})
    bad = plan((0, 0, 25, 3, "MOVE"), (0, 80, 25, 4, "CAPTURE"), (0, 0, 25, 4, "MOVE"), (0, 0, 0, 2, "LAND"))
    fixed = repair(bad, constraints)
    assert fixed.repaired
    assert any("over an obstacle" in a for a in fixed.actions)


def test_speed_altitude_and_landing_are_clamped():
    bad = plan((0, 0, 25, 3, "MOVE"), (0, 80, 150, 30, "CAPTURE"), (0, 0, 4, 2, "LAND"))
    world = load_world("lodhi")
    constraints, _ = constraints_for(world, None)
    pad = survey(world, 0, 0).position
    config = DEFAULT_CONFIG.model_copy(update={"home": pad})
    fixed = repair(bad, constraints, world=world, config=config)
    assert fixed.repaired
    wps = fixed.plan.waypoints()
    assert max(w.speed for w in wps) <= constraints.max_velocity
    assert max(w.z for w in wps) <= constraints.max_altitude
    assert wps[-1].z == pytest.approx(pad[2], abs=0.01)


@pytest.mark.parametrize("name", ["lodhi", "chowk", "munnar", "delhi"])
def test_repair_never_returns_a_plan_the_verifier_refuses(name):
    """Random routes across real places, some through buildings and hills: repaired, or honestly not."""
    world = load_world(name)
    constraints, _ = constraints_for(world, None)
    pad = survey(world, 0, 0).position
    config = DEFAULT_CONFIG.model_copy(update={"home": pad})
    rng = random.Random(name)
    minx, miny, maxx, maxy = world.bounds
    repaired = 0
    for _ in range(12):
        points = [(pad[0], pad[1], pad[2] + 30, 3, "MOVE")]
        for _ in range(rng.randint(1, 3)):
            points.append((
                rng.uniform(minx + 60, maxx - 60) * 0.5, rng.uniform(miny + 60, maxy - 60) * 0.5,
                rng.uniform(5, 60) + world.elevation_at(0, 0), rng.uniform(3, 14), rng.choice(["MOVE", "CAPTURE"]),
            ))
        points += [(pad[0], pad[1], pad[2] + 30, 6, "MOVE"), (*pad, 1.5, "LAND")]
        outcome = repair(plan(*points), constraints, world=world, config=config)
        if outcome.repaired:
            repaired += 1
            assert verify(outcome.plan, constraints, config, world).passed
        else:
            assert outcome.plan is None
    assert repaired >= 6  # most of these are fixable


def test_trim_for_battery_gives_up_phases_then_heads_home():
    constraints = DEFAULT_CONSTRAINTS.model_copy(update={"battery_budget_pct": 5.0})
    long = Plan.model_validate({"subtasks": [
        {"name": "far_survey", "waypoints": [{"x": 150, "y": 150, "z": 30, "speed": 6, "action": "CAPTURE"}]},
        {"name": "near_survey", "waypoints": [{"x": 20, "y": 20, "z": 30, "speed": 6, "action": "CAPTURE"}]},
        {"name": "land", "waypoints": [{"x": 0, "y": 0, "z": 0, "speed": 1.5, "action": "LAND"}]},
    ]})
    config = DEFAULT_CONFIG.model_copy(update={"home": (10.0, 10.0, 30.0)})
    trimmed = trim_for_battery(long, constraints, config=config, landing=(0.0, 0.0, 0.0))
    assert trimmed.repaired
    assert "far_survey" not in [s.name for s in trimmed.plan.subtasks]
    assert verify(trimmed.plan, constraints, config).passed
    assert route_length(trimmed.plan, config.home) < route_length(long, config.home)


def test_relaxed_lets_intent_give_way_but_never_safety():
    from waypoint.intent import MissionIntent, Target

    intent = MissionIntent(targets=[Target(name="corner", x=150, y=150)])
    p = plan((0, 0, 25, 3, "MOVE"), (0, 0, 0, 2, "LAND"))
    result = verify(p, DEFAULT_CONSTRAINTS, intent=intent)
    assert not result.passed and relaxed(result).passed
    unsafe = verify(plan((0, 0, 500, 3, "MOVE"), (0, 0, 0, 2, "LAND")), DEFAULT_CONSTRAINTS, intent=intent)
    assert not relaxed(unsafe).passed
    assert math.isclose(len(relaxed(result).violations), len(result.violations))
    assert Point(0, 0).equals(Point(0, 0))
