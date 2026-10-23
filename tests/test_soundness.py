"""The verifier is sound with respect to its world model: it never passes what a dense check fails.

Each check is exact geometry, so these tests compare it against brute force:
walk the route in 25 cm steps and look for anything the check missed.
"""

from __future__ import annotations

import math
import random

import pytest

from waypoint.verifier.checks import lowest_clearance
from waypoint.world.models import Terrain, World


def _dense(start, end, world, step=0.25):
    length = math.dist(start, end)
    steps = max(1, int(length / step))
    worst = None
    for i in range(steps + 1):
        f = i / steps
        x = start[0] + (end[0] - start[0]) * f
        y = start[1] + (end[1] - start[1]) * f
        z = start[2] + (end[2] - start[2]) * f
        c = z - world.elevation_at(x, y)
        worst = c if worst is None else min(worst, c)
    return worst


def _ridge_world(seed: int) -> World:
    rng = random.Random(seed)
    heights = [[rng.uniform(-10, 60) for _ in range(9)] for _ in range(9)]
    return World(terrain=Terrain(origin_x=-100, origin_y=-100, spacing=25, heights=heights))


@pytest.mark.parametrize("seed", range(12))
def test_exact_clearance_is_never_above_a_dense_walk(seed):
    world = _ridge_world(seed)
    rng = random.Random(1000 + seed)
    for _ in range(40):
        # Some legs run off the grid, where the ground is clamped to its edge.
        start = (rng.uniform(-140, 140), rng.uniform(-140, 140), rng.uniform(0, 90))
        end = (rng.uniform(-140, 140), rng.uniform(-140, 140), rng.uniform(0, 90))
        exact = lowest_clearance(start, end, world)
        dense = _dense(start, end, world)
        assert exact is not None
        # Sound: never more optimistic than what the dense walk found.
        assert exact[0] <= dense + 1e-6
        # Exact: and not pessimistic either, beyond what 25 cm steps can miss.
        assert exact[0] >= dense - 0.5


def test_a_ridge_between_five_metre_samples_is_caught():
    """A spike one grid post wide, crossed where the old 5 m samples straddled it."""
    heights = [[0.0] * 5 for _ in range(5)]
    heights[2][2] = 40.0  # a post at (0, 0) on a 2 m grid: a narrow peak
    world = World(terrain=Terrain(origin_x=-4, origin_y=-4, spacing=2, heights=heights))
    worst = lowest_clearance((-20.0, 0.0, 30.0), (20.0, 0.0, 30.0), world)
    assert worst is not None
    clearance, x, y = worst
    assert clearance == pytest.approx(-10.0)
    assert (x, y) == pytest.approx((0.0, 0.0))


def _wall_world():
    from waypoint.planner.models import GeoPoint
    from waypoint.world.models import Building

    wall = Building(footprint=[GeoPoint(x=40, y=-20), GeoPoint(x=60, y=-20), GeoPoint(x=60, y=20), GeoPoint(x=40, y=20)],
                    height=50, name="the hall")
    return World(buildings=[wall])


def _plan(*points):
    from waypoint.planner.models import Plan

    return Plan.model_validate({"subtasks": [{"name": "r", "waypoints": [
        {"x": x, "y": y, "z": z, "speed": s, "action": a} for x, y, z, s, a in points
    ]}]})


def test_a_climb_steeper_than_the_aircraft_can_fly_is_judged_as_flown():
    """The straight line clears the roof; the aircraft, climbing at 4 m/s, does not -- and flies into it."""
    from waypoint.planner import DEFAULT_CONSTRAINTS
    from waypoint.sim.executor import Executor
    from waypoint.verifier import verify

    world = _wall_world()
    plan = _plan((0, 0, 20, 3, "MOVE"), (70, 0, 110, 12, "MOVE"), (70, 0, 0, 2, "LAND"))
    flown = Executor(plan, constraints=DEFAULT_CONSTRAINTS, world=world)
    list(flown.run())
    assert flown.crashed  # the physics says so
    result = verify(plan, DEFAULT_CONSTRAINTS, world=world)
    assert not result.passed
    assert any("climbs faster" in v.description for v in result.blocking)


def test_stopping_just_above_a_roof_is_refused():
    from waypoint.planner import DEFAULT_CONSTRAINTS
    from waypoint.verifier import verify

    world = _wall_world()
    plan = _plan((0, 0, 60, 3, "MOVE"), (50, 0, 60, 4, "MOVE"), (50, 0, 51, 3, "CAPTURE"),
                 (50, 0, 60, 3, "MOVE"), (0, 0, 60, 4, "MOVE"), (0, 0, 0, 2, "LAND"))
    result = verify(plan, DEFAULT_CONSTRAINTS, world=world)
    assert any("only 1.0 m above the roof" in v.description for v in result.blocking)


def test_a_pad_brushing_a_wall_is_not_a_landing_site():
    """The live batch landed 0.36 m from the National Stadium's wall, and the descent drifted into it."""
    from waypoint.scenario import load_world
    from waypoint.world.landing import find, survey

    world = load_world("delhi")
    assert survey(world, 256.5, -125.14).obstructed == "the National Stadium"
    moved = find(world, 256.5, -125.14)
    assert moved.usable and moved.moved > 0


def test_grounded_ends_are_left_out():
    world = World(terrain=Terrain(origin_x=-50, origin_y=-50, spacing=25, heights=[[0.0] * 5 for _ in range(5)]))
    # A vertical landing: the last metres are meant to be near the ground.
    assert lowest_clearance((0, 0, 20), (0, 0, 0), world, skip_end=5.0)[0] == pytest.approx(5.0)
    assert lowest_clearance((0, 0, 20), (0, 0, 0), world)[0] == pytest.approx(0.0)
