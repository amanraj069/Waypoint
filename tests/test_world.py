"""The environment: terrain relief, buildings, and what they do to a flight.

These are the tests for flying somewhere rather than in empty space -- the
ground is not flat, buildings occupy volume, and z is height above the launch
pad rather than above whatever happens to be underneath.
"""

from __future__ import annotations

import math

import pytest

from waypoint.planner import Action, GeoPoint, Plan, SubTask, Waypoint
from waypoint.sim import Executor, FaultEvent, FaultSchedule, FaultType, SimConfig, Trigger
from waypoint.verifier import ViolationType, verify
from waypoint.world import Building, Terrain, World, canyon, city, rolling_terrain
from waypoint.world.osm import build_world, height_of, to_local


def wp(x, y, z, speed=5.0, action=Action.MOVE) -> Waypoint:
    return Waypoint(x=x, y=y, z=z, speed=speed, action=action)


def plan_of(*waypoints: Waypoint) -> Plan:
    return Plan(mission_text="test", subtasks=[SubTask(name="leg", waypoints=list(waypoints))])


def box(x0, y0, x1, y1) -> list[GeoPoint]:
    return [GeoPoint(x=x0, y=y0), GeoPoint(x=x1, y=y0), GeoPoint(x=x1, y=y1), GeoPoint(x=x0, y=y1)]


@pytest.fixture
def tower_world() -> World:
    """One 50 m tower on flat ground, 100 m north of the pad."""
    return World(
        name="one tower",
        buildings=[Building(footprint=box(-15, 85, 15, 115), height=50, name="the tower")],
    )


@pytest.fixture
def hill_world() -> World:
    """A ridge 45 m high running north-south at x = 50.

    The pad at (0, 0) is on the flat, and the ground drops back to zero beyond
    the ridge -- so a leg can have both ends in the clear and still fly into it.
    """
    profile = [0.0, 0.0, 0.0, 45.0, 0.0]          # x = -100, -50, 0, 50, 100
    return World(
        name="ridge",
        terrain=Terrain(origin_x=-100, origin_y=-100, spacing=50, heights=[profile] * 5),
    )


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def test_elevation_interpolates_between_grid_posts(hill_world):
    assert hill_world.elevation_at(0, 0) == pytest.approx(0.0)      # the pad, on the flat
    assert hill_world.elevation_at(50, 0) == pytest.approx(45.0)    # the ridge line
    assert hill_world.elevation_at(25, 0) == pytest.approx(22.5)    # halfway up the slope


def test_elevation_clamps_outside_the_grid(hill_world):
    assert hill_world.elevation_at(-9999, 0) == pytest.approx(0.0)
    assert hill_world.elevation_at(9999, 0) == pytest.approx(0.0)
    assert hill_world.elevation_at(50, 9999) == pytest.approx(45.0)


def test_agl_is_not_the_same_as_z(hill_world):
    """The whole point of terrain: 30 m up is not 30 m clear."""
    assert hill_world.agl(0, 0, 30) == pytest.approx(30.0)      # over the flat
    assert hill_world.agl(50, 0, 30) == pytest.approx(-15.0)    # inside the ridge


def test_terrain_can_be_levelled_on_the_pad():
    rough = rolling_terrain(half_extent=200, spacing=50, amplitude=30, seed=3)
    levelled = rough.levelled_at(0, 0)

    assert levelled.elevation_at(0, 0) == pytest.approx(0.0, abs=0.01)
    low, high = levelled.relief
    assert high - low == pytest.approx(rough.relief[1] - rough.relief[0], abs=0.05)


def test_a_ragged_terrain_grid_is_rejected():
    with pytest.raises(ValueError, match="same number of columns"):
        Terrain(heights=[[0.0, 1.0], [0.0]])


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


def test_a_building_occupies_volume(tower_world):
    assert tower_world.collides(0, 100, 20) is not None      # inside it
    assert tower_world.collides(0, 100, 60) is None          # above the roof
    assert tower_world.collides(60, 100, 20) is None         # beside it


def test_the_surface_is_the_roof_where_a_building_stands(tower_world):
    assert tower_world.surface_at(0, 100) == pytest.approx(50.0)
    assert tower_world.surface_at(0, 0) == pytest.approx(0.0)
    assert tower_world.agl(0, 100, 70) == pytest.approx(20.0)


def test_a_building_sits_on_its_base():
    raised = Building(footprint=box(0, 0, 10, 10), height=20, base=35, name="clifftop")
    assert raised.roof == 55
    world = World(buildings=[raised])
    assert world.collides(5, 5, 40) is not None
    assert world.collides(5, 5, 60) is None


def test_the_generated_city_leaves_the_launch_pad_clear():
    vista = city(seed=11)
    assert vista.collides(0, 0, 1) is None
    assert vista.elevation_at(0, 0) == pytest.approx(0.0, abs=0.05)
    assert len(vista.buildings) > 20


def test_the_city_is_the_same_city_every_time():
    first, second = city(seed=5), city(seed=5)
    assert first.model_dump() == second.model_dump()
    assert city(seed=6).model_dump() != first.model_dump()


def test_a_world_round_trips_through_json():
    vista = city(seed=2)
    assert World.model_validate_json(vista.model_dump_json()).model_dump() == vista.model_dump()


# ---------------------------------------------------------------------------
# The verifier, once there is something to hit
# ---------------------------------------------------------------------------


def test_a_route_through_a_building_is_rejected(tower_world, constraints):
    plan = plan_of(wp(0, 0, 20, 3), wp(0, 200, 20), wp(0, 0, 0, 2, Action.LAND))
    result = verify(plan, constraints, world=tower_world)

    assert result.passed is False
    buildings = result.of_type(ViolationType.BUILDING)
    assert buildings
    assert "the tower" in buildings[0].description
    assert "50 m" in buildings[0].description


def test_a_waypoint_inside_a_building_is_rejected(tower_world, constraints):
    plan = plan_of(wp(0, 0, 20, 3), wp(0, 100, 25), wp(0, 0, 0, 2, Action.LAND))
    result = verify(plan, constraints, world=tower_world)

    violation = result.of_type(ViolationType.BUILDING)[0]
    assert violation.waypoint_index == 1
    assert "is inside" in violation.description


def test_flying_over_the_roof_is_fine(tower_world, constraints):
    plan = plan_of(wp(0, 0, 60, 3), wp(0, 140, 60), wp(0, 0, 60, 4), wp(0, 0, 0, 2, Action.LAND))
    result = verify(plan, constraints, world=tower_world)

    assert result.of_type(ViolationType.BUILDING) == []


def test_a_leg_that_flies_into_a_hillside_is_rejected(hill_world, constraints):
    """Both ends are clear; the ridge between them is not."""
    plan = plan_of(wp(0, 0, 25, 3), wp(100, 0, 25, 5), wp(0, 0, 0, 2, Action.LAND))
    result = verify(plan, constraints, world=hill_world)

    terrain = result.of_type(ViolationType.TERRAIN)
    assert terrain
    assert terrain[0].severity.value == "CRITICAL"
    assert "hillside" in terrain[0].description


def test_a_waypoint_under_the_ground_is_rejected(hill_world, constraints):
    plan = plan_of(wp(0, 0, 20, 3), wp(50, 0, 20), wp(0, 0, 0, 2, Action.LAND))
    result = verify(plan, constraints, world=hill_world)

    violation = result.of_type(ViolationType.TERRAIN)[0]
    assert "underground" in violation.description
    assert violation.waypoint_index == 1


def test_a_route_that_climbs_over_the_ridge_is_fine(hill_world, constraints):
    # Out over the ridge and back over it again -- the return leg has to climb
    # too, which is exactly the mistake this check is here to catch.
    plan = plan_of(wp(0, 0, 25, 3), wp(50, 0, 70, 4), wp(100, 0, 25, 4),
                   wp(50, 0, 70, 4), wp(0, 0, 25, 4), wp(0, 0, 0, 2, Action.LAND))
    result = verify(plan, constraints, world=hill_world)

    assert result.of_type(ViolationType.TERRAIN) == []


def test_without_a_world_the_sky_is_empty(tower_world, constraints):
    """The same plan that hits a tower passes when no world is given."""
    plan = plan_of(wp(0, 0, 20, 3), wp(0, 140, 20), wp(0, 0, 0, 2, Action.LAND))

    assert verify(plan, constraints).of_type(ViolationType.BUILDING) == []
    assert verify(plan, constraints, world=tower_world).of_type(ViolationType.BUILDING) != []


# ---------------------------------------------------------------------------
# The simulator, once there is something to hit
# ---------------------------------------------------------------------------


def test_the_aircraft_crashes_into_a_building(tower_world):
    executor = Executor(plan_of(wp(0, 0, 20, 3), wp(0, 200, 20), wp(0, 0, 0, 2, Action.LAND)),
                        world=tower_world)
    states = executor.run_to_list()

    assert executor.crashed is True
    collision = next(e for e in executor.events if e.type == "COLLISION")
    assert "the tower" in collision.description
    assert states[-1].velocity == 0.0
    assert states[-1].y < 100  # stopped at the near face, not beyond it


def test_a_rooftop_crossing_survives(tower_world):
    executor = Executor(
        plan_of(wp(0, 0, 60, 3), wp(0, 140, 60), wp(0, 0, 60, 5), wp(0, 0, 0, 2, Action.LAND)),
        world=tower_world,
    )
    states = executor.run_to_list()

    assert executor.crashed is False
    assert states[-1].z == pytest.approx(0.0, abs=0.05)


def test_collisions_can_be_switched_off(tower_world):
    executor = Executor(
        plan_of(wp(0, 0, 20, 3), wp(0, 200, 20), wp(0, 0, 0, 2, Action.LAND)),
        world=tower_world,
        config=SimConfig(collisions=False),
    )
    executor.run_to_list()
    assert executor.crashed is False


def test_telemetry_carries_the_ground_and_the_height_above_it(hill_world):
    """Over the ridge, z and height-above-ground are two different numbers."""
    executor = Executor(
        plan_of(wp(0, 0, 70, 3), wp(50, 0, 70, 4), wp(0, 0, 70, 4), wp(0, 0, 0, 2, Action.LAND)),
        world=hill_world,
    )
    states = executor.run_to_list()

    assert all(s.agl == pytest.approx(s.z - s.ground) for s in states)
    over_ridge = [s for s in states if s.ground > 40]
    assert over_ridge, "the route should pass over the ridge"
    assert over_ridge[0].agl == pytest.approx(over_ridge[0].z - over_ridge[0].ground)
    assert over_ridge[0].agl < over_ridge[0].z   # higher ground, less clearance
    assert over_ridge[0].as_record()["agl"] is not None


def test_the_aircraft_starts_on_the_ground_not_inside_the_hill():
    """Launching where the terrain is high puts the pad at that elevation."""
    raised = World(
        name="plateau",
        terrain=Terrain(origin_x=-50, origin_y=-50, spacing=50, heights=[[12.0] * 3] * 3),
    )
    executor = Executor(plan_of(wp(0, 0, 40, 3), wp(0, 0, 12, 2, Action.LAND)), world=raised)
    states = executor.run_to_list()

    assert states[0].z == pytest.approx(12.0)
    assert executor.crashed is False


def test_gps_drift_flies_a_verified_route_into_a_wall():
    """The point of the whole exercise: a safe plan, made unsafe by a fault.

    Chandni Chowk is 30 m wide between the shopfronts, so a route down the
    middle of it has 15 m of margin either side -- and a GPS that slides 1.2 m/s
    for half a minute spends all of it.
    """
    street = canyon()
    route = plan_of(wp(-220, 0, 14, 4), wp(230, 0, 14, 6), wp(0, 0, 0, 2, Action.LAND))
    drift = FaultSchedule.of(
        FaultEvent(type=FaultType.GPS_DRIFT, trigger=Trigger(time=3.0), magnitude=1.2,
                   direction=0.0, duration=30.0, name="gps slip")
    )

    clean = Executor(route, world=street)
    clean.run_to_list()
    assert clean.crashed is False

    drifting = Executor(route, world=street, faults=drift)
    drifting.run_to_list()
    assert drifting.crashed is True
    assert any(e.type == "COLLISION" for e in drifting.events)


# ---------------------------------------------------------------------------
# OpenStreetMap import (no network: the conversion is what is tested)
# ---------------------------------------------------------------------------


def test_building_height_comes_from_whichever_tag_exists():
    assert height_of({"height": "42"}) == 42.0
    assert height_of({"height": "42 m"}) == 42.0
    assert height_of({"building:levels": "10"}) == pytest.approx(32.0)
    assert height_of({"height": "nonsense"}) == 9.0  # falls back rather than crashing
    assert height_of({}) == 9.0


def test_lat_lon_converts_to_metres_about_the_origin():
    x, y = to_local(51.5009, -0.1246, 51.5000, -0.1246)
    assert x == pytest.approx(0.0, abs=0.01)
    assert y == pytest.approx(99.5, abs=1.0)  # ~0.0009 deg of latitude


def test_an_overpass_payload_becomes_a_world():
    payload = {
        "elements": [
            {"type": "node", "id": 1, "lat": 51.5000, "lon": -0.1000},
            {"type": "node", "id": 2, "lat": 51.5000, "lon": -0.0995},
            {"type": "node", "id": 3, "lat": 51.5004, "lon": -0.0995},
            {"type": "node", "id": 4, "lat": 51.5004, "lon": -0.1000},
            {"type": "way", "id": 9, "nodes": [1, 2, 3, 4, 1],
             "tags": {"building": "office", "building:levels": "12", "name": "Example House"}},
        ]
    }
    world = build_world(payload, 51.5000, -0.1000, "somewhere")

    assert world.source == "osm"
    assert world.origin_lat == 51.5
    assert [b.name for b in world.buildings] == ["Example House"]
    assert world.buildings[0].height == pytest.approx(38.4)
    assert math.isclose(world.buildings[0].footprint[0].x, 0.0, abs_tol=0.01)


def test_a_payload_with_no_buildings_says_so():
    from waypoint.world.osm import OSMError

    with pytest.raises(OSMError, match="No buildings"):
        build_world({"elements": []}, 0.0, 0.0, "empty")


# ---------------------------------------------------------------------------
# Lodhi Garden: somewhere with names you can put in an instruction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def the_park() -> World:
    from waypoint.world import lodhi_garden

    return lodhi_garden()


def test_the_park_has_trees_and_they_are_solid(the_park):
    trees = [b for b in the_park.buildings if b.kind == "tree"]
    assert len(trees) > 40

    # The ashoka rows are trees; the gap between them is flyable.
    assert the_park.collides(-13, 90, 8) is not None
    assert the_park.collides(0, 90, 8) is None


def test_the_old_grove_must_be_flown_over_not_through(the_park):
    hits = [
        the_park.collides(-182 + dx, 300 + dy, 9)
        for dx in range(-40, 41, 10)
        for dy in range(-40, 41, 10)
    ]
    assert any(hits), "the old grove should be dense enough to stop a drone at 9 m"
    assert the_park.collides(-182, 300, 40) is None, "and clear above the canopy"


def test_the_launch_pad_is_clear(the_park):
    assert the_park.collides(0, 0, 1) is None
    assert the_park.elevation_at(0, 0) == pytest.approx(0.0, abs=0.05)


def test_landmarks_reach_the_planner_with_their_positions(the_park):
    described = the_park.describe()

    for name in ("Bara Gumbad", "the lake", "the old grove", "Safdarjung's Tomb",
                 "the ashoka avenue"):
        assert name in described
    assert "(5, 215)" in described          # where Bara Gumbad is
    assert "refer to any of them by name" in described


def test_a_landmark_can_be_looked_up_however_it_is_typed(the_park):
    assert the_park.find("Bara Gumbad").kind == "monument"
    assert the_park.find("bara gumbad").kind == "monument"
    assert the_park.find("Bara-Gumbad!").kind == "monument"
    assert the_park.find("the duck pond") is None


def test_the_lake_is_a_landmark_not_an_obstacle(the_park):
    lake = the_park.find("the lake")
    assert lake.kind == "water"
    # Open water: a low pass over the middle hits nothing.
    assert the_park.collides(lake.x, lake.y, 8) is None


def test_the_park_round_trips_with_its_landmarks(the_park):
    restored = World.model_validate_json(the_park.model_dump_json())
    assert [lm.name for lm in restored.landmarks] == [lm.name for lm in the_park.landmarks]
    assert [b.kind for b in restored.buildings] == [b.kind for b in the_park.buildings]


# ---------------------------------------------------------------------------
# The four built-in places
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["delhi", "lodhi", "chowk", "munnar"])
def test_every_built_in_world_is_flyable_from_its_pad(key):
    """Whatever else a place is, the aircraft has to be able to leave it."""
    from waypoint.world import build

    place = build(key)
    assert place.collides(0, 0, 1) is None, "the pad is inside something"
    assert place.elevation_at(0, 0) == pytest.approx(0.0, abs=0.05), "the pad is the datum"
    assert place.buildings, "a world with nothing in it teaches nothing"
    assert place.landmarks, "a world nobody can name is a world nobody can fly"


@pytest.mark.parametrize("key", ["delhi", "lodhi", "chowk", "munnar"])
def test_every_building_carries_a_name(key):
    """A mission is said out loud, so everything solid has to be sayable.

    This is the difference between "photograph the Jama Masjid" and a field of
    four hundred identical boxes called 'building'.
    """
    from waypoint.world import build

    for structure in build(key).buildings:
        assert structure.name.strip(), f"an unnamed {structure.kind} in {key}"


@pytest.mark.parametrize("key", ["delhi", "lodhi", "chowk", "munnar"])
def test_every_world_is_the_same_world_every_time(key):
    """A scenario has to stay comparable while the planner changes underneath it."""
    from waypoint.world import build

    assert build(key).model_dump() == build(key).model_dump()


def test_the_landmarks_of_a_place_are_really_there():
    """A landmark that names a building must sit on that building.

    Landmarks are descriptive and buildings are solid, so nothing enforces the
    relationship -- which is exactly why it is worth checking that 'Bara Gumbad'
    points at Bara Gumbad and not at the lawn beside it.
    """
    from waypoint.world import lodhi_garden

    garden = lodhi_garden()
    for name in ("Bara Gumbad", "Sheesh Gumbad", "Safdarjung's Tomb"):
        landmark = garden.find(name)
        assert landmark is not None
        structure = next(b for b in garden.buildings if b.name == name)
        cx, cy = structure.centre
        assert math.dist((cx, cy), (landmark.x, landmark.y)) < landmark.radius


def test_the_secretariat_roof_is_higher_than_the_secretariat():
    """Raisina Hill is the reason the Central Vista is worth flying.

    The North Block is 42 m tall and its roof is two-thirds as high again above
    the pad, because the ground under it has climbed. A planner that reads the
    building height and forgets the terrain flies into it.
    """
    from waypoint.world import lutyens_delhi

    vista = lutyens_delhi()
    north_block = next(b for b in vista.buildings if b.name == "the North Block")
    assert north_block.height == pytest.approx(42.0)
    assert north_block.roof > 60.0
    assert vista.elevation_at(-548, 0) > vista.elevation_at(120, 0) + 20


def test_chandni_chowk_is_only_flyable_down_the_middle():
    """The street is the whole scenario: 30 m of clear air and no way round."""
    from shapely.geometry import LineString

    from waypoint.world import chandni_chowk

    street = chandni_chowk()
    down_the_middle = LineString([(-240, 0), (280, 0)])
    assert street.obstructions(down_the_middle) == []

    for offset in (-20, 20):
        beside_it = LineString([(-240, offset), (280, offset)])
        assert street.obstructions(beside_it), f"{offset} m off centre should hit shopfronts"


def test_the_mattupetty_dam_reaches_the_water_it_holds():
    """The dam's height is derived from the terrain, so it has to hold up."""
    from waypoint.world import munnar

    hills = munnar()
    dam = next(b for b in hills.buildings if "dam wall" in b.name)
    lake = hills.find("Mattupetty Lake")
    assert dam.roof > hills.elevation_at(lake.x, lake.y)


# ---------------------------------------------------------------------------
# Naming a real place, where the mapper did not
# ---------------------------------------------------------------------------


def test_osm_buildings_fall_back_through_the_names_they_do_have():
    """`building=yes` is a tag value, not a name, and neither is 'building'."""
    from waypoint.world.osm import name_of

    assert name_of({"name": "Gateway of India"}, 1) == "Gateway of India"
    assert name_of({"addr:housenumber": "12", "addr:street": "Colaba Causeway"}, 1) == (
        "12 Colaba Causeway"
    )
    assert name_of({"addr:street": "Colaba Causeway"}, 1) == "a block on Colaba Causeway"
    assert name_of({"building": "apartments"}, 7) == "apartments 7"
    assert name_of({"building": "yes"}, 7) == "unnamed block 7"
    assert name_of({}, 7) == "unnamed block 7"


def test_an_imported_place_gets_landmarks_from_its_named_buildings():
    """Otherwise the planner is handed 400 footprints and no way to say which."""
    from waypoint.world.osm import build_world

    payload = {
        "elements": [
            {"type": "node", "id": 1, "lat": 0.0000, "lon": 0.0000},
            {"type": "node", "id": 2, "lat": 0.0000, "lon": 0.0009},
            {"type": "node", "id": 3, "lat": 0.0009, "lon": 0.0009},
            {"type": "node", "id": 4, "lat": 0.0009, "lon": 0.0000},
            {
                "type": "way",
                "id": 10,
                "nodes": [1, 2, 3, 4, 1],
                "tags": {"name": "Rajabai Clock Tower", "height": "85"},
            },
            {"type": "way", "id": 11, "nodes": [1, 2, 3, 4, 1], "tags": {"building": "yes"}},
        ]
    }
    place = build_world(payload, 0.0, 0.0, name="fort")

    assert [b.name for b in place.buildings] == ["Rajabai Clock Tower", "unnamed block 2"]
    assert [lm.name for lm in place.landmarks] == ["Rajabai Clock Tower"]
    assert "Rajabai Clock Tower" in place.describe()


# ---------------------------------------------------------------------------
# Landing sites: is this a patch of ground you could set down on?
# ---------------------------------------------------------------------------


def slope_world(gradient: float) -> World:
    """Ground that climbs `gradient` metres for every metre east."""
    return World(
        name="slope",
        terrain=Terrain(
            origin_x=-100,
            origin_y=-100,
            spacing=20,
            heights=[[col * 20 * gradient for col in range(11)] for _ in range(11)],
        ),
    )


def test_flat_ground_is_landable():
    from waypoint.world.landing import survey

    site = survey(World(name="flat"), 10, 10)

    assert site.usable
    assert site.slope_deg == 0
    assert site.drop == 0
    assert site.elevation == 0


def test_a_slope_is_measured_and_refused():
    from waypoint.world.landing import MAX_SLOPE_DEG, survey

    site = survey(slope_world(0.5), 0, 0)      # 1 in 2: about 27 degrees

    assert not site.usable
    assert site.slope_deg > MAX_SLOPE_DEG
    assert "too uneven" in site.why_not() or "slopes" in site.why_not()
    assert f"{site.drop:.1f}" in site.why_not()


def test_a_gentle_slope_is_still_landable():
    from waypoint.world.landing import survey

    site = survey(slope_world(0.02), 0, 0)     # about 1 degree

    assert site.usable
    assert site.slope_deg < 3


def test_the_elevation_is_the_ground_under_the_pad():
    from waypoint.world.landing import survey

    site = survey(slope_world(0.05), 40, 0)

    assert site.elevation == pytest.approx(slope_world(0.05).elevation_at(40, 0), abs=0.01)
    assert site.position == (site.x, site.y, site.elevation)


def test_standing_on_a_building_is_not_a_site(tower_world):
    from waypoint.world.landing import survey

    site = survey(tower_world, 0, 100)

    assert not site.usable
    assert site.obstructed == "the tower"
    assert "standing there" in site.why_not()


def test_an_unusable_pick_moves_to_the_nearest_ground_that_works():
    from waypoint.world.landing import find

    # Flat everywhere except a steep ridge down the middle.
    profile = [0.0, 0.0, 30.0, 0.0, 0.0]
    ridged = World(
        name="ridge",
        terrain=Terrain(origin_x=-100, origin_y=-100, spacing=50, heights=[profile] * 5),
    )
    site = find(ridged, 0, 0)               # on the ridge

    assert site.usable
    assert site.moved > 0
    assert "moved" in site.describe()


def test_a_usable_pick_is_left_where_it_was_asked_for():
    from waypoint.world.landing import find

    site = find(World(name="flat"), 25, -40)

    assert site.usable
    assert (site.x, site.y) == (25, -40)
    assert site.moved == 0


def test_nowhere_to_land_comes_back_unusable_rather_than_wrong():
    from waypoint.world.landing import find

    site = find(slope_world(0.5), 0, 0, search=40, step=10)

    assert not site.usable
    assert site.moved == 0          # it did not pretend to find somewhere


# --- and the verifier refuses to fly a plan that lands on one --------------


def test_a_plan_that_lands_on_a_slope_is_rejected(constraints):
    steep = slope_world(0.5)
    plan = plan_of(wp(0, 0, 40, 3), wp(0, 0, steep.elevation_at(0, 0), 2, Action.LAND))
    result = verify(plan, constraints, world=steep)

    landing = result.of_type(ViolationType.LANDING)
    assert landing
    assert landing[0].severity.value == "CRITICAL"
    assert "cannot be landed on" in landing[0].description


def test_a_land_waypoint_hovering_above_the_ground_is_rejected(constraints):
    ground = World(
        name="shelf",
        terrain=Terrain(origin_x=-100, origin_y=-100, spacing=100, heights=[[12.0] * 3] * 3),
    )
    plan = plan_of(wp(0, 0, 40, 3), wp(0, 0, 25, 2, Action.LAND))     # 13 m up
    result = verify(plan, constraints, world=ground)

    landing = result.of_type(ViolationType.LANDING)
    assert landing
    assert "above the ground" in landing[0].description
    assert "z = 12" in landing[0].description


def test_landing_properly_on_level_ground_passes(constraints):
    ground = World(
        name="shelf",
        terrain=Terrain(origin_x=-100, origin_y=-100, spacing=100, heights=[[12.0] * 3] * 3),
    )
    plan = plan_of(wp(0, 0, 40, 3), wp(0, 0, 12, 2, Action.LAND))
    assert verify(plan, constraints, world=ground).of_type(ViolationType.LANDING) == []


def test_without_a_world_landing_is_not_second_guessed(constraints):
    """No terrain, no opinion: the checks that need ground are skipped."""
    plan = plan_of(wp(0, 0, 40, 3), wp(0, 0, 0, 2, Action.LAND))
    assert verify(plan, constraints).of_type(ViolationType.LANDING) == []
