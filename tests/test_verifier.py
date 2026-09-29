"""Verifier tests.

Every plan here is hand-built, so each assertion says exactly which violations
a known plan must produce -- and, just as importantly, which it must not.

Most tests run with `NO_TURN_MODEL`, a config whose turn limit never bites.
An out-and-back route reverses direction at its far end, which is a real (if
mild) turn finding; switching the turn model off keeps the other tests asserting
on exactly the violation they are about. The turn tests use the real defaults,
and the good-plan test runs under both.
"""

from __future__ import annotations

import pytest

from waypoint.planner import Action, GeoPoint, MissionConstraints, Obstacle, Plan, SubTask, Waypoint
from waypoint.verifier import (
    DEFAULT_CONFIG,
    Severity,
    VerificationResult,
    VerifierConfig,
    Violation,
    ViolationType,
    verify,
)

# A turn model that never fires: every turn is allowed at every speed.
NO_TURN_MODEL = VerifierConfig(max_turn_angle_deg=180.0, turn_reference_speed=1.0e6)


def wp(x: float, y: float, z: float, speed: float = 4.0, action: Action = Action.MOVE) -> Waypoint:
    return Waypoint(x=x, y=y, z=z, speed=speed, action=action)


def plan_of(*waypoints: Waypoint, name: str = "mission") -> Plan:
    """A one-subtask plan holding exactly the waypoints given."""
    return Plan(mission_text="test mission", subtasks=[SubTask(name=name, waypoints=list(waypoints))])


def circuit(*, corner: Waypoint | None = None) -> Plan:
    """The reference good route: climb, fly a square, come home, land.

    Every turn is 90 degrees and nothing reverses, so the route is clean under
    the default turn model too. `corner` replaces the far corner (index 2),
    which is how each bad-plan test injects exactly one fault.
    """
    return Plan(
        mission_text="fly the square",
        subtasks=[
            SubTask(name="takeoff", waypoints=[wp(0, 0, 20, 3)]),
            SubTask(
                name="survey",
                waypoints=[
                    wp(80, 0, 20, 6),
                    corner if corner is not None else wp(80, 80, 20, 4, Action.CAPTURE),
                    wp(0, 80, 20, 6),
                ],
            ),
            SubTask(
                name="return_and_land",
                waypoints=[wp(0, 0, 20, 6), wp(0, 0, 0, 2, Action.LAND)],
            ),
        ],
    )


def types_of(result: VerificationResult) -> list[ViolationType]:
    return [v.type for v in result.violations]


@pytest.fixture
def good_plan() -> Plan:
    return circuit()


# ---------------------------------------------------------------------------
# The good plan: no false positives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [DEFAULT_CONFIG, NO_TURN_MODEL], ids=["defaults", "no-turn"])
def test_good_plan_produces_no_violations(good_plan, constraints, config):
    result = verify(good_plan, constraints, config)
    assert result.violations == []
    assert result.passed is True


def test_good_plan_stays_clean_with_zones_and_obstacles_out_of_the_way(good_plan, constraints):
    envelope = constraints.model_copy(
        update={
            "no_fly_zones": [square(-40, -40, -20, -20)],
            "obstacles": [Obstacle(x=140, y=140, radius=5, name="silo")],
            "min_separation": 10.0,
        }
    )
    result = verify(good_plan, envelope, NO_TURN_MODEL)
    assert result.violations == []
    assert result.passed is True


def test_takeoff_and_landing_may_sit_on_the_ground(constraints):
    """z = 0 is legal for the first waypoint and for a LAND, despite min_altitude."""
    assert constraints.min_altitude > 0
    plan = plan_of(
        wp(0, 0, 0, 2),
        wp(0, 0, 20, 3),
        wp(40, 0, 20, 4),
        wp(0, 0, 20, 4),
        wp(0, 0, 0, 2, Action.LAND),
    )
    result = verify(plan, constraints, NO_TURN_MODEL)
    assert result.violations == []


# ---------------------------------------------------------------------------
# Geofence
# ---------------------------------------------------------------------------


def square(x0: float, y0: float, x1: float, y1: float) -> list[GeoPoint]:
    return [
        GeoPoint(x=x0, y=y0),
        GeoPoint(x=x1, y=y0),
        GeoPoint(x=x1, y=y1),
        GeoPoint(x=x0, y=y1),
    ]


def test_waypoint_outside_the_geofence_is_caught(constraints):
    plan = circuit(corner=wp(300, 80, 20, 4))
    result = verify(plan, constraints, NO_TURN_MODEL)

    assert result.passed is False
    assert types_of(result) == [ViolationType.GEOFENCE]
    violation = result.violations[0]
    assert violation.waypoint_index == 2
    assert violation.severity is Severity.ERROR
    assert "outside the geofence" in violation.description
    assert "150" in violation.description or "m beyond" in violation.description


def test_every_out_of_bounds_waypoint_is_reported(constraints):
    plan = plan_of(wp(0, 0, 20, 3), wp(400, 0, 20), wp(0, 400, 20), wp(0, 0, 0, 2, Action.LAND))
    result = verify(plan, constraints, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.GEOFENCE, ViolationType.GEOFENCE]
    assert [v.waypoint_index for v in result.violations] == [1, 2]


def test_a_leg_that_cuts_the_corner_of_a_concave_geofence_is_caught():
    """Both ends inside an L-shaped fence, but the straight line between them is not."""
    l_shape = MissionConstraints(
        geofence=[
            GeoPoint(x=0, y=0),
            GeoPoint(x=100, y=0),
            GeoPoint(x=100, y=40),
            GeoPoint(x=40, y=40),
            GeoPoint(x=40, y=100),
            GeoPoint(x=0, y=100),
        ],
        max_altitude=100.0,
        min_altitude=5.0,
        max_velocity=10.0,
        battery_budget_pct=75.0,
    )
    plan = plan_of(wp(10, 10, 20, 3), wp(90, 10, 20), wp(10, 90, 20), wp(10, 10, 20))
    result = verify(plan, l_shape, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.GEOFENCE]
    violation = result.violations[0]
    assert violation.waypoint_index == 2  # the leg's destination
    assert "leaves the geofence" in violation.description


def test_geofence_margin_warns_without_failing_the_plan(good_plan, constraints):
    tight = constraints.model_copy(update={"geofence": square(-50, -50, 90, 150)})
    result = verify(good_plan, tight, NO_TURN_MODEL.model_copy(update={"geofence_margin": 25.0}))

    assert result.passed is True  # a warning does not reject the plan
    assert types_of(result) == [ViolationType.GEOFENCE, ViolationType.GEOFENCE]
    assert {v.severity for v in result.violations} == {Severity.WARNING}
    assert [v.waypoint_index for v in result.violations] == [1, 2]


def test_a_self_intersecting_geofence_is_rejected_as_a_bad_constraint(good_plan, constraints):
    bowtie = constraints.model_copy(
        update={
            "geofence": [
                GeoPoint(x=0, y=0),
                GeoPoint(x=100, y=100),
                GeoPoint(x=100, y=0),
                GeoPoint(x=0, y=100),
            ]
        }
    )
    with pytest.raises(ValueError, match="geofence is not a valid polygon"):
        verify(good_plan, bowtie)


# ---------------------------------------------------------------------------
# No-fly zones
# ---------------------------------------------------------------------------


def test_waypoint_inside_a_no_fly_zone_is_caught(constraints):
    envelope = constraints.model_copy(update={"no_fly_zones": [square(60, 60, 100, 100)]})
    result = verify(circuit(), envelope, NO_TURN_MODEL)

    assert result.passed is False
    assert types_of(result) == [ViolationType.NO_FLY_ZONE]
    violation = result.violations[0]
    assert violation.waypoint_index == 2
    assert violation.severity is Severity.CRITICAL
    assert "inside no-fly zone 1" in violation.description


def test_a_leg_crossing_a_no_fly_zone_is_caught(constraints):
    """Neither end is in the zone, but the straight line between them is."""
    envelope = constraints.model_copy(update={"no_fly_zones": [square(30, -10, 50, 10)]})
    result = verify(circuit(), envelope, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.NO_FLY_ZONE]
    violation = result.violations[0]
    assert violation.waypoint_index == 1  # leg from the climb waypoint to (80, 0)
    assert "crosses no-fly zone 1" in violation.description
    assert "Route around" in violation.description


def test_each_no_fly_zone_is_numbered_in_its_own_feedback(constraints):
    envelope = constraints.model_copy(
        update={"no_fly_zones": [square(30, -10, 50, 10), square(60, 60, 100, 100)]}
    )
    result = verify(circuit(), envelope, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.NO_FLY_ZONE, ViolationType.NO_FLY_ZONE]
    assert "no-fly zone 1" in result.violations[0].description
    assert "no-fly zone 2" in result.violations[1].description


def test_a_zone_the_route_never_touches_is_not_flagged(good_plan, constraints):
    envelope = constraints.model_copy(update={"no_fly_zones": [square(-45, -45, -25, -25)]})
    assert verify(good_plan, envelope, NO_TURN_MODEL).violations == []


# ---------------------------------------------------------------------------
# Altitude
# ---------------------------------------------------------------------------


def test_waypoint_above_the_ceiling_is_caught(constraints):
    result = verify(circuit(corner=wp(80, 80, 120, 4)), constraints, NO_TURN_MODEL)

    assert result.passed is False
    assert types_of(result) == [ViolationType.ALTITUDE]
    violation = result.violations[0]
    assert violation.waypoint_index == 2
    assert "above the 100 m ceiling" in violation.description


def test_waypoint_below_the_floor_is_caught(constraints):
    result = verify(circuit(corner=wp(80, 80, 2, 4)), constraints, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.ALTITUDE]
    assert result.violations[0].waypoint_index == 2
    assert "below the 5 m minimum altitude" in result.violations[0].description


def test_a_mid_route_waypoint_on_the_ground_is_caught(constraints):
    """z = 0 is only excused for the launch waypoint and for a LAND."""
    result = verify(circuit(corner=wp(80, 80, 0, 4)), constraints, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.ALTITUDE]
    assert result.violations[0].waypoint_index == 2


def test_negative_altitude_is_caught(constraints):
    result = verify(circuit(corner=wp(80, 80, -10, 4)), constraints, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.ALTITUDE]
    assert "negative altitude" in result.violations[0].description


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------


def test_commanded_speed_above_the_limit_is_caught(constraints):
    result = verify(circuit(corner=wp(80, 80, 20, 25)), constraints, NO_TURN_MODEL)

    assert result.passed is False
    assert types_of(result) == [ViolationType.MAX_VELOCITY]
    violation = result.violations[0]
    assert violation.waypoint_index == 2
    assert "25 m/s" in violation.description
    assert "10 m/s limit" in violation.description


def test_a_steep_climb_implies_a_speed_above_the_limit(constraints):
    """The commanded speed is legal, but it is a *ground* speed on a steep leg."""
    plan = plan_of(
        wp(0, 0, 20, 3),
        wp(5, 0, 100, 9),
        wp(100, 0, 100, 4),
        wp(0, 0, 20, 4),
        wp(0, 0, 0, 2, Action.LAND),
    )
    result = verify(plan, constraints, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.MAX_VELOCITY]
    violation = result.violations[0]
    assert violation.waypoint_index == 1
    assert "implies" in violation.description
    assert "Slow this waypoint" in violation.description


def test_a_shallow_climb_at_a_legal_speed_is_fine(constraints):
    plan = plan_of(wp(0, 0, 20, 3), wp(100, 0, 60, 8), wp(0, 0, 20, 8), wp(0, 0, 0, 2, Action.LAND))
    assert verify(plan, constraints, NO_TURN_MODEL).violations == []


# ---------------------------------------------------------------------------
# Turn feasibility
# ---------------------------------------------------------------------------


def sharp_turn_plan(speed: float, action: Action = Action.MOVE) -> Plan:
    """A 90 degree turn at (60, 0) flown at `speed`, then a slow turn home."""
    return plan_of(
        wp(0, 0, 20, 3),
        wp(60, 0, 20, speed, action),
        wp(60, 60, 20, 3),
        wp(0, 0, 20, 3),
        wp(0, 0, 0, 2, Action.LAND),
    )


def test_a_turn_too_sharp_for_the_speed_is_flagged(constraints):
    result = verify(sharp_turn_plan(speed=10.0), constraints, DEFAULT_CONFIG)

    assert types_of(result) == [ViolationType.TURN_FEASIBILITY]
    violation = result.violations[0]
    assert violation.waypoint_index == 1
    assert violation.severity is Severity.WARNING
    assert "turns 90 deg" in violation.description
    assert "Slow to at most" in violation.description
    # A warning is reported but does not by itself reject the plan.
    assert result.passed is True


def test_the_same_turn_flown_slowly_is_fine(constraints):
    assert verify(sharp_turn_plan(speed=4.0), constraints, DEFAULT_CONFIG).violations == []


def test_a_turn_at_a_hover_waypoint_is_free(constraints):
    """The aircraft stops to hover, so it can turn on the spot."""
    result = verify(sharp_turn_plan(speed=10.0, action=Action.HOVER), constraints, DEFAULT_CONFIG)
    assert result.violations == []


def test_a_turn_no_speed_can_fly_says_so(constraints):
    plan = plan_of(
        wp(0, 0, 20, 3),
        wp(60, 0, 20, 3),
        wp(0, 10, 20, 3),
        wp(0, 0, 20, 3),
        wp(0, 0, 0, 2, Action.LAND),
    )
    result = verify(plan, constraints, DEFAULT_CONFIG)

    assert types_of(result) == [ViolationType.TURN_FEASIBILITY]
    assert result.violations[0].waypoint_index == 1
    assert "No speed makes a turn that sharp flyable" in result.violations[0].description


def test_the_turn_limit_is_configurable(constraints):
    strict = VerifierConfig(max_turn_angle_deg=45.0, turn_reference_speed=100.0)
    result = verify(sharp_turn_plan(speed=4.0), constraints, strict)

    # Both corners of the route are sharper than 45 deg now.
    assert set(types_of(result)) == {ViolationType.TURN_FEASIBILITY}
    assert [v.waypoint_index for v in result.violations] == [1, 2]
    assert "can only turn 45 deg" in result.violations[0].description


# ---------------------------------------------------------------------------
# Battery / range
# ---------------------------------------------------------------------------


def test_a_route_beyond_the_battery_budget_is_rejected(good_plan, constraints):
    thirsty = constraints.model_copy(update={"battery_budget_pct": 5.0})
    result = verify(good_plan, thirsty, NO_TURN_MODEL)

    assert result.passed is False
    assert types_of(result) == [ViolationType.BATTERY]
    violation = result.violations[0]
    assert violation.waypoint_index is None  # a whole-plan finding
    assert violation.severity is Severity.ERROR
    assert "360 m" in violation.description  # the route length it measured
    assert "exceeds the 5% budget" in violation.description
    assert "Shorten the route by at least" in violation.description


def test_the_energy_model_is_configurable(good_plan, constraints):
    thirsty = constraints.model_copy(update={"battery_budget_pct": 5.0})
    efficient = NO_TURN_MODEL.model_copy(update={"meters_per_battery_pct": 200.0})

    assert verify(good_plan, thirsty, efficient).violations == []


def test_hovering_costs_battery(constraints):
    """Two plans, same route: the one that hovers needs more battery."""
    route = [wp(0, 0, 20, 3), wp(100, 0, 20, 5), wp(0, 0, 20, 5), wp(0, 0, 0, 2, Action.LAND)]
    hovering = list(route)
    hovering[1] = wp(100, 0, 20, 5, Action.HOVER)

    config = NO_TURN_MODEL.model_copy(
        update={"hover_seconds_per_waypoint": 600.0}  # 20% of battery at the default rate
    )
    budget = constraints.model_copy(update={"battery_budget_pct": 10.0})

    assert verify(plan_of(*route), budget, config).violations == []
    result = verify(plan_of(*hovering), budget, config)
    assert types_of(result) == [ViolationType.BATTERY]
    assert "hovers 600 s" in result.violations[0].description


def test_a_reserve_requirement_is_honoured(good_plan, constraints):
    budget = constraints.model_copy(update={"battery_budget_pct": 10.0})
    assert verify(good_plan, budget, NO_TURN_MODEL).violations == []

    with_reserve = NO_TURN_MODEL.model_copy(update={"reserve_battery_pct": 20.0})
    result = verify(good_plan, budget, with_reserve)
    assert types_of(result) == [ViolationType.BATTERY]
    assert "20% reserve" in result.violations[0].description


# ---------------------------------------------------------------------------
# Minimum separation
# ---------------------------------------------------------------------------


def with_obstacle(constraints: MissionConstraints, obstacle: Obstacle, separation: float):
    return constraints.model_copy(update={"obstacles": [obstacle], "min_separation": separation})


def test_a_waypoint_inside_an_obstacle_keep_out_is_caught(constraints):
    envelope = with_obstacle(constraints, Obstacle(x=78, y=80, radius=2, name="crane"), 10.0)
    result = verify(circuit(), envelope, NO_TURN_MODEL)

    assert result.passed is False
    assert types_of(result) == [ViolationType.MIN_SEPARATION]
    violation = result.violations[0]
    assert violation.waypoint_index == 2
    assert violation.severity is Severity.CRITICAL
    assert "'crane'" in violation.description
    assert "keep-out radius" in violation.description


def test_a_leg_that_shaves_an_obstacle_is_caught(constraints):
    """Both waypoints are clear; the straight line between them is not."""
    envelope = with_obstacle(constraints, Obstacle(x=80, y=40, radius=5, name="mast"), 10.0)
    result = verify(circuit(), envelope, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.MIN_SEPARATION]
    violation = result.violations[0]
    assert violation.waypoint_index == 2  # the leg's destination
    assert "passes within" in violation.description


def test_an_obstacle_the_route_keeps_clear_of_is_not_flagged(good_plan, constraints):
    envelope = with_obstacle(constraints, Obstacle(x=140, y=40, radius=5, name="mast"), 10.0)
    assert verify(good_plan, envelope, NO_TURN_MODEL).violations == []


def test_flying_over_a_short_obstacle_is_allowed(good_plan, constraints):
    envelope = with_obstacle(
        constraints, Obstacle(x=80, y=40, radius=5, height=10, name="shed"), 10.0
    )
    assert verify(good_plan, envelope, NO_TURN_MODEL).violations == []


def test_an_obstacle_taller_than_the_route_still_bites(good_plan, constraints):
    envelope = with_obstacle(
        constraints, Obstacle(x=80, y=40, radius=5, height=30, name="tower"), 10.0
    )
    result = verify(good_plan, envelope, NO_TURN_MODEL)

    assert types_of(result) == [ViolationType.MIN_SEPARATION]
    assert "fly above its 30 m top" in result.violations[0].description


def test_separation_is_the_obstacle_radius_plus_the_margin(constraints):
    """The route passes 28 m from a 2 m obstacle: clear at 5 m separation, not at 30 m."""
    obstacle = Obstacle(x=100, y=100, radius=2, name="pole")

    clear = verify(circuit(), with_obstacle(constraints, obstacle, 5.0), NO_TURN_MODEL)
    assert clear.violations == []

    result = verify(circuit(), with_obstacle(constraints, obstacle, 30.0), NO_TURN_MODEL)
    assert set(types_of(result)) == {ViolationType.MIN_SEPARATION}
    assert "32.0 m" in result.violations[0].description  # 2 m radius + 30 m separation


def test_obstacles_without_a_keep_out_radius_are_ignored(good_plan, constraints):
    envelope = with_obstacle(constraints, Obstacle(x=80, y=40, name="survey marker"), 0.0)
    assert verify(good_plan, envelope, NO_TURN_MODEL).violations == []


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------


def test_every_violation_in_a_thoroughly_bad_plan_is_reported(constraints):
    envelope = constraints.model_copy(
        update={
            "no_fly_zones": [square(30, -10, 50, 10)],
            "obstacles": [Obstacle(x=200, y=200, radius=5, name="mast")],
            "min_separation": 20.0,
            "battery_budget_pct": 6.0,
        }
    )
    plan = plan_of(
        wp(0, 0, 20, 3),
        wp(300, 0, 20, 25),  # outside the fence, too fast, past the no-fly zone
        wp(200, 200, 200, 4),  # above the ceiling, on top of the obstacle, outside the fence
        wp(0, 0, 0, 2, Action.LAND),
    )
    result = verify(plan, envelope, NO_TURN_MODEL)

    assert result.passed is False
    assert set(types_of(result)) == {
        ViolationType.GEOFENCE,
        ViolationType.NO_FLY_ZONE,
        ViolationType.ALTITUDE,
        ViolationType.MAX_VELOCITY,
        ViolationType.BATTERY,
        ViolationType.MIN_SEPARATION,
    }


def test_the_verifier_is_deterministic(constraints):
    envelope = constraints.model_copy(update={"no_fly_zones": [square(30, -10, 50, 10)]})
    plan = circuit(corner=wp(300, 80, 250, 25))

    first = verify(plan, envelope)
    second = verify(plan, envelope)
    assert first.model_dump() == second.model_dump()


def test_violations_read_as_planner_feedback(constraints):
    result = verify(circuit(corner=wp(300, 80, 20, 4)), constraints, NO_TURN_MODEL)

    assert result.messages() == [v.description for v in result.violations]
    assert str(result.violations[0]) == result.violations[0].description
    # Enough context to act on without seeing the plan again.
    assert "waypoint 2" in result.violations[0].description
    assert 'subtask "survey"' in result.violations[0].description


def test_result_partitions_blocking_violations_from_warnings(constraints):
    plan = circuit(corner=wp(300, 80, 20, 10))
    tight = constraints.model_copy(update={"geofence": square(-50, -50, 400, 150)})
    result = verify(plan, tight, DEFAULT_CONFIG.model_copy(update={"geofence_margin": 25.0}))

    assert [v.severity for v in result.blocking] == [Severity.ERROR] * len(result.blocking)
    assert result.warnings
    assert all(not v.blocking for v in result.warnings)
    assert result.violations == result.blocking + result.warnings or set(
        result.blocking + result.warnings
    ) == set(result.violations)


def test_of_type_selects_one_kind_of_violation(constraints):
    result = verify(circuit(corner=wp(300, 80, 250, 4)), constraints, NO_TURN_MODEL)

    assert len(result.of_type(ViolationType.GEOFENCE)) == 1
    assert len(result.of_type(ViolationType.ALTITUDE)) == 1
    assert result.of_type(ViolationType.BATTERY) == []


def test_verification_result_defaults_to_no_violations():
    assert VerificationResult(passed=True).violations == []


def test_verification_result_carries_violations():
    violation = Violation(
        type=ViolationType.ALTITUDE,
        waypoint_index=3,
        description="altitude ceiling exceeded",
        severity=Severity.ERROR,
    )
    result = VerificationResult(passed=False, violations=[violation])

    assert result.passed is False
    assert result.violations[0].description == "altitude ceiling exceeded"
    assert result.violations[0].blocking is True


def test_a_violation_round_trips_through_json():
    violation = Violation(
        type=ViolationType.NO_FLY_ZONE, waypoint_index=1, description="in the zone"
    )
    assert Violation.model_validate_json(violation.model_dump_json()) == violation
