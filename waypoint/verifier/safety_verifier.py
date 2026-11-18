"""The deterministic safety verifier.

`verify()` is the only gate between the LLM planner and the aircraft. It is
plain geometry and arithmetic -- given the same plan and the same constraints
it returns the same violations, every time, with no model in the loop.

    result = verify(plan, constraints)
    if not result.passed:
        plan = replan(mission_text, plan, result.messages(), constraints)

Every violation's `description` is written to be handed straight back to the
planner as the reason its plan was rejected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from waypoint.planner.models import MissionConstraints, Plan
from waypoint.verifier.checks import (
    check_altitude,
    check_battery,
    check_buildings,
    check_geofence,
    check_landing,
    check_no_fly_zones,
    check_separation,
    check_terrain,
    check_turns,
    check_velocity,
    flatten,
    legs_of,
)
from waypoint.verifier.models import DEFAULT_CONFIG, VerificationResult, VerifierConfig, Violation
from waypoint.world.models import World

if TYPE_CHECKING:
    from waypoint.intent.models import MissionIntent


def verify(
    plan: Plan,
    constraints: MissionConstraints,
    config: VerifierConfig | None = None,
    world: World | None = None,
    intent: "MissionIntent | None" = None,
) -> VerificationResult:
    """Check a plan against the safety envelope.

    Runs every check -- the report lists all of a plan's problems, not just the
    first one, so one replan can fix them together. Raises ValueError only if
    the *constraints* are unusable (e.g. a self-intersecting geofence); a bad
    plan produces violations, never an exception.

    Pass a `world` to also check the route against the ground and the buildings
    standing on it; with none, the checks that need an environment are skipped.

    Pass an `intent` to also check that the plan does what the mission asked
    (`waypoint.intent.check`): its violations are INTENT, blocking like the
    safety ones, and listed after them.
    """
    config = config or DEFAULT_CONFIG
    flat = flatten(plan)
    legs = legs_of(flat, config)

    violations: list[Violation] = [
        *check_geofence(flat, legs, constraints, config),
        *check_no_fly_zones(flat, legs, constraints),
        *check_altitude(flat, constraints, world),
        *check_velocity(legs, constraints),
        *check_turns(flat, config),
        *check_battery(legs, flat, constraints, config),
        *check_separation(flat, legs, constraints),
        *check_buildings(flat, legs, world, config),
        *check_terrain(flat, legs, world, config),
        *check_landing(flat, world, config),
    ]
    if intent is not None:
        from waypoint.intent.check import check_intent  # noqa: PLC0415 -- intent imports the verifier

        violations += check_intent(plan, intent, constraints=constraints, world=world, config=config)

    return VerificationResult(
        passed=not any(v.blocking for v in violations),
        violations=violations,
    )
