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

from planner.models import MissionConstraints, Plan
from verifier.checks import (
    check_altitude,
    check_battery,
    check_geofence,
    check_no_fly_zones,
    check_separation,
    check_turns,
    check_velocity,
    flatten,
    legs_of,
)
from verifier.models import DEFAULT_CONFIG, VerificationResult, VerifierConfig, Violation


def verify(
    plan: Plan,
    constraints: MissionConstraints,
    config: VerifierConfig | None = None,
) -> VerificationResult:
    """Check a plan against the safety envelope.

    Runs every check -- the report lists all of a plan's problems, not just the
    first one, so one replan can fix them together. Raises ValueError only if
    the *constraints* are unusable (e.g. a self-intersecting geofence); a bad
    plan produces violations, never an exception.
    """
    config = config or DEFAULT_CONFIG
    flat = flatten(plan)
    legs = legs_of(flat, config)

    violations: list[Violation] = [
        *check_geofence(flat, legs, constraints, config),
        *check_no_fly_zones(flat, legs, constraints),
        *check_altitude(flat, constraints),
        *check_velocity(legs, constraints),
        *check_turns(flat, config),
        *check_battery(legs, flat, constraints, config),
        *check_separation(flat, legs, constraints),
    ]

    return VerificationResult(
        passed=not any(v.blocking for v in violations),
        violations=violations,
    )
