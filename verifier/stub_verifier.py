"""Stub verifier.

Stands in for the deterministic safety verifier (geofence, altitude envelope,
battery margin, no-fly zones). Always passes for now.
"""

from __future__ import annotations

from planner.models import Plan
from verifier.models import VerificationResult


def verify(plan: Plan) -> VerificationResult:
    """Always report the plan as safe. Real checks land here later."""
    return VerificationResult(passed=True, violations=[])
