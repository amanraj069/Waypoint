"""PEACE's constraint enforcement service (peace/services/constraint_enforcement_service.py).

The paper, section III-F: "A dedicated safety layer validates every motion
command before it reaches the autopilot, enforcing two geometric constraints:
an altitude band and a horizontal geofence around the takeoff point.
Out-of-band altitude targets are clamped; geofence breaches abort the
offending tool calls and surface a structured failure to the replanner."

The fence is a square, +-`horizontal_limit` in x and in y from home, not a
circle: that is how the released code checks it, and how its system prompt
states it.
"""

from __future__ import annotations

from compare.peace.exceptions import SafetyViolationError


class ConstraintEnforcementService:
    """Validates and clamps positions against configurable constraint limits.

    Raises SafetyViolationError for hard violations (position out of geofence).
    Provides clamp_altitude() for soft violations (altitude adjustments).
    """

    def __init__(self, min_altitude: float, max_altitude: float, horizontal_limit: float) -> None:
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.horizontal_limit = horizontal_limit

    def validate_position(self, x: float, y: float, z: float) -> None:
        """Raise SafetyViolationError if (x, y) violates the horizontal geofence.

        Altitude is treated as a soft constraint: callers run the value through
        clamp_altitude() before validate_position(), so the altitude bounds are
        not re-checked here.
        """
        if abs(x) > self.horizontal_limit or abs(y) > self.horizontal_limit:
            raise SafetyViolationError(
                f"Position ({x:.1f}, {y:.1f}) exceeds ±{self.horizontal_limit}m geofence limit"
            )

    def clamp_altitude(self, z: float) -> float:
        """Return altitude clamped to the safe range."""
        return max(self.min_altitude, min(self.max_altitude, z))
