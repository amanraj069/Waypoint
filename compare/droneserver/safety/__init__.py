"""DroneServer's safety and security layer.

Modules
-------
``config``      SafetySettings - every limit and per-component switch   (vendored)
``tiers``       criticality table (read_only / normal / critical / emergency)   (vendored)
``validation``  parameter bounds, state preconditions, rate limiting   (vendored)
``geofence``    server-side fence (pure geometry), independent of firmware   (vendored)
``tokens``      single-use confirmation tokens for critical tools   (vendored)
``auth``        API keys and scopes   (vendored)
``audit``       append-only JSONL log   (vendored)
``state``       cached vehicle-state snapshot used by preconditions   (adapted to the simulated vehicle)
``middleware``  the pipeline; wraps every tool at registration   (adapted: one layer per session)
"""

from compare.droneserver.safety.config import SafetySettings
from compare.droneserver.safety.middleware import SafetyLayer, Session, guard
from compare.droneserver.safety.tiers import TOOL_TIERS, Tier, effective_tier

__all__ = ["SafetyLayer", "SafetySettings", "Session", "TOOL_TIERS", "Tier", "effective_tier", "guard"]
