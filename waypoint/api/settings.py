"""Where things live, and what the service is allowed to talk to."""

from __future__ import annotations

import os

from waypoint.paths import FAULTS_DIR, PLANS_DIR, RUNS_DIR, WORLDS_DIR  # noqa: F401 -- re-exported

#: The dev server, and whatever else the operator names. The API is a local
#: tool, so the default list is localhost and nothing else.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "WAYPOINT_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001",
    ).split(",")
    if origin.strip()
]

#: How many missions may be in the air at once. Planning is a network call and
#: flying is CPU-bound, so a small number keeps one user's mission from waiting
#: behind somebody else's while still not thrashing the machine.
MAX_CONCURRENT_MISSIONS = int(os.environ.get("WAYPOINT_MAX_MISSIONS", "3"))

#: Missions are kept in memory so their event log can be replayed; this is how
#: many of the finished ones to hold on to.
MISSION_HISTORY = 50
