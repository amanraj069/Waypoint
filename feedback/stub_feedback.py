"""Stub feedback monitor.

Stands in for telemetry ingestion and perception. Real version will watch for
battery margin loss, wind, GPS degradation and newly detected obstacles, and
trigger autonomous replanning. For now it never asks for a replan.
"""

from __future__ import annotations

from feedback.models import Feedback
from sim.state import MissionState


def observe(state: MissionState) -> Feedback:
    """Inspect one tick of telemetry and decide whether to replan."""
    return Feedback(replan_required=False, reasons=[])
