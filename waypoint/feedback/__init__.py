"""In-flight feedback: watch the telemetry, and say when the plan has to change."""

from waypoint.feedback.models import FeedbackConfig, ReplanTrigger, TelemetrySnapshot, TriggerKind
from waypoint.feedback.monitor import (
    estimated_battery_need,
    needs_replan,
    remaining_route,
    summarize_for_llm,
)

__all__ = [
    "FeedbackConfig",
    "ReplanTrigger",
    "TelemetrySnapshot",
    "TriggerKind",
    "estimated_battery_need",
    "needs_replan",
    "remaining_route",
    "summarize_for_llm",
]
