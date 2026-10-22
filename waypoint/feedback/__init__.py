"""In-flight feedback: watch the telemetry, see what is around, and say when the plan has to change."""

from waypoint.feedback.gazebo_camera import GazeboCameraPerceptionSource
from waypoint.feedback.models import FeedbackConfig, ReplanTrigger, TelemetrySnapshot, TriggerKind
from waypoint.feedback.monitor import (
    battery_drain_ratio,
    estimated_battery_need,
    needs_replan,
    path_deviation,
    remaining_route,
    summarize_for_llm,
)
from waypoint.feedback.perception import (
    PerceptionEvent,
    PerceptionKind,
    PerceptionSeverity,
    PerceptionSource,
    SymbolicPerceptionSource,
)

__all__ = [
    "FeedbackConfig",
    "GazeboCameraPerceptionSource",
    "PerceptionEvent",
    "PerceptionKind",
    "PerceptionSeverity",
    "PerceptionSource",
    "ReplanTrigger",
    "SymbolicPerceptionSource",
    "TelemetrySnapshot",
    "TriggerKind",
    "battery_drain_ratio",
    "estimated_battery_need",
    "needs_replan",
    "path_deviation",
    "remaining_route",
    "summarize_for_llm",
]
