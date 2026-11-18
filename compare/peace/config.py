"""PEACE's settings (peace/core/config.py), with the released defaults.

Only the settings that shape the agent's behaviour are kept; the ROS topic
names and the Ollama endpoint have no meaning here. Every value below is the
default in the released code, except `action_enable_control`, which the
release ships False as a safety override ("set to true to allow MAVROS
commands") and which every flight needs True.

`PeaceSettings.from_env()` reads the same variable names the release reads
from its .env (AGENT_MAX_REPLAN_DEPTH, ACTION_MAX_ALTITUDE, ...).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class PeaceSettings(BaseModel):
    model_config = {"extra": "forbid"}

    # Agent
    agent_name: str = "PEACE"
    agent_max_replan_depth: int = Field(
        default=2, description="Maximum number of full-replan cycles per mission (prevents infinite replan loops)"
    )
    agent_enable_memory: bool = Field(default=True, description="Include robot state history in the LLM prompt")
    agent_memory_window: int = Field(default=6, description="Number of past snapshots to include in the LLM prompt")

    # Language model
    language_model: str = Field(default="qwen2.5:14b", description="The model the release plans with, through Ollama")
    language_model_temperature: float = 0.7
    language_model_num_predict: int = 2000
    language_model_timeout: int = 120

    # Vision
    use_yolo: bool = False
    vision_inference_rate: float = Field(default=0.2, description="Vision inference rate in Hz")
    vision_confidence_threshold: float = Field(default=0.6, description="Confidence threshold for detections (0-1)")
    yolo_confidence_threshold: float = 0.5
    max_depth: float = Field(default=100.0, description="Maximum depth distance in metres")
    min_depth: float = Field(default=0.1, description="Minimum depth distance in metres")

    # Logging
    log_enable: bool = True
    log_output_dir: str = "./logs"

    # Context
    language_context_window: int = Field(default=10, description="Snapshots kept in the context window")
    language_max_detections: int = Field(default=10, description="Maximum detections included in context")

    # Action
    action_enable_control: bool = Field(default=True, description="Enable flight control commands")
    action_waypoint_tolerance: float = 0.5
    action_max_altitude: float = Field(default=50.0, description="Maximum altitude limit in metres")
    action_min_altitude: float = Field(default=0.5, description="Minimum altitude limit in metres")
    action_horizontal_limit: float = Field(
        default=500.0, description="Maximum horizontal distance from home in metres (+-x and +-y)"
    )
    action_command_timeout: float = Field(default=30.0, description="Timeout for command execution in seconds")
    action_setpoint_stream_rate: float = 10.0
    action_status_timer_interval: float = 5.0

    @classmethod
    def from_env(cls, **overrides) -> "PeaceSettings":
        """The defaults, overridden by the release's environment variables, then by `overrides`."""
        values: dict = {}
        for name, info in cls.model_fields.items():
            raw = os.environ.get(name.upper())
            if raw is None or not raw.strip():
                continue
            kind = info.annotation
            if kind is bool:
                values[name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif kind is int:
                values[name] = int(raw)
            elif kind is float:
                values[name] = float(raw)
            else:
                values[name] = raw
        values.update(overrides)
        return cls(**values)
