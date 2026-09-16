"""Telemetry / perception feedback types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Feedback(BaseModel):
    """What the feedback layer concluded from one tick of telemetry."""

    replan_required: bool = False
    reasons: list[str] = Field(default_factory=list)
