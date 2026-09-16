"""Result type returned by the deterministic safety verifier."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    passed: bool
    violations: list[str] = Field(default_factory=list)
