"""Shared test fixtures.

Nothing in the suite is allowed to reach the network: `no_network` replaces the
Gemini client constructor with a tripwire, so a test that forgets to inject a
fake client fails loudly instead of quietly dialling out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from google import genai

from waypoint.planner import GeoPoint, MissionConstraints


@dataclass
class FakeError:
    code: str | None = None
    message: str | None = None


@dataclass
class FakeInteraction:
    """Stands in for google.genai.interactions.Interaction."""

    output_text: str | None = None
    status: str = "completed"
    errors: list[FakeError] = field(default_factory=list)


class FakeClient:
    """Stands in for genai.Client, recording the request it was given."""

    def __init__(self, *interactions: FakeInteraction):
        self._interactions = list(interactions)
        self.calls: list[dict] = []
        self.interactions = _FakeInteractions(self)

    def _next(self, kwargs: dict) -> FakeInteraction:
        self.calls.append(kwargs)
        if not self._interactions:
            raise AssertionError("FakeClient received more calls than it has responses")
        return self._interactions.pop(0)

    @property
    def last_call(self) -> dict:
        return self.calls[-1]

    @property
    def last_prompt(self) -> str:
        return self.last_call["input"]


class _FakeInteractions:
    def __init__(self, client: FakeClient):
        self._client = client

    def create(self, **kwargs: Any) -> FakeInteraction:
        return self._client._next(kwargs)


def json_response(payload: Any) -> FakeInteraction:
    """A completed interaction whose body is the JSON encoding of `payload`."""
    return FakeInteraction(output_text=json.dumps(payload))


VALID_PAYLOAD = {
    "subtasks": [
        {
            "name": "takeoff",
            "waypoints": [{"x": 0.0, "y": 0.0, "z": 25.0, "speed": 3.0, "action": "MOVE"}],
        },
        {
            "name": "survey_north_field",
            "waypoints": [
                {"x": 0.0, "y": 80.0, "z": 25.0, "speed": 4.0, "action": "CAPTURE"},
                {"x": 40.0, "y": 80.0, "z": 25.0, "speed": 4.0, "action": "CAPTURE"},
            ],
        },
        {
            "name": "return_and_land",
            "waypoints": [{"x": 0.0, "y": 0.0, "z": 0.0, "speed": 2.0, "action": "LAND"}],
        },
    ]
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any test that tries to build a real Gemini client."""

    def tripwire(*args, **kwargs):
        raise AssertionError("test attempted to construct a real Gemini client")

    monkeypatch.setattr(genai, "Client", tripwire)
    monkeypatch.setattr("waypoint.planner.llm_planner._client", None)


@pytest.fixture(autouse=True)
def isolated_runs_dir(monkeypatch, tmp_path):
    """Backstop: no test may write into the project's real runs/ directory."""
    monkeypatch.setattr("waypoint.orchestrator.loop.RUNS_DIR", tmp_path / "runs")


@pytest.fixture
def constraints() -> MissionConstraints:
    return MissionConstraints(
        geofence=[
            GeoPoint(x=-50.0, y=-50.0),
            GeoPoint(x=150.0, y=-50.0),
            GeoPoint(x=150.0, y=150.0),
            GeoPoint(x=-50.0, y=150.0),
        ],
        max_altitude=100.0,
        min_altitude=5.0,
        max_velocity=10.0,
        battery_budget_pct=75.0,
    )


@pytest.fixture
def sample_plan():
    """A valid Plan built locally -- for tests of sim/verifier that need a plan,
    not of the planner that produces one."""
    from waypoint.planner import Plan

    return Plan.model_validate({**VALID_PAYLOAD, "mission_text": "survey the north field"})


def http_error(status: int, message: str = "boom"):
    """Build a real SDK error carrying the given HTTP status."""
    from google.genai._gaos.lib import compat_errors

    response = compat_errors.httpx.Response(
        status, request=compat_errors.httpx.Request("POST", "https://example.invalid")
    )
    cls = {
        429: compat_errors.RateLimitError,
        500: compat_errors.InternalServerError,
        503: compat_errors.InternalServerError,
        400: compat_errors.BadRequestError,
        401: compat_errors.AuthenticationError,
        403: compat_errors.PermissionDeniedError,
    }.get(status, compat_errors.APIStatusError)
    return cls(message, response=response, body=None)


class SequenceClient:
    """A client that raises or returns per call, recording the model asked for."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.models: list[str] = []
        self.interactions = self

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        if not self._outcomes:
            raise AssertionError("SequenceClient ran out of outcomes")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def fake_planner(mission_text: str, constraints: MissionConstraints, **_):
    """A planner that returns VALID_PAYLOAD without calling the API."""
    from waypoint.planner import Plan

    return Plan.model_validate({**VALID_PAYLOAD, "mission_text": mission_text})
