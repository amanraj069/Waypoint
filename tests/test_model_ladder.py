"""The planner shifts down the model ladder when a model is unavailable."""

from __future__ import annotations

import logging

import pytest

from conftest import VALID_PAYLOAD, SequenceClient, http_error, json_response
from waypoint.planner import MissionConstraints, PlannerAPIError, plan
from waypoint.planner.config import MODEL_LADDER, get_model, get_model_ladder

# Only the two tests below pin the configured order; every behaviour test reads
# the ladder by position, so reordering it (or setting GEMINI_MODEL in .env,
# which the suite ignores) changes what they expect without breaking them.
FIRST, SECOND, THIRD, FOURTH = MODEL_LADDER[:4]


def test_the_ladder_starts_at_37_flash():
    assert MODEL_LADDER[0] == "gemini-3.7-flash"
    assert get_model() == "gemini-3.7-flash"


def test_the_ladder_is_the_configured_preference_order():
    assert MODEL_LADDER == (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    )


def test_an_override_leads_the_ladder_without_losing_fallbacks(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    ladder = get_model_ladder()
    assert ladder[0] == "gemini-2.5-pro"
    assert ladder[1:] == MODEL_LADDER


def test_a_blank_override_is_ignored(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "  ")
    assert get_model_ladder() == MODEL_LADDER


def test_the_happy_path_never_leaves_the_first_model(constraints):
    client = SequenceClient(json_response(VALID_PAYLOAD))

    plan("survey the north field", constraints, client=client)

    assert client.models == [FIRST]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_an_overloaded_model_shifts_down_one_rung(status, constraints):
    """This is the 500 'experiencing high demand' case."""
    client = SequenceClient(http_error(status), json_response(VALID_PAYLOAD))

    result = plan("survey the north field", constraints, client=client)

    assert client.models == [FIRST, SECOND]
    assert result.subtasks


def test_it_keeps_shifting_until_a_model_answers(constraints):
    client = SequenceClient(
        http_error(503), http_error(503), http_error(429), json_response(VALID_PAYLOAD)
    )

    plan("survey the north field", constraints, client=client)

    assert client.models == [FIRST, SECOND, THIRD, FOURTH]


def test_exhausting_the_ladder_raises_with_every_attempt_listed(constraints):
    client = SequenceClient(*[http_error(503) for _ in MODEL_LADDER])

    with pytest.raises(PlannerAPIError) as exc_info:
        plan("survey the north field", constraints, client=client)

    message = str(exc_info.value)
    for model in MODEL_LADDER:
        assert model in message
    assert client.models == list(MODEL_LADDER)


@pytest.mark.parametrize("status", [400, 401, 403])
def test_our_own_errors_fail_fast_without_burning_the_ladder(status, constraints):
    """A bad request or bad key repeats on every model -- don't spend eight calls on it."""
    client = SequenceClient(http_error(status))

    with pytest.raises(PlannerAPIError):
        plan("survey the north field", constraints, client=client)

    assert client.models == [FIRST]


def test_a_connection_error_shifts_too(constraints):
    from google.genai._gaos.lib import compat_errors

    exc = compat_errors.APIConnectionError(
        message="connection reset",
        request=compat_errors.httpx.Request("POST", "https://example.invalid"),
    )
    client = SequenceClient(exc, json_response(VALID_PAYLOAD))

    plan("survey the north field", constraints, client=client)

    assert client.models == [FIRST, SECOND]


def test_a_shift_is_logged_so_the_operator_sees_it(constraints, caplog):
    client = SequenceClient(http_error(500), json_response(VALID_PAYLOAD))

    with caplog.at_level(logging.WARNING, logger="waypoint.planner.llm_planner"):
        plan("survey the north field", constraints, client=client)

    assert f"{FIRST} unavailable" in caplog.text
    assert f"shifting down to {SECOND}" in caplog.text


def test_a_malformed_plan_does_not_shift(constraints):
    """A bad plan is a planning failure, not an availability failure."""
    from waypoint.planner import PlanValidationError

    client = SequenceClient(json_response({"subtasks": []}))

    with pytest.raises(PlanValidationError):
        plan("survey the north field", constraints, client=client)

    assert client.models == [FIRST]
