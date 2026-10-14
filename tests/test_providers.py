"""The provider chain: OpenAI first when its key is set, Gemini after it.

Both clients are faked; nothing here reaches the network.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

from conftest import VALID_PAYLOAD, SequenceClient, http_error, json_response
from waypoint.planner import PlannerAPIError, PlannerOptions, PlannerResponseError, plan, replan
from waypoint.planner.config import OPENAI_MODEL_LADDER, describe_providers, get_providers
from waypoint.planner.llm_planner import OPENAI_RESPONSE_FORMAT
from waypoint.planner.prompts import SYSTEM_PROMPT


def completion(payload=VALID_PAYLOAD, *, refusal=None, finish_reason="stop"):
    content = None if refusal else json.dumps(payload)
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def status_error(cls, status: int):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.invalid"))
    return cls(f"{status} from the fake", response=response, body=None)


class FakeOpenAI:
    """Answers chat.completions.create from a script, recording every request."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def models(self) -> list[str]:
        return [call["model"] for call in self.calls]


@pytest.fixture
def keys(monkeypatch):
    """Both providers configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")


def use_openai(monkeypatch, fake: FakeOpenAI) -> FakeOpenAI:
    monkeypatch.setattr("waypoint.planner.llm_planner.get_openai_client", lambda: fake)
    return fake


def use_gemini(monkeypatch, fake) -> None:
    monkeypatch.setattr("waypoint.planner.llm_planner.get_client", lambda: fake)


# ---------------------------------------------------------------------------
# Which providers, in which order
# ---------------------------------------------------------------------------


def test_openai_comes_first_when_both_keys_are_set(keys):
    assert get_providers() == ("openai", "gemini")
    assert describe_providers().startswith(
        "openai [COST=HIGH, effort=low] (gpt-5-mini, gpt-4o-mini, gpt-5-nano)"
    )


def test_without_an_openai_key_it_is_gemini_alone(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    assert get_providers() == ("gemini",)


def test_a_blank_openai_key_does_not_count(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    assert get_providers() == ("gemini",)


def test_with_no_key_at_all_planning_says_which_to_set(monkeypatch):
    import os

    for name in list(os.environ):
        if name.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(name)
    with pytest.raises(PlannerResponseError, match="OPENAI_API_KEY or GEMINI_API_KEY"):
        plan("survey", _constraints())


def test_the_openai_ladder_is_configurable(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-nano")
    from waypoint.planner.config import get_openai_model_ladder

    assert get_openai_model_ladder() == ("gpt-4.1-nano", *OPENAI_MODEL_LADDER)
    monkeypatch.setenv("OPENAI_MODELS", "a, b")
    assert get_openai_model_ladder()[1:] == ("a", "b")


# ---------------------------------------------------------------------------
# OpenAI serves the plan
# ---------------------------------------------------------------------------


def test_openai_plans_with_the_strict_schema_and_the_same_prompt(keys, monkeypatch):
    fake = use_openai(monkeypatch, FakeOpenAI(completion()))
    use_gemini(monkeypatch, SequenceClient())  # must not be called

    result = plan("survey the north field", _constraints())

    assert [s.name for s in result.subtasks] == ["takeoff", "survey_north_field", "return_and_land"]
    (call,) = fake.calls
    assert call["model"] == OPENAI_MODEL_LADDER[0]
    assert call["response_format"] == OPENAI_RESPONSE_FORMAT
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "survey the north field" in call["messages"][1]["content"]
    assert call["reasoning_effort"] == "low"  # gpt-5-mini's default


def test_a_non_reasoning_model_is_not_sent_reasoning_effort(keys, monkeypatch):
    monkeypatch.setenv("OPENAI_MODELS", "gpt-4o-mini")
    fake = use_openai(monkeypatch, FakeOpenAI(completion()))
    plan("survey", _constraints())
    assert "reasoning_effort" not in fake.calls[0]


def test_the_ablations_apply_to_openai_too(keys, monkeypatch):
    fake = use_openai(monkeypatch, FakeOpenAI(completion()))
    result = plan("survey", _constraints(), options=PlannerOptions(hierarchical=False))
    assert [s.name for s in result.subtasks] == ["route"]
    assert "Stage 1" not in fake.calls[0]["messages"][0]["content"]


def test_an_inflight_replan_goes_to_openai(keys, monkeypatch):
    fake = use_openai(monkeypatch, FakeOpenAI(completion()))
    from waypoint.planner import Plan

    previous = Plan.model_validate({**VALID_PAYLOAD, "mission_text": "survey"})
    replan("survey", previous, [], _constraints(), situation="a crane appeared", home=(0, 40, 25))
    assert "a crane appeared" in fake.calls[0]["messages"][1]["content"]


# ---------------------------------------------------------------------------
# Shifting and falling back
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("error", [
    lambda: status_error(openai.RateLimitError, 429),
    lambda: status_error(openai.InternalServerError, 503),
    lambda: status_error(openai.NotFoundError, 404),
    lambda: openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.invalid")),
])
def test_an_unavailable_openai_model_shifts_down_its_ladder(keys, monkeypatch, error):
    fake = use_openai(monkeypatch, FakeOpenAI(error(), completion()))
    plan("survey", _constraints())
    assert fake.models == list(OPENAI_MODEL_LADDER[:2])


def test_a_bad_openai_key_falls_back_to_gemini_at_once(keys, monkeypatch, caplog):
    fake = use_openai(monkeypatch, FakeOpenAI(status_error(openai.AuthenticationError, 401)))
    gemini = SequenceClient(json_response(VALID_PAYLOAD))
    use_gemini(monkeypatch, gemini)

    with caplog.at_level(logging.WARNING, logger="waypoint.planner.llm_planner"):
        result = plan("survey", _constraints())

    assert result.subtasks
    assert fake.models == [OPENAI_MODEL_LADDER[0]]  # no point trying a cheaper model with the same key
    assert len(gemini.models) == 1
    assert "openai could not plan; falling back to gemini" in caplog.text


def test_an_exhausted_openai_ladder_falls_back_to_gemini(keys, monkeypatch):
    fake = use_openai(monkeypatch, FakeOpenAI(*[status_error(openai.RateLimitError, 429)] * len(OPENAI_MODEL_LADDER)))
    use_gemini(monkeypatch, SequenceClient(json_response(VALID_PAYLOAD)))
    assert plan("survey", _constraints()).subtasks
    assert fake.models == list(OPENAI_MODEL_LADDER)


def test_when_everything_fails_every_attempt_is_listed(keys, monkeypatch):
    use_openai(monkeypatch, FakeOpenAI(status_error(openai.AuthenticationError, 401)))
    use_gemini(monkeypatch, SequenceClient(http_error(401)))
    with pytest.raises(PlannerAPIError, match="OpenAI and Gemini") as exc_info:
        plan("survey", _constraints())
    message = str(exc_info.value)
    assert "openai gpt-5-mini: AuthenticationError" in message and "gemini " in message


def test_a_refusal_is_a_planning_failure_not_a_fallback(keys, monkeypatch):
    use_openai(monkeypatch, FakeOpenAI(completion(refusal="I can't help with that")))
    gemini = SequenceClient()
    use_gemini(monkeypatch, gemini)
    with pytest.raises(PlannerResponseError, match="refused"):
        plan("survey", _constraints())
    assert gemini.models == []


def test_a_truncated_answer_is_a_planning_failure(keys, monkeypatch):
    use_openai(monkeypatch, FakeOpenAI(completion(finish_reason="length")))
    with pytest.raises(PlannerResponseError, match="ran out of tokens"):
        plan("survey", _constraints())


def test_an_invalid_plan_from_openai_is_not_retried_elsewhere(keys, monkeypatch):
    from waypoint.planner import PlanValidationError

    use_openai(monkeypatch, FakeOpenAI(completion({"subtasks": []})))
    gemini = SequenceClient()
    use_gemini(monkeypatch, gemini)
    with pytest.raises(PlanValidationError):
        plan("survey", _constraints())
    assert gemini.models == []


def test_an_explicit_gemini_client_pins_the_request_to_gemini(keys, monkeypatch):
    fake = use_openai(monkeypatch, FakeOpenAI())
    gemini = SequenceClient(json_response(VALID_PAYLOAD))
    plan("survey", _constraints(), client=gemini)
    assert fake.calls == [] and len(gemini.models) == 1


def _constraints():
    from waypoint.planner import DEFAULT_CONSTRAINTS

    return DEFAULT_CONSTRAINTS


# ---------------------------------------------------------------------------
# COST tiers, reasoning effort, usage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cost", "ladder"), [
    (None, ("gpt-5-mini", "gpt-4o-mini", "gpt-5-nano")),
    ("HIGH", ("gpt-5-mini", "gpt-4o-mini", "gpt-5-nano")),
    ("high", ("gpt-5-mini", "gpt-4o-mini", "gpt-5-nano")),
    ("LOW", ("gpt-5-nano", "gpt-4.1-nano")),
])
def test_cost_picks_the_openai_ladder(monkeypatch, cost, ladder):
    from waypoint.planner.config import get_openai_model_ladder

    if cost is not None:
        monkeypatch.setenv("COST", cost)
    assert get_openai_model_ladder() == ladder


def test_low_cost_plans_with_nano_and_falls_back_to_41_nano(keys, monkeypatch):
    monkeypatch.setenv("COST", "LOW")
    fake = use_openai(monkeypatch, FakeOpenAI(status_error(openai.InternalServerError, 503), completion()))
    plan("survey", _constraints())
    assert fake.models == ["gpt-5-nano", "gpt-4.1-nano"]
    assert fake.calls[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in fake.calls[1]  # gpt-4.1-nano does not reason


def test_an_explicit_ladder_still_wins_over_cost(monkeypatch):
    from waypoint.planner.config import get_openai_model_ladder

    monkeypatch.setenv("COST", "LOW")
    monkeypatch.setenv("OPENAI_MODELS", "gpt-4o-mini")
    assert get_openai_model_ladder() == ("gpt-4o-mini",)


def test_a_mistyped_cost_is_refused(monkeypatch):
    from waypoint.planner.config import get_openai_model_ladder

    monkeypatch.setenv("COST", "CHEAP")
    with pytest.raises(ValueError, match="COST='CHEAP'"):
        get_openai_model_ladder()


def test_reasoning_effort_is_low_unless_set(keys, monkeypatch):
    from waypoint.planner.config import get_reasoning_effort

    assert get_reasoning_effort() == "low"
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "minimal")
    assert get_reasoning_effort() == "minimal"
    fake = use_openai(monkeypatch, FakeOpenAI(completion()))
    plan("survey", _constraints())
    assert fake.calls[0]["reasoning_effort"] == "minimal"
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "lots")
    from waypoint.planner.config import get_reasoning_effort

    with pytest.raises(ValueError):
        get_reasoning_effort()


def test_every_call_is_counted_with_its_tokens(keys, monkeypatch):
    from waypoint.planner.llm_planner import usage_since, usage_snapshot

    reply = completion()
    reply.usage = SimpleNamespace(
        prompt_tokens=2400, completion_tokens=600,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2000),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    use_openai(monkeypatch, FakeOpenAI(reply, completion()))
    before = usage_snapshot()
    plan("survey", _constraints())
    plan("survey", _constraints())  # no usage reported: counted as a call, no tokens
    spent = usage_since(before)
    assert spent == {"openai:gpt-5-mini": {"calls": 2, "input": 2400, "cached": 2000, "output": 600, "reasoning": 0}}
