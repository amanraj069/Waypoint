"""Hierarchical LLM planner: OpenAI first when its key is set, then Google Gemini.

Every request pins the response to JSON against the same strict schema, so the
reply either decodes and validates into a `Plan` or raises. There is no
free-text fallback path.

Providers are tried in the order `config.get_providers()` gives (OpenAI, then
Gemini, each only if its key is set), and each has its own model ladder. A
model that is overloaded, out of quota or unreachable shifts down its ladder;
a provider that fails outright (every rung gone, or a bad key or request, which
no other model of the same provider would fix) hands over to the next
provider. A reply that is not a valid plan does not: that is a planning
failure, not an availability one, and it is raised as it is.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from typing import TYPE_CHECKING

from waypoint.planner.config import (
    API_KEY_VAR,
    OPENAI_KEY_VAR,
    get_api_key,
    get_model_ladder,
    get_openai_key,
    get_openai_model_ladder,
    get_providers,
    get_reasoning_effort,
)
from waypoint.planner.errors import PlannerAPIError, PlannerResponseError, PlanValidationError
from waypoint.planner.models import MissionConstraints, Plan
from waypoint.planner.prompts import (
    DEFAULT_OPTIONS,
    PLAN_SCHEMA,
    PlannerOptions,
    build_inflight_replan_prompt,
    build_plan_prompt,
    build_replan_prompt,
    flatten,
    system_prompt,
)

RESPONSE_FORMAT = {
    "type": "text",
    "mime_type": "application/json",
    "schema": PLAN_SCHEMA,
}

if TYPE_CHECKING:  # avoids a cycle: world imports planner.models
    from google import genai
    from waypoint.intent.models import MissionIntent
    from waypoint.world.models import World

log = logging.getLogger(__name__)

# Statuses that mean no usable plan came back.
_FAILED_STATUSES = {"failed", "incomplete", "cancelled", "budget_exceeded"}

# HTTP statuses where a different model is likely to succeed: the model is
# overloaded or we are out of quota on it. A 4xx like 400/401/403 is our fault
# and shifting models would just repeat it.
_SHIFTABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _api_error_types() -> tuple[type[BaseException], ...]:
    """Every exception type the SDK may raise for a failed call.

    client.interactions raises from google.genai._gaos.lib.compat_errors, whose
    classes do not inherit from the public google.genai.errors.APIError, so
    catching the public base alone misses quota and auth failures. The private
    base is looked up defensively: if the SDK moves it, we still catch the
    public one rather than failing to import.
    """
    from google.genai import errors as genai_errors

    types: list[type[BaseException]] = [genai_errors.APIError]
    try:
        from google.genai._gaos.lib.compat_errors import (  # type: ignore[import-not-found]
            APIError as _InteractionsAPIError,
        )
    except ImportError:  # pragma: no cover - depends on SDK internals
        pass
    else:
        types.append(_InteractionsAPIError)
    return tuple(types)


#: Filled in on the first call, because importing the SDK costs a quarter of a
#: second and replaying a flight never needs it.
API_ERRORS: tuple[type[BaseException], ...] = ()


def api_errors() -> tuple[type[BaseException], ...]:
    global API_ERRORS
    if not API_ERRORS:
        API_ERRORS = _api_error_types()
    return API_ERRORS


def _should_shift(exc: BaseException) -> bool:
    """Whether trying the next model down the ladder could plausibly help."""
    status = getattr(exc, "status_code", None)
    if status is None:
        # Connection and timeout errors carry no status; another model is on
        # different capacity, so it is worth one try.
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "NoResponseError"}
    return status in _SHIFTABLE_STATUSES

def _http_options():
    """The SDK retries 429s and 5xxs by itself with backoff, for minutes, and
    never raises -- so a spent quota looked like a mission stuck in planning
    forever. Fail fast instead: the model ladder is the retry policy, and a
    failure the operator can see beats a spinner."""
    from google.genai import types as genai_types

    return genai_types.HttpOptions(retry_options=genai_types.HttpRetryOptions(attempts=0))

#: Seconds to wait for one plan. client.interactions ignores HttpOptions.timeout,
#: so this is passed on every call.
REQUEST_TIMEOUT_S = 120.0

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Return a process-wide Gemini client.

    The key comes from GEMINI_API_KEY1, GEMINI_API_KEY2, ... (or a plain
    GEMINI_API_KEY) via planner.config, and is handed to the SDK explicitly: the
    SDK on its own only looks at GEMINI_API_KEY / GOOGLE_API_KEY, so numbered
    keys would otherwise be invisible to it.
    """
    global _client
    if _client is None:
        api_key = get_api_key()
        if api_key is None:
            raise PlannerResponseError(
                f"{API_KEY_VAR} is not set. Put your Gemini API key in .env "
                f"(see .env.example) or export {API_KEY_VAR} before planning."
            )
        from google import genai

        _client = genai.Client(api_key=api_key, http_options=_http_options())
    return _client


def plan(
    mission_text: str,
    constraints: MissionConstraints,
    *,
    client: genai.Client | None = None,
    world: "World | None" = None,
    home: tuple[float, float, float] | None = None,
    landing: tuple[float, float, float] | None = None,
    options: PlannerOptions = DEFAULT_OPTIONS,
    intent: "MissionIntent | None" = None,
) -> Plan:
    """Turn a natural language mission into a validated hierarchical Plan.

    `world` describes the terrain and buildings the mission is flown in, so the
    model can plan around them rather than being corrected by the verifier.
    `home` is where the aircraft is standing; the plan has to start and end
    there, and it is not always the origin. `options` are the experiment
    harness's ablations; the defaults are the planner as designed.
    """
    return _request_plan(
        mission_text=mission_text,
        user_prompt=build_plan_prompt(mission_text, constraints, world, home, landing, options, intent),
        client=client,
        options=options,
    )


def replan(
    mission_text: str,
    previous_plan: Plan,
    violation_report: list[str],
    constraints: MissionConstraints,
    *,
    client: genai.Client | None = None,
    world: "World | None" = None,
    home: tuple[float, float, float] | None = None,
    landing: tuple[float, float, float] | None = None,
    situation: str | None = None,
    options: PlannerOptions = DEFAULT_OPTIONS,
    intent: "MissionIntent | None" = None,
) -> Plan:
    """Ask for a new plan, telling the model exactly why the last one will not do.

    On the ground, `violation_report` is the verifier's reasons for rejecting
    `previous_plan`. In the air, pass `situation` (the feedback layer's
    `summarize_for_llm`) and `home` as the aircraft's current position: the new
    plan then starts from there, and `violation_report` holds the verifier's
    reasons only if an earlier in-flight attempt was also rejected.
    """
    if situation is not None:
        prompt = build_inflight_replan_prompt(
            mission_text, previous_plan, situation, constraints, world, home, landing,
            violation_report, options, intent,
        )
    else:
        prompt = build_replan_prompt(
            mission_text, previous_plan, violation_report, constraints, world, home, landing, options, intent
        )
    return _request_plan(mission_text=mission_text, user_prompt=prompt, client=client, options=options)


def _request_plan(
    *,
    mission_text: str,
    user_prompt: str,
    client: genai.Client | None,
    options: PlannerOptions = DEFAULT_OPTIONS,
) -> Plan:
    """Ask each provider in turn for a plan, and validate the first answer."""
    payload = request_json(
        user_prompt,
        system=system_prompt(options),
        gemini_format=RESPONSE_FORMAT,
        openai_format=OPENAI_RESPONSE_FORMAT,
        client=client,
        what="plan",
    )
    result = _build_plan(payload, mission_text)
    return result if options.hierarchical else flatten(result)


def request_json(
    user_prompt: str,
    *,
    system: str,
    gemini_format: dict,
    openai_format: dict,
    client: genai.Client | None = None,
    what: str = "answer",
) -> dict:
    """One structured request: each provider in turn, down its model ladder; the decoded JSON object.

    The plan and the mission intent both go through here, so they share the
    provider fallback, the fail-fast clients and the token accounting.
    An explicit `client` is a Gemini client and pins the request to Gemini, so
    callers (and tests) that hand one in get exactly the provider they built.
    """
    providers = ("gemini",) if client is not None else get_providers()
    if not providers:
        raise PlannerResponseError(
            f"No planner is configured: set {OPENAI_KEY_VAR} or {API_KEY_VAR} in .env "
            f"(see .env.example) before planning."
        )
    attempts: list[str] = []
    for position, provider in enumerate(providers):
        is_last = position == len(providers) - 1
        try:
            if provider == "openai":
                text, model = _ask_openai(user_prompt, system, openai_format, attempts)
            else:
                text, model = _ask_gemini(user_prompt, system, gemini_format, client, attempts)
        except _ProviderUnavailable:
            if is_last:
                names = " and ".join({"openai": "OpenAI", "gemini": "Gemini"}[p] for p in providers)
                raise PlannerAPIError(
                    f"The {names} API call failed and no model could serve it.\nTried:\n  "
                    + "\n  ".join(attempts)
                ) from None
            log.warning("%s could not %s; falling back to %s", provider, "plan" if what == "plan" else "answer", providers[position + 1])
            continue
        log.info("%s by %s %s", what, provider, model)
        return _decode(text)
    raise PlannerAPIError("No planner provider was tried.")  # pragma: no cover


class _ProviderUnavailable(Exception):
    """This provider cannot serve the request; the next one might."""


def _ask_gemini(user_prompt: str, system: str, response_format: dict, client, attempts: list[str]) -> tuple[str, str]:
    client = client or get_client()
    ladder = get_model_ladder()
    for position, model in enumerate(ladder):
        try:
            interaction = client.interactions.create(
                model=model,
                system_instruction=system,
                input=user_prompt,
                response_format=response_format,
                timeout=REQUEST_TIMEOUT_S,
            )
        except api_errors() as exc:
            attempts.append(f"gemini {model}: {type(exc).__name__}: {exc}")
            if not _should_shift(exc) or position == len(ladder) - 1:
                raise _ProviderUnavailable from exc
            log.warning(
                "%s unavailable (%s); shifting down to %s",
                model,
                type(exc).__name__,
                ladder[position + 1],
            )
            continue
        if position:
            log.info("planned on %s after %d model shift(s)", model, position)
        _count(f"gemini:{model}", getattr(interaction, "usage", None))
        return _gemini_text(interaction), model
    raise _ProviderUnavailable("the Gemini model ladder is empty")


# -- OpenAI -------------------------------------------------------------------

#: The plan schema as an OpenAI structured-output format. PLAN_SCHEMA already
#: meets strict mode's rules: every object closed, every property required.
OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "flight_plan", "strict": True, "schema": PLAN_SCHEMA},
}

_openai_client = None


def get_openai_client():
    """A process-wide OpenAI client that fails fast, like the Gemini one.

    The SDK retries 429s and 5xxs on its own; the model ladder and the
    provider fallback are the retry policy here, so its retries are off.
    """
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # noqa: PLC0415 -- only paid for when OpenAI is configured

        _openai_client = OpenAI(api_key=get_openai_key(), max_retries=0, timeout=REQUEST_TIMEOUT_S)
    return _openai_client


def _ask_openai(user_prompt: str, system: str, response_format: dict, attempts: list[str]) -> tuple[str, str]:
    import openai  # noqa: PLC0415

    client = get_openai_client()
    ladder = get_openai_model_ladder()
    for position, model in enumerate(ladder):
        extra = {"reasoning_effort": get_reasoning_effort()} if _is_reasoning_model(model) else {}
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
                **extra,
            )
        except openai.OpenAIError as exc:
            attempts.append(f"openai {model}: {type(exc).__name__}: {exc}")
            if not _openai_should_shift(exc) or position == len(ladder) - 1:
                raise _ProviderUnavailable from exc
            log.warning(
                "%s unavailable (%s); shifting down to %s", model, type(exc).__name__, ladder[position + 1]
            )
            continue
        if position:
            log.info("planned on %s after %d model shift(s)", model, position)
        _count(f"openai:{model}", getattr(completion, "usage", None))
        return _openai_text(completion, model), model
    raise _ProviderUnavailable("the OpenAI model ladder is empty")


# -- usage --------------------------------------------------------------------

#: Every successful planner call this process has made, per provider:model:
#: calls, input, cached input, output and (of the output) reasoning tokens.
#: The experiment harness reads it before and after each flight.
USAGE: dict[str, dict[str, int]] = {}
USAGE_FIELDS = ("calls", "input", "cached", "output", "reasoning")


def usage_snapshot() -> dict[str, dict[str, int]]:
    return {model: dict(counts) for model, counts in USAGE.items()}


def usage_since(before: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """What was spent since `before`, a `usage_snapshot()`: only models that were used."""
    spent = {}
    for model, counts in USAGE.items():
        then = before.get(model, {})
        delta = {k: counts.get(k, 0) - then.get(k, 0) for k in USAGE_FIELDS}
        if delta["calls"]:
            spent[model] = delta
    return spent


def _count(model: str, usage: Any) -> None:
    """Add one call to USAGE. Token counts are read defensively; a provider may not report them."""
    counts = USAGE.setdefault(model, dict.fromkeys(USAGE_FIELDS, 0))
    counts["calls"] += 1
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "total_input_tokens", None) or 0
    output = getattr(usage, "completion_tokens", None) or getattr(usage, "total_output_tokens", None) or 0
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    output_details = getattr(usage, "completion_tokens_details", None)
    counts["input"] += int(prompt or 0)
    counts["output"] += int(output or 0)
    counts["cached"] += int(getattr(prompt_details, "cached_tokens", 0) or 0)
    counts["reasoning"] += int(getattr(output_details, "reasoning_tokens", 0) or 0)


def _is_reasoning_model(model: str) -> bool:
    """gpt-5 and the o-series think before answering; how long is OPENAI_REASONING_EFFORT."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _openai_should_shift(exc: BaseException) -> bool:
    """A different OpenAI model may help: it is overloaded, rate limited, missing, or unreachable."""
    import openai  # noqa: PLC0415

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    # 404: this model is not available to this key; a cheaper one may be.
    return status in _SHIFTABLE_STATUSES or status == 404


def _openai_text(completion, model: str) -> str:
    choice = completion.choices[0] if completion.choices else None
    if choice is None:
        raise PlannerResponseError(f"OpenAI ({model}) returned no choices. No plan was produced.")
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        raise PlannerResponseError(f"OpenAI ({model}) refused to plan: {refusal}")
    if choice.finish_reason == "length":
        raise PlannerResponseError(f"OpenAI ({model}) ran out of tokens before finishing the plan.")
    text = choice.message.content
    if not text or not text.strip():
        raise PlannerResponseError(f"OpenAI ({model}) returned an empty response. No plan was produced.")
    return text


def _gemini_text(interaction: Any) -> str:
    """The body of a Gemini reply, or why there isn't one."""
    status = getattr(interaction, "status", None)
    if status in _FAILED_STATUSES:
        raise PlannerResponseError(
            f"Gemini returned status {status!r} instead of a plan: "
            f"{_describe_errors(interaction)}"
        )

    text = getattr(interaction, "output_text", None)
    if not text or not text.strip():
        raise PlannerResponseError(
            f"Gemini returned an empty response (status={status!r}). No plan was produced."
        )
    return text


def _decode(text: str) -> dict:
    """The JSON object a reply holds, or why it is not one."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerResponseError(
            f"The planner did not return valid JSON ({exc}). First 200 characters:\n{text[:200]}"
        ) from exc

    if not isinstance(payload, dict):
        raise PlannerResponseError(
            f"The planner returned a JSON {type(payload).__name__}, expected an object."
        )
    return payload


def _describe_errors(interaction: Any) -> str:
    errors = getattr(interaction, "errors", None) or []
    messages = [
        f"{getattr(e, 'code', None) or 'error'}: {getattr(e, 'message', None) or e}"
        for e in errors
    ]
    return "; ".join(messages) if messages else "no error detail given"


def _build_plan(payload: dict, mission_text: str) -> Plan:
    """Validate the decoded response against the Plan model."""
    if "mission_text" in payload:
        # The model does not own this field; we do.
        payload = {k: v for k, v in payload.items() if k != "mission_text"}
    subtasks = payload.get("subtasks")
    if isinstance(subtasks, list):
        # A phase with no waypoints flies nothing (models sometimes emit an empty
        # "land" phase after putting the LAND in the phase before). Dropping it
        # loses nothing; a plan with nothing left still fails below.
        payload = {**payload, "subtasks": [
            s for s in subtasks if not (isinstance(s, dict) and s.get("waypoints") == [])
        ]}
    try:
        result = Plan.model_validate({**payload, "mission_text": mission_text})
    except ValidationError as exc:
        raise PlanValidationError(
            f"The model returned a plan that is not a valid Plan:\n{exc}"
        ) from exc

    # An empty plan is schema-valid but not flyable, and would execute silently.
    if not result.subtasks:
        raise PlanValidationError(
            "The model returned a plan with no subtasks. That is not a valid Plan."
        )
    return result
