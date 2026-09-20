"""Hierarchical LLM planner backed by Google Gemini.

The request pins the response to `application/json` against a strict schema, so
the reply either decodes and validates into a `Plan` or raises. There is no
free-text fallback path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import ValidationError

from typing import TYPE_CHECKING

from planner.config import API_KEY_VAR, get_api_key, get_model_ladder
from planner.errors import PlannerAPIError, PlannerResponseError, PlanValidationError
from planner.models import MissionConstraints, Plan
from planner.prompts import (
    PLAN_SCHEMA,
    SYSTEM_PROMPT,
    build_plan_prompt,
    build_replan_prompt,
)

RESPONSE_FORMAT = {
    "type": "text",
    "mime_type": "application/json",
    "schema": PLAN_SCHEMA,
}

if TYPE_CHECKING:  # avoids a cycle: world imports planner.models
    from world.models import World

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


API_ERRORS = _api_error_types()


def _should_shift(exc: BaseException) -> bool:
    """Whether trying the next model down the ladder could plausibly help."""
    status = getattr(exc, "status_code", None)
    if status is None:
        # Connection and timeout errors carry no status; another model is on
        # different capacity, so it is worth one try.
        return type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "NoResponseError"}
    return status in _SHIFTABLE_STATUSES

# The SDK retries 429s and 5xxs by itself with backoff, for minutes, and never
# raises -- so a spent quota looked like a mission stuck in planning forever.
# Fail fast instead: the model ladder above is the retry policy, and a failure
# the page can show beats a spinner.
HTTP_OPTIONS = genai_types.HttpOptions(retry_options=genai_types.HttpRetryOptions(attempts=0))

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
        _client = genai.Client(api_key=api_key, http_options=HTTP_OPTIONS)
    return _client


def plan(
    mission_text: str,
    constraints: MissionConstraints,
    *,
    client: genai.Client | None = None,
    world: "World | None" = None,
    home: tuple[float, float, float] | None = None,
    landing: tuple[float, float, float] | None = None,
) -> Plan:
    """Turn a natural language mission into a validated hierarchical Plan.

    `world` describes the terrain and buildings the mission is flown in, so the
    model can plan around them rather than being corrected by the verifier.
    `home` is where the aircraft is standing; the plan has to start and end
    there, and it is not always the origin.
    """
    return _request_plan(
        mission_text=mission_text,
        user_prompt=build_plan_prompt(mission_text, constraints, world, home, landing),
        client=client,
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
) -> Plan:
    """Ask for a corrected plan, telling the model exactly why the last one failed."""
    return _request_plan(
        mission_text=mission_text,
        user_prompt=build_replan_prompt(
            mission_text, previous_plan, violation_report, constraints, world, home, landing
        ),
        client=client,
    )


def _request_plan(
    *,
    mission_text: str,
    user_prompt: str,
    client: genai.Client | None,
) -> Plan:
    client = client or get_client()
    ladder = get_model_ladder()
    attempts: list[str] = []

    for position, model in enumerate(ladder):
        try:
            interaction = client.interactions.create(
                model=model,
                system_instruction=SYSTEM_PROMPT,
                input=user_prompt,
                response_format=RESPONSE_FORMAT,
                timeout=REQUEST_TIMEOUT_S,
            )
        except API_ERRORS as exc:
            attempts.append(f"{model}: {type(exc).__name__}: {exc}")
            is_last = position == len(ladder) - 1
            if not _should_shift(exc) or is_last:
                raise PlannerAPIError(
                    "The Gemini API call failed and no model on the ladder could "
                    "serve it.\nTried:\n  " + "\n  ".join(attempts)
                ) from exc
            log.warning(
                "%s unavailable (%s); shifting down to %s",
                model,
                type(exc).__name__,
                ladder[position + 1],
            )
            continue

        if position:
            log.info("planned on %s after %d model shift(s)", model, position)
        payload = _extract_payload(interaction)
        return _build_plan(payload, mission_text)

    raise PlannerAPIError("The model ladder is empty; nothing to plan with.")


def _extract_payload(interaction: Any) -> dict:
    """Decode the JSON body, or explain why there isn't one."""
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

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerResponseError(
            f"Gemini did not return valid JSON ({exc}). First 200 characters:\n{text[:200]}"
        ) from exc

    if not isinstance(payload, dict):
        raise PlannerResponseError(
            f"Gemini returned a JSON {type(payload).__name__}, expected an object."
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
