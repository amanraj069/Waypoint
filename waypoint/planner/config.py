"""Planner configuration, loaded from the environment or a .env file.

API keys are read from OPENAI_API_KEY and GEMINI_API_KEY1, GEMINI_API_KEY2,
... and never stored in the repository. OpenAI is tried first when its key is
set, then Gemini. `.env` is gitignored; `.env.example` documents what belongs
in it.
"""

from __future__ import annotations

import os
import re

from dotenv import find_dotenv, load_dotenv

from waypoint.paths import ROOT

# Real environment variables win over .env, so an exported key or CI secret
# always overrides the file. A .env near where you ran the command comes first,
# then the project's own, so a script run from anywhere still finds the key.
load_dotenv(find_dotenv(usecwd=True), override=False)
load_dotenv(ROOT / ".env", override=False)

API_KEY_VAR = "GEMINI_API_KEY"
_NUMBERED_KEY = re.compile(rf"^{API_KEY_VAR}(\d+)$")

# Models are tried in this order, most capable first. When one is overloaded,
# missing, or out of quota on every key, the planner shifts down to the next
# rather than failing the mission.
MODEL_LADDER: tuple[str, ...] = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

DEFAULT_MODEL = MODEL_LADDER[0]


def get_model_ladder() -> tuple[str, ...]:
    """The models to try, in order.

    GEMINI_MODELS (comma separated) replaces the ladder outright. GEMINI_MODEL
    moves one model to the front; the rest of the ladder still follows it, so
    an override changes the preference without giving up the fallbacks.
    """
    configured = [m.strip() for m in (os.environ.get("GEMINI_MODELS") or "").split(",")]
    base = tuple(m for m in configured if m) or MODEL_LADDER
    override = (os.environ.get("GEMINI_MODEL") or "").strip()
    if not override:
        return base
    return (override, *(m for m in base if m != override))


def get_model() -> str:
    """The model tried first."""
    return get_model_ladder()[0]


def get_api_keys() -> tuple[str, ...]:
    """Every configured key, in pool order.

    GEMINI_API_KEY1, GEMINI_API_KEY2, ... are read in numeric order (not
    lexical, so KEY10 follows KEY9). Blank values are skipped, so the
    placeholder lines in a freshly copied .env fail with a clear message
    instead of a confusing 401. A plain GEMINI_API_KEY, the old single-key
    setting, is still honoured after the numbered ones.
    """
    numbered = sorted(
        (int(match.group(1)), value.strip())
        for name, value in os.environ.items()
        if (match := _NUMBERED_KEY.match(name)) and value and value.strip()
    )
    keys = [value for _, value in numbered]
    legacy = (os.environ.get(API_KEY_VAR) or "").strip()
    if legacy:
        keys.append(legacy)
    return tuple(dict.fromkeys(keys))  # the same key twice is one key


def get_api_key() -> str | None:
    """The first configured key, or None if there are none."""
    keys = get_api_keys()
    return keys[0] if keys else None


# ---------------------------------------------------------------------------
# OpenAI: tried before Gemini whenever its key is set
# ---------------------------------------------------------------------------

OPENAI_KEY_VAR = "OPENAI_API_KEY"

COST_VAR = "COST"

#: The OpenAI ladder for each COST tier. HIGH: gpt-5-mini plans, and gpt-4o-mini
#: then gpt-5-nano stand in only when it cannot answer. LOW: gpt-5-nano plans,
#: with gpt-4.1-nano as its fallback. OPENAI_MODELS (comma separated) replaces
#: the tier's ladder; OPENAI_MODEL moves one model to the front of it.
OPENAI_LADDERS: dict[str, tuple[str, ...]] = {
    "HIGH": ("gpt-5-mini", "gpt-4o-mini", "gpt-5-nano"),
    "LOW": ("gpt-5-nano", "gpt-4.1-nano"),
}
DEFAULT_COST = "HIGH"
OPENAI_MODEL_LADDER: tuple[str, ...] = OPENAI_LADDERS[DEFAULT_COST]

REASONING_EFFORT_VAR = "OPENAI_REASONING_EFFORT"
#: How long gpt-5 and o-series models think before answering; OPENAI_REASONING_EFFORT
#: changes it. `low` is the only setting that planned well and in time: gpt-5-nano
#: at `minimal` completed 38% of a 40-flight batch (80% at `low`), and at `medium`
#: it wrote ~5,000 output tokens a call and timed out on 31% of calls at 120 s.
DEFAULT_REASONING_EFFORT = "low"
REASONING_EFFORTS = ("minimal", "low", "medium", "high")


def get_cost() -> str:
    """The COST tier, HIGH unless set; anything else is a mistake worth saying so."""
    value = (os.environ.get(COST_VAR) or DEFAULT_COST).strip().upper() or DEFAULT_COST
    if value not in OPENAI_LADDERS:
        raise ValueError(f"{COST_VAR}={value!r} is not a cost tier; use one of {', '.join(OPENAI_LADDERS)}")
    return value


def get_reasoning_effort() -> str:
    """OPENAI_REASONING_EFFORT if set, else `low`; anything else is a mistake worth saying so."""
    value = (os.environ.get(REASONING_EFFORT_VAR) or DEFAULT_REASONING_EFFORT).strip().lower()
    if value not in REASONING_EFFORTS:
        raise ValueError(
            f"{REASONING_EFFORT_VAR}={value!r} is not a reasoning effort; use one of {', '.join(REASONING_EFFORTS)}"
        )
    return value


def get_openai_key() -> str | None:
    """The OpenAI key, or None when it is unset or blank."""
    return (os.environ.get(OPENAI_KEY_VAR) or "").strip() or None


def get_openai_model_ladder() -> tuple[str, ...]:
    configured = [m.strip() for m in (os.environ.get("OPENAI_MODELS") or "").split(",")]
    base = tuple(m for m in configured if m) or OPENAI_LADDERS[get_cost()]
    override = (os.environ.get("OPENAI_MODEL") or "").strip()
    if not override:
        return base
    return (override, *(m for m in base if m != override))


def get_providers() -> tuple[str, ...]:
    """The planner's providers in the order they are tried: OpenAI, then Gemini.

    A provider is only in the chain when its key is set, so with no OpenAI key
    the planner is Gemini alone, exactly as before.
    """
    chain = []
    if get_openai_key():
        chain.append("openai")
    if get_api_key():
        chain.append("gemini")
    return tuple(chain)


def describe_providers() -> str:
    """The chain as one line, e.g. `openai (gpt-5-mini, ...) -> gemini (...)`, for logs and health checks."""
    ladders = {"openai": get_openai_model_ladder(), "gemini": get_model_ladder()}
    chain = get_providers()
    if not chain:
        return "no planner configured"
    labels = {"openai": f"openai [COST={get_cost()}, effort={get_reasoning_effort()}]", "gemini": "gemini"}
    return " -> ".join(f"{labels[name]} ({', '.join(ladders[name])})" for name in chain)
