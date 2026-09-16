"""Planner configuration, loaded from the environment or a .env file.

The API key is read from GEMINI_API_KEY and never stored in the repository.
`.env` is gitignored; `.env.example` documents what belongs in it.
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

# Real environment variables win over .env, so an exported key or CI secret
# always overrides the file.
load_dotenv(find_dotenv(usecwd=True), override=False)

API_KEY_VAR = "GEMINI_API_KEY"

# Models are tried in this order. When one is overloaded or rate limited the
# planner shifts down to the next rather than failing the mission. Order is a
# deliberate preference list, not strict version order.
MODEL_LADDER: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

DEFAULT_MODEL = MODEL_LADDER[0]


def get_model_ladder() -> tuple[str, ...]:
    """The models to try, in order.

    GEMINI_MODEL moves one model to the front; the rest of the ladder still
    follows it, so an override changes the preference without giving up the
    fallbacks.
    """
    override = (os.environ.get("GEMINI_MODEL") or "").strip()
    if not override:
        return MODEL_LADDER
    return (override, *(m for m in MODEL_LADDER if m != override))


def get_model() -> str:
    """The model tried first."""
    return get_model_ladder()[0]


def get_api_key() -> str | None:
    """The configured key, or None if it is unset or blank.

    A blank value is treated as unset so the placeholder line in a freshly
    copied .env fails with a clear message instead of a confusing 401.
    """
    return (os.environ.get(API_KEY_VAR) or "").strip() or None
