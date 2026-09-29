"""Planner configuration, loaded from the environment or a .env file.

API keys are read from GEMINI_API_KEY1, GEMINI_API_KEY2, ... and never stored
in the repository. `.env` is gitignored; `.env.example` documents what belongs
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
