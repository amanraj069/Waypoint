"""The comparison suite never dials out: every test hands the system a scripted model.

Like Waypoint's own suite, the real clients are replaced with tripwires and
the keys are cleared from the environment, so a test that forgets to inject a
scripted model fails instead of spending tokens.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEY1", "GEMINI_API_KEY2", "GEMINI_API_KEY3"):
        monkeypatch.delenv(name, raising=False)

    def tripwire(*args, **kwargs):
        raise AssertionError("a test tried to build a real LLM client; inject a ScriptedChat")

    import compare.common.llm as llm

    monkeypatch.setattr(llm, "default_client", tripwire)


def mission_plan(steps: list[dict], reasoning: str = "scripted") -> str:
    """A MissionPlan as PEACE's planner returns it."""
    return json.dumps(
        {
            "reasoning": reasoning,
            "steps": [{"tool": s["tool"], "args": s.get("args", {}), "reasoning": s.get("reasoning", "")} for s in steps],
            "safety_checks": ["scripted"],
            "mission_id": "",
            "timestamp": 0.0,
            "user_query": None,
        }
    )
