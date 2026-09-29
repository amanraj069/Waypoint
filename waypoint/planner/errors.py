"""Planner failure modes.

Every one of these means "the planner did not produce a usable Plan". They are
deliberately loud: a bad plan must never reach the verifier as if it were good.
"""

from __future__ import annotations


class PlannerError(Exception):
    """Base class for all planner failures."""


class PlannerResponseError(PlannerError):
    """The model's response was not a usable plan submission.

    Raised for a refusal, a truncated response, or a reply that contains no
    tool_use block for the plan tool.
    """


class PlanValidationError(PlannerError):
    """The model called the plan tool but the arguments did not form a valid Plan."""


class PlannerAPIError(PlannerError):
    """The Gemini API call itself failed: auth, quota, network, or a server error.

    Distinct from a bad plan -- the model never got to answer. Retrying may help.
    """
