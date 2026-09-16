"""Planner tests. Every Gemini call is faked; nothing here touches the network."""

from __future__ import annotations

import copy

import pytest

from conftest import FakeClient, FakeError, FakeInteraction
from conftest import VALID_PAYLOAD, json_response
from planner import (
    Action,
    PlannerAPIError,
    GeoPoint,
    MissionConstraints,
    Plan,
    PlannerError,
    PlannerResponseError,
    PlanValidationError,
    SubTask,
    Waypoint,
    plan,
    replan,
)
from planner.config import API_KEY_VAR, get_model
from planner.llm_planner import RESPONSE_FORMAT, get_client
from planner.prompts import PLAN_SCHEMA


# --- happy path -------------------------------------------------------------


def test_plan_maps_tool_output_onto_the_plan_model(constraints):
    client = FakeClient(json_response(VALID_PAYLOAD))

    result = plan("survey the north field", constraints, client=client)

    assert isinstance(result, Plan)
    assert [s.name for s in result.subtasks] == [
        "takeoff",
        "survey_north_field",
        "return_and_land",
    ]
    assert len(result.waypoints()) == 4
    assert result.waypoints()[-1].action is Action.LAND


def test_mission_text_comes_from_the_caller_not_the_model(constraints):
    payload = {**copy.deepcopy(VALID_PAYLOAD), "mission_text": "something the model made up"}
    client = FakeClient(json_response(payload))

    result = plan("survey the north field", constraints, client=client)

    assert result.mission_text == "survey the north field"


# --- request shape ----------------------------------------------------------


def test_request_pins_the_response_to_the_strict_json_schema(constraints):
    client = FakeClient(json_response(VALID_PAYLOAD))

    plan("survey the north field", constraints, client=client)

    call = client.last_call
    assert call["model"] == get_model()
    assert call["response_format"] == RESPONSE_FORMAT
    assert call["response_format"]["mime_type"] == "application/json"
    assert call["response_format"]["schema"] is PLAN_SCHEMA


def test_model_is_overridable_from_the_environment(constraints, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    client = FakeClient(json_response(VALID_PAYLOAD))

    plan("survey the north field", constraints, client=client)

    assert client.last_call["model"] == "gemini-3.1-pro-preview"


def test_prompt_carries_the_mission_and_every_constraint(constraints):
    client = FakeClient(json_response(VALID_PAYLOAD))

    plan("survey the north field", constraints, client=client)

    prompt = client.last_prompt
    assert "survey the north field" in prompt
    for fragment in ["100", "5", "10", "75", "(-50, -50)", "(150, 150)"]:
        assert fragment in prompt
    assert "subtask" in client.last_call["system_instruction"]


def test_system_prompt_demands_two_stage_decomposition(constraints):
    client = FakeClient(json_response(VALID_PAYLOAD))

    plan("survey the north field", constraints, client=client)

    system = client.last_call["system_instruction"]
    assert "Stage 1" in system and "Stage 2" in system
    assert "example" in system.lower()


# --- the schema is what makes the mapping reliable --------------------------


def _walk_objects(schema: dict):
    if schema.get("type") == "object":
        yield schema
    for sub in schema.get("properties", {}).values():
        yield from _walk_objects(sub)
    if "items" in schema:
        yield from _walk_objects(schema["items"])


def test_every_object_in_the_schema_is_closed_and_fully_required():
    for obj in _walk_objects(PLAN_SCHEMA):
        assert obj.get("additionalProperties") is False
        assert set(obj.get("required", [])) == set(obj.get("properties", {}))


def test_schema_waypoint_fields_match_the_waypoint_model():
    waypoint_schema = PLAN_SCHEMA["properties"]["subtasks"]["items"]["properties"][
        "waypoints"
    ]["items"]
    assert set(waypoint_schema["properties"]) == set(Waypoint.model_fields)
    assert waypoint_schema["properties"]["action"]["enum"] == [a.value for a in Action]


# --- malformed model output must raise, never produce a bad plan ------------


def test_free_text_instead_of_json_raises(constraints):
    client = FakeClient(FakeInteraction(output_text="Here is my plan: fly north, then land."))

    with pytest.raises(PlannerResponseError, match="did not return valid JSON"):
        plan("survey the north field", constraints, client=client)


def test_json_fenced_in_markdown_raises(constraints):
    """A fenced block is not valid JSON -- it must not be silently unwrapped."""
    client = FakeClient(FakeInteraction(output_text='```json\n{"subtasks": []}\n```'))

    with pytest.raises(PlannerResponseError, match="did not return valid JSON"):
        plan("survey the north field", constraints, client=client)


def test_empty_response_raises(constraints):
    client = FakeClient(FakeInteraction(output_text="   "))

    with pytest.raises(PlannerResponseError, match="empty response"):
        plan("survey the north field", constraints, client=client)


def test_missing_output_raises(constraints):
    client = FakeClient(FakeInteraction(output_text=None))

    with pytest.raises(PlannerResponseError, match="empty response"):
        plan("survey the north field", constraints, client=client)


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled", "budget_exceeded"])
def test_a_failed_interaction_raises_with_the_error_detail(status, constraints):
    client = FakeClient(
        FakeInteraction(
            output_text=None,
            status=status,
            errors=[FakeError(code="SAFETY", message="blocked by safety filter")],
        )
    )

    with pytest.raises(PlannerResponseError) as exc_info:
        plan("survey the north field", constraints, client=client)

    assert status in str(exc_info.value)
    assert "blocked by safety filter" in str(exc_info.value)


def test_a_failed_interaction_without_detail_still_raises_clearly(constraints):
    client = FakeClient(FakeInteraction(output_text=None, status="failed"))

    with pytest.raises(PlannerResponseError, match="no error detail given"):
        plan("survey the north field", constraints, client=client)


def test_non_object_json_raises(constraints):
    client = FakeClient(json_response(["takeoff", "survey", "land"]))

    with pytest.raises(PlannerResponseError, match="expected an object"):
        plan("survey the north field", constraints, client=client)


@pytest.mark.parametrize(
    "mutation,description",
    [
        ({"subtasks": [{"waypoints": []}]}, "subtask missing its name"),
        ({"subtasks": [{"name": "takeoff"}]}, "subtask missing its waypoints"),
        ({"subtasks": [{"name": "takeoff", "waypoints": []}]}, "empty waypoint list"),
        (
            {
                "subtasks": [
                    {"name": "t", "waypoints": [{"x": 0, "y": 0, "z": 1, "speed": 1, "action": "TELEPORT"}]}
                ]
            },
            "action outside the enum",
        ),
        (
            {
                "subtasks": [
                    {"name": "t", "waypoints": [{"x": 0, "y": 0, "z": 1, "speed": 0, "action": "MOVE"}]}
                ]
            },
            "zero speed",
        ),
        (
            {
                "subtasks": [
                    {"name": "t", "waypoints": [{"x": "north", "y": 0, "z": 1, "speed": 1, "action": "MOVE"}]}
                ]
            },
            "non-numeric coordinate",
        ),
        ({"subtasks": "takeoff then land"}, "subtasks is not a list"),
    ],
)
def test_invalid_payloads_raise_plan_validation_error(mutation, description, constraints):
    client = FakeClient(json_response(mutation))

    with pytest.raises(PlanValidationError) as exc_info:
        plan("survey the north field", constraints, client=client)

    assert "not a valid Plan" in str(exc_info.value), description


@pytest.mark.parametrize("payload", [{}, {"subtasks": []}])
def test_an_empty_plan_is_rejected(payload, constraints):
    """Schema-valid but unflyable: it would execute zero waypoints in silence."""
    client = FakeClient(json_response(payload))

    with pytest.raises(PlanValidationError, match="no subtasks"):
        plan("survey the north field", constraints, client=client)


def test_a_waypoint_without_an_action_defaults_to_move(constraints):
    """The schema requires `action`, but Waypoint defaults it rather than failing."""
    payload = {"subtasks": [{"name": "t", "waypoints": [{"x": 0, "y": 0, "z": 1, "speed": 1}]}]}
    client = FakeClient(json_response(payload))

    result = plan("survey the north field", constraints, client=client)

    assert result.waypoints()[0].action is Action.MOVE


def test_planner_errors_share_a_base_class(constraints):
    client = FakeClient(json_response({"subtasks": "not a list"}))

    with pytest.raises(PlannerError):
        plan("survey the north field", constraints, client=client)


# --- replan -----------------------------------------------------------------


@pytest.fixture
def rejected_plan() -> Plan:
    return Plan(
        mission_text="survey the north field",
        subtasks=[
            SubTask(
                name="survey",
                waypoints=[Waypoint(x=500.0, y=0.0, z=300.0, speed=40.0, action=Action.CAPTURE)],
            )
        ],
    )


def test_replan_returns_a_corrected_plan(constraints, rejected_plan):
    client = FakeClient(json_response(VALID_PAYLOAD))

    result = replan(
        "survey the north field",
        rejected_plan,
        ["waypoint 1 is outside the geofence"],
        constraints,
        client=client,
    )

    assert isinstance(result, Plan)
    assert result.mission_text == "survey the north field"


def test_replan_prompt_includes_the_rejected_plan_and_every_violation(constraints, rejected_plan):
    client = FakeClient(json_response(VALID_PAYLOAD))
    violations = ["waypoint 1 is outside the geofence", "speed 40 m/s exceeds the 10 m/s limit"]

    replan("survey the north field", rejected_plan, violations, constraints, client=client)

    prompt = client.last_prompt
    assert "REJECTED" in prompt
    for violation in violations:
        assert violation in prompt
    assert "(500, 0, 300)" in prompt
    assert "CAPTURE" in prompt


def test_replan_uses_the_same_strict_schema(constraints, rejected_plan):
    client = FakeClient(json_response(VALID_PAYLOAD))

    replan("survey the north field", rejected_plan, ["too fast"], constraints, client=client)

    assert client.last_call["response_format"] == RESPONSE_FORMAT


def test_replan_rejects_malformed_corrections(constraints, rejected_plan):
    client = FakeClient(json_response({"subtasks": [{"name": "x", "waypoints": []}]}))

    with pytest.raises(PlanValidationError):
        replan("survey the north field", rejected_plan, ["too fast"], constraints, client=client)


# --- API / transport failures ---


class _ExplodingClient:
    """A client whose create() raises the given exception."""

    def __init__(self, exc: BaseException):
        self.interactions = self
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc


def test_a_rate_limit_is_wrapped_in_a_planner_error(constraints):
    """The interactions SDK raises from a private module -- it must not escape raw."""
    from google.genai._gaos.lib import compat_errors

    response = compat_errors.httpx.Response(
        429, request=compat_errors.httpx.Request("POST", "https://example.invalid")
    )
    client = _ExplodingClient(
        compat_errors.RateLimitError("quota exceeded", response=response, body=None)
    )

    with pytest.raises(PlannerAPIError, match="Gemini API call failed"):
        plan("survey the north field", constraints, client=client)


def test_a_public_sdk_error_is_wrapped_too(constraints):
    from google.genai import errors as genai_errors

    client = _ExplodingClient(genai_errors.APIError(503, {"message": "backend down"}))

    with pytest.raises(PlannerAPIError, match="Gemini API call failed"):
        plan("survey the north field", constraints, client=client)


def test_api_errors_are_planner_errors(constraints):
    """So callers can catch PlannerError and get every failure mode."""
    assert issubclass(PlannerAPIError, PlannerError)


def test_a_genuine_bug_is_not_swallowed(constraints):
    """Wrapping must not turn a programming error into a plausible API failure."""
    client = _ExplodingClient(TypeError("unexpected keyword argument"))

    with pytest.raises(TypeError):
        plan("survey the north field", constraints, client=client)


# --- credentials ------------------------------------------------------------


def test_get_client_requires_the_env_var(monkeypatch):
    monkeypatch.delenv(API_KEY_VAR, raising=False)

    with pytest.raises(PlannerResponseError, match=API_KEY_VAR):
        get_client()


def test_a_blank_key_counts_as_unset(monkeypatch):
    """The placeholder line in a freshly copied .env must fail clearly, not 401."""
    monkeypatch.setenv(API_KEY_VAR, "   ")

    with pytest.raises(PlannerResponseError, match=API_KEY_VAR):
        get_client()


def test_no_api_key_is_baked_into_the_source():
    from pathlib import Path

    for path in Path("planner").rglob("*.py"):
        text = path.read_text()
        assert "AIza" not in text, f"{path} looks like it contains a Google API key"


# --- constraints ------------------------------------------------------------


def test_constraints_reject_an_inverted_altitude_envelope():
    with pytest.raises(ValueError, match="min_altitude"):
        MissionConstraints(
            geofence=[GeoPoint(x=0, y=0), GeoPoint(x=1, y=0), GeoPoint(x=0, y=1)],
            max_altitude=10.0,
            min_altitude=50.0,
            max_velocity=5.0,
            battery_budget_pct=50.0,
        )


def test_constraints_reject_a_degenerate_geofence():
    with pytest.raises(ValueError):
        MissionConstraints(
            geofence=[GeoPoint(x=0, y=0), GeoPoint(x=1, y=0)],
            max_altitude=100.0,
            min_altitude=0.0,
            max_velocity=5.0,
            battery_budget_pct=50.0,
        )


def test_constraints_reject_an_impossible_battery_budget():
    with pytest.raises(ValueError):
        MissionConstraints(
            geofence=[GeoPoint(x=0, y=0), GeoPoint(x=1, y=0), GeoPoint(x=0, y=1)],
            max_altitude=100.0,
            min_altitude=0.0,
            max_velocity=5.0,
            battery_budget_pct=120.0,
        )


def test_describe_renders_every_field(constraints):
    described = constraints.describe()
    for fragment in ["Geofence", "Altitude", "Speed", "Battery", "100", "75"]:
        assert fragment in described
