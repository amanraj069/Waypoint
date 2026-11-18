"""PEACE, rebuilt: the released behaviour, checked against the released code's rules."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from compare.common.flight import make_stack
from compare.common.llm import ScriptedChat
from compare.common.sensors import SceneObject
from compare.peace import PeaceAgent, PeaceSettings
from compare.peace.evaluation.run_dataset import CATEGORIES, filter_prompts, load_prompts, run
from compare.peace.exceptions import SafetyViolationError
from compare.peace.prompts import LANGUAGE_SYSTEM_PROMPT, format_world_context
from compare.peace.schemas import MissionPlan, RobotState, WorldModel
from compare.peace.services.constraint_enforcement import ConstraintEnforcementService
from compare.peace.services.depth import DepthProjectionService
from compare.peace.services.mission_logger import COLUMNS
from compare.peace.services.world_model import WorldModelAggregatorService
from compare.peace.tools.uav_tools import UAVTools
from compare.tests.conftest import mission_plan

TAKEOFF = {"tool": "takeoff", "args": {"height": 3.0}}


def agent_with(replies, **kwargs) -> tuple[PeaceAgent, ScriptedChat]:
    llm = ScriptedChat(replies)
    return PeaceAgent(make_stack(), llm, kwargs.pop("settings", None), **kwargs), llm


# -- the released artefacts ------------------------------------------------------


def test_the_tool_set_is_the_released_one():
    assert set(UAVTools.tool_names()) == {
        "forward", "backward", "left", "right", "rotate", "waypoint", "navigate_to_object", "localize_object",
        "up", "down", "arm", "disarm", "takeoff", "land",
    }


def test_the_system_prompt_is_the_released_one_but_for_its_dashes():
    # The release's language.txt, with each of its 21 em dashes back in place, hashes to the release's file.
    em_dash = chr(0x2014)
    restored = LANGUAGE_SYSTEM_PROMPT.replace(" - ", f" {em_dash} ")
    lines = LANGUAGE_SYSTEM_PROMPT.splitlines()
    assert len(lines) == 96
    assert em_dash not in LANGUAGE_SYSTEM_PROMPT
    assert restored.count(em_dash) == 21
    assert hashlib.sha256(restored.encode()).hexdigest() == (
        "f2e724fb65e328cefa38b2ff3318fc13b6f859bdd5b5915827b7ffbca7487428"
    )
    for tool in UAVTools.tool_names():
        assert f"- {tool}(" in LANGUAGE_SYSTEM_PROMPT


def test_the_released_defaults():
    s = PeaceSettings()
    assert (s.action_min_altitude, s.action_max_altitude, s.action_horizontal_limit) == (0.5, 50.0, 500.0)
    assert s.agent_max_replan_depth == 2
    assert s.action_command_timeout == 30.0
    assert (s.agent_enable_memory, s.agent_memory_window) == (True, 6)
    assert (s.language_model_temperature, s.language_model_num_predict) == (0.7, 2000)
    assert s.vision_inference_rate == 0.2


def test_settings_read_the_released_environment_names(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_REPLAN_DEPTH", "0")
    monkeypatch.setenv("ACTION_MAX_ALTITUDE", "20")
    monkeypatch.setenv("AGENT_ENABLE_MEMORY", "false")
    s = PeaceSettings.from_env()
    assert (s.agent_max_replan_depth, s.action_max_altitude, s.agent_enable_memory) == (0, 20.0, False)


def test_the_dataset_is_the_published_108_prompts():
    prompts = load_prompts()
    assert len(prompts) == 108
    counts = {c: sum(1 for p in prompts if p["category"] == c) for c in CATEGORIES}
    assert counts == {
        "takeoff_and_landing": 12, "relative_motion": 12, "altitude_and_attitude_control": 12,
        "waypoint_navigation": 13, "geometric_path_following": 13, "object_search_and_localisation": 12,
        "open_vocabulary_navigation": 13, "composite_multi_step_mission": 13, "safety_violation": 8,
    }
    assert prompts[0]["prompt"] == "Fly to position x = 3, y = 4, z = 3."
    assert len(filter_prompts(prompts, category="safety_violation", limit=3)) == 3
    assert filter_prompts(prompts, start_id="sv_007")[0]["id"] == "sv_007"


# -- constraint enforcement (paper section III-F) -------------------------------


def test_altitude_is_clamped_and_the_fence_is_a_square():
    svc = ConstraintEnforcementService(0.5, 50.0, 500.0)
    assert svc.clamp_altitude(80.0) == 50.0
    assert svc.clamp_altitude(-2.0) == 0.5
    assert svc.clamp_altitude(12.0) == 12.0
    svc.validate_position(500.0, -500.0, 5.0)  # on the boundary: inside
    svc.validate_position(450.0, 450.0, 5.0)  # 636 m from home, inside the square
    with pytest.raises(SafetyViolationError, match="exceeds"):
        svc.validate_position(500.1, 0.0, 5.0)
    with pytest.raises(SafetyViolationError):
        svc.validate_position(0.0, -750.0, 5.0)


def test_an_out_of_band_altitude_is_clamped_not_refused():
    agent, _ = agent_with([mission_plan([TAKEOFF, {"tool": "waypoint", "args": {"x": 0, "y": 0, "z": 80}}])])
    result = agent.run("Fly to position x = 0, y = 0, z = 80.")
    assert result.completed and not result.safety_aborts
    assert agent.stack.position[2] <= 50.5
    assert result.replan_count == 0


def test_a_fence_breach_aborts_the_step_and_goes_to_the_replanner():
    agent, llm = agent_with(
        [
            mission_plan([TAKEOFF, {"tool": "waypoint", "args": {"x": 800, "y": 0, "z": 5}}, {"tool": "land"}]),
            mission_plan([{"tool": "land"}]),
        ]
    )
    result = agent.run("Take off to 3 metres, then fly to x = 800, y = 0, z = 5, then land.")
    assert result.completed
    assert result.replan_count == 1 and llm.calls == 2
    assert [s["tool"] for s in result.steps] == ["takeoff", "waypoint", "land"]
    assert result.steps[1]["result"].startswith("ERROR: Safety violation - Position (800.0, 0.0) exceeds")
    assert max(abs(s.x) for s in agent.stack.states) < 5.0, "the refused command never moved the aircraft"
    replan = llm.requests[1]["messages"][1]["content"]
    assert replan.startswith("Original query: Take off to 3 metres")
    assert "Completed steps:\n  - takeoff({'height': 3.0}): " in replan
    assert "Remaining steps (not yet executed):\n  - waypoint(" in replan
    assert "Failure reason: ERROR: Safety violation" in replan
    assert replan.endswith(
        "Produce a revised MissionPlan for the remaining objectives only. "
        "Return ONLY the JSON mission plan, no additional text."
    )


def test_body_relative_motion_is_checked_too():
    agent, _ = agent_with(
        [mission_plan([TAKEOFF, {"tool": "forward", "args": {"dist": 800}}])], settings=PeaceSettings(agent_max_replan_depth=0)
    )
    result = agent.run("Move 800 metres forward.")
    assert not result.completed
    assert len(result.safety_aborts) == 1


# -- planner-executor (paper section III-B) ---------------------------------------


def test_one_llm_call_plans_the_whole_mission():
    agent, llm = agent_with(
        [mission_plan([{"tool": "takeoff", "args": {"height": 5.0}}, {"tool": "forward", "args": {"dist": 3.0}}])]
    )
    result = agent.run("Take off to 5 meters then move forward 3 meters")
    assert result.completed and result.message == "Mission complete"
    assert llm.calls == 1 and result.llm_calls == 1
    x, y, z = agent.stack.position
    assert abs(x - 3.0) < 0.5 and abs(y) < 0.3 and abs(z - 5.0) < 0.5
    assert agent.stack.mode == "AUTO.LOITER", "the executor stops safely: loiter, stream off"
    request = llm.requests[0]
    assert request["schema"] == MissionPlan.model_json_schema()
    assert (request["temperature"], request["max_tokens"]) == (0.7, 2000)
    system, user = request["messages"]
    assert system == {"role": "system", "content": LANGUAGE_SYSTEM_PROMPT}
    assert user["content"].startswith("Take off to 5 meters then move forward 3 meters\n\nPosition: (0.00m, 0.00m, 0.00m), Yaw=0.0°")
    assert "--- Ego State History (oldest → newest) ---" in user["content"]


def test_replanning_is_bounded_at_two():
    bad = mission_plan([{"tool": "waypoint", "args": {"x": 900, "y": 0, "z": 5}}])
    agent, llm = agent_with([bad, bad, bad, bad])
    result = agent.run("go far")
    assert not result.completed
    assert llm.calls == 3, "the first plan and two replans, then the mission fails"
    assert result.replan_count == 2
    assert result.message.startswith("ERROR: Safety violation")


def test_a_plan_that_does_not_validate_fails_the_mission():
    for reply in ("not json", json.dumps({"steps": []}), mission_plan([])):
        agent, llm = agent_with([reply])
        result = agent.run("anything")
        assert result.state == "failed" and llm.calls == 1
        assert agent.stack.on_ground


def test_an_unknown_tool_or_bad_arguments_is_a_tool_failure():
    agent, _ = agent_with(
        [mission_plan([{"tool": "fly_home"}]), mission_plan([{"tool": "takeoff", "args": {"altitude": 3}}]), mission_plan([TAKEOFF])]
    )
    result = agent.run("x")
    assert result.steps[0]["result"] == "ERROR: Unknown tool 'fly_home'"
    assert result.steps[1]["result"].startswith("ERROR: Bad arguments for 'takeoff'")
    assert result.completed and result.replan_count == 2


def test_an_arrival_timeout_is_not_a_failure():
    """As released: only "failed" or a leading "ERROR" fails a step, so a timeout lets the plan carry on."""
    agent, llm = agent_with([mission_plan([TAKEOFF, {"tool": "waypoint", "args": {"x": 450, "y": 0, "z": 5}}, {"tool": "land"}])])
    result = agent.run("go 450 m east")
    assert "(arrival timeout)" in result.steps[1]["result"]
    assert result.completed and llm.calls == 1
    assert 300 < agent.stack.position[0] < 450


def test_waypoint_without_takeoff_does_not_arm():
    agent, _ = agent_with([mission_plan([{"tool": "waypoint", "args": {"x": 3, "y": 4, "z": 3}}])])
    result = agent.run("Fly to position x = 3, y = 4, z = 3.")
    assert "(arrival timeout)" in result.steps[0]["result"]
    assert agent.stack.on_ground and not agent.stack.armed


def test_relative_motion_follows_the_heading():
    agent, _ = agent_with(
        [mission_plan([TAKEOFF, {"tool": "rotate", "args": {"degrees": 90}}, {"tool": "forward", "args": {"dist": 6}},
                       {"tool": "right", "args": {"dist": 4}}, {"tool": "up", "args": {"dist": 2}}])]
    )
    result = agent.run("turn left, go, strafe, climb")
    assert result.completed
    assert result.steps[1]["result"] == "Rotated +90°. Yaw=90°"
    x, y, z = agent.stack.position
    assert abs(x - 4.0) < 0.6 and abs(y - 6.0) < 0.6 and abs(z - 5.0) < 0.5


def test_land_waits_for_touchdown():
    agent, _ = agent_with([mission_plan([TAKEOFF, {"tool": "land", "args": {"x": 6.0, "y": 0.0}}])])
    result = agent.run("take off and land 6 m east")
    assert result.completed
    assert result.steps[1]["result"].startswith("Landed at (6.0,0.0)")
    assert agent.stack.on_ground and not agent.stack.armed
    assert abs(agent.stack.position[0] - 6.0) < 0.6


# -- perception (paper sections III-C, III-D) -------------------------------------


def test_depth_projection_puts_an_object_where_it_is():
    svc = DepthProjectionService(fx=432.6, fy=432.6, cx=320, cy=240, width=640, height=480)
    import numpy as np

    from compare.peace.schemas import Object

    depth = np.zeros((480, 640), dtype=np.float32)
    depth[200:300, 300:340] = 10.0
    (seen,) = svc.project_detections([Object(id=0, label="person", bbox=[300, 200, 340, 300], confidence=90)], depth, 1.0, 2.0, 3.0, math.radians(90))
    assert seen.depth_valid and seen.depth_meters == 10.0
    assert abs(seen.world_pos[0] - 1.0) < 0.05 and abs(seen.world_pos[1] - 12.0) < 0.05 and seen.world_pos[2] == 3.0
    (low,) = svc.project_detections([Object(id=0, label="person", bbox=[300, 200, 340, 300], confidence=50)], depth, 0, 0, 0, 0)
    assert not low.depth_valid, "below the confidence threshold the position is not trusted"
    (none,) = svc.project_detections([Object(id=0, label="car", bbox=[10, 10, 40, 40], confidence=90)], depth, 0, 0, 0, 0)
    assert not none.depth_valid and none.depth_meters is None


def test_lidar_sectors_are_the_released_eight():
    svc = WorldModelAggregatorService()
    samples = 360
    ranges = [float("inf")] * samples
    ranges[180] = 4.0  # 0 rad: dead ahead
    ranges[90] = 7.0  # -pi/2: to the right
    ranges[0] = 9.0  # -pi: behind
    svc.update_lidar(ranges, -math.pi, 2 * math.pi / samples, 0.1, 30.0)
    lidar = svc.get_world_model().lidar_ranges
    assert (lidar["front"], lidar["right"], lidar["back"], lidar["min_range"]) == (4.0, 7.0, 9.0, 4.0)
    assert lidar["left"] == float("inf")
    context = format_world_context(RobotState(), svc.get_world_model())
    assert "Closest obstacle: 4.0m\n- Front: 4.0m\n- Front-Right: infm\n- Right: 7.0m" in context


def test_detections_reach_the_prompt_with_their_depth_flag():
    world = WorldModel(
        detections=[
            {"label": "person", "world_x": 8.0, "world_y": 6.0, "depth_valid": True, "confidence": 0.9},
            {"label": "car", "world_x": 0.0, "world_y": 0.0, "depth_valid": False, "confidence": 0.8},
        ]
    )
    context = format_world_context(RobotState(), world)
    assert "1. person at (8.0m, 6.0m) - valid position, confidence=90%" in context
    assert "2. car at (0.0m, 0.0m) - position UNCERTAIN (depth sensor failure), confidence=80%" in context


def test_search_then_approach_an_object():
    person = SceneObject("person", -9.0, 5.0)
    agent, _ = agent_with(
        [mission_plan([TAKEOFF, {"tool": "localize_object", "args": {"label": "person"}},
                       {"tool": "navigate_to_object", "args": {"label": "person", "stop_distance": 2.0}}, {"tool": "land"}])],
        objects=[person], settings=PeaceSettings(vision_inference_rate=2.0),
    )
    result = agent.run("find the person and approach to 2 m")
    assert result.completed
    assert result.steps[1]["result"].startswith("Found 'person' at yaw=")
    assert abs(math.dist(agent.stack.true_position[:2], (person.x, person.y)) - 2.0) < 0.8


def test_an_object_that_is_not_there_fails_the_search_after_a_full_turn():
    agent, _ = agent_with(
        [mission_plan([TAKEOFF, {"tool": "localize_object", "args": {"label": "bench"}}])],
        objects=[SceneObject("person", 9.0, 0.0)], settings=PeaceSettings(agent_max_replan_depth=0),
    )
    began = agent.stack.now
    result = agent.run("find the bench")
    assert result.steps[1]["result"] == "ERROR: 'bench' not found after full 360° scan"
    assert not result.completed
    assert agent.stack.now - began >= 36 * 1.5, "36 steps of 10 degrees, 1.5 s to settle at each"


def test_an_object_beyond_the_depth_sensor_cannot_be_approached():
    agent, _ = agent_with(
        [mission_plan([TAKEOFF, {"tool": "navigate_to_object", "args": {"label": "car"}}])],
        objects=[SceneObject("car", 30.0, 0.0, width=1.8, height=1.5)],
        settings=PeaceSettings(agent_max_replan_depth=0, vision_inference_rate=2.0),
    )
    result = agent.run("approach the car")
    assert result.steps[1]["result"] == "ERROR: Object 'car' not found or depth invalid"
    detections = agent.world_model_svc.get_world_model().detections
    assert detections and detections[0]["label"] == "car" and not detections[0]["depth_valid"]


# -- the mission log and the dataset runner ----------------------------------------


def test_the_mission_log_has_the_released_columns(tmp_path):
    llm = ScriptedChat([mission_plan([TAKEOFF])])
    agent = PeaceAgent(make_stack(), llm, log_dir=str(tmp_path))
    agent.run("Take off to 3 metres.", metadata={"prompt_id": "tk_x", "task_category": "takeoff_and_landing"})
    agent.close()
    with (tmp_path / "agent_log.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == COLUMNS
    row = rows[0]
    assert row["final_state"] == "completed" and row["replan_count"] == "0"
    assert json.loads(row["metadata_json"]) == {"prompt_id": "tk_x", "task_category": "takeoff_and_landing"}
    assert json.loads(row["steps_json"])[0]["tool"] == "takeoff"
    assert json.loads(row["final_robot_state_json"])["position_z"] > 2.0


def test_the_dataset_runner_counts_terminal_states(tmp_path):
    def brain(messages):
        query = messages[1]["content"]
        if "Original query" in query:
            return mission_plan([{"tool": "up", "args": {"dist": 0.1}}])
        if "x = 3, y = 4, z = 3" in query:
            return mission_plan([TAKEOFF, {"tool": "waypoint", "args": {"x": 3, "y": 4, "z": 3}}])
        if "x = 600" in query:
            return mission_plan([{"tool": "waypoint", "args": {"x": 600, "y": 0, "z": 5}}])
        if "z = 80" in query:
            return mission_plan([{"tool": "waypoint", "args": {"x": 0, "y": 0, "z": 80}}])
        return "no plan"

    prompts = [p for p in load_prompts() if p["id"] in ("wp_001", "rel_001", "sv_001", "sv_004")]
    summary = run(prompts, ScriptedChat(brain), out_dir=tmp_path, progress=lambda _: None)
    assert summary["terminal_states"] == {"completed": 3, "failed": 1}
    assert summary["by_category"]["waypoint_navigation"]["reached"] == 1
    assert summary["by_category"]["safety_violation"]["enforced"] == {"clamped": 1, "aborted": 1}
    assert summary["within_limits"] == 4
    assert summary["llm_calls"] == 5 and summary["replans"] == 1
    assert len((tmp_path / "missions.jsonl").read_text().splitlines()) == 4
    assert (tmp_path / "agent_log.csv").exists() and (tmp_path / "summary.json").exists()


def test_the_aircraft_carries_over_between_prompts_unless_reset(tmp_path):
    plans = [mission_plan([TAKEOFF, {"tool": "forward", "args": {"dist": 5}}]), mission_plan([{"tool": "forward", "args": {"dist": 5}}])]
    prompts = [{"id": "a", "category": "relative_motion", "prompt": "one"}, {"id": "b", "category": "relative_motion", "prompt": "two"}]
    carried = run(prompts, ScriptedChat(list(plans)), progress=lambda _: None)
    assert carried["missions"] == 2
    reset = run(prompts, ScriptedChat(list(plans)), reset_between=True, reset_mode="teleport", out_dir=tmp_path, progress=lambda _: None)
    rows = [json.loads(line) for line in (tmp_path / "missions.jsonl").read_text().splitlines()]
    assert rows[1]["final_position"][0] < 1.0, "a fresh aircraft on the pad never armed, so it did not move"
    assert reset["terminal_states"] == {"completed": 2}


def test_the_model_is_called_as_the_release_calls_it():
    """No reasoning effort of Waypoint's, and the MissionPlan schema itself, as Ollama's `format` gives it."""
    from compare.common.llm import OpenAIChat
    from compare.peace.agent import release_protocol

    configured = OpenAIChat(("gpt-5-nano",), api_key="unused", provider="openai", reasoning_effort="low", client=object())
    released = release_protocol(configured)
    assert released.reasoning_effort is None and released.loose_schema is True
    assert configured.reasoning_effort == "low" and configured.loose_schema is False  # the shared client is untouched
