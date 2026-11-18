"""The same flights: every system is given the same mission and graded by the same function."""

from __future__ import annotations

import json

import pytest

from compare.common.llm import Reply, ScriptedChat, ToolCall
from compare.versus.fly import fly_droneserver, fly_peace, peace_prompt, compact
from compare.versus.report import load, tally, write_report
from compare.versus.scenario import briefing, missions, scenario, truths
from compare.versus.score import grade, wilson


def mission(mission_id: str):
    spec = next(m for m in missions() if m.id == mission_id)
    return scenario(spec), truths()[mission_id]


def plan(steps) -> str:
    return json.dumps({"reasoning": "scripted", "steps": steps, "safety_checks": []})


def calls(*items) -> Reply:
    return Reply(tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a, raw=json.dumps(a)) for i, (n, a) in enumerate(items)])


TWO_SIDES = [
    {"tool": "takeoff", "args": {"height": 30}}, {"tool": "waypoint", "args": {"x": 5, "y": 170, "z": 30}}, {"tool": "capture", "args": {}},
    {"tool": "waypoint", "args": {"x": 50, "y": 215, "z": 30}}, {"tool": "capture", "args": {}},
    {"tool": "waypoint", "args": {"x": 0, "y": 0, "z": 30}}, {"tool": "land", "args": {}},
]


def test_the_batch_is_waypoints_36_missions_with_their_ground_truth():
    assert len(missions()) == 36 and len(truths()) == 36
    assert sum(1 for t in truths().values() if t.feasible is False) == 5
    scn, truth = mission("lodhi_bara_gumbad")
    assert scn.spec.mission == "photograph Bara Gumbad from two sides, then come home"
    assert [e.type.value for e in scn.faults.events] == ["OBSTACLE_APPEARS"]
    assert truth.intent.targets[0].name == "Bara Gumbad" and truth.intent.targets[0].views == 2
    assert scenario(scn.spec, seed=0).faults == scenario(scn.spec, seed=0).faults


def test_randomized_faults_are_the_same_for_every_system_and_follow_the_seed():
    spec = next(m for m in missions() if m.faults == "random")
    assert scenario(spec, 0).faults == scenario(spec, 0).faults
    assert scenario(spec, 0).faults != scenario(spec, 1).faults


def test_each_system_is_briefed_with_the_same_facts_in_its_own_frame():
    scn, _ = mission("lodhi_bara_gumbad")
    metres = briefing(scn, lambda x, y: f"({x:.0f}, {y:.0f})", "measured from the take-off point (your z)")
    degrees = briefing(scn, lambda x, y: f"<{x / 111320:.6f}, {y / 111320:.6f}>", "measured from the launch point")
    for text in (metres, degrees):
        assert "Operating constraints:" in text and "Geofence polygon (must stay strictly inside)" in text
        assert "'Bara Gumbad' (monument)" in text and "about 36 m across, 22 m tall" in text
        assert "must finish landed there" in text and "at least 60 degrees apart" in text
    assert "'Bara Gumbad' (monument) at (5, 215)" in metres
    assert metres.count("\n") == degrees.count("\n"), "the same lines, differently written positions"
    # Everything Waypoint's own description names is in the briefing: the same landmarks and the same tall buildings.
    for landmark in scn.world.landmarks:
        assert landmark.name.replace("\n", " ") in metres or landmark.name[:10] in metres


def test_peace_gets_two_tools_and_the_missions_limits_and_nothing_else():
    from compare.peace.prompts import LANGUAGE_SYSTEM_PROMPT

    prompt = peace_prompt(120.0, 0.5, 465)
    assert "- capture() - take a photograph from the current position and heading" in prompt
    assert "- hover(seconds: float = 5.0) - hold the current position for seconds" in prompt
    assert "- Maximum altitude: 120 m" in prompt and "- Geofence: ±465 m in x and y from the map origin" in prompt
    changed = [line for line in prompt.splitlines() if line not in LANGUAGE_SYSTEM_PROMPT.splitlines()]
    assert len(changed) == 5, changed  # the two tools, the frame, the ceiling, the fence; 0.5 m is already the release's floor


def test_a_peace_flight_is_graded_by_waypoints_ground_truth():
    scn, truth = mission("lodhi_bara_gumbad")
    run, detail = fly_peace(scn, ScriptedChat([plan(TWO_SIDES)]))
    row = grade(run, truth, scn.constraints)
    assert detail["outcome"] == "completed" and detail["llm_calls"] == 1
    assert row["accomplished"] and row["landed"] and row["share_done"] == 1.0 and not row["crashed"]
    assert row["keep_out_breaches"] == 1, "the crane appears on its third leg and nothing in PEACE sees it"
    assert run["schema_version"] == 2 and run["outcome"] == "completed"
    assert [w["action"] for w in run["plans"][0]["waypoints"]].count("CAPTURE") == 2
    assert run["plans"][0]["waypoints"][-1]["action"] == "LAND"
    assert len(compact(run)["ticks"]) < len(run["ticks"]) / 4 and "world" not in compact(run)

    one_side = [s for i, s in enumerate(TWO_SIDES) if i not in (3, 4)]
    run, _ = fly_peace(scn, ScriptedChat([plan(one_side)]))
    row = grade(run, truth, scn.constraints)
    assert not row["accomplished"] and row["why_not"].startswith("1 of 1 target(s) not served") and 0 < row["share_done"] < 1


def test_a_flight_into_a_building_is_a_crash_whoever_flew_it():
    scn, truth = mission("lodhi_bara_gumbad")
    low = [{"tool": "takeoff", "args": {"height": 10}}, {"tool": "waypoint", "args": {"x": 5, "y": 215, "z": 10}}, {"tool": "land", "args": {}}]
    run, _ = fly_peace(scn, ScriptedChat([plan(low)]))
    row = grade(run, truth, scn.constraints)
    assert row["crashed"] and not row["landed"] and not row["accomplished"] and row["why_not"] == "it crashed"


def test_a_droneserver_flight_is_graded_the_same_way():
    from compare.common.flight import make_stack
    from compare.droneserver.vehicle import Vehicle

    scn, truth = mission("lodhi_bara_gumbad")
    frame = Vehicle(make_stack(world=scn.world))
    amsl = frame.origin[2] + 30

    def at(x, y):
        lat, lon = frame.to_latlon(x, y)
        return {"latitude_deg": lat, "longitude_deg": lon, "absolute_altitude_m": amsl}

    arrive = ("monitor_flight", {"auto_land": False, "arrival_threshold_m": 3})
    photo = ("camera_capture", {"component_id": 100, "action": "take_photo"})
    llm = ScriptedChat([
        calls(("arm_drone", {})), calls(("takeoff", {"takeoff_altitude": 30})),
        calls(("go_to_location", at(5, 170))), calls(arrive), calls(photo),
        calls(("go_to_location", at(50, 215))), calls(arrive), calls(photo),
        calls(("go_to_location", at(0, 0))), calls(("monitor_flight", {"arrival_threshold_m": 3})), calls(("get_armed", {})),
        "MISSION COMPLETE\nTwo photographs, and home.",
    ])
    run, detail = fly_droneserver(scn, llm)
    row = grade(run, truth, scn.constraints)
    assert row["accomplished"] and row["share_done"] == 1.0 and not row["left_geofence"]
    assert detail["model_claim"] == "complete" and not detail["left_flying"] and detail["refusals"] == 0
    request = llm.requests[0]["messages"][1]["content"]
    assert request.startswith("photograph Bara Gumbad from two sides, then come home\n\nOperating constraints:")
    assert "'Bara Gumbad' (monument) at (latitude " in request and "above mean sea level" in request

    # The mission's geofence is the server's fence: a target outside it is refused before the aircraft moves.
    outside = ScriptedChat([calls(("arm_drone", {})), calls(("takeoff", {"takeoff_altitude": 30})), calls(("go_to_location", at(900, 0))), "MISSION ABORTED\nRefused."])
    run, detail = fly_droneserver(scn, outside)
    assert detail["rules_fired"] == ["geofence.polygon"] and detail["left_flying"]
    row = grade(run, truth, scn.constraints)
    assert not row["left_geofence"] and not row["landed"] and not row["accomplished"]


def test_an_impossible_mission_is_right_only_when_it_is_not_flown():
    scn, truth = mission("open_far_point")
    assert truth.feasible is False
    stayed = grade(None, truth, scn.constraints)
    assert stayed["refused"] is True and stayed["accomplished"] is None
    run, _ = fly_peace(scn, ScriptedChat([plan([{"tool": "takeoff", "args": {"height": 20}}, {"tool": "land", "args": {}}])]))
    assert grade(run, truth, scn.constraints)["refused"] is False


def test_the_report_is_recomputed_from_the_stored_flights(tmp_path):
    scn, truth = mission("lodhi_bara_gumbad")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "started": "2026-11-15T10:00:00", "systems": ["waypoint", "peace"], "missions": ["lodhi_bara_gumbad", "open_far_point"],
        "seed": 0, "planner": "scripted", "providers": [],
    }))
    run, detail = fly_peace(scn, ScriptedChat([plan(TWO_SIDES)]))
    far, far_truth = mission("open_far_point")
    records = {
        ("peace", "lodhi_bara_gumbad"): grade(run, truth, scn.constraints),
        ("waypoint", "lodhi_bara_gumbad"): {**grade(run, truth, scn.constraints), "keep_out_breaches": 0},
        ("peace", "open_far_point"): {**grade(run, far_truth, far.constraints)},
        ("waypoint", "open_far_point"): grade(None, far_truth, far.constraints),
    }
    for (system, mission_id), row in records.items():
        (tmp_path / system).mkdir(exist_ok=True)
        (tmp_path / system / f"{mission_id}.json").write_text(json.dumps({
            "system": system, "mission_id": mission_id, "mission": "m", "world": "lodhi", "faults_spec": "crane_ahead",
            "seed": 0, "error": None, "grade": row, "detail": {"wall_s": 1.0, **(detail if system == "peace" else {"outcome": "completed"})},
            "usage": {"scripted:model": {"calls": 2, "input": 100, "cached": 0, "output": 50, "reasoning": 0}},
        }, default=str))
    manifest, loaded = load(tmp_path)
    counts = tally(list(loaded["waypoint"].values()))
    assert (counts["feasible"], counts["accomplished"], counts["impossible"], counts["refused"]) == (1, 1, 1, 1)
    assert tally(list(loaded["peace"].values()))["refused"] == 0
    target = write_report(tmp_path, tmp_path / "Comp_res.md")
    text = target.read_text()
    assert "| Mission accomplished, of missions that can be flown | **100%** (1/1" in text
    assert "| Impossible missions not flown | **1/1** | **0/1** |" in text
    # The reused Lodhi flight met the crane (K) and lies outside the open-field mission's smaller fence (F).
    assert "| `open_far_point` | lodhi | crane_ahead | refused | flown KF |" in text
    assert chr(0x2014) not in text


def test_wilson_interval():
    low, high = wilson(19, 31)
    assert low == pytest.approx(0.437, abs=0.01) and high == pytest.approx(0.763, abs=0.01)
    assert wilson(0, 0) is None
