"""The comparison systems, and scoring flights against hand-written ground truth."""

from __future__ import annotations

from conftest import VALID_PAYLOAD
from waypoint.experiments.spec import Expectation, MissionSpec, prepare
from waypoint.experiments.truth import Truth, extraction_match, planned_landing, score
from waypoint.intent import ground
from waypoint.orchestrator import run_mission
from waypoint.orchestrator.options import NO_GATE, RUNTIME_GUARD
from waypoint.planner import DEFAULT_CONSTRAINTS, Plan
from waypoint.sim import FaultSchedule
from waypoint.sim.backend import CustomSimBackend


def survey() -> Plan:
    return Plan.model_validate({**VALID_PAYLOAD, "mission_text": "survey the north field"})


def outside_and_high() -> Plan:
    """The survey, but one corner 400 m out of the fence and everything 150 m up."""
    plan = survey()
    plan.subtasks[1].waypoints[1].x = 400.0
    for subtask in plan.subtasks[:-1]:
        for wp in subtask.waypoints:
            wp.z = 150.0
    return plan


class Planners:
    def __init__(self, first: Plan):
        self.first = first
        self.calls: list[dict] = []

    def planner(self, mission, constraints, **kwargs):
        self.calls.append({"kind": "plan"})
        return self.first

    def replanner(self, mission, previous, violations, constraints, **kwargs):
        self.calls.append({"kind": "inflight" if kwargs.get("situation") else "ground", "situation": kwargs.get("situation")})
        return survey()


def fly(planners, system, tmp_path):
    return run_mission(
        "survey the north field", DEFAULT_CONSTRAINTS, FaultSchedule(), CustomSimBackend(),
        planner=planners.planner, replanner=planners.replanner, system=system, verbose=False, runs_dir=tmp_path,
    )


def test_the_runtime_guard_clamps_altitudes_and_refuses_a_command_out_of_the_fence(tmp_path):
    planners = Planners(outside_and_high())
    log = fly(planners, RUNTIME_GUARD, tmp_path)
    # Nothing was refused before flight: the plan flew at once, clamped to the ceiling.
    assert log.of("verification")[0]["passed"]
    assert max(s.z for s in log.report.states) <= DEFAULT_CONSTRAINTS.max_altitude + 1.5  # climb overshoot
    # Then the guard refused the waypoint outside the fence, and the planner was asked from the air.
    triggers = log.of("replan_triggered")
    assert triggers and triggers[0]["trigger"] == "COMMAND_REJECTED"
    assert [c["kind"] for c in planners.calls] == ["plan", "inflight"]
    assert "outside the geofence" in planners.calls[1]["situation"]
    assert log.outcome == "completed"


def test_no_gate_flies_the_plan_as_written(tmp_path):
    planners = Planners(outside_and_high())
    log = fly(planners, NO_GATE, tmp_path)
    assert log.of("verification")[0]["passed"]
    assert any(v["severity"] in ("ERROR", "CRITICAL") for v in log.of("verification")[0]["violations"])
    assert max(s.x for s in log.report.states) > 300  # it left the fence


def test_a_flight_is_scored_on_what_it_reached_not_on_what_it_planned():
    spec = MissionSpec(id="t", mission="photograph Bara Gumbad from two sides", world="lodhi",
                       expect=Expectation.model_validate({"targets": [{"landmark": "Bara Gumbad", "views": 2}]}))
    scenario = prepare(spec)
    truth = Truth(ground(spec.expect.payload(), scenario.world, landing=scenario.landing), True, scenario.world)
    pad = scenario.landing

    def run(*reached):
        events = [{"type": "WAYPOINT", "description": "", "payload": {"action": a, "position": list(p)}} for p, a in reached]
        return {"events": events, "ticks": [{"x": pad[0], "y": pad[1], "z": pad[2]}],
                "plans": [{"waypoints": [{"x": pad[0], "y": pad[1], "z": pad[2], "action": "LAND"}]}]}

    both = run(((5, 170, 30), "CAPTURE"), ((5, 255, 25), "CAPTURE"), (pad, "LAND"))
    one = run(((5, 170, 30), "CAPTURE"), (pad, "LAND"))
    assert score(truth, both)["accomplished"]
    graded = score(truth, one)
    assert not graded["accomplished"] and graded["score"] == 0.5
    assert planned_landing(both) == (pad[0], pad[1])


def test_extraction_is_matched_target_by_target():
    spec = MissionSpec(id="t", mission="m", world="lodhi",
                       expect=Expectation.model_validate({"targets": [{"landmark": "Bara Gumbad", "views": 2},
                                                                      {"landmark": "Sheesh Gumbad"}]}))
    scenario = prepare(spec)
    truth = ground(spec.expect.payload(), scenario.world)
    right = {"targets": [{"name": "Bara Gumbad", "landmark": True, "action": "CAPTURE", "views": 2, "sides": [], "overhead": False},
                         {"name": "Sheesh Gumbad", "landmark": True, "action": "CAPTURE", "views": 1, "sides": [], "overhead": False}]}
    half = {"targets": right["targets"][:1]}
    wrong_views = {"targets": [{**right["targets"][0], "views": 1}, right["targets"][1]]}
    assert extraction_match(truth, right) == {"exact": True, "target_f1": 1.0}
    assert extraction_match(truth, half)["target_f1"] == 0.6667 and not extraction_match(truth, half)["exact"]
    assert not extraction_match(truth, wrong_views)["exact"]
