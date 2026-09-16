"""Orchestrator tests. The planner is injected, so no API call is made."""

from __future__ import annotations

import pytest

from conftest import fake_planner
from orchestrator import run_mission
from orchestrator.run import main
from planner import DEFAULT_CONSTRAINTS, MissionConstraints, Plan


@pytest.fixture
def stub_plan(monkeypatch):
    """Point the CLI's default planner at the fake one."""
    monkeypatch.setattr("orchestrator.loop.default_planner", fake_planner)


def test_run_mission_completes(capsys, tmp_path):
    report = run_mission("survey the north field", planner=fake_planner, runs_dir=tmp_path)
    assert report.verification.passed
    assert report.states
    assert report.completed
    assert "landed" in capsys.readouterr().out


def test_run_mission_prints_every_tick(capsys, tmp_path):
    report = run_mission("survey the north field", planner=fake_planner, runs_dir=tmp_path)
    out = capsys.readouterr().out
    for state in report.states:
        assert f"t={state.timestamp:6.1f}s" in out


def test_quiet_mode_prints_nothing(capsys, tmp_path):
    run_mission("survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path)
    assert capsys.readouterr().out == ""


def test_planner_receives_the_constraints(tmp_path):
    seen = {}

    def spy(mission_text: str, constraints: MissionConstraints) -> Plan:
        seen["mission"] = mission_text
        seen["constraints"] = constraints
        return fake_planner(mission_text, constraints)

    run_mission("survey the north field", verbose=False, planner=spy, runs_dir=tmp_path)

    assert seen["mission"] == "survey the north field"
    assert seen["constraints"] == DEFAULT_CONSTRAINTS


def test_custom_constraints_reach_the_planner(constraints, tmp_path):
    seen = {}

    def spy(mission_text: str, c: MissionConstraints) -> Plan:
        seen["constraints"] = c
        return fake_planner(mission_text, c)

    run_mission("survey the north field", constraints, verbose=False, planner=spy, runs_dir=tmp_path)

    assert seen["constraints"] == constraints


def test_cli_exits_zero(capsys, stub_plan):
    assert main(["--mission", "survey the north field"]) == 0
    assert "Mission complete" in capsys.readouterr().out


def test_cli_requires_a_mission():
    with pytest.raises(SystemExit):
        main([])
