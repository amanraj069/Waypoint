"""Run-file persistence."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from conftest import VALID_PAYLOAD, fake_planner
from orchestrator import run_mission
from orchestrator.runlog import build_run, read_run, run_path, slugify, write_run
from planner import DEFAULT_CONSTRAINTS


@pytest.mark.parametrize(
    "text,expected",
    [
        ("survey the north field", "survey-the-north-field"),
        ("Survey  the   NORTH field!", "survey-the-north-field"),
        ("inspect dam #3 (west side)", "inspect-dam-3-west-side"),
        ("../../etc/passwd", "etc-passwd"),
        ("!!!", "mission"),
        ("", "mission"),
    ],
)
def test_slugify(text, expected):
    assert slugify(text) == expected


def test_slug_is_length_capped():
    slug = slugify("survey " * 40)
    assert len(slug) <= 48
    assert not slug.endswith("-")


def test_run_path_format(tmp_path):
    when = datetime(2026, 9, 16, 14, 30, 5)
    path = run_path("survey the north field", when, tmp_path)
    assert path.name == "survey-the-north-field_20260916T143005.json"
    assert path.parent == tmp_path


def test_write_run_creates_the_directory(tmp_path):
    report = run_mission("survey the north field", verbose=False, planner=fake_planner, runs_dir=None)
    target = tmp_path / "nested" / "runs"

    path = write_run(report, runs_dir=target)

    assert path.exists()
    assert path.parent == target


def test_run_mission_writes_a_run_file_and_records_the_path(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    )

    assert report.run_file is not None
    assert report.run_file.exists()
    assert report.run_file.parent == tmp_path


def test_runs_dir_none_writes_nothing(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=None
    )

    assert report.run_file is None
    assert list(tmp_path.iterdir()) == []


def test_the_run_file_is_self_contained(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    )
    run = json.loads(report.run_file.read_text())

    assert run["mission_text"] == "survey the north field"
    assert run["completed"] is True
    assert run["verification"] == {"passed": True, "violations": []}
    assert run["constraints"]["max_altitude"] == DEFAULT_CONSTRAINTS.max_altitude
    assert run["created_at"]
    assert run["schema_version"] == 1


def test_every_tick_is_recorded_with_full_telemetry(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    )
    run = json.loads(report.run_file.read_text())

    assert len(run["ticks"]) == len(report.states)
    for tick, state in zip(run["ticks"], report.states):
        assert tick == {
            "x": state.x, "y": state.y, "z": state.z,
            "velocity": state.velocity, "heading": state.heading,
            "battery_pct": state.battery_pct, "timestamp": state.timestamp,
        }


def test_waypoints_keep_their_subtask_names(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    )
    run = json.loads(report.run_file.read_text())

    expected = [
        (s["name"], w["action"])
        for s in VALID_PAYLOAD["subtasks"]
        for w in s["waypoints"]
    ]
    assert [(w["subtask"], w["action"]) for w in run["waypoints"]] == expected
    assert [w["index"] for w in run["waypoints"]] == list(range(len(expected)))


def test_ticks_are_printed_as_well_as_collected(capsys, tmp_path):
    """Saving must not have replaced the console output."""
    report = run_mission("survey the north field", planner=fake_planner, runs_dir=tmp_path)
    out = capsys.readouterr().out

    assert report.states
    for state in report.states:
        assert f"t={state.timestamp:6.1f}s" in out
    assert str(report.run_file) in out


def test_read_run_round_trips(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    )
    assert read_run(report.run_file) == build_run(report, datetime.fromisoformat(
        json.loads(report.run_file.read_text())["created_at"]
    ))


def test_read_run_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No run file"):
        read_run(tmp_path / "nope.json")


def test_read_run_rejects_non_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")

    with pytest.raises(ValueError, match="not valid JSON"):
        read_run(bad)


def test_read_run_rejects_unrelated_json(tmp_path):
    bad = tmp_path / "other.json"
    bad.write_text('{"hello": "world"}')

    with pytest.raises(ValueError, match="does not look like a mission run file"):
        read_run(bad)
