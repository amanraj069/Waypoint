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
    assert run["schema_version"] == 2


def test_every_tick_is_recorded_with_full_telemetry(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    )
    run = json.loads(report.run_file.read_text())

    assert len(run["ticks"]) == len(report.states)
    for tick, state in zip(run["ticks"], report.states):
        assert tick["x"] == state.x and tick["y"] == state.y and tick["z"] == state.z
        assert (tick["vx"], tick["vy"], tick["vz"]) == state.velocity_vector
        assert tick["velocity"] == state.velocity
        assert tick["heading"] == state.heading
        assert tick["battery_pct"] == state.battery_pct
        assert tick["timestamp"] == pytest.approx(state.timestamp)
        assert tick["subtask"] == state.subtask


def test_the_run_file_carries_the_simulator_settings(tmp_path):
    """Schema 2: everything needed to re-fly the run exactly, and to animate it."""
    from sim import FaultEvent, FaultSchedule, FaultType, Trigger, Wind

    faults = FaultSchedule.of(
        FaultEvent(type=FaultType.WIND_GUST, trigger=Trigger(time=3.0), magnitude=6.0, duration=2.0)
    )
    report = run_mission(
        "survey the north field",
        verbose=False,
        planner=fake_planner,
        runs_dir=tmp_path,
        wind=Wind(vector=(2.0, 0.0, 0.0)),
        faults=faults,
    )
    run = json.loads(report.run_file.read_text())

    assert run["sim"]["wind"]["vector"] == [2.0, 0.0, 0.0]
    assert run["sim"]["config"]["dt"] == 0.1
    assert run["sim"]["distance_flown_m"] > 0
    assert [f["type"] for f in run["faults"]] == ["WIND_GUST"]
    assert any(e["type"] == "FAULT" for e in run["events"])
    assert run["events"][-1]["type"] == "LANDED"


def test_obstacles_are_recorded_for_the_viewer(tmp_path):
    from planner import Obstacle
    from sim import FaultEvent, FaultSchedule, FaultType, Trigger

    constraints = DEFAULT_CONSTRAINTS.model_copy(
        update={"obstacles": [Obstacle(x=200, y=200, radius=4, name="silo")], "min_separation": 5.0}
    )
    faults = FaultSchedule.of(
        FaultEvent(
            type=FaultType.OBSTACLE_APPEARS,
            trigger=Trigger(time=2.0),
            obstacle=Obstacle(x=10, y=40, radius=6, name="crane"),
        )
    )
    report = run_mission(
        "survey the north field",
        constraints,
        verbose=False,
        planner=fake_planner,
        runs_dir=tmp_path,
        faults=faults,
    )
    run = json.loads(report.run_file.read_text())

    assert [o["name"] for o in run["obstacles"]] == ["silo", "crane"]
    assert run["obstacles"][0]["appeared_at"] == 0.0  # known from the start
    assert run["obstacles"][1]["appeared_at"] >= 2.0  # found mid-flight


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
    """Saving must not have replaced the console output.

    The simulator ticks at 10 Hz, so the console shows a sample -- but always
    the first tick and the touchdown.
    """
    report = run_mission("survey the north field", planner=fake_planner, runs_dir=tmp_path)
    out = capsys.readouterr().out

    assert report.states
    printed = [line for line in out.splitlines() if "t=" in line and "pos=" in line]
    assert 1 < len(printed) < len(report.states)
    assert f"t={report.states[0].timestamp:6.1f}s" in out
    assert f"batt={report.states[-1].battery_pct:5.1f}%" in out
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
