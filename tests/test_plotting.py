"""3D visualization. Builds real figures; never opens a browser."""

from __future__ import annotations

import re

import pytest

from conftest import fake_planner
from waypoint.orchestrator import run_mission
from waypoint.orchestrator.plotting import ACTION_STYLE, build_figure, html_path_for, render
from waypoint.orchestrator.runlog import build_run


@pytest.fixture
def run(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    ).report
    return build_run(report)


@pytest.fixture
def no_browser(monkeypatch):
    opened = []
    monkeypatch.setattr("waypoint.orchestrator.plotting.webbrowser.open", opened.append)
    return opened


def _trace(fig, name):
    return next(t for t in fig.data if t.name == name)


# --- the flown path ---------------------------------------------------------


def test_the_flown_path_has_one_point_per_tick(run):
    flown = _trace(build_figure(run), "Flown path")
    assert len(flown.x) == len(run["ticks"])
    assert list(flown.z) == [t["z"] for t in run["ticks"]]


def test_the_flown_path_is_a_continuous_line(run):
    assert "lines" in _trace(build_figure(run), "Flown path").mode


def test_the_flown_path_is_coloured_by_battery_along_its_length(run):
    flown = _trace(build_figure(run), "Flown path")
    battery = [t["battery_pct"] for t in run["ticks"]]

    assert list(flown.line.color) == battery
    assert list(flown.marker.color) == battery
    assert flown.marker.cmin == 0 and flown.marker.cmax == 100
    assert flown.marker.colorbar.title.text == "Battery %"


def test_hover_data_carries_every_telemetry_field(run):
    flown = _trace(build_figure(run), "Flown path")

    assert [list(row) for row in flown.customdata] == [
        [t["timestamp"], t["velocity"], t["heading"], t["battery_pct"]] for t in run["ticks"]
    ]
    for field in ("customdata[0]", "customdata[1]", "customdata[2]", "customdata[3]"):
        assert field in flown.hovertemplate
    for axis in ("%{x:", "%{y:", "%{z:"):
        assert axis in flown.hovertemplate


# --- planned waypoints ------------------------------------------------------


def test_every_action_gets_its_own_labelled_trace(run):
    fig = build_figure(run)
    actions = {w["action"] for w in run["waypoints"]}

    for action in actions:
        trace = _trace(fig, action)
        expected = [w for w in run["waypoints"] if w["action"] == action]
        assert len(trace.x) == len(expected)
        assert list(trace.text) == [w["subtask"] for w in expected]
        assert trace.marker.symbol == ACTION_STYLE[action][0]


def test_waypoint_labels_show_subtask_and_action(run):
    fig = build_figure(run)
    capture = _trace(fig, "CAPTURE")

    assert "survey_north_field" in capture.text
    assert "<b>CAPTURE</b>" in capture.hovertemplate
    assert "customdata[0]" in capture.hovertemplate  # subtask name


def test_each_action_is_visually_distinct():
    symbols = [s for s, _ in ACTION_STYLE.values()]
    colors = [c for _, c in ACTION_STYLE.values()]
    assert len(set(symbols)) == len(symbols)
    assert len(set(colors)) == len(colors)


def test_the_planned_route_is_drawn_separately_from_the_flown_one(run):
    fig = build_figure(run)
    planned = _trace(fig, "Planned route")

    assert len(planned.x) == len(run["waypoints"])
    assert planned.mode == "lines"


# --- layout -----------------------------------------------------------------


def test_axes_are_labelled_and_undistorted(run):
    scene = build_figure(run).layout.scene

    assert "altitude" in scene.zaxis.title.text
    assert "east" in scene.xaxis.title.text
    assert "north" in scene.yaxis.title.text
    # Without this the altitude axis is stretched and the path misleads.
    assert scene.aspectmode == "data"


def test_the_title_names_the_mission(run):
    assert "survey the north field" in build_figure(run).layout.title.text


# --- output -----------------------------------------------------------------


def test_render_writes_a_standalone_html_beside_the_run(run, tmp_path, no_browser):
    run_file = tmp_path / "a_run.json"

    out = render(run, run_file, open_browser=False)

    assert out == tmp_path / "a_run.html"
    html = out.read_text()
    assert html.lstrip().lower().startswith("<!doctype html>")
    # Plotly is inlined, so the file works with no network. (The bundle itself
    # mentions the CDN URL in its config defaults, so test for a remote <script>
    # tag rather than for the string.)
    assert not re.search(r'<script[^>]+src="https?://', html)
    assert "<script" in html and len(html) > 500_000


def test_render_opens_the_browser_by_default(run, tmp_path, no_browser):
    out = render(run, tmp_path / "a_run.json", open_browser=True)

    assert no_browser == [out.resolve().as_uri()]


def test_render_can_skip_the_browser(run, tmp_path, no_browser):
    render(run, tmp_path / "a_run.json", open_browser=False)

    assert no_browser == []


def test_html_path_sits_beside_the_run_file():
    assert html_path_for("runs/x_20260916T120000.json") .name == "x_20260916T120000.html"


# --- failure modes ----------------------------------------------------------


def test_a_run_with_no_ticks_fails_clearly(run):
    """A mission that fails verification never flies, so there is nothing to draw."""
    run["ticks"] = []

    with pytest.raises(ValueError, match="no telemetry ticks"):
        build_figure(run)


def test_a_run_that_was_never_saved_fails_clearly(run, no_browser):
    with pytest.raises(ValueError, match="nowhere to save"):
        render(run, None)


def test_a_run_with_no_waypoints_still_plots_the_path(run, tmp_path, no_browser):
    """Degraded, not broken -- the telemetry is the point."""
    run["waypoints"] = []

    fig = build_figure(run)

    assert _trace(fig, "Flown path")
    assert all(t.name != "Planned route" for t in fig.data)
