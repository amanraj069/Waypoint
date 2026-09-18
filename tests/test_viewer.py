"""The 3D viewer's data contract and the little server that feeds it.

The viewer itself is browser code, so what is tested here is the seam: the run
index the dropdown reads, and the keys `viewer/index.html` expects a run file to
have. Those are what break silently when the run schema moves.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from conftest import fake_planner
from orchestrator import run_mission

ROOT = Path(__file__).resolve().parent.parent
VIEWER_HTML = ROOT / "viewer" / "index.html"


@pytest.fixture(scope="module")
def serve_viewer():
    """Load scripts/serve_viewer.py, which is a script rather than a package."""
    spec = importlib.util.spec_from_file_location("serve_viewer", ROOT / "scripts" / "serve_viewer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["serve_viewer"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runs_dir(monkeypatch, tmp_path, serve_viewer):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(serve_viewer, "RUNS_DIR", runs)
    monkeypatch.setattr(serve_viewer, "VIEWER_DIR", tmp_path / "viewer")
    return runs


def write_run_file(runs_dir: Path, name: str, **overrides) -> Path:
    payload = {
        "mission_text": "survey the north field",
        "ticks": [{"timestamp": 0.0, "x": 0, "y": 0, "z": 0}],
        "sim": {"duration_s": 42.0},
        **overrides,
    }
    path = runs_dir / name
    path.write_text(json.dumps(payload))
    return path


# --- the run index ---------------------------------------------------------


def test_runs_without_telemetry_are_not_offered(serve_viewer, runs_dir):
    """A rejected plan never flew, so there is nothing to animate."""
    write_run_file(runs_dir, "flown.json")
    write_run_file(runs_dir, "rejected.json", ticks=[])

    assert [p.name for p in serve_viewer.flyable_runs()] == ["flown.json"]


def test_unreadable_files_are_skipped(serve_viewer, runs_dir):
    write_run_file(runs_dir, "good.json")
    (runs_dir / "broken.json").write_text("{not json")

    assert [p.name for p in serve_viewer.flyable_runs()] == ["good.json"]


def test_the_index_labels_each_run_for_the_dropdown(serve_viewer, runs_dir):
    write_run_file(runs_dir, "flown.json")
    serve_viewer.write_index(serve_viewer.flyable_runs())

    index = json.loads((serve_viewer.VIEWER_DIR / "runs.json").read_text())
    assert index[0]["url"] == "/runs/flown.json"
    assert "survey the north field" in index[0]["label"]
    assert "42s" in index[0]["label"]


def test_an_empty_runs_directory_reports_rather_than_serving(serve_viewer, runs_dir, capsys):
    assert serve_viewer.serve(None, port=0, open_browser=False) == 1
    assert "No run files" in capsys.readouterr().err


# --- the run file contract -------------------------------------------------


def test_a_real_run_carries_everything_the_viewer_draws(tmp_path):
    report = run_mission(
        "survey the north field", verbose=False, planner=fake_planner, runs_dir=tmp_path
    )
    run = json.loads(report.run_file.read_text())

    # Every key the viewer reads out of a run file.
    assert run["ticks"] and run["waypoints"]
    assert {"x", "y", "z", "timestamp", "velocity", "heading", "battery_pct", "subtask"} <= set(
        run["ticks"][0]
    )
    assert {"x", "y", "z", "action", "subtask"} <= set(run["waypoints"][0])
    assert run["constraints"]["geofence"]
    assert isinstance(run["constraints"]["no_fly_zones"], list)
    assert isinstance(run["obstacles"], list)
    assert all({"t", "type", "description"} <= set(event) for event in run["events"])


def test_the_viewer_page_is_self_contained_and_needs_no_build_step():
    html = VIEWER_HTML.read_text()

    assert "<script type=\"importmap\">" in html  # three.js straight from a CDN
    assert "three@0.160.0" in html
    assert "OrbitControls" in html
    assert "<script src=" not in html  # nothing bundled, nothing to compile


def test_every_element_the_viewer_script_touches_exists_on_the_page():
    html = VIEWER_HTML.read_text()
    module = re.search(r'<script type="module">(.*?)</script>', html, re.S).group(1)

    defined = set(re.findall(r'id="([\w-]+)"', html))
    used = set(re.findall(r"\$\('([\w-]+)'\)", module))
    assert used, "the viewer should be wiring up elements by id"
    assert used <= defined, f"missing from the page: {sorted(used - defined)}"


# --- the live session (scripts/explore.py) ---------------------------------


@pytest.fixture(scope="module")
def explore():
    """Load scripts/explore.py the same way, without starting a server."""
    spec = importlib.util.spec_from_file_location("explore", ROOT / "scripts" / "explore.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["explore"] = module
    spec.loader.exec_module(module)
    return module


def test_a_world_preview_is_a_run_with_no_flight_in_it(explore, monkeypatch, tmp_path):
    """What you look around before anything has flown."""
    from planner import DEFAULT_CONSTRAINTS
    from world import city

    monkeypatch.setattr(explore, "VIEWER", tmp_path)
    url = explore.preview_file(city(seed=1), DEFAULT_CONSTRAINTS)

    assert url.startswith("/viewer/preview.json")
    preview = json.loads((tmp_path / "preview.json").read_text())
    assert preview["ticks"] == []                      # nothing has flown
    assert preview["world"]["buildings"]               # but there is a world to see
    assert preview["constraints"]["geofence"]
    assert "nothing has flown yet" in preview["mission_text"]


def test_the_session_file_tracks_what_the_terminal_is_doing(explore, monkeypatch, tmp_path):
    monkeypatch.setattr(explore, "VIEWER", tmp_path)
    session = explore.Session()

    session.write("planning: survey the roofs")
    state = json.loads((tmp_path / "session.json").read_text())
    assert state["message"] == "planning: survey the roofs"
    assert state["run"] is None

    session.write("landed", run="/runs/some-run.json")
    state = json.loads((tmp_path / "session.json").read_text())
    assert state["run"] == "/runs/some-run.json"

    # A later message keeps pointing at the same run, so the page does not reload.
    session.write("ready")
    assert json.loads((tmp_path / "session.json").read_text())["run"] == "/runs/some-run.json"

    session.clear()
    assert not (tmp_path / "session.json").exists()


def test_the_viewer_polls_a_session_only_when_asked():
    html = VIEWER_HTML.read_text()
    assert "session.json" in html
    assert "params.get('session')" in html      # opt-in, so a plain replay makes no requests


def test_the_viewer_can_show_a_world_with_no_flight():
    html = VIEWER_HTML.read_text()
    assert "const preview = !data.ticks.length && !!data.world;" in html
    assert "setTransport(" in html


def test_buildings_are_placed_with_the_same_convention_as_everything_else():
    """A building must land on its own plot, not its north-south mirror image.

    `toScene` maps ENU (x, y, z) to (x, z, -y). ExtrudeGeometry plus
    rotateX(-90) already does that, so flipping y in the footprint puts the
    building on the wrong side of the map -- where the ground is a different
    height, which is what made masts hang in the air.
    """
    html = VIEWER_HTML.read_text()
    shape_line = re.search(r"new THREE\.Shape\(\s*building\.footprint\.map\(\(p\) => ([^)]*\))", html)

    assert shape_line, "buildingMesh should build its shape from the footprint"
    assert "-p.y" not in shape_line.group(1), "footprint y must not be negated"
    assert "new THREE.Vector2(p.x, p.y)" in shape_line.group(1)


def test_trees_are_drawn_as_trees_and_landmarks_are_labelled():
    """A tree is a box to the simulator; drawing it as one is a lie you can see."""
    html = VIEWER_HTML.read_text()

    assert "function treeMesh(" in html
    assert "building.kind === 'tree' ? treeMesh(building) : buildingMesh(building)" in html
    assert "function labelSprite(" in html          # landmark names, readable in the scene
    assert "function elevationAt(" in html          # so labels sit on the ground


# --- switching worlds from the page ----------------------------------------


def test_the_world_index_describes_what_is_on_offer(serve_viewer, tmp_path, monkeypatch):
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    monkeypatch.setattr(serve_viewer, "WORLDS_DIR", worlds)
    monkeypatch.setattr(serve_viewer, "VIEWER_DIR", tmp_path / "viewer")
    monkeypatch.setattr(serve_viewer, "BUILT_IN_WORLDS", ("lodhi",))

    entries = serve_viewer.world_index()

    assert [e["name"] for e in entries] == ["lodhi garden"]     # written out on demand
    assert entries[0]["url"] == "/worlds/lodhi.json"
    assert "trees" in entries[0]["detail"]
    assert "Bara Gumbad" in entries[0]["landmarks"]

    serve_viewer.write_world_index()
    written = json.loads((tmp_path / "viewer" / "worlds.json").read_text())
    assert written == entries


def test_a_page_can_ask_the_session_to_switch_worlds(serve_viewer):
    """`/command?world=...` reaches whatever registered a handler."""
    seen = []
    serve_viewer.COMMAND_HANDLERS["world"] = lambda spec: seen.append(spec) or f"now in {spec}"
    try:
        assert serve_viewer.COMMAND_HANDLERS["world"]("/worlds/chowk.json") == "now in /worlds/chowk.json"
        assert seen == ["/worlds/chowk.json"]
    finally:
        serve_viewer.COMMAND_HANDLERS.clear()


def test_commands_nobody_is_listening_for_are_harmless(serve_viewer):
    assert serve_viewer.COMMAND_HANDLERS.get("world") is None


def test_the_viewer_offers_a_world_picker():
    html = VIEWER_HTML.read_text()

    assert 'id="worlds"' in html
    assert "async function openWorld(" in html
    assert "/command?world=" in html              # tells a live session to follow
    assert "populateWorldList()" in html


def test_switching_worlds_gives_the_preview_a_new_url(explore, monkeypatch, tmp_path):
    """Otherwise a polling page never notices the world changed."""
    from planner import DEFAULT_CONSTRAINTS
    from world import chandni_chowk, lodhi_garden

    monkeypatch.setattr(explore, "VIEWER", tmp_path)
    first = explore.preview_file(lodhi_garden(), DEFAULT_CONSTRAINTS)
    second = explore.preview_file(chandni_chowk(), DEFAULT_CONSTRAINTS)

    assert first != second
    assert second.startswith("/viewer/preview.json?v=")
    assert json.loads((tmp_path / "preview.json").read_text())["world"]["name"] == "chandni chowk"


def test_a_busy_port_steps_forward_instead_of_failing(serve_viewer):
    """A viewer left running in another terminal must not block a new one."""
    first, first_port = serve_viewer.start_server(0)          # 0 = any free port
    try:
        second, second_port = serve_viewer.start_server(first_port)
        try:
            assert second_port != first_port
            assert second_port > first_port
        finally:
            second.server_close()
    finally:
        first.server_close()


def test_it_gives_up_after_a_reasonable_number_of_ports(serve_viewer):
    held, port = serve_viewer.start_server(0)
    try:
        with pytest.raises(OSError, match="are all in use"):
            serve_viewer.start_server(port, tries=1)
    finally:
        held.server_close()
