"""The HTTP surface: worlds to fly in, missions to fly, runs to replay.

Nothing here talks to Gemini. A mission is started from a bundled plan, which
is the path that exists precisely so the whole pipeline can be exercised -- and
demonstrated -- without an API key.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from world.generate import BUILT_IN


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Write runs somewhere disposable, so a test does not fill runs/."""
    directory = tmp_path / "runs"
    directory.mkdir()
    monkeypatch.setattr("api.missions.RUNS_DIR", directory)
    return directory


# ---------------------------------------------------------------------------
# What the server says it can do
# ---------------------------------------------------------------------------


def test_health_says_whether_a_mission_can_be_planned_at_all(client):
    body = client.get("/api/health").json()

    assert body["ok"] is True
    assert isinstance(body["planner_ready"], bool)
    assert body["stages"][0] == "queued"
    assert body["stages"][-1] == "failed"


def test_a_mission_in_english_is_refused_without_a_key(client, monkeypatch):
    """Better a clear 503 than a 500 from the planner four seconds later."""
    monkeypatch.setattr("api.main.get_api_key", lambda: None)

    response = client.post("/api/missions", json={"mission": "survey the north lawn"})

    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_an_empty_mission_is_refused(client):
    assert client.post("/api/missions", json={}).status_code == 422


def test_an_unknown_world_is_a_404(client):
    assert client.get("/api/worlds/atlantis").status_code == 404
    response = client.post("/api/missions", json={"plan": "lodhi_garden_tour", "world": "atlantis"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


def test_the_world_list_leads_with_the_built_in_places(client):
    cards = client.get("/api/worlds").json()

    assert [card["id"] for card in cards][: len(BUILT_IN)] == list(BUILT_IN)
    delhi = cards[0]
    assert delhi["name"] == "central vista"
    assert "India Gate" in delhi["landmarks"]
    assert delhi["tallest_roof"] > 60


def test_a_world_comes_back_whole(client):
    world = client.get("/api/worlds/chowk").json()

    assert world["name"] == "chandni chowk"
    assert any(b["name"] == "Fatehpuri Masjid" for b in world["buildings"])
    assert all(b["name"] for b in world["buildings"])


def test_a_world_preview_is_a_run_with_nothing_flown_in_it(client):
    """One shape for a place and for a flight, so the 3D view has one code path."""
    preview = client.get("/api/worlds/munnar/preview").json()

    assert preview["ticks"] == []
    assert preview["waypoints"] == []
    assert preview["preview"] is True
    assert preview["world"]["name"] == "munnar"
    # Munnar has 200 m of relief, so the default 120 m ceiling cannot stand.
    assert preview["constraints"]["max_altitude"] > 200
    assert any("ceiling raised" in note for note in preview["constraint_notes"])


def test_the_geofence_is_widened_to_whatever_world_is_loaded(client):
    preview = client.get("/api/worlds/delhi/preview").json()
    fence = preview["constraints"]["geofence"]

    assert min(p["x"] for p in fence) < -500      # Rashtrapati Bhavan is inside it
    assert any("geofence widened" in note for note in preview["constraint_notes"])


# ---------------------------------------------------------------------------
# Flying one
# ---------------------------------------------------------------------------


def stream(client: TestClient, mission_id: str) -> list[dict]:
    """Drain a mission's event stream into a list.

    The trailing frame is an SSE *named* event, which a browser delivers to its
    own listener rather than to `onmessage`; dropping it here keeps the list the
    same as what the page actually reduces over.
    """
    events = []
    with client.stream("GET", f"/api/missions/{mission_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[len("data: ") :])
                if event.get("kind") != "close":
                    events.append(event)
    return events


def test_a_bundled_plan_flies_the_whole_pipeline_without_a_planner(client, runs_dir):
    started = client.post("/api/missions", json={"plan": "chandni_chowk_run"})
    assert started.status_code == 202
    mission_id = started.json()["id"]

    events = stream(client, mission_id)
    kinds = [event["kind"] for event in events]
    assert "world" in kinds
    assert "plan" in kinds
    assert "verification" in kinds
    assert "telemetry" in kinds

    stages = [event["stage"] for event in events if event["kind"] == "stage"]
    assert stages == ["planning", "verifying", "flying"]

    verdict = next(event for event in events if event["kind"] == "verification")
    assert verdict["passed"] is True

    end = events[-1]
    assert end["kind"] == "end"
    assert end["stage"] == "done"
    assert end["completed"] is True
    assert end["distance_m"] > 500
    assert (runs_dir / f"{end['run']}.json").exists()


def test_the_plan_decides_the_world_when_nobody_says(client, runs_dir):
    """A plan is only waypoints, so the world it was written for comes with it."""
    mission_id = client.post("/api/missions", json={"plan": "munnar_ridge_run"}).json()["id"]
    events = stream(client, mission_id)

    world = next(event for event in events if event["kind"] == "world")
    assert world["world"]["name"] == "munnar"
    assert events[-1]["stage"] == "done"


def test_a_late_watcher_still_sees_the_whole_mission(client, runs_dir):
    """Opening the page halfway through must not be a different experience."""
    mission_id = client.post("/api/missions", json={"plan": "lodhi_garden_tour"}).json()["id"]
    stream(client, mission_id)                       # let it finish

    snapshot = client.get(f"/api/missions/{mission_id}").json()
    assert snapshot["status"] == "done"
    assert snapshot["events"][0]["seq"] == 0
    assert snapshot["events"][-1]["kind"] == "end"

    # And the stream replays from the beginning for somebody who just arrived.
    assert [event["seq"] for event in stream(client, mission_id)] == [
        event["seq"] for event in snapshot["events"]
    ]


def test_a_plan_flown_in_the_wrong_world_is_rejected_rather_than_flown(client, runs_dir):
    """The verifier is the gate, and the API does not get to skip it."""
    mission_id = client.post(
        "/api/missions", json={"plan": "chandni_chowk_run", "world": "munnar", "retries": 0}
    ).json()["id"]
    events = stream(client, mission_id)

    verdict = next(event for event in events if event["kind"] == "verification")
    assert verdict["passed"] is False
    assert verdict["violations"]
    assert events[-1]["stage"] == "failed"
    assert "nothing was flown" in events[-1]["message"]
    assert list(runs_dir.glob("*.json")) == []


def test_a_mission_that_was_never_started_is_a_404(client):
    assert client.get("/api/missions/m999-nope").status_code == 404
    assert client.get("/api/missions/m999-nope/events").status_code == 404


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def test_a_flown_run_can_be_listed_read_and_deleted(client, runs_dir, monkeypatch):
    monkeypatch.setattr("api.catalog.RUNS_DIR", runs_dir)
    mission_id = client.post("/api/missions", json={"plan": "chandni_chowk_run"}).json()["id"]
    run_id = stream(client, mission_id)[-1]["run"]

    cards = client.get("/api/runs").json()
    card = next(card for card in cards if card["id"] == run_id)
    assert card["world"] == "chandni chowk"
    assert card["flyable"] is True
    assert card["phases"][0] == "takeoff"

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["ticks"], "a flown run carries its telemetry"
    assert run["world"]["name"] == "chandni chowk"

    assert client.delete(f"/api/runs/{run_id}").json() == {"deleted": run_id}
    assert client.get(f"/api/runs/{run_id}").status_code == 404


# --- picking where the aircraft starts -------------------------------------


def test_the_launch_point_defaults_to_the_origin():
    from api.missions import home_for
    from api.catalog import load_world

    assert home_for(None, None) == (0.0, 0.0, 0.0)
    world = load_world("lodhi")
    assert home_for(world, None)[:2] == (0.0, 0.0)


def test_a_chosen_launch_point_stands_on_the_ground_under_it():
    from api.missions import home_for
    from api.catalog import load_world

    world = load_world("lodhi")           # gentle ground: the pick is kept
    x, y, z = home_for(world, (150.0, -120.0))

    assert (x, y) == (150.0, -120.0)
    assert z == pytest.approx(world.elevation_at(150.0, -120.0), abs=0.01)
    assert z != 0.0, "this world is not flat, so a moved pad should not be at zero"


def test_a_pad_on_a_hillside_is_moved_to_ground_it_can_use():
    """Picking a slope is not refused -- it is corrected, and says so."""
    from api.missions import site_for
    from api.catalog import load_world
    from world.landing import survey

    world = load_world("munnar")          # a hill station: most of it is steep
    asked = survey(world, 150.0, -120.0)
    assert not asked.usable, "this spot should be too steep to land on"

    site = site_for(world, (150.0, -120.0), "land")

    assert site.usable
    assert site.moved > 0
    assert (site.x, site.y) != (150.0, -120.0)
    assert site.elevation == pytest.approx(world.elevation_at(site.x, site.y), abs=0.01)
    assert "moved" in site.describe()


def test_a_pad_on_a_roof_is_moved_off_it():
    from api.missions import site_for
    from api.catalog import load_world

    world = load_world("lodhi")
    building = world.buildings[0]
    site = site_for(world, building.centre, "take off")

    assert site.usable
    assert world.collides(site.x, site.y, site.elevation + 0.5) is None


def test_nowhere_to_land_is_refused_rather_than_guessed():
    from api.missions import MissionError, site_for
    from world.models import Terrain, World

    # A world that is steep everywhere: there is no good answer to give.
    cliff = World(
        name="cliff",
        terrain=Terrain(
            origin_x=-200,
            origin_y=-200,
            spacing=20,
            heights=[[col * 30.0 for col in range(21)] for _ in range(21)],
        ),
    )
    with pytest.raises(MissionError, match="nowhere within 120 m"):
        site_for(cliff, (0.0, 0.0), "land")


def test_the_planner_is_told_where_it_starts():
    from planner.prompts import build_plan_prompt
    from planner import DEFAULT_CONSTRAINTS

    prompt = build_plan_prompt("survey it", DEFAULT_CONSTRAINTS, None, (120.0, -60.0, 3.5))

    assert "(120, -60, 3.5)" in prompt
    assert "must finish landed there" in prompt


def test_the_planner_is_told_when_it_lands_somewhere_else():
    from planner.prompts import build_plan_prompt
    from planner import DEFAULT_CONSTRAINTS

    prompt = build_plan_prompt(
        "survey it", DEFAULT_CONSTRAINTS, None, (10.0, 20.0, 1.5), (300.0, -40.0, 12.0)
    )

    assert "starts on the ground at (10, 20, 1.5)" in prompt
    assert "LAND at exactly (300, -40, 12)" in prompt
    assert "flat enough to set down on" in prompt


def test_the_body_accepts_a_landing_point():
    from api.main import MissionBody

    body = MissionBody(mission="x", home=(0.0, 0.0), landing=(120.0, 45.0))
    assert body.landing == (120.0, 45.0)
    assert MissionBody(mission="x").landing is None


def test_a_bundled_plan_will_not_take_a_chosen_pad():
    """Its waypoints are absolute; flying it from elsewhere is a different mission."""
    import uuid

    from api.missions import Mission, MissionError, MissionRequest, run_mission

    for pads in ({"home": (80.0, 40.0)}, {"landing": (80.0, 40.0)}):
        mission = Mission(
            id=str(uuid.uuid4()),
            request=MissionRequest(plan_id="lodhi_garden_tour", world="lodhi", **pads),
        )
        with pytest.raises(MissionError, match="write the mission in English"):
            run_mission(mission)


def test_the_body_accepts_a_launch_point(client):
    """The field exists and is typed as a pair of metres."""
    from api.main import MissionBody

    body = MissionBody(mission="x", home=(12.5, -30.0))
    assert body.home == (12.5, -30.0)
    assert MissionBody(mission="x").home is None
