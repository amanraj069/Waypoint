"""Reading what is already on disk: worlds, runs and bundled plans.

Everything here is a pure function of the filesystem, so the API's lists and
the CLI's lists are the same lists.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from api.settings import PLANS_DIR, RUNS_DIR, WORLDS_DIR
from world.generate import BUILDERS, BUILT_IN, build
from world.models import World


def ensure_built_in_worlds() -> None:
    """Materialise the shipped worlds, so every world is a file like any other."""
    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    for key in BUILT_IN:
        path = WORLDS_DIR / f"{key}.json"
        if not path.exists():
            path.write_text(build(key).model_dump_json(indent=2) + "\n")


def world_ids() -> list[str]:
    ensure_built_in_worlds()
    return sorted(path.stem for path in WORLDS_DIR.glob("*.json"))


def load_world(world_id: str) -> World:
    """A world by file stem, or by built-in name. Raises KeyError if unknown."""
    ensure_built_in_worlds()
    path = WORLDS_DIR / f"{Path(world_id).name}.json"
    if path.exists():
        return World.model_validate_json(path.read_text())
    if world_id in BUILDERS:
        return build(world_id)
    raise KeyError(world_id)


def world_card(world_id: str) -> dict:
    """The summary a picker needs: enough to choose between places."""
    world = load_world(world_id)
    summary = world.summary()
    minx, miny, maxx, maxy = world.bounds
    described = BUILDERS.get(world_id, (None, ""))[1]
    return {
        "id": world_id,
        "name": world.name,
        "source": world.source,
        "blurb": described if not described.startswith("alias for") else "",
        "buildings": summary["buildings"] - summary["trees"],
        "trees": summary["trees"],
        "landmarks": summary["landmarks"],
        "tallest_roof": summary["tallest_roof"],
        "elevation_min": summary["elevation_min"],
        "elevation_max": summary["elevation_max"],
        "bounds": [round(minx), round(miny), round(maxx), round(maxy)],
        "has_terrain": world.terrain is not None,
        "origin": (
            {"lat": world.origin_lat, "lon": world.origin_lon}
            if world.origin_lat is not None
            else None
        ),
        "built_in": world_id in BUILT_IN,
    }


def world_cards() -> list[dict]:
    """Every world, the four hand-authored places first."""
    cards = [world_card(world_id) for world_id in world_ids()]
    order = {key: index for index, key in enumerate(BUILT_IN)}
    cards.sort(key=lambda card: (order.get(card["id"], len(order)), card["id"]))
    return cards


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def run_ids() -> list[str]:
    """Run file stems, newest first."""
    if not RUNS_DIR.exists():
        return []
    return [
        path.stem
        for path in sorted(
            RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    ]


def run_path(run_id: str) -> Path:
    return RUNS_DIR / f"{Path(run_id).name}.json"


def load_run(run_id: str) -> dict:
    path = run_path(run_id)
    if not path.exists():
        raise KeyError(run_id)
    return json.loads(path.read_text())


def run_card(run_id: str) -> dict | None:
    """The summary a run list needs, without shipping every telemetry tick."""
    try:
        run = load_run(run_id)
    except (KeyError, json.JSONDecodeError, OSError):
        return None

    sim = run.get("sim") or {}
    ticks = run.get("ticks") or []
    world = run.get("world") or {}
    waypoints = run.get("waypoints") or []
    phases: list[str] = []
    for waypoint in waypoints:
        if not phases or phases[-1] != waypoint.get("subtask"):
            phases.append(waypoint.get("subtask"))
    when = datetime.fromtimestamp(run_path(run_id).stat().st_mtime)

    return {
        "id": run_id,
        "mission_text": run.get("mission_text") or run_id,
        "created_at": run.get("created_at") or when.isoformat(timespec="seconds"),
        "world": world.get("name"),
        "duration_s": sim.get("duration_s"),
        "distance_m": sim.get("distance_flown_m"),
        "battery_pct": sim.get("battery_remaining_pct"),
        "completed": run.get("completed"),
        "crashed": bool(sim.get("crashed")),
        "replans": run.get("replans", 0),
        "verified": (run.get("verification") or {}).get("passed"),
        "ticks": len(ticks),
        "waypoints": len(waypoints),
        "captures": sum(1 for w in waypoints if w.get("action") == "CAPTURE"),
        "phases": phases,
        "flyable": bool(ticks),
    }


def run_cards(limit: int | None = None) -> list[dict]:
    cards = [card for card in (run_card(run_id) for run_id in run_ids()) if card]
    return cards[:limit] if limit else cards


def delete_run(run_id: str) -> None:
    path = run_path(run_id)
    if not path.exists():
        raise KeyError(run_id)
    path.unlink()


# ---------------------------------------------------------------------------
# Bundled plans: the demo that needs no API key
# ---------------------------------------------------------------------------

#: Which world each bundled plan was written for. A plan is just waypoints, so
#: nothing in the file itself says where it belongs.
PLAN_WORLDS = {
    "central_vista_survey": "delhi",
    "lodhi_garden_tour": "lodhi",
    "chandni_chowk_run": "chowk",
    "munnar_ridge_run": "munnar",
}


def plan_cards() -> list[dict]:
    """Every plan in examples/plans, ready to fly without a planner."""
    if not PLANS_DIR.exists():
        return []
    cards = []
    for path in sorted(PLANS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        subtasks = payload.get("subtasks") or []
        cards.append(
            {
                "id": path.stem,
                "mission_text": payload.get("mission_text") or path.stem,
                "world": PLAN_WORLDS.get(path.stem),
                "subtasks": [s.get("name") for s in subtasks],
                "waypoints": sum(len(s.get("waypoints") or []) for s in subtasks),
            }
        )
    return cards


def plan_path(plan_id: str) -> Path:
    return PLANS_DIR / f"{Path(plan_id).name}.json"
