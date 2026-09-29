"""The waypoint API: worlds to fly in, missions to fly, runs to replay.

    .venv/bin/python -m uvicorn api.main:app --reload --port 8000

Every route is a thin wrapper around the same code the CLI runs. The one thing
that is genuinely of the server is `GET /api/missions/{id}/events`, which
streams a mission's progress as server-sent events so the page can show
planning, verification, replanning and flight as they happen rather than
handing over a finished run and pretending it was instant.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from waypoint.api import catalog
from waypoint.api.missions import MISSIONS, MissionError, MissionRequest, STAGES
from waypoint.api.settings import ALLOWED_ORIGINS
from waypoint.planner.config import get_api_key, get_model

app = FastAPI(
    title="waypoint",
    version="1.0.0",
    summary="Safety-aware hierarchical LLM mission planning for UAVs.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """What this server can do, so the page can say so before you try it.

    `planner_ready` is the difference between "type a mission" and "fly one of
    the bundled plans": without a Gemini key everything except the planner still
    works, and the page should offer that rather than a broken text box.
    """
    return {
        "ok": True,
        "planner_ready": get_api_key() is not None,
        "planner_model": get_model(),
        "stages": list(STAGES),
        "worlds": len(catalog.world_ids()),
        "runs": len(catalog.run_ids()),
        "plans": len(catalog.plan_cards()),
    }


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


@app.get("/api/worlds")
def list_worlds() -> list[dict]:
    return catalog.world_cards()


@app.get("/api/worlds/{world_id}")
def get_world(world_id: str) -> dict:
    """The whole world: terrain grid, every building, every landmark."""
    try:
        return catalog.load_world(world_id).model_dump(mode="json")
    except KeyError:
        raise HTTPException(404, f"no world called '{world_id}'") from None


@app.get("/api/worlds/{world_id}/preview")
def preview_world(world_id: str) -> dict:
    """The world shaped like a run, with nothing flown in it yet.

    The 3D view can then load a place to look around and a finished flight
    through exactly the same code, which is why a world preview is not its own
    kind of thing.
    """
    try:
        world = catalog.load_world(world_id)
    except KeyError:
        raise HTTPException(404, f"no world called '{world_id}'") from None

    from waypoint.api.missions import constraints_for  # noqa: PLC0415 -- avoids a cycle

    constraints, notes = constraints_for(world, None)
    return {
        "schema_version": 2,
        "mission_text": f"{world.name}: nothing has flown here yet",
        "preview": True,
        "constraints": constraints.model_dump(mode="json"),
        "constraint_notes": notes,
        "world": world.model_dump(mode="json"),
        "waypoints": [],
        "ticks": [],
        "events": [],
        "obstacles": [],
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@app.get("/api/worlds/{world_id}/site")
def survey_site(
    world_id: str,
    x: float = Query(description="Metres east of the world's origin."),
    y: float = Query(description="Metres north of the world's origin."),
    snap: bool = Query(default=True, description="Move to usable ground if this spot will not do."),
) -> dict:
    """Can the aircraft stand here?

    A client asks this before it places a pad, so that where a pad may go is
    decided in one place -- here -- rather than being guessed somewhere else
    and disagreeing with the flight.
    """
    try:
        world = catalog.load_world(world_id)
    except KeyError:
        raise HTTPException(404, f"no world called '{world_id}'") from None

    from waypoint.world.landing import find, survey  # noqa: PLC0415 -- avoids a cycle

    asked = survey(world, x, y)
    site = find(world, x, y) if snap and not asked.usable else asked
    return {
        **site.model_dump(mode="json"),
        "usable": site.usable,
        "asked": {"x": x, "y": y, "usable": asked.usable, "why_not": asked.why_not()},
        "reason": site.describe(),
    }


@app.get("/api/runs")
def list_runs(limit: int = Query(default=60, ge=1, le=500)) -> list[dict]:
    return catalog.run_cards(limit=limit)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Any:
    """The whole run file, telemetry included -- what the replay animates."""
    try:
        return JSONResponse(catalog.load_run(run_id))
    except KeyError:
        raise HTTPException(404, f"no run called '{run_id}'") from None
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(500, f"{run_id} could not be read: {exc}") from exc


@app.delete("/api/runs/{run_id}")
def remove_run(run_id: str) -> dict:
    try:
        catalog.delete_run(run_id)
    except KeyError:
        raise HTTPException(404, f"no run called '{run_id}'") from None
    return {"deleted": run_id}


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@app.get("/api/plans")
def list_plans() -> list[dict]:
    """The bundled plans: something to fly when there is no planner key."""
    return catalog.plan_cards()


# ---------------------------------------------------------------------------
# Missions
# ---------------------------------------------------------------------------


class MissionBody(BaseModel):
    """What to fly. Give it `mission` text, or a bundled `plan` to skip the LLM."""

    mission: str = Field(default="", description="The mission, in plain English.")
    plan: str | None = Field(default=None, description="A bundled plan id, instead of text.")
    world: str | None = Field(default=None, description="A world id; omit for open space.")
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    turbulence: float = Field(default=0.0, ge=0, le=5)
    faults: str = Field(default="none", description="'none', 'sample', or a schedule name.")
    retries: int = Field(default=2, ge=0, le=5, description="Corrections to ask for.")
    max_altitude: float | None = Field(default=None, gt=0)
    blind: bool = Field(default=False, description="No wind compensation: fly blind.")
    home: tuple[float, float] | None = Field(
        default=None,
        description="Where to launch from, as (x, y) in metres. Defaults to the origin.",
    )
    landing: tuple[float, float] | None = Field(
        default=None,
        description="Where to put down, as (x, y) in metres. Defaults to where it started.",
    )


@app.post("/api/missions", status_code=202)
def start_mission(body: MissionBody) -> dict:
    """Start a mission and return immediately; watch it on the events stream."""
    if not body.mission.strip() and not body.plan:
        raise HTTPException(422, "give a mission in plain English, or a bundled plan id")
    if body.plan and not catalog.plan_path(body.plan).exists():
        raise HTTPException(404, f"no bundled plan called '{body.plan}'")
    if body.mission.strip() and get_api_key() is None:
        raise HTTPException(
            503,
            "there is no GEMINI_API_KEY in this server's environment, so a mission "
            "written in English cannot be planned. Fly a bundled plan instead.",
        )
    if body.world:
        try:
            catalog.load_world(body.world)
        except KeyError:
            raise HTTPException(404, f"no world called '{body.world}'") from None

    mission = MISSIONS.start(
        MissionRequest(
            mission=body.mission.strip(),
            plan_id=body.plan,
            world=body.world,
            wind=body.wind,
            turbulence=body.turbulence,
            faults=body.faults,
            retries=body.retries,
            max_altitude=body.max_altitude,
            blind=body.blind,
            home=body.home,
            landing=body.landing,
        )
    )
    return {"id": mission.id, "status": mission.status}


@app.get("/api/missions")
def list_missions() -> list[dict]:
    return MISSIONS.list()


@app.get("/api/missions/{mission_id}")
def get_mission(mission_id: str) -> dict:
    """Everything that has happened so far, in one response.

    The stream is nicer to watch, but a snapshot is what a page needs after a
    reload, and what a script needs when it does not want to hold a connection.
    """
    try:
        return MISSIONS.get(mission_id).snapshot()
    except KeyError:
        raise HTTPException(404, f"no mission called '{mission_id}'") from None


@app.post("/api/missions/{mission_id}/cancel")
def cancel_mission(mission_id: str) -> dict:
    try:
        mission = MISSIONS.cancel(mission_id)
    except KeyError:
        raise HTTPException(404, f"no mission called '{mission_id}'") from None
    return {"id": mission.id, "status": mission.status, "cancelled": True}


@app.get("/api/missions/{mission_id}/events")
def mission_events(mission_id: str, request: Request, since: int = 0) -> StreamingResponse:
    """Server-sent events: the mission's whole log, then each new line.

    `since` replays from an offset, so a reconnecting page picks up where it
    left off instead of re-rendering the flight from the beginning.
    """
    try:
        mission = MISSIONS.get(mission_id)
    except KeyError:
        raise HTTPException(404, f"no mission called '{mission_id}'") from None

    def stream() -> Iterator[bytes]:
        for event in mission.follow(start=since):
            if event.get("kind") == "heartbeat":
                yield b": still here\n\n"       # keeps proxies from timing us out
                continue
            yield f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n".encode()
        yield b'event: close\ndata: {"kind": "close"}\n\n'

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
