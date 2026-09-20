# waypoint

**Safety-Aware Hierarchical LLM for Autonomous UAV Mission Planning and
Adaptation**. Say a mission in plain English, watch a language model decompose
it, a deterministic verifier check it, and a simulator fly it through a real
place. The planner, the verifier and the simulator are real; only the feedback
layer is still mocked.

```
natural language mission
        -> world      (terrain + buildings)        [REAL: four Indian places, or OpenStreetMap]
        -> planner    (LLM hierarchical planner)   [REAL: Gemini, strict JSON response schema]
        -> verifier   (deterministic safety check) [REAL: shapely geometry, no model in the loop]
        -> sim        (UAV executor)               [REAL: point-mass physics, wind, faults, collisions]
        -> feedback   (telemetry / perception)     [stub: never requests a replan]
        -> replan loop back into planner           [wired: a rejected plan goes back with the reasons]
```

There are two ways in. **The studio** is a web application: pick a place, look
around it in 3D, type what you want, and watch each stage happen:

```bash
.venv/bin/python scripts/studio.py
```

**The CLI** does the same thing at a prompt, and every stage is also importable
on its own. Both drive the same pipeline; neither is a wrapper around the other.

## Layout

| Path            | Role |
| --------------- | ---- |
| `world/`        | `World`, `Terrain`, `Building`; procedural cities and an OpenStreetMap importer |
| `planner/`      | `plan()` / `replan()` against Gemini; `Waypoint`, `SubTask`, `Plan`, `Action`, `MissionConstraints` |
| `verifier/`     | `verify(plan, constraints) -> VerificationResult` (`passed`, `violations`); the checks live in `verifier/checks.py` |
| `sim/`          | `UAVState`, `Executor`, point-mass `dynamics`, `Wind`, `FaultSchedule`; `execute(plan)` yields a state per 0.1 s tick |
| `feedback/`     | `observe(state) -> Feedback` (`replan_required`, `reasons`) |
| `orchestrator/` | `run_mission(mission_text)` and the `python -m orchestrator.run` CLI |
| `tests/`        | pytest suite, one module per component |
| `examples/`     | ready-to-fly plans and a fault schedule, so the demo needs no API key |
| `worlds/`       | saved environments built by `scripts/make_world.py` |
| `experiments/`  | scratch scenarios and logged runs |
| `api/`          | FastAPI over the pipeline: worlds, missions (streamed as they happen), runs |
| `web/`          | The studio: Next.js + react-three-fiber, the 3D view and the mission box |
| `viewer/`       | `index.html`: the original single-file three.js replay, no build step |
| `gazebo/`       | Gazebo as the 3D view: a run becomes an SDF world, and the flight is replayed in it |

## Configuration

Secrets live in `.env`, which is gitignored. `.env.example` documents the keys:

| Variable | Required | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | yes | Gemini API key ([get one](https://aistudio.google.com/apikey)) |
| `GEMINI_MODEL` | no | Moves one model to the front of the ladder; defaults to `gemini-3.6-flash` |

A real environment variable overrides `.env`, so exported keys and CI secrets
win. A blank key is treated as unset, so the placeholder in a fresh `.env`
fails with a clear message rather than a 401.

## Setup

```bash
git clone https://github.com/amanraj069/Waypoint.git && cd Waypoint
uv venv --python 3.11                   # or: python3.11 -m venv .venv
uv pip install -e ".[dev,web]"          # or: .venv/bin/pip install -e ".[dev,web]"
```

That is everything the simulator, the API and the old viewer need. The studio
also needs Node 20 or newer, once:

```bash
cd web && npm install && cd ..
```

If you only want the CLI, `".[dev]"` is enough; `web` is just FastAPI and
uvicorn, and nothing outside `api/` imports them. A Gemini key is only
required to *plan* a new mission; flying and replaying work without one:

```bash
cp .env.example .env                    # then put your Gemini key in it
```

**Every command below is written as `python`, meaning the interpreter in the
virtualenv.** On a fresh macOS shell there is usually no `python` at all, so
either prefix each command:

```bash
.venv/bin/python scripts/simulate.py ...      # always works
```

…or activate the virtualenv once per terminal, after which plain `python` means
the right one:

```bash
source .venv/bin/activate
python scripts/simulate.py ...
```

`uv run python scripts/...` works too. A system `python3` will start but will
not have the dependencies.

## Run

### 0. The studio

One command starts the API, the web app, and a browser on it:

```bash
.venv/bin/python scripts/studio.py
```

```
api   http://127.0.0.1:8000
      5 worlds, 19 runs, planner gemini-3.6-flash
web   http://localhost:3000

ready. Ctrl-C to stop both.
```

The page has two columns. On the left, one sidebar holding everything *about*
the mission: **where** (the places below, plus anything you have imported),
**this place**, and the pipeline as it runs: **pipeline**, **plan**,
**verifier** with each violation spelled out, and the **log**. On the right, the
place itself, the transport bar, and **the mission box**, where you type the
instruction; every named place in the world is a chip under the box, so you can
click "Bara Gumbad" rather than guess what it is called.

Every panel folds, and so does the whole sidebar: `⟨⟨` collapses it to a rail
and gives the 3D view the full width, `⟩⟩` brings it back. All of it is
remembered, so the layout you arrive at is the layout you left.

**Folding things away.** Every panel folds; click its title. A mission you are
watching does not need the world list, and a place you are studying does not
need the log. What you fold stays folded: the state is kept per panel in
`localStorage`, so the layout you arrive at is the layout you left.

**Light or dark.** The switch is in the top bar, and it remembers. With no
choice stored it follows your system. Both palettes are defined once as CSS
variables in `web/app/globals.css`, and the 3D view reads the *same* variables,
so the scene can never be dark while its chrome is light: the ground, the
buildings and the name plates all change with the page. A small script in the
layout applies your stored choice before the first paint, so there is no flash
of the wrong palette on the way in.

**Choosing where it takes off.** Both pads are on screen from the moment a
world loads: a green one for the start, an amber one for the landing, each a
lit disc with a beam and a name plate so you can find them from across a city.
Until you move the landing pad it sits on the launch pad, which is what "comes
home" looks like.

**Drag them.** Grab either pad and pull it across the ground. It follows the
terrain as it goes, re-reading the ground height under it on every frame, and
turns red the moment it is over something it could not stand on. Let go and the
server surveys the spot properly and has the last word. If you would rather
aim than drag, **start** and **landing** in the toolbar turn the pointer into a
crosshair for one click; ✕ puts either back to its default.

That one point is shared by all three stages, which is the only way it can mean
anything: the planner is told the aircraft is standing there and writes a plan
that climbs from it and lands back on it, the verifier measures the first leg
from it, and the simulator starts the aircraft on it. The altitude comes from
the ground under the spot you picked, so a pad on a hillside is at the height of
that hillside; you never have to work it out.

**And where it puts down.** **set landing** does the same for the other end, in
amber so the two pads are never confused. Leave it alone and the aircraft comes
home; set it and the plan ends somewhere else.

A landing point gets more scrutiny than a launch point, because a patch of
ground that is perfectly clear can still be useless: ground that falls away
under the skids is how a multirotor tips over on touchdown. So the spot is
*surveyed* rather than assumed: the ground is sampled across the aircraft's
own 6 m footprint, and a site is only usable if it slopes no more than 8° and
rises or falls no more than 0.9 m across that footprint.

Click somewhere that will not do and the pad does not go there. It moves to the
nearest ground that will, within 120 m, and says what happened:

```
moved the landing 112 m to level ground: the ground falls 3.4 m
across the 6 m pad (30°), too uneven to set down on
```

If nothing within 120 m would do (a 45° face in Munnar, say), it refuses
outright rather than putting you somewhere surprising. The survey lives in
`world/landing.py` and runs on the server, so the pad you are looking at cannot
disagree with the flight you get; the browser asks
`/api/worlds/{id}/site?x=&y=` on every click rather than guessing for itself.

Two things it will refuse. Starting inside a building (it moves you off it), and
moving either pad under a *bundled* plan, whose waypoints are absolute and whose
last one lands where it was written; for that, write the mission in English
instead.

When the flight is written the middle column loads it and plays it. `/runs` is
every flight this machine has made; opening one gives the full replay, with the
onboard photographs rendered from the aircraft's own pose at each `CAPTURE`.

**Without a Gemini key the box is disabled and the bundled plans are not**: each
world ships with a verified plan you can fly, which exercises the verifier, the
simulator, the fault injection and the replay without spending a token.

Flags: `--api-port`, `--web-port`, `--no-open`, `--api-only`, `--reload`.

The API is a plain HTTP service if you would rather drive it yourself
(`http://127.0.0.1:8000/api/docs` documents itself):

| Route | |
| --- | --- |
| `GET /api/health` | what this server can do, and whether it has a planner key |
| `GET /api/worlds` · `/api/worlds/{id}` · `/api/worlds/{id}/preview` | the places |
| `POST /api/missions` | `{"mission": "...", "world": "lodhi"}` or `{"plan": "lodhi_garden_tour"}` |
| `GET /api/missions/{id}/events` | server-sent events: every stage as it happens |
| `GET /api/runs` · `/api/runs/{id}` | the flights |

```bash
curl -X POST localhost:8000/api/missions -H 'content-type: application/json' \
     -d '{"mission":"photograph Bara Gumbad from two sides, then come home","world":"lodhi"}'
curl -N localhost:8000/api/missions/m1-49e180/events
```

### 1. Look around, then just say what you want, at a prompt

The quickest way in. It opens the world in your browser with nothing flying in
it, so you can orbit around and see what you are dealing with, then you type
missions at a prompt and watch each one appear in the page you are already
looking at.

```bash
.venv/bin/python scripts/explore.py --world lodhi          # named places to point at
.venv/bin/python scripts/explore.py --world munnar         # open country
.venv/bin/python scripts/explore.py --world worlds/gateway_of_india.json
.venv/bin/python scripts/explore.py --world chowk --wind 4,0,0 --faults sample
```

Lodhi Garden is the one to start with, because everything in it has a name:

```
mission> photograph Bara Gumbad from two sides, then fly north up the ashoka
         avenue and make a low pass over the lake, then come home

  1. takeoff                [0] (0, 0, 25) MOVE
  2. fly_ashoka_avenue      [1] (0, 90, 25)    [2] (0, 170, 25)
  3. photograph_bara_gumbad [3] (5, 185, 25) CAPTURE   [4] (35, 215, 25) CAPTURE
  4. low_pass_lake          [5] (115, 310, 15)  [6] (115, 360, 8) CAPTURE ...
  169s | 933 m flown | 74.0% battery left | landed
```

It flanked the 22 m tomb to get two angles, ran the avenue at 25 m (over the
16 m ashokas, not between them) and dropped to 8 m for the water.

```
central vista (procedural): 66 obstacles (44 trees), tallest roof 80 m, ground -0..26 m
viewer: http://localhost:8765/viewer/index.html?session=1
have a look around, then tell it what to do. /help for commands.

mission> climb over the Secretariat and photograph Rashtrapati Bhavan from the east, then come home

planning...
  1. takeoff              [0] (0, 0, 35) at 3 m/s MOVE
  2. transit_up_the_vista [1] (-200, 0, 55) ...
  3. photograph_rashtrapati_bhavan  [3] (-455, 0, 95) CAPTURE ...
  verification passed

flying...
  ...loading in the browser
```

The browser follows along by itself: it shows the world first, says *planning*,
then *flying*, then loads the finished run and plays it. Type another mission
and it does the whole thing again.

Note the altitude the planner chose: 95 m, to clear a 42 m building standing
on ground that has itself climbed 24 m up Raisina Hill. It reads the terrain
and the structures out of the world description before it commits to a route.

If the verifier rejects a plan, `explore.py` feeds the violations straight back
to the planner and asks for a correction (`--retries`, 2 by default) before
giving up. That is the replan loop closing: the model proposes, the deterministic
checker disposes, and the reasons are specific enough to fix.

**Switching worlds from the page.** The control bar has a world picker
alongside the run picker. It lists everything in `worlds/` (the four built-in
ones are written there the first time you serve), and choosing one shows it
immediately. If a session is running, the page also tells it to switch, so your
*next mission flies in the world you just picked*. It works the other way too:
`/world` at the prompt reloads the page.

| At the prompt | |
| --- | --- |
| *anything else* | a mission, in plain English |
| `/world <spec>` | switch worlds: `delhi`, `lodhi`, `chowk`, `munnar`, or a `worlds/*.json` |
| `/worlds` | list what the picker offers |
| `/wind x,y,z` | set the steady wind in m/s |
| `/faults <spec>` | `sample`, `none`, or a schedule JSON |
| `/retries <n>` | corrections to ask for after a rejection |
| `/again` | re-fly the last mission |
| `/help` `/quit` | |

### 2. Watch the 3D demo (no API key, ~10 seconds)

One command flies a bundled plan through the simulator with a bundled fault
schedule, writes a run file, and opens the three.js viewer on it:

```bash
python scripts/simulate.py \
    --plan examples/plans/north_field_and_tower.json \
    --faults examples/faults/storm.json \
    --serve
```

Your browser opens on `http://localhost:8765/viewer/index.html?run=/runs/<file>`
and the flight animates immediately: the aircraft climbs, flies the survey
pattern, gets shoved sideways by a squall, watches a crane appear beneath it,
drifts off course when its GPS slips, and lands. The terminal prints the flight
log alongside. Press **Ctrl-C** in the terminal when you are done; that stops
the server.

Without `--serve` the run is still written and the command prints the exact
line to replay it later.

### 3. Replay a past run

Run files are self-contained, so nothing is re-simulated:

```bash
python scripts/serve_viewer.py                       # the newest run
python scripts/serve_viewer.py runs/my-run.json      # a specific one
python scripts/serve_viewer.py --port 9000 --no-open # different port, no browser
```

Two pickers in the control bar: **run** replays any past flight, **world** opens
any environment in `worlds/` on its own, with nothing flying in it, so you can
look around before deciding what to ask for.

The viewer's dropdown lists every run in `runs/`, so you can switch between
flights without restarting the server. You can also drag any run `.json`
straight onto the page.

**In the viewer:** `Space` plays and pauses, `←`/`→` step a second at a time,
the slider scrubs, the speed selector runs up to 8×, **Follow** locks the camera
to the aircraft, and clicking any line in the flight log jumps to that moment.
Drag to orbit, scroll to zoom.

### 4. Fly your own scenario

```bash
# fly through a real place instead of empty space
python scripts/simulate.py --plan examples/plans/lodhi_garden_tour.json --world lodhi

# steady 4 m/s easterly with gusty air
python scripts/simulate.py --plan examples/plans/north_field_and_tower.json \
    --wind 4,0,0 --turbulence 0.8

# no wind estimation: the aircraft cannot feel the wind and gets blown off course
python scripts/simulate.py --plan examples/plans/north_field_and_tower.json \
    --wind 2,0,0 --blind

# a clean reference flight, no faults at all
python scripts/simulate.py --plan examples/plans/north_field_and_tower.json --faults none
```

`--faults` takes `sample` (the built-in demo schedule, the default), `none`, or
a path to a schedule JSON; copy `examples/faults/storm.json` and edit it. Other
flags: `--dt` for tick length, `--runs-dir`, `--quiet`, `--force` to fly a plan
that failed verification.

For a sense of scale on the same plan: calm and fault-free is 142 s and lands
with 81% battery; `storm.json` costs 12 more points of battery and pushes the
touchdown 20 m off the pad; `--wind 2,0,0 --blind` takes 453 s because the
aircraft keeps overshooting into the wind. Push `--blind` past about 3 m/s and
it never converges; it drifts downwind until the battery empties and lands
wherever it is, which is realistic but writes a very large run file.

### 5. Give it your own mission in English (needs a Gemini key)

Hand it a sentence and watch the whole pipeline run: the planner decomposes the
mission into named phases, the verifier checks them, the simulator flies them,
and the viewer animates the result.

```bash
python scripts/simulate.py \
    "inspect the three wind turbines along the ridge 120 m north of base, \
     photograph each one from two angles, then come home" \
    --save examples/plans/wind_turbines.json \
    --serve
```

The terminal shows the decomposition before anything flies:

```
  1. takeoff  (1 waypoint(s))
       [0] (0, 0, 20) at 3 m/s  MOVE
  2. transit_to_ridge  (1 waypoint(s))
       [1] (-30, 100, 30) at 8 m/s  MOVE
  3. inspect_turbine_1  (2 waypoint(s))
       [2] (-30, 110, 30) at 3 m/s  CAPTURE
       [3] (-20, 120, 30) at 3 m/s  CAPTURE
  ...
  7 subtask(s), 10 waypoint(s)
```

…then the verifier's verdict, then the flight. In the viewer, the **Mission
plan** panel lists those same phases down the left side and lights each one up
as the aircraft flies it, so the model's decomposition and the flight are the
same object seen two ways. Click a phase to jump to it.

`--save` writes the generated plan to a file, so you can re-fly the same mission
as often as you like (different wind, different faults) without spending
another token:

```bash
python scripts/simulate.py --plan examples/plans/wind_turbines.json --wind 5,0,0 --serve
```

The verifier runs on the model's output either way. If the plan breaks the
envelope the flight is refused and the report says which waypoint and why, which
is the point of having a deterministic gate between the LLM and the aircraft.

Or run the orchestrator loop, which prints each stage and writes a run file:

```bash
python -m orchestrator.run --mission "survey the north field"
python -m orchestrator.run --mission "survey the north field" --plot   # plotly 3D
```

Exits 0 when the mission ends landed and stationary. `--quiet` suppresses
per-tick telemetry.

### 6. Exercise one stage at a time

```bash
python scripts/try_planner.py "survey the north field"          # planner only (spends tokens)
python scripts/try_verifier.py --plan examples/plans/north_field_and_tower.json \
    --shrink 0.3 --hazards                                      # verifier only, no tokens
```

### 7. Watch it in Gazebo

Gazebo (gz-sim, Harmonic or newer; Homebrew's `gz-jetty` works) can stand in for
the browser views. Every run file becomes a Gazebo world: the terrain as a mesh
coloured by ground cover, each building as a named model, the trees, the
geofence and no-fly zones as glass, the planned route as beads coloured by
action, and the launch and landing pads. The aircraft is then flown through it
from the run's ticks, with a chase camera and a magenta trail, and obstacles
that appear mid-flight are spawned when the flight reaches them.

Gazebo's Python bindings are built for Homebrew's Python, not the project's, so
the process that talks to Gazebo runs in a small side environment. Make it once:

```bash
/opt/homebrew/bin/python3.14 -m venv --system-site-packages .venv-gz
.venv-gz/bin/python -m pip install protobuf
```

Then:

```bash
# fly a plan and watch it in Gazebo instead of the browser
python scripts/simulate.py --plan examples/plans/lodhi_garden_tour.json --world lodhi --gazebo

# replay the newest run, or any run file, at any speed
python scripts/gazebo_view.py --latest --speed 2
python scripts/gazebo_view.py runs/<mission>_<stamp>.json

# a place on its own, nothing flying
python scripts/gazebo_view.py --world munnar

# plan with Gemini, then watch the flight
python -m orchestrator.run --mission "survey the north field" --gazebo
```

The launcher starts the gz server and the GUI as separate processes (on macOS
they cannot share one) plus `gazebo/bridge.py`. It prints the flight log as it
plays, and when the flight is over it leaves the window open until you close it.
The scene is written to `~/.cache/waypoint/gazebo/<mission>/`, because SDF mesh
URIs do not survive the spaces in a path like `VS Code/`.

The physics you watch is still ours: Gazebo runs with gravity off, and the
aircraft is posed from the simulator's ticks, so the flight in the window is
exactly the flight the verifier passed and the run file recorded. Putting PX4
SITL underneath instead (Gazebo as PX4's physics, MAVSDK uploading the plan)
would be a new `sim` backend; the world export here is the half of that already
done.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No run files with telemetry in runs/` | Fly one first: `python scripts/simulate.py --plan examples/plans/north_field_and_tower.json` |
| `port 8765 is already in use` | It just moves to 8766 and tells you; the URL it prints is the one to open. To reclaim the port: `lsof -ti:8765 \| xargs kill` |
| Viewer page loads but stays empty | three.js comes from a CDN, so the first load needs internet; check the browser console |
| `zsh: command not found: python` | macOS has no bare `python`; use `.venv/bin/python`, or `source .venv/bin/activate` first |
| `ModuleNotFoundError: shapely` (or `pydantic`) | You are on a system `python3`; use `.venv/bin/python` |
| `no Python can import Gazebo's bindings` | Make `.venv-gz` as shown in [Watch it in Gazebo](#7-watch-it-in-gazebo), or point `WAYPOINT_GZ_PYTHON` at a Python that can `import gz.transport` |
| Gazebo window opens but the aircraft never moves | A previous run is still holding the world: `pkill -f gz-sim-main`, then try again |
| Plan rejected before flying | That is the verifier doing its job; the report says which waypoint and why. `--force` flies it anyway |

## Planner

`plan(mission_text, constraints)` calls Gemini and is the only path to a `Plan`.
The request pins the reply to `application/json` against a strict schema
(`additionalProperties: false`, every field required), so the response either
decodes and validates into the `Plan` model or raises:

| Failure | Exception |
| --- | --- |
| Blocked, truncated, empty, non-JSON, or fenced-in-markdown response | `PlannerResponseError` |
| Overload, quota, auth or network failure (after exhausting the model ladder) | `PlannerAPIError` |
| Tool arguments that don't form a valid `Plan` (bad enum, speed <= 0, empty plan, wrong types) | `PlanValidationError` |

Both subclass `PlannerError`. There is no free-text fallback and no silent
default: a bad plan never reaches the verifier looking like a good one.

`replan(mission_text, previous_plan, violation_report, constraints)` re-prompts
with the rejected plan and the verifier's exact reasons, for the feedback loop.

### Model ladder

Gemini models go through spells of overload (`500 ... experiencing high demand`)
and free-tier quota limits. Rather than failing the mission, the planner shifts
down `planner/config.py:MODEL_LADDER`:

```
gemini-3.6-flash -> gemini-3.5-flash -> gemini-3.7-flash -> gemini-2.5-flash -> gemini-2.5-flash-lite
```

It shifts on 408/429/500/502/503/504 and on connection or timeout errors, and
logs a warning each time. It does **not** shift on 400/401/403; a bad request
or a bad key fails identically on every model, so those fail fast on the first
rung. A malformed *plan* does not shift either: that is a planning failure, not
an availability one. If every rung fails, `PlannerAPIError` lists each attempt.

Manual test against the live API (spends tokens):

```bash
python scripts/try_planner.py "survey the north field and return to base"
```

## Verifier

`verify(plan, constraints)` is the gate between the planner and the aircraft. It
is 100% deterministic (plain `shapely` geometry and arithmetic, no model, no
sampling), so the same plan and constraints always produce the same report. It
runs every check rather than stopping at the first failure, so one replan can
fix everything at once.

| Check | What it rejects |
| --- | --- |
| `GEOFENCE` | A waypoint outside the fence, or a leg that leaves a concave fence with both ends inside it |
| `NO_FLY_ZONE` | A waypoint inside a no-fly polygon, or a leg that crosses one |
| `ALTITUDE` | A waypoint outside `[min_altitude, max_altitude]` (z = 0 is excused for the launch waypoint and for a `LAND`) |
| `MAX_VELOCITY` | A commanded speed over the limit, or a climb steep enough that the commanded *ground* speed implies an over-limit speed along the path |
| `TURN_FEASIBILITY` | A corner sharper than the airframe can fly at the speed commanded through it |
| `BATTERY` | A route whose linear energy estimate exceeds `battery_budget_pct` |
| `MIN_SEPARATION` | A waypoint or leg inside an obstacle's keep-out radius (`obstacle.radius + min_separation`) |
| `BUILDING` | A waypoint inside a building, or a leg that flies through one (needs a [world](#the-world)) |
| `TERRAIN` | A waypoint underground or below the ground-clearance minimum, or a leg that flies into a hillside |
| `LANDING` | A LAND waypoint on ground too uneven to set down on, or one that is not at ground level |

Each `Violation` carries its `type`, the `waypoint_index` it belongs to (0-based
into the flattened plan, `None` for whole-plan findings like battery), a
`severity`, and a `description` written to be handed straight back to the model:

```
waypoint 3 (subtask "survey_north_field") at (80, 150, 30) is outside the
geofence, 55.0 m beyond its boundary. Move this waypoint back inside the
geofence polygon, leaving a few metres of margin.
```

`result.passed` is False when any **blocking** violation (`ERROR` or `CRITICAL`)
was found. `WARNING` findings (a marginal turn, a waypoint hugging the fence)
are reported but do not reject a plan. `result.messages()` hands the
descriptions to `planner.replan()`.

The mission envelope lives on `MissionConstraints` (geofence, altitude bounds,
max velocity, battery budget, `no_fly_zones`, `obstacles`, `min_separation`).
The *aircraft* model lives on `VerifierConfig` and is fully configurable:

| Field | Default | Meaning |
| --- | --- | --- |
| `meters_per_battery_pct` | 40 | Metres of flight per 1% of battery |
| `hover_seconds_per_battery_pct` | 30 | Seconds of hovering per 1% of battery |
| `hover_seconds_per_waypoint` | 2 | Dwell charged to each `HOVER` waypoint |
| `reserve_battery_pct` | 0 | Battery that must still be unspent at touchdown |
| `max_turn_angle_deg` | 150 | Turn the aircraft can make at or below the reference speed |
| `turn_reference_speed` | 5 | Above this, the allowed turn shrinks in proportion to speed |
| `geofence_margin` | 0 | Warn when a waypoint is inside the fence but within this margin |
| `home` | (0, 0, 0) | Start and end point, so the first leg is home → waypoint 0 |

Bad *constraints* (a self-intersecting geofence, a degenerate no-fly polygon)
raise `ValueError`. A bad *plan* never raises; it comes back as violations.

Manual check against a real plan (the first form spends tokens):

```bash
python scripts/try_verifier.py "survey the north field and return to base" --save /tmp/plan.json
python scripts/try_verifier.py --plan /tmp/plan.json --shrink 0.3 --hazards
```

It verifies the plan against the normal envelope, then against a geofence
shrunk about its centre (plus, with `--hazards`, a no-fly zone across the route
and an obstacle on its far end) and prints the full violation report.


## The world

Until a world is loaded the aircraft flies in empty space above a flat plane,
where a plan only has to be right in x, y and z. A `World` gives the mission
somewhere to be: ground that rises and falls, and buildings that occupy volume.

Four places ship with the project, hand-authored from their published
dimensions rather than downloaded, so they are identical on every machine:

| Name | Place | What it is for |
| --- | --- | --- |
| `delhi` | The Central Vista, New Delhi | Terrain that matters: the Secretariat is 42 m tall on ground that has climbed 24 m, so its roof is 66 m up |
| `lodhi` | Lodhi Garden, New Delhi | Named things to point at: five tombs, a lake, an avenue of ashokas, and Safdarjung's 47 m dome over the wall |
| `chowk` | Chandni Chowk, Old Delhi | A 30 m street between unbroken shopfronts: the hard case for wind and GPS drift |
| `munnar` | The Kannan Devan hills, Kerala | 235 m of relief, a reservoir and a dam, and almost nothing built |

```bash
python scripts/make_world.py delhi                         # India Gate to Rashtrapati Bhavan
python scripts/make_world.py lodhi                         # tombs, groves and a lake
python scripts/make_world.py chowk --street 24             # a narrower bazaar
python scripts/make_world.py munnar --relief-m 300         # steeper hills
python scripts/make_world.py osm --lat 18.9220 --lon 72.8347 --radius 420 \
    --name gateway_of_india --relief
```

All four can also be used by name without saving anything: `--world lodhi`
builds one on the spot. The older names `city`, `park`, `canyon` and `hills`
still work and mean `delhi`, `lodhi`, `chowk` and `munnar`.

Distances are not always the real ones. The Central Vista is 2.6 km end to end
and the aircraft has about 3.2 km of battery, so it is laid out at roughly a
third scale; Munnar's 1100 m of relief is compressed to 235 m. **Heights are
real** (India Gate is 42 m, the Jama Masjid's minarets are 41 m, Safdarjung's
dome is 47 m) because the heights are what the planner has to reason about.
Each builder's docstring says what it compressed.

### Landmarks: how you say what you want

A world can carry **landmarks**: named places with a position and an extent.
They are not solid (anything you can hit is a building); they exist so a mission
can be *said*. The planner is given the list, so "photograph Bara Gumbad"
becomes waypoints instead of a shrug:

```
- Named places here. The mission may refer to any of them by name:
  - 'Bara Gumbad' (monument) at (5, 215), about 36 m across, 22 m tall, a 15th-century
    domed tomb with its mosque attached to the west
  - 'the lake' (water) at (115, 360), about 104 m across, good for a low pass
  - 'the old grove' (grove) at (-182, 300), about 128 m across, 21 m tall -- fly over it,
    not through it
  - 'the ashoka avenue' (path) at (0, 90), 16 m tall, the two rows stand 26 m apart
  - "Safdarjung's Tomb" (monument) at (-372, 216), 47 m tall, over the west wall, and the
    tallest thing anywhere near
```

In the viewer each one is ringed on the ground and labelled in 3D, so you can
read the name, then type it. Press `l` to hide the labels.

The garden's 61 trees are real obstacles: the ashoka rows will stop a drone at
8 m, the ~17 m gap down the middle will not, and the old grove of neem, peepal
and banyan is dense enough that the only way across it is over the top.

The four built-in places need nothing but Python and are the same every time,
so a scenario stays comparable while you change the planner. The OSM importer
pulls real building footprints from OpenStreetMap through Overpass (the one
part of this project that needs the internet; it tries three mirrors and fails
with a clear message). Heights come from the `height` tag, or
`building:levels × 3.2 m`, or a 9 m default.

**Every building carries a name**, in the built-in worlds and the imported ones.
Most OSM buildings are anonymous, so the importer falls back from the `name` tag
through the address to the building type (`12 Colaba Causeway`, `apartments 7`,
`unnamed block 213`) and promotes the named, notable ones to landmarks, so a
place nobody hand-authored still has things a mission can refer to by name.

Worlds are saved to `worlds/<name>.json` and embedded in each run file, so a
replay never has to rebuild or re-download anything.

**Altitude means height above the launch pad**, not height above whatever is
underneath. That is the one thing to hold on to: over a 40 m hill, `z = 30` is
underground, and `world.agl(x, y, z)` is what tells you the difference.
Procedural terrain is levelled so the pad sits at zero.

### What it changes

| Layer | With a world |
| --- | --- |
| Planner | The prompt describes the relief and the tallest roofs, so it plans over or around them |
| Verifier | Two new checks: `BUILDING` (a waypoint inside one, or a leg that flies through one) and `TERRAIN` (underground, or below the ground-clearance minimum, including legs that fly into a hillside between two clear waypoints) |
| Simulator | The aircraft can hit things. A `COLLISION` event ends the flight, and `report.completed` is False |
| Pads | Where the aircraft may stand is surveyed, not assumed: no steeper than 8°, and no more than 0.9 m of rise and fall across its 6 m footprint (`world/landing.py`) |
| Telemetry | Every tick carries `ground` and `agl` alongside `z` |
| Viewer | The grid is replaced by the real terrain mesh and extruded buildings |

The interesting case is the one the verifier cannot save you from:

```bash
python scripts/simulate.py --plan examples/plans/chandni_chowk_run.json --world chowk \
    --faults examples/faults/gps_slip.json
```

```
t=  2.9s  FAULT      gps slip: GPS drifts 1.2 m/s toward bearing 0 deg for 30s
t= 47.7s  COLLISION  struck 'Fatehpuri Masjid' at (-286, 0), 26 m tall
                     at (-255.2, 0.0, 14.0) doing 6.0 m/s -- its roof is at 26 m
```

The plan is verified safe (it runs down the middle of the street and stops
20 m short of the mosque that closes its west end), and the aircraft hits the
mosque anyway, because GPS drift moves where it *thinks* it is while Fatehpuri
Masjid stays where it has been since 1650. That is a capability you cannot test
in empty space.

### Flying a real place

`worlds/gateway_of_india.json` ships with the project: 118 real footprints from
OpenStreetMap around Colaba, Mumbai, 80 of them carrying the name their mapper
gave them: the Taj Mahal Palace, the Royal Bombay Yacht Club, Regal Cinema,
Dhanraj Mahal, the Gateway of India itself. Build another anywhere:

```bash
python scripts/make_world.py osm --lat 28.6129 --lon 77.2295 --radius 400 --name india_gate
.venv/bin/python scripts/explore.py --world worlds/india_gate.json
```

The trade is accuracy for completeness: the footprints are survey-accurate and
the heights usually are not, because most buildings have no `height` tag and
fall back to `building:levels × 3.2 m` or a 9 m default. OSM says the Taj Mahal
Palace is 19 m; it is nearer 85. The hand-authored worlds are the other way
round: approximate plans, real heights.

The verifier names whatever it rejects, which is what makes a rejection
actionable:

```
[CRITICAL] [BUILDING] the leg from waypoint 4 ... passes through 'The Torch'
at (-184, 143), 325.1 m tall: over the footprint the route is as low as 200 m,
below the 325 m roof. Route around the building, or climb above its roof.
```

When `--world` is given the geofence widens to the world's bounds, since the
default practice area is smaller than a city; `--keep-geofence` turns that off.
The API does the same, and also raises the ceiling when the world needs it
(nothing could be flown over Anamudi under a 120 m ceiling) and says so.


## Simulator

`Executor(plan, constraints=..., config=..., wind=..., faults=...)` flies a
verified plan and yields a `UAVState` every `dt` (0.1 s by default). It is a
small custom simulator rather than PX4/Gazebo: pure Python, no binary
dependencies, fast enough to fly a mission in well under a second, and
identical on macOS, Linux and Windows.

Three things make it a simulator rather than an interpolator:

* **It accelerates.** The controller asks for a velocity each tick and the
  airframe moves toward it within `max_acceleration`, capped at the commanded
  speed and at `sqrt(2 * a * distance)` on approach so it can stop on a
  waypoint instead of orbiting it. Nothing teleports.
* **It flies on its estimate.** The controller steers on the *reported*
  position, so GPS drift takes the aircraft somewhere other than where it
  believes it is going.
* **It can run out.** Battery drains with distance flown plus a baseline draw;
  at 0% the plan is abandoned and the aircraft lands.

```python
from sim import Executor, FaultEvent, FaultSchedule, FaultType, Trigger, Wind

executor = Executor(
    plan,
    constraints=constraints,
    wind=Wind(vector=(3.0, 0.0, 0.0), variation=(1.5, 0.0, 0.0), period=20.0),
    faults=FaultSchedule.of(
        FaultEvent(type=FaultType.WIND_GUST, trigger=Trigger(time=12.0),
                   magnitude=9.0, direction=90.0, duration=6.0, name="squall"),
    ),
)
states = list(executor.run())      # one UAVState per tick
executor.events                    # faults, waypoints reached, separation, touchdown
```

### Wind

`Wind.at(t)` is a pure function of time: a steady `vector`, plus a `variation`
amplitude swept as a sine over `period`, plus optional seeded `turbulence`. Two
runs of one scenario are identical, turbulence included.

`SimConfig.wind_compensation` (0.75 by default) is how much of the wind the
autopilot senses and cancels. What it misses blows the aircraft off the line, so
gusts hurt more than steady wind. Set it to 0 to fly blind.

### Fault injection

A `FaultSchedule` is a list of `FaultEvent`s, each with a `Trigger`. Triggers
fire on `time` or on a flight condition -- `battery_below`, `at_waypoint`,
`beyond_distance`, `above_altitude` -- and combine with AND.

| Fault | Effect |
| --- | --- |
| `GPS_DRIFT` | Reported position slides from the true one at `magnitude` m/s, so the aircraft flies off course believing it is on track |
| `OBSTACLE_APPEARS` | An `Obstacle` joins the live obstacle list mid-flight, as perception would find one |
| `WIND_GUST` | A `magnitude` m/s gust from `direction`, for `duration` seconds |
| `BATTERY_DRAIN_SPIKE` | An extra `magnitude` % per second of drain, for `duration` seconds |

Schedules are plain pydantic models, so a scenario can be written as JSON and
replayed exactly:

```bash
python scripts/simulate.py --plan /tmp/plan.json --faults my_scenario.json
```

## 3D views

The view draws **on demand**: while a replay is playing it animates, and the
moment it is paused (or while you are simply looking at a place), it stops
drawing entirely and costs nothing. Orbiting, seeking, toggling labels and
loading a world each ask for the frames they need. Nothing idles: the rotors
only turn while the aircraft is flying, because a scene that is never still can
never stop being drawn.

Geometry is merged by kind, so a city is a handful of draw calls rather than one
per building plus one per outline: the Gateway of India's 118 structures go
from 238 meshes to 3. Photographs render into an offscreen target, so the canvas
does not have to preserve its drawing buffer on every frame of every flight for
the sake of the occasional shot.


There are two, and they draw the same run file. The studio's is a React port in
`web/components/Viewport.tsx`, built on react-three-fiber, with the scene
construction kept framework-free in `web/lib/scene/`. The original is
`viewer/index.html`, and it still works exactly as it did.

`viewer/index.html` is a single file of vanilla JS that animates a run in 3D
with three.js (loaded from a CDN via an import map -- no build step, no
`node_modules`, but it does want an internet connection the first time). It
draws the geofence, no-fly zones, obstacles as they appear, the planned route,
the flown path as it is flown, and a ghost marker showing where GPS *said* the
aircraft was.

`serve_viewer.py` uses Python's own `http.server`, so there is nothing to
install. It writes `viewer/runs.json` so the viewer's dropdown lists past runs,
then opens `viewer/index.html?run=/runs/<file>`. Run files are self-contained
(`runs/<mission>_<stamp>.json`, schema 2), so a past flight replays without
re-simulating.

See [Run](#run) for the commands and the viewer's controls.


## Test

```bash
.venv/bin/python -m pytest          # 279 tests, about 2 seconds
cd web && npx tsc --noEmit          # the studio type-checks
```

Every test mocks the Gemini client; an autouse fixture replaces the real
constructor with a tripwire, so a test that forgets to inject a fake client
fails rather than dialling out. `tests/test_api.py` flies whole missions through
the HTTP API using the bundled plans, so the pipeline, the event stream and the
run files are covered without a key and without a network.

## Where the real pieces go

- `planner/`: done. Tune `planner/prompts.py` (system prompt + example) rather
  than the call site.
- `verifier/`: done. Tune the aircraft model in `VerifierConfig` rather than
  the checks; add a new check as a function in `verifier/checks.py` plus a
  `ViolationType`.
- `sim/`: done. Tune the airframe in `SimConfig` and the disturbances in
  `Wind` / `FaultSchedule`; bridging to PX4/Gazebo later means reimplementing
  `Executor.run()` and keeping the generator interface.
- `feedback/stub_feedback.py`: set `replan_required=True` to trigger the
  adaptation loop. `orchestrator/loop.py` counts these in `MissionReport.replans`;
  `run_mission(planner=...)` already takes an injected planner for it. The
  *verification* half of that loop is closed: `api/missions.py` and
  `scripts/explore.py` both hand a rejected plan back to the planner with the
  violations, but the *in-flight* half, replanning because of what the aircraft
  saw, still goes through this stub.
- `api/`: the routes are thin. New capability belongs in the packages they call,
  not here; `api/missions.py` is the only file with logic of its own, and that
  logic is the stage-by-stage reporting.
- `web/`: `lib/scene/build.ts` is the 3D, `lib/useMission.ts` folds the event
  stream into what the page draws, and `components/MissionBox.tsx` is the front
  door. There is no state library: the playhead is a plain object in
  `lib/playback.ts` because it moves at frame rate and React does not.
