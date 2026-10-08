# waypoint

**Safety-Aware Hierarchical LLM for Autonomous UAV Mission Planning and
Adaptation**. Say a mission in plain English, watch a language model decompose
it, a deterministic verifier check it, and a simulator fly it through a real
place, replanning in the air when something gets in the way. The planner, the
verifier, the simulator and the in-flight feedback are all real.

```
natural language mission
        -> world      (terrain + buildings)        [REAL: four Indian places, or OpenStreetMap]
        -> planner    (LLM hierarchical planner)   [REAL: OpenAI, then Gemini; strict JSON schema]
        -> verifier   (deterministic safety check) [REAL: shapely geometry, no model in the loop]
        -> sim        (UAV executor)               [REAL: point-mass physics, wind, faults, collisions]
        -> feedback   (perception + monitor)       [REAL: new obstacle on the route, battery short or failing, off route]
        -> replan loop back into planner           [REAL: on the ground after a rejection, in the air from where it is]
```

There are two ways in. **The prompt** opens a place, takes a mission in plain
English, and flies it:

```bash
.venv/bin/python scripts/explore.py --world lodhi
```

**The CLI** does each stage on its own, and every stage is also importable.
Both drive the same pipeline; neither is a wrapper around the other. You watch
the result in **Gazebo**, which is the only 3D view: every run becomes an SDF
world and the flight is replayed in it.

## Layout

```
waypoint/                 the repository
├── waypoint/             all the code, one importable package
│   ├── world/            where it flies: terrain, buildings, landmarks
│   ├── planner/          English -> hierarchical Plan, via OpenAI or Gemini
│   ├── verifier/         deterministic safety check of a Plan
│   ├── sim/              point-mass physics, wind, fault injection
│   ├── feedback/         perception sources, and the in-flight monitor: needs_replan(), summarize_for_llm()
│   ├── orchestrator/     the closed loop, its MissionLog, the CLI, run files
│   ├── backends.py       ExecutorBackend: all the loop may ask of what flies
│   ├── gazebo/           a run as a Gazebo world, the flight replayed in it
│   ├── api/              the same pipeline over HTTP
│   ├── experiments/      batch runs, ablations, metrics and plots
│   ├── scenario.py       short specs ("lodhi", "sample", "4,0,0") -> inputs
│   └── paths.py          where data, worlds and runs live on disk
├── scripts/              the commands you run
├── data/
│   ├── worlds/           saved environments, one JSON per place
│   ├── plans/            verified plans that fly without an API key
│   └── faults/           fault schedules to replay
├── experiments/          the batch file and how to reproduce each metric; results in runs/ (gitignored)
├── runs/                 every flight flown (gitignored)
└── tests/                pytest suite, one module per component
```

| Package | What to reach for |
| --- | --- |
| `waypoint.world` | `World`, `Terrain`, `Building`; procedural cities and an OpenStreetMap importer |
| `waypoint.planner` | `plan()` / `replan()` against OpenAI, falling back to Gemini; `Waypoint`, `SubTask`, `Plan`, `Action`, `MissionConstraints` |
| `waypoint.verifier` | `verify(plan, constraints) -> VerificationResult` (`passed`, `violations`); the checks live in `verifier/checks.py` |
| `waypoint.sim` | `UAVState`, `Executor`, point-mass `dynamics`, `Wind`, `FaultSchedule`; `execute(plan)` yields a state per 0.1 s tick |
| `waypoint.backends` | `ExecutorBackend` (`prepare`, `run(plan)`, ...) and `make_backend("custom")`; `px4` is reserved |
| `waypoint.feedback` | `needs_replan(history) -> ReplanTrigger | None`, `summarize_for_llm(snapshot, trigger)`, `path_deviation()`; `PerceptionSource` and `SymbolicPerceptionSource` |
| `waypoint.orchestrator` | `run_mission(mission, constraints, faults, backend) -> MissionLog` and the `run_full` CLI |
| `waypoint.gazebo` | `write_scene(run)` and `play(run)`: the 3D view |
| `waypoint.api` | FastAPI over the pipeline: worlds, missions (streamed as they happen), runs |
| `waypoint.experiments` | `run_batch(config)` and `python -m waypoint.experiments run/report`: catch rate, replan success and latency, safety vs optimality, ablations |

Every path in the code comes from `waypoint/paths.py`, so a command behaves the
same whichever directory you run it from. Bundled plans, fault schedules and
worlds can be named rather than spelled out: `--plan lodhi_garden_tour`,
`--faults storm`, `--world gateway_of_india`.

## Configuration

Secrets live in `.env`, which is gitignored. `.env.example` documents the keys:

| Variable | Required | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | one of the two | OpenAI API key; when set, the planner tries OpenAI first ([get one](https://platform.openai.com/api-keys)) |
| `GEMINI_API_KEY` (or `GEMINI_API_KEY1`, `2`, ...) | one of the two | Gemini API key, the fallback, or the only planner when there is no OpenAI key ([get one](https://aistudio.google.com/apikey)) |
| `COST` | no | `HIGH` (default): OpenAI plans with `gpt-5-mini`, then `gpt-4o-mini`, then `gpt-5-nano`. `LOW`: `gpt-5-nano`, then `gpt-4.1-nano` |
| `OPENAI_REASONING_EFFORT` | no | How long `gpt-5`/`o` models think: `minimal`, `low`, `medium`, `high`. Unset, every model uses `low`; set, it applies to every model. Thinking is billed as output |
| `OPENAI_MODEL` | no | Moves one model to the front of the `COST` tier's ladder |
| `OPENAI_MODELS` | no | Replaces the OpenAI ladder outright, comma separated (wins over `COST`) |
| `GEMINI_MODEL` / `GEMINI_MODELS` | no | The same, for the Gemini ladder |

A real environment variable overrides `.env`, so exported keys and CI secrets
win. A blank key is treated as unset, so the placeholder in a fresh `.env`
fails with a clear message rather than a 401.

## Setup

```bash
git clone https://github.com/amanraj069/Waypoint.git && cd Waypoint
uv venv --python 3.11                   # or: python3.11 -m venv .venv
uv pip install -e ".[dev,api]"          # or: .venv/bin/pip install -e ".[dev,api]"
```

That is everything the planner, the verifier and the simulator need. To *watch*
a flight you also want Gazebo (gz-sim, Harmonic or newer):

```bash
brew install gz-jetty                   # macOS; any gz-sim >= Harmonic works
/opt/homebrew/bin/python3.14 -m venv --system-site-packages .venv-gz
.venv-gz/bin/python -m pip install protobuf
```

Gazebo's Python bindings are built for Homebrew's Python rather than the
project's, so the one process that talks to Gazebo runs in that small side
environment; everything else runs in `.venv`. If you only want the CLI,
`".[dev]"` is enough; `api` is just FastAPI and uvicorn, and nothing outside
`waypoint/api/` imports them. An LLM key (OpenAI or Gemini) is only
required to *plan* a new mission; flying and replaying work without one:

```bash
cp .env.example .env                    # then put your OpenAI and/or Gemini key in it
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

### 0. Watch a flight in Gazebo (no API key, ~10 seconds)

One command flies a bundled plan through the simulator and opens Gazebo on it:

```bash
python scripts/simulate.py --plan data/plans/munnar_ridge_run.json \
    --world munnar --gazebo
```

Every run file becomes a Gazebo world: the terrain as a mesh coloured by ground
cover, each building as a named model, the trees, the geofence and no-fly zones
as glass, the planned route as beads coloured by action, and the launch and
landing pads. The aircraft is then flown through it from the run's ticks, with
a chase camera and a magenta trail; obstacles that appear mid-flight are
spawned when the flight reaches them. The terminal prints the flight log as it
plays, and the window stays open when the flight ends, so you can look around.

```bash
# replay the newest run, or any run file, at any speed
python scripts/gazebo_view.py --latest --speed 2
python scripts/gazebo_view.py runs/<mission>_<stamp>.json

# a place on its own, nothing flying
python scripts/gazebo_view.py --world munnar

# no window at all: the flight log only
python scripts/gazebo_view.py --latest --headless
```

The launcher starts the gz server and the GUI as separate processes (on macOS
they cannot share one) plus `waypoint/gazebo/bridge.py`, which is the only code that
speaks gz-transport. The scene is written to
`~/.cache/waypoint/gazebo/<mission>/`, because SDF mesh URIs do not survive the
spaces in a path like `VS Code/`.

The physics you watch is still ours: Gazebo runs with gravity off, and the
aircraft is posed from the simulator's ticks, so the flight in the window is
exactly the flight the verifier passed and the run file recorded. Putting PX4
SITL underneath instead (Gazebo as PX4's physics, MAVSDK uploading the plan)
would be a new `sim` backend; the world export here is the half of that already
done.

The API is a plain HTTP service if you would rather drive the pipeline yourself
(`uvicorn api.main:app`, then `http://127.0.0.1:8000/api/docs` documents itself):

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

### 1. Look, choose the pads, say what to do (needs an LLM key)

The whole thing, from one prompt:

```bash
.venv/bin/python scripts/explore.py --world munnar
```

It prints the place, its size and every named landmark, then waits. Three steps:

```
mission> /look                             1. Gazebo opens on the world, pads marked
                                              (close the window to come back)
mission> /start the Kanan Devan tea estate 2. where it takes off: a landmark, or x,y
  start: the Kanan Devan tea estate (-202.6, -97.4), ground at 43.5 m;
         moved 32 m to the nearest flat, clear ground
mission> /land Echo Point                     and where it finishes (default: the start)
  landing: Echo Point (263.0, 112.9), ground at 67.2 m; moved 96 m ...
mission> /look                                look again: green pad start, amber pad landing
mission> photograph the Tata Tea Museum from above, then fly over Munnar town
         and land at the landing pad       3. the mission, in plain English
```

The mission then runs through the closed loop (see [5b](#5b-the-closed-loop-replan-in-the-air-needs-an-llm-key)):
Gemini plans it from your start pad to your landing pad, the verifier checks it
(and asks for corrections if it fails), the simulator flies it, the monitor
replans in the air if something gets in the way, and Gazebo opens on the
finished flight. Close the window and type the next mission.

Pads are surveyed rather than assumed: a point on a slope, in water or inside a
building moves to the nearest flat, clear ground within 120 m, and it says how
far. A point outside the geofence, or a landmark name that matches nothing (or
more than one thing), is refused with the reason. Everything can be set up
front instead: `--start`, `--land`, `--faults crane_ahead`, `--wind 3,0,0`.

| At the prompt | |
| --- | --- |
| `/look` | Gazebo on the world, pads marked, nothing flying |
| `/landmarks` | the named places, with their coordinates |
| `/start <where>` | launch pad: a landmark name (or part of one), or `x,y` in metres |
| `/land <where>` | landing pad: a landmark, `x,y`, or `start` for back where it began |
| `/pads` | show both |
| *anything else* | a mission, in plain English |
| `/again` | fly the last mission again |
| `/world <name>` | switch places (resets the pads); `/worlds` lists them |
| `/wind x,y,z` | steady wind in m/s |
| `/faults <spec>` | `none`, `crane_ahead`, `storm`, `gps_slip`, or a schedule JSON |
| `/help` `/quit` | |

Coordinates are metres from the world's origin: x east, y north. The world's
extent is printed when it loads.

### 2. Replay a past run

Run files are self-contained, so nothing is re-simulated:

```bash
python scripts/gazebo_view.py --latest              # the newest run
python scripts/gazebo_view.py runs/my-run.json      # a specific one
python scripts/gazebo_view.py --world lodhi         # a place, nothing flying
python scripts/gazebo_view.py --latest --speed 4    # four times as fast
```

**In the window:** drag to orbit, scroll to zoom, and the camera follows the
aircraft unless you pass `--no-follow`. The entity tree lists every building by
name, so you can find "Bara Gumbad" in the world you are looking at. The flight
log prints in the terminal as the flight plays.

### 3. Fly without a window

`--headless` runs the gz server and the flight without the GUI, which is how
the replay is exercised in CI and on a machine with no display:

```bash
python scripts/gazebo_view.py --latest --headless
```

### 4. Fly your own scenario

```bash
# fly through a real place instead of empty space
python scripts/simulate.py --plan data/plans/lodhi_garden_tour.json --world lodhi

# steady 4 m/s easterly with gusty air
python scripts/simulate.py --plan data/plans/north_field_and_tower.json \
    --wind 4,0,0 --turbulence 0.8

# no wind estimation: the aircraft cannot feel the wind and gets blown off course
python scripts/simulate.py --plan data/plans/north_field_and_tower.json \
    --wind 2,0,0 --blind

# a clean reference flight, no faults at all
python scripts/simulate.py --plan data/plans/north_field_and_tower.json --faults none
```

`--faults` takes `sample` (the built-in demo schedule, the default), `none`, or
a path to a schedule JSON; copy `data/faults/storm.json` and edit it. Other
flags: `--dt` for tick length, `--runs-dir`, `--quiet`, `--force` to fly a plan
that failed verification.

For a sense of scale on the same plan: calm and fault-free is 142 s and lands
with 81% battery; `storm.json` costs 12 more points of battery and pushes the
touchdown 20 m off the pad; `--wind 2,0,0 --blind` takes 453 s because the
aircraft keeps overshooting into the wind. Push `--blind` past about 3 m/s and
it never converges; it drifts downwind until the battery empties and lands
wherever it is, which is realistic but writes a very large run file.

### 5. Give it your own mission in English (needs an LLM key)

Hand it a sentence and watch the whole pipeline run: the planner decomposes the
mission into named phases, the verifier checks them, the simulator flies them,
and Gazebo animates the result.

```bash
python scripts/simulate.py \
    "inspect the three wind turbines along the ridge 120 m north of base, \
     photograph each one from two angles, then come home" \
    --save data/plans/wind_turbines.json \
    --gazebo
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

…then the verifier's verdict, then the flight. The same phases label the route
in Gazebo: each waypoint is a bead coloured by its action, so the model's
decomposition and the flight are the same object seen two ways.

`--save` writes the generated plan to a file, so you can re-fly the same mission
as often as you like (different wind, different faults) without spending
another token:

```bash
python scripts/simulate.py --plan data/plans/wind_turbines.json --wind 5,0,0 --gazebo
```

The verifier runs on the model's output either way. If the plan breaks the
envelope the flight is refused and the report says which waypoint and why, which
is the point of having a deterministic gate between the LLM and the aircraft.

Or run the closed loop, which is the next section.

### 5b. The closed loop: replan in the air (needs an LLM key)

```bash
python -m waypoint.orchestrator.run_full \
    --mission "survey the north field: photograph the corners of the field 80 m north of base, then come home and land" \
    --faults crane_ahead --gazebo
```

`crane_ahead` puts a 60 m crane 35 m up the route, wherever the route goes, as
the aircraft heads for its third waypoint. The console follows every phase:

```
==> PLANNING  ...
==> VERIFYING  plan 0
    passed
==> EXECUTING  plan 0
    t=   17.1s  FAULT          crane spotted: 'mobile crane' appears 35 m ahead on the route, radius 8 m
==> REPLAN TRIGGERED  t=17.2s  'mobile crane' appeared at (-21.27, 95), radius 8 m, 60 m tall;
                      the route passes 3.7 m from it, inside its 8 m keep-out
==> REPLANNING  from (-24.6, 64.0, 20.0) with 97.5% battery
      [0] (-30, 95, 20) @ 3 m/s  CAPTURE      <- the corner photo, moved clear of the crane
      [1] (-30, 105, 20) @ 5 m/s  MOVE        <- round the north side of it
      ...
==> VERIFYING  plan 1
    passed
==> RESUMING  plan 1
==> MISSION COMPLETE  landed after 78.7s, 316 m, 90.8% battery, 1 in-flight replan(s)
```

That is a real run. `--gazebo` then opens it: the crane appears at t=17 s, the
new route appears beside the old one at the same moment (cyan; the one it
replaced stays grey), and the trail shows the aircraft follow it round.

What the loop does, and where it is written down:

- **On the ground**, a plan the verifier rejects goes back to the planner with
  the reasons, up to `--max-attempts` times (3).
- **In the air**, every tick (`--check-every`) the perception source reports
  what is around the aircraft, and the feedback monitor looks at the recent
  telemetry. The flight pauses when one of three things is true: something
  was perceived that the plan was not verified against, and the rest of the
  route passes through its keep-out; the battery left will not cover the rest
  of the route plus a reserve (costed at the observed drain rate when the
  battery has lately been draining faster than modelled, as a failing cell
  does); or the aircraft has been more than 15 m off its planned leg for 3 s
  without a break. The planner gets the situation
  (`summarize_for_llm`: where it is, what is done, what is left, why) and plans
  the rest *from where it is*; that plan gets its own verify-and-correct loop,
  against the new obstacle and the battery budget that is left, then flying
  resumes on the same backend.
- **Hard stops**: more than `--max-replans` (3) interruptions, a replacement that
  never passes, or no battery budget left, and it comes down (`failsafe`): where
  it is if the ground below is flat and clear, otherwise on the nearest clear
  ground (the same survey the pads get, keeping clear of obstacles seen in
  flight), crossing above every roof and hill on the way.
  A flat battery or a crash ends the mission too.

Two files land in `runs/`: the run file Gazebo replays (with every plan that
flew, under `plans`), and the mission log, `<run>.log.jsonl`, one decision per
line: `mission_started`, `plan_proposed`, `verification`, `execution_started`,
`perception` (an obstacle seen for the first time), `replan_triggered`, `execution_resumed`, `failsafe`, `mission_finished`. It
records the backend, so runs from different backends can be compared:

```bash
jq -c '{kind, source, passed, trigger, outcome}' runs/*.log.jsonl
```

`--backend custom` is the default; `--backend px4` is reserved for PX4 SITL with
Gazebo as its physics and says so. The loop only ever calls `ExecutorBackend`,
so that backend is a new class in `waypoint/backends.py`, not a change to the
loop. `--plan lodhi_garden_tour` starts from a saved plan instead of the
planner (replans in the air still use it). Exits 0 when the mission completes.
`python -m waypoint.orchestrator.run` is the same command by its older name.

### 6. Exercise one stage at a time

```bash
python scripts/try_planner.py "survey the north field"          # planner only (spends tokens)
python scripts/try_verifier.py --plan data/plans/north_field_and_tower.json \
    --shrink 0.3 --hazards                                      # verifier only, no tokens
```

### 7. Run experiments (needs an LLM key)

```bash
python -m waypoint.experiments run --limit 5                 # a first look
python -m waypoint.experiments run --all-conditions          # 20 missions x 4 ablation conditions
python -m waypoint.experiments report experiments/runs/<stamp>
```

Each batch writes per-mission and per-replan CSVs, a summary, and plots of the
verifier's catch rate, replan latency by fault type, how replans resolved, and
path length against an unconstrained baseline. `--flat` and
`--no-violation-feedback` are the ablations. [experiments/README.md](experiments/README.md)
defines every metric and how to reproduce it.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No run files with telemetry in runs/` | Fly one first: `python scripts/simulate.py --plan data/plans/north_field_and_tower.json` |
| `` `gz` is not on PATH `` | Install Gazebo: `brew install gz-jetty` (or any gz-sim >= Harmonic) |
| `zsh: command not found: python` | macOS has no bare `python`; use `.venv/bin/python`, or `source .venv/bin/activate` first |
| `ModuleNotFoundError: shapely` (or `pydantic`) | You are on a system `python3`; use `.venv/bin/python` |
| `no Python can import Gazebo's bindings` | Make `.venv-gz` as shown in [Setup](#setup), or point `WAYPOINT_GZ_PYTHON` at a Python that can `import gz.transport` |
| Gazebo window opens but the aircraft never moves | A previous run is still holding the world: `pkill -f gz-sim-main`, then try again |
| Plan rejected before flying | That is the verifier doing its job; the report says which waypoint and why. `--force` flies it anyway |

## Planner

`plan(mission_text, constraints)` calls an LLM and is the only path to a `Plan`.
OpenAI is tried first when `OPENAI_API_KEY` is set, then Gemini when its key
is; with only one key, that provider is the planner. Both requests pin the
reply to JSON against the same strict schema (`additionalProperties: false`,
every field required; OpenAI's structured outputs in `strict` mode, Gemini's
`application/json` response format), so the response either decodes and
validates into the `Plan` model or raises:

| Failure | Exception |
| --- | --- |
| Blocked, truncated, empty, non-JSON, or fenced-in-markdown response | `PlannerResponseError` |
| Overload, quota, auth or network failure (after every provider has failed) | `PlannerAPIError` |
| A refusal (OpenAI) | `PlannerResponseError` |
| Tool arguments that don't form a valid `Plan` (bad enum, speed <= 0, empty plan, wrong types) | `PlanValidationError` |

Both subclass `PlannerError`. There is no free-text fallback and no silent
default: a bad plan never reaches the verifier looking like a good one.

`replan(mission_text, previous_plan, violation_report, constraints)` re-prompts
with the rejected plan and the verifier's exact reasons, for the feedback loop.

### Providers and model ladders

Each provider has a ladder of models, and the providers form a chain:

```
openai:  COST=HIGH  gpt-5-mini -> gpt-4o-mini -> gpt-5-nano   (default)
         COST=LOW   gpt-5-nano -> gpt-4.1-nano                (OPENAI_MODEL / OPENAI_MODELS override either)
   then
gemini:  waypoint/planner/config.py:MODEL_LADDER               (GEMINI_MODEL / GEMINI_MODELS)
```

Within a provider, an overloaded, rate-limited, missing (OpenAI 404) or
unreachable model shifts down to the next rung. When a provider fails outright
(every rung gone, or a bad key or bad request, which no cheaper model of the
same provider would fix) the next provider takes over, with a warning on the
console. A reply that is not a valid plan, or an OpenAI refusal, ends the
attempt there: that is a planning failure, and asking someone else would hide
it. Reasoning models (`gpt-5*`, `o*`) are asked for `reasoning_effort: low`
(`OPENAI_REASONING_EFFORT` overrides it). Thinking is billed as output, but
`low` is the only setting that planned well and in time: at `minimal`
gpt-5-nano completed 38% of a 40-flight batch against 80% at `low`, and at
`medium` it timed out on 31% of calls. The
`client=` argument to `plan`/`replan` is a Gemini client and pins the request
to Gemini.

Gemini models go through spells of overload (`500 ... experiencing high demand`)
and free-tier quota limits. Rather than failing the mission, the planner shifts
down `waypoint/planner/config.py:MODEL_LADDER`:

```
gemini-3.7-flash -> gemini-3.6-flash -> gemini-3.5-flash -> gemini-3.5-flash-lite
  -> gemini-3.1-flash-lite -> gemini-3.0-flash -> gemini-2.5-flash -> gemini-2.5-flash-lite
```

`GEMINI_MODEL` in `.env` moves one model to the front without losing the rest.

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
| `MAX_VELOCITY` | A commanded speed over the limit. Speed is along the flown path, as the simulator flies it; a steep climb is flown slower (capped at the climb rate), never faster |
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

In Gazebo each one is a named model, so the entity tree reads like the place
does: find the name there, then type it.

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

Worlds are saved to `data/worlds/<name>.json` and embedded in each run file, so a
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
| Pads | Where the aircraft may stand is surveyed, not assumed: no steeper than 8°, and no more than 0.9 m of rise and fall across its 6 m footprint (`waypoint/world/landing.py`) |
| Telemetry | Every tick carries `ground` and `agl` alongside `z` |
| Viewer | The grid is replaced by the real terrain mesh and extruded buildings |

The interesting case is the one the verifier cannot save you from:

```bash
python scripts/simulate.py --plan data/plans/chandni_chowk_run.json --world chowk \
    --faults data/faults/gps_slip.json
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

`data/worlds/gateway_of_india.json` ships with the project: 118 real footprints from
OpenStreetMap around Colaba, Mumbai, 80 of them carrying the name their mapper
gave them: the Taj Mahal Palace, the Royal Bombay Yacht Club, Regal Cinema,
Dhanraj Mahal, the Gateway of India itself. Build another anywhere:

```bash
python scripts/make_world.py osm --lat 28.6129 --lon 77.2295 --radius 400 --name india_gate
.venv/bin/python scripts/explore.py --world data/worlds/india_gate.json
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

## The 3D view

Gazebo draws every run, and `waypoint/gazebo/scene.py` is what turns a run file into
something it can open. Two files come out, into
`~/.cache/waypoint/gazebo/<mission>/`:

`world.sdf` holds everything that is not ground: each building as its own named
model (so the entity tree reads like the place does), the trees, the geofence
and no-fly zones as glass, the planned route as beads coloured by action, the
launch and landing pads, and the aircraft. Nothing in it is a physics body
except the aircraft, and the world has no gravity.

`terrain.obj` is the ground, written as a mesh rather than an SDF heightmap
because a heightmap cannot carry ground cover and wants a power-of-two image.
The mesh is sampled on a 4 m grid (capped at 320 cells a side, so a city still
loads in a second or two), each cell coloured by what covers it: tea terraces,
lawn, water, road, forest. Normals come from the height field's own slope, so
hills are lit like hills. Cells are written as quads, which Gazebo triangulates
on load: the same mesh from half as many lines of OBJ.

`waypoint/gazebo/bridge.py` is the only code that speaks gz-transport, and it runs under
Homebrew's Python because that is what Gazebo's bindings are built for. The
launcher feeds it one JSON command per line over a pipe: a pose 30 times a
second (interpolated between the simulator's 10 Hz ticks so the aircraft moves
smoothly), new trail points as the flight lays them down, and an obstacle to
spawn when the flight reaches one. The bridge keeps the trail itself and
redraws it on a timer that eases off as it grows, because a marker is always
redrawn from all of its points.

Run files are self-contained (`runs/<mission>_<stamp>.json`, schema 2), so a
past flight replays without re-simulating.

## Test

```bash
.venv/bin/python -m pytest          # 389 tests, about 8 seconds
```

Every test mocks the planner's clients; autouse fixtures replace the real
Gemini and OpenAI constructors with tripwires and clear `OPENAI_API_KEY` and the
model overrides from the environment, so your `.env` cannot steer the suite, so a test that forgets to inject a fake client
fails rather than dialling out. `tests/test_api.py` flies whole missions through
the HTTP API using the bundled plans, so the pipeline, the event stream and the
run files are covered without a key and without a network.

## Where the real pieces go

- `waypoint/planner/`: done. Tune `waypoint/planner/prompts.py` (system prompt + example) rather
  than the call site.
- `waypoint/verifier/`: done. Tune the aircraft model in `VerifierConfig` rather than
  the checks; add a new check as a function in `waypoint/verifier/checks.py` plus a
  `ViolationType`.
- `waypoint/sim/`: done. Tune the airframe in `SimConfig` and the disturbances in
  `Wind` / `FaultSchedule`. `CustomSimBackend` (`sim/backend.py`) is its
  `ExecutorBackend`; `Executor.resume(plan)` is what lets a new plan take over
  mid-flight.
- `waypoint/feedback/`: a new reason to replan is a function in `monitor.py`
  returning a `ReplanTrigger`, plus a `TriggerKind`; `needs_replan` tries them in
  order (obstacle, battery, off route). Its thresholds live on `FeedbackConfig`.
  How the aircraft sees is a `PerceptionSource` (`perception.py`):
  `SymbolicPerceptionSource` reads the backend's obstacles, optionally within a
  `sensor_range`, and `run_mission(perception=...)` takes any other, such as a
  detector on a Gazebo camera feed. GPS degradation is the obvious next trigger.
- `waypoint/orchestrator/loop.py`: the closed loop. It touches the aircraft only
  through `ExecutorBackend`, and every decision goes through `MissionLog.add`.
  `scripts/explore.py` flies every mission through it; `waypoint/api/missions.py`
  still runs its own (ground-only) replan loop, and moving it onto
  `run_mission` is the next step.
- `waypoint/api/`: the routes are thin. New capability belongs in the packages they call,
  not here; `waypoint/api/missions.py` is the only file with logic of its own, and that
  logic is the stage-by-stage reporting.
- `waypoint/gazebo/`: `scene.py` writes the world, `launch.py` runs the three processes
  and flies the ticks, `bridge.py` is the only thing that talks to Gazebo. A
  PX4/Gazebo backend would be a new `ExecutorBackend`, not a change here: the
  world export is the half of that job already done.
