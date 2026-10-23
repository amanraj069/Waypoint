# waypoint

**Safety-Aware Hierarchical LLM for Autonomous UAV Mission Planning and
Adaptation**. Say a mission in plain English, watch a language model decompose
it, a deterministic verifier check it, and a simulator fly it through a real
place, replanning in the air when something gets in the way. The planner, the
verifier, the simulator and the in-flight feedback are all real, and the same
loop can fly the real PX4 autopilot in Gazebo instead of the simulator.

```
natural language mission
        -> world      (terrain + buildings)        [REAL: four Indian places, or OpenStreetMap]
        -> intent     (what the mission asks for)  [REAL: LLM reads it under a strict schema, the world grounds it;
                                                    a mission that cannot be flown as asked is refused here]
        -> planner    (LLM hierarchical planner)   [REAL: OpenAI, then Gemini; strict JSON schema]
        -> verifier   (deterministic checks)       [REAL: exact geometry, no model in the loop: safety AND intent]
        -> repair     (deterministic fixes)        [REAL: detours, climbs, clamps, added viewpoints; LLM only if this fails]
        -> sim        (UAV executor)               [REAL: point-mass physics, wind, faults, collisions;
                                                    or PX4 SITL flying an x500 in Gazebo, --backend px4]
        -> shield     (runtime safety filter)      [REAL: control barrier functions on every velocity command]
        -> feedback   (perception + monitor)       [REAL: new obstacle on the route, battery short or failing, off route;
                                                    symbolic, or a depth camera on PX4]
        -> replan     (tiered, from where it is)   [REAL: repair in milliseconds first, the LLM second, hovering meanwhile]
```

Each capability after the planner is a switch (`--system`, section
[The system](#the-system-what-runs-around-the-llm)); `full` is the default
everywhere, and `classic` is the loop as first published, for comparison.

There are two ways in. **The prompt** opens a place, takes a mission in plain
English, and flies it:

```bash
.venv/bin/python scripts/explore.py --world lodhi
```

**The CLI** does each stage on its own, and every stage is also importable.
Both drive the same pipeline; neither is a wrapper around the other. You watch
the result in **Gazebo**, which is the only 3D view: every run becomes an SDF
world and the flight is replayed in it. With `--backend px4` the flight is not
replayed but flown live there, by PX4 (section 5c).

## Layout

```
waypoint/                 the repository
├── waypoint/             all the code, one importable package
│   ├── world/            where it flies: terrain, buildings, landmarks
│   ├── intent/           what a mission asks for: extraction, grounding, the INTENT check
│   ├── planner/          English -> hierarchical Plan, via OpenAI or Gemini
│   ├── verifier/         deterministic safety check of a Plan, and deterministic repair
│   ├── sim/              point-mass physics, wind, fault injection (gazebo_backend.py: the PX4 backend's spec name)
│   ├── feedback/         perception sources, and the in-flight monitor: needs_replan(), summarize_for_llm()
│   ├── orchestrator/     the closed loop, its MissionLog, the CLI, run files
│   ├── backends.py       ExecutorBackend: all the loop may ask of what flies
│   ├── gazebo/           a run as a Gazebo world, the flight replayed in it
│   ├── px4/              PX4 SITL in Gazebo as a flight backend (--backend px4)
│   ├── api/              the same pipeline over HTTP
│   ├── experiments/      batch runs, ablations, metrics and plots
│   ├── scenario.py       short specs ("lodhi", "sample", "4,0,0") -> inputs
│   └── paths.py          where data, worlds and runs live on disk
├── scripts/              the commands you run; setup_px4.sh and px4_check.py for the PX4 backend
│   └── px4-patches/      three small build fixes PX4 v1.17 needs against current Homebrew
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
| `waypoint.intent` | `extract_intent(mission, world)`, `check_intent(plan, intent)`, `assess(intent, ...)`, `evaluate(...)`; `MissionIntent`, `Target` |
| `waypoint.verifier` | `verify(plan, constraints, intent=...) -> VerificationResult` (`passed`, `violations`); the checks live in `verifier/checks.py`, deterministic repair in `verifier/repair.py` |
| `waypoint.sim` | `UAVState`, `Executor`, point-mass `dynamics`, `Wind`, `FaultSchedule`; `execute(plan)` yields a state per 0.1 s tick; `SafetyShield`, the runtime filter |
| `waypoint.backends` | `ExecutorBackend` (`prepare`, `run(plan)`, `close()`, ...) and `make_backend("custom" \| "px4")` |
| `waypoint.px4` | `Px4Backend` (also `waypoint.sim.gazebo_backend.Px4GazeboBackend`): PX4's x500 flown in offboard mode through Gazebo; `Sitl` starts and stops PX4 and Gazebo; `Link` is the MAVLink side |
| `waypoint.feedback` | `needs_replan(history) -> ReplanTrigger | None`, `summarize_for_llm(snapshot, trigger)`, `path_deviation()`; `PerceptionSource`, `SymbolicPerceptionSource`, and `GazeboCameraPerceptionSource` (obstacles from the PX4 depth camera) |
| `waypoint.orchestrator` | `run_mission(mission, constraints, faults, backend, system=...) -> MissionLog`, `SystemOptions` (`FULL`, `CLASSIC`, ...) and the `run_full` CLI |
| `waypoint.gazebo` | `write_scene(run)` and `play(run)`: the 3D view |
| `waypoint.api` | FastAPI over the pipeline: worlds, missions (streamed as they happen), runs |
| `waypoint.experiments` | `run_batch(config)` and `python -m waypoint.experiments run/report`: catch rate, mission success against hand-written ground truth, replan success and latency, safety vs optimality, systems and ablations |

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
| `WAYPOINT_PX4_DIR`, `_GUI`, `_SPEED`, `_HOLD`, `_FAULTS`, `_MODEL` | no | Only for `--backend px4`: where PX4 is, whether to open the window, speed factor, what happens while replanning, how faults reach PX4, which airframe. See section 5c |

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
environment; everything else runs in `.venv`. To fly missions on the real PX4
autopilot as well, `scripts/setup_px4.sh` builds PX4 SITL against this Gazebo
(a large clone and a full build the first time; section 5c). If you only want the CLI,
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
exactly the flight the verifier passed and the run file recorded. To fly the
mission on the real autopilot instead, with Gazebo as PX4's physics, use
`--backend px4` (section 5c).

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

`--backend custom` is the default; `--backend px4` flies the same loop on PX4
SITL in Gazebo (section 5c). The loop only ever calls `ExecutorBackend`, so the
two are interchangeable. `--plan lodhi_garden_tour` starts from a saved plan instead of the
planner (replans in the air still use it). Exits 0 when the mission completes.
`python -m waypoint.orchestrator.run` is the same command by its older name.

### 5c. Fly it on PX4 SITL in Gazebo

`--backend px4` replaces the point-mass simulator with the real autopilot: PX4
flies its x500 quadrotor through Gazebo's physics, in a Gazebo world written
from the same `World` and `MissionConstraints` the planner and verifier used,
and the closed loop replans it in the air exactly as before
(`orchestrator/loop.py` does not know which backend it is driving). The run
file, the mission log and the experiment metrics are the same, so the two
backends compare like for like.

#### Setup (macOS on Apple silicon)

Gazebo first, if it is not there: `brew tap osrf/simulation && brew install
gz-jetty` (or `gz-harmonic`; PX4 v1.17 supports both). Then:

```bash
scripts/setup_px4.sh            # clone PX4 v1.17.0, patch, build SITL, install pymavlink here
scripts/setup_px4.sh --check    # just report what is there and what is missing
python scripts/px4_check.py     # the end-to-end check below, about 3 minutes
```

The script is safe to run again: every step checks first. It does, in order:

1. Checks that `gz`, `cmake`, `ninja` and `ccache` are installed, and installs
   the last three with Homebrew if not.
2. `git clone --recursive --branch v1.17.0
   https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot`
   (`WAYPOINT_PX4_DIR` and `PX4_VERSION` change both).
3. Makes PX4's Python tools in a venv inside the checkout.
4. Applies `scripts/px4-patches/*.patch`.
5. Runs `make px4_sitl`.
6. Runs `pip install -e ".[px4]"` here (pymavlink).

Why not PX4's own `Tools/setup/macos.sh`:
- it installs the ARM cross-compilers for flight-controller hardware, which
  SITL does not need;
- its `pip install --user` fails on Homebrew's Python (PEP 668).

It still works if you prefer it; run the patch and build steps after it.

The three patches exist because PX4 v1.17 does not build unmodified against
current Homebrew. Each one is a few lines and says why:
- `01` compiles PX4's generated Gazebo messages as C++17, which protobuf's
  abseil needs, without `-Werror`.
- `02` builds the gz bridge without `-Werror`.
- `03` builds the gz plugins without `-Werror`, and skips the optical-flow
  plugin unless an OpenCV older than 5 is installed (it uses OpenCV's removed
  C API).

Nothing here uses optical flow, so OpenCV is not needed at all.

To check PX4 on its own before involving waypoint, its baseline target flies
the x500 in PX4's default world, with PX4's interactive shell:

```bash
cd ~/PX4-Autopilot && PATH="$PWD/.venv/bin:$PATH" make px4_sitl gz_x500
pxh> commander takeoff        # needs NAV_DLL_ACT 0 and COM_RC_IN_MODE 4 without a ground station
```

#### If the build fails

- **Gazebo version detection.** PX4's Gazebo integration is version-sensitive.
  If the build stops there, compare the Gazebo versions PX4 knows about with
  what is installed:
  - PX4 side: `Tools/simulation/gz/README.md` and the `find_package(gz-...)`
    lines in `src/modules/simulation/gz_bridge/CMakeLists.txt`;
  - installed side: `gz sim --versions`.

  PX4 v1.17 accepts Harmonic (gz-sim 8) and newer.
- **Gazebo itself stops working after a `brew install` or `brew upgrade`**
  (for example `Library not loaded: libprotobuf.<version>.dylib`). Something
  upgraded protobuf ahead of the Gazebo bottles. Installing OpenCV does exactly
  this. Upgrade every Gazebo formula to the bottles rebuilt for the new
  protobuf:

  ```bash
  brew upgrade $(brew list --formula | grep -E '^(gz-|sdformat)')
  ```

  If Homebrew then refuses with "conflicting formulae" for `ogre2.3`, the
  rebuilt bottles want `ogre2.3-with-freeimage` instead:

  ```bash
  brew uninstall --ignore-dependencies osrf/simulation/ogre2.3
  ```

  Then run the upgrade again, and rebuild PX4 with `scripts/setup_px4.sh`.

#### Run it

```bash
# the crane scenario on the real autopilot, headless
python -m waypoint.orchestrator.run_full --backend px4 --faults crane_ahead \
    --mission "survey the north field: photograph the corners of the field 80 m north of base, then come home and land"

# watch it: the window follows the aircraft, and stays open after landing until you close it
WAYPOINT_PX4_GUI=1 python -m waypoint.orchestrator.run_full --backend px4 --world lodhi \
    --mission "take a round of the garden, photograph Bara Gumbad, and come home"

# a batch, faster than real time
WAYPOINT_PX4_SPEED=4 python -m waypoint.experiments run --backend px4 --limit 5
```

`from waypoint.sim.gazebo_backend import Px4GazeboBackend` is the same
backend, by the name the Stage 8 spec gives it; `make_backend("px4")` is the
usual way in.

| Setting | Default | Meaning |
| --- | --- | --- |
| `WAYPOINT_PX4_DIR` | `~/PX4-Autopilot` | The checkout with `make px4_sitl` built |
| `WAYPOINT_PX4_GUI` | off | `1` opens the Gazebo window, following the aircraft |
| `WAYPOINT_PX4_SPEED` | `1` | Simulation speed factor (PX4 runs in lockstep with Gazebo) |
| `WAYPOINT_PX4_HOLD` | `freeze` | While the planner thinks: `freeze` pauses Gazebo (and so PX4's clock), `hover` hovers in real time |
| `WAYPOINT_PX4_FAULTS` | `emulate` | `emulate` imitates every fault, comparably with the custom simulator; `px4` injects into PX4 what PX4 accepts (below) |
| `WAYPOINT_PX4_MODEL` | `x500` | `x500_depth` carries an OAK-D Lite (RGB and depth), `x500_mono_cam` one camera |

#### The world PX4 flies in

The backend does not drop a file into PX4's own worlds directory. For every
flight it:

1. writes the world with `gazebo.scene.write_scene(..., px4=True)` (the same
   writer as the replay) into
   `~/.cache/waypoint/px4/<run>/scene/world.sdf`. A path without spaces, because
   SDF URIs do not survive the space in `VS Code/`.
2. starts `gz sim -s -r` on it, plus `gz sim -g` for the window.
3. starts PX4 in standalone mode, which attaches to a running Gazebo rather than
   launching its own:
   - `PX4_GZ_STANDALONE=1`
   - `PX4_GZ_WORLD=waypoint` (the world's name)
   - `PX4_SIM_MODEL=gz_x500`
   - `PX4_GZ_MODEL_POSE=<launch pad>`
   - `PX4_GZ_MODELS=<checkout>/Tools/simulation/gz/models`

   PX4 runs as `bin/px4 -d build/px4_sitl_default/etc -w <fresh dir>`, so no
   parameters carry over between flights.

The world holds the terrain, buildings and trees, plus everything the verifier
reasoned about: the geofence walls, the no-fly volumes, the planned obstacles,
the launch pad, and the landing pad once a plan names it. Each plan's route is
drawn as it takes over, and each obstacle that appears in flight is drawn when
it appears. Only the ground has collision: the fence, the zones and the
obstacles are glass, seen and never hit, because keeping out of them is the
verifier's and the monitor's job.

Only one PX4 flight can run at a time, because PX4's MAVLink port and the
world's name are fixed. Each flight takes about 20 s to start; a start that
stalls (PX4 now and then does) is retried up to three times. If a flight is
killed hard and leaves Gazebo behind, `pkill -f gz-sim-main` clears it; the
backend refuses to start next to a stray `waypoint` world rather than fly in
the wrong one.

#### How it maps onto the simulator, so the two compare

* **Control.** PX4 is flown in offboard mode, over MAVLink (pymavlink), leg by
  leg. It uses velocity setpoints toward the next waypoint at the waypoint's
  commanded speed, slowing to arrive, and steers on the position the aircraft
  believes it has. The last 3 m of a landing are PX4's own landing mode.
  (Uploading the plan as a PX4 mission instead would lose waypoint-by-waypoint
  progress, which the `at_waypoint` fault triggers and the monitor need, and
  the paused clock while the planner replans.)
* **Telemetry.** PX4's local position, relative to where it stood on the pad,
  as one `UAVState` per 0.1 s of simulated time. Positions are interpolated
  between PX4's own samples, which arrive every 16 or 24 ms.
* **Battery.** The simulator's model (metres and seconds per percent, plus
  drain spikes), not PX4's simulated battery. Both backends then spend the
  budget the verifier checked.
* **Crashes.** Judged against the `World` the same way. Buildings are drawn but
  are not physics bodies, so the backend detects an impact and cuts the motors.
  If one of PX4's own failsafes takes the aircraft (it leaves offboard by
  itself), the backend records `PX4_FAILSAFE`, stops steering, and lets PX4
  land it.
* **Faults (`emulate`, the default).** One schedule drives both backends:
  - an obstacle joins the obstacle list and is drawn in Gazebo;
  - a gust is a drag force on the airframe, which PX4 has to fight;
  - a battery spike drains the shared battery model;
  - GPS drift offsets the position the backend steers on, so the aircraft flies
    off its line believing it is on it, as in the simulator.
* **Faults (`px4`).** Each fault is first offered to PX4 through
  `MAV_CMD_INJECT_FAILURE` (this sets `SYS_FAILURE_EN=1`):
  - **Battery:** PX4 v1.17 with Gazebo accepts the battery failure. PX4's own
    battery then collapses and its failsafe lands the aircraft; the backend
    reads that battery and reports `battery_depleted`.
  - **GPS:** PX4 does not answer the GPS failure. It is handled only by PX4's
    older simulator bridge, not the Gazebo one, so `failure gps off` does
    nothing here, and the drift is imitated instead.
  - **Gusts and obstacles:** PX4 has no injection for them, so they are always
    imitated.

  Every fault's event in the run file says how it was delivered. Runs in this
  mode test PX4's failsafes, and are not meant to compare with the custom
  backend.
* **Replanning.** The aircraft holds position and, by default, Gazebo is
  paused, so the planner's thinking costs no flight time, as in the simulator.

#### The end-to-end check

`scripts/px4_check.py` flies one scenario through the unmodified loop with
scripted planners (no API key). The scenario has a fence, a no-fly zone, a
planned water tower, and a crane that appears 40 m up the first leg. It prints
PASS or FAIL for each step:

1. the PX4 world shows the fence, the zone and the tower, as glass;
2. PX4 and Gazebo boot;
3. the plan is flown and lands;
4. telemetry streams back, one state per 0.1 s, inside the fence;
5. the crane triggers a replan, the flight resumes on the detour, and it keeps
   clear;
6. the custom simulator verifies and replans the same way, with a similar
   flight.

`--gui` to watch, `--speed N`, `--keep` to keep the world and PX4's logs.
`WAYPOINT_PX4_TESTS=1 pytest tests/test_px4.py` also flies real PX4.

#### Cameras, for perception that is not symbolic

With `WAYPOINT_PX4_MODEL=x500_depth`, Gazebo publishes the OAK-D Lite's frames,
headless or with the window:
- RGB, 1920 x 1080, on `.../camera_link/sensor/IMX214/image`;
- depth, 640 x 480, on `/depth_camera`.

`waypoint.feedback.GazeboCameraPerceptionSource` turns the depth frames into
obstacles, so the loop can see what no map has:

```python
from waypoint.feedback.gazebo_camera import GazeboCameraPerceptionSource
backend = Px4Backend(model="x500_depth")
camera = GazeboCameraPerceptionSource.on(backend)      # starts with the flight
run_mission(mission, constraints, faults, backend, perception=camera, system=FULL)
```

- `waypoint/gazebo/depth_bridge.py` (gz-only, under `.venv-gz`) writes the newest
  depth frame and the airframe's full pose (attitude included: it tilts as it
  accelerates) to one file, replaced atomically.
- `waypoint/feedback/depth.py` back-projects it into the world, takes away
  everything the map explains (ground, buildings, known obstacles, the drawn
  fence and no-fly walls), clusters what is left on a 1 m grid and fits a
  circle to each cluster's visible arc. Detection is geometric, not learned:
  anything solid the map lacks is an obstacle, whatever it looks like.
- The source confirms a cluster over two frames, then reports it once, frozen;
  a clearly bigger later estimate is reported as well.
- Worlds flown with a camera airframe draw no route beads or pad beams (the
  camera would see them) and show obstacles that appear as solid.

The depth sensor reaches about 19 m, which is why the in-flight repair
(milliseconds) matters with it. The end-to-end check, with the crane known to
the camera alone and the shield off, so nothing else covers for it:

```bash
python scripts/camera_check.py            # about 3 minutes; --gui to watch
```

On this machine: the crane was seen 23 m from its centre, placed 0.0 m from
where it stands (radius 6.5 m against 6.0 m), and flown round by repair with no
keep-out breach.

### 6. Exercise one stage at a time

```bash
python scripts/try_planner.py "survey the north field"          # planner only (spends tokens)
python scripts/try_verifier.py --plan data/plans/north_field_and_tower.json \
    --shrink 0.3 --hazards                                      # verifier only, no tokens
```

### 7. Run experiments (needs an LLM key)

```bash
python -m waypoint.experiments run --limit 5                 # a first look, the full system
python -m waypoint.experiments run --systems "full;classic;runtime-guard" --repeats 2
python -m waypoint.experiments run --all-conditions          # 36 missions x 4 planner ablations
python -m waypoint.experiments report experiments/runs/<stamp>
python -m waypoint.experiments report experiments/runs/<old> --missions experiments/missions.json
```

Every mission of the batch carries hand-written ground truth (`expect`) that no
planner sees; flights are scored on it from their run files, so "mission
accomplished" means the same for every system, and batches flown before the
ground truth existed can be re-scored (the last line).

Each batch writes per-mission and per-replan CSVs, a summary, and plots of the
verifier's catch rate, replan latency by fault type, how replans resolved, and
path length against an unconstrained baseline. `--flat` and
`--no-violation-feedback` are the ablations. `--backend px4` flies the same batch
on PX4 SITL in Gazebo (one flight at a time, about 20 s to start each;
`WAYPOINT_PX4_SPEED=4` runs it faster than real time).
[experiments/README.md](experiments/README.md) defines every metric and how to
reproduce it.

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
| `no PX4 SITL build at ...` | `scripts/setup_px4.sh`, or point `WAYPOINT_PX4_DIR` at your PX4-Autopilot checkout |
| `the px4 backend needs pymavlink` | `pip install -e ".[px4]"` (the setup script does this) |
| `another Gazebo is already serving a world called 'waypoint'` | A replay or an earlier flight is still running: close it, or `pkill -f gz-sim-main` |
| `PX4's startup script did not come up ... (gave up after 3 attempts)` | `pkill -f gz-sim-main; pkill -f bin/px4`, then retry; `python scripts/px4_check.py --keep` keeps PX4's log to read |
| `Library not loaded: libprotobuf...` from any `gz` command | Homebrew upgraded protobuf ahead of Gazebo: see "If the build fails" in section 5c |

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
| `BUILDING` | A waypoint inside a building or less than `roof_clearance` above its roof, or a leg that crosses one lower than its roof plus `roof_clearance` (needs a [world](#the-world)) |
| `TERRAIN` | A waypoint underground or below the ground-clearance minimum, or a leg that flies into a hillside |
| `LANDING` | A LAND waypoint on ground too uneven to set down on, within 2 m of anything standing, or not at ground level |
| `INTENT` | Only with an `intent`: the plan does not do what the mission asked (a target not photographed, too few sides, the wrong order, an avoid zone crossed, not ending on the pad); see [Intent](#intent-what-the-mission-asks-for) |

Three things make the checks exact rather than approximate:

- **Terrain is not sampled.** Bilinear ground is a quadratic along a straight
  leg inside each grid cell, so the minimum clearance is solved for, cell by
  cell. No ridge can hide between samples (`tests/test_soundness.py` checks it
  against a 25 cm walk).
- **Legs are judged as flown.** A leg that climbs faster than the airframe can
  (`max_climb_rate`, 4 m/s) is flown below its straight line, so the building
  and terrain checks use the climb-limited profile. Re-flying every verified
  plan from the published batches in calm air found two plans that passed the
  old checks and crashed; both are refused now.
- **Room to stop.** A waypoint over a roof needs `roof_clearance` above it: an
  aircraft arriving there sinks past it before it holds.

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
| `min_ground_clearance` | 5 | Metres every leg stays above the ground |
| `roof_clearance` | 3 | Metres over a roof a waypoint sits, and a leg crosses |
| `max_climb_rate` | 4 | Vertical speed cap; steeper legs are judged on the profile actually flown |
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


## The system: what runs around the LLM

The LLM decomposes the mission and writes the route. Everything around it is
deterministic, and each piece is a switch on `SystemOptions`
(`waypoint/orchestrator/options.py`), so it can be turned off and measured:

| Switch | `full` | `classic` | What it does |
| --- | --- | --- | --- |
| `intent` | on | off | Read the mission into checkable requirements; refuse what cannot be flown; check every plan against what is left |
| `repair` | on | off | Fix a rejected plan geometrically before asking the LLM again |
| `replan` | `tiered` | `llm` | In the air, repair (or trim for battery) first; the LLM only if that fails |
| `hold` | `hover` | `freeze` | While the LLM thinks in the air, the aircraft hovers and burns battery; `freeze` stops the clock |
| `shield` | on | off | The runtime safety filter under the controller |
| `monitor_energy` | on | off | Verify in-flight plans with the monitor's energy model, so a new plan does not trip it again |
| `subtask_feedback` | on | off | Tell the planner which subtask each violation is in, and which to keep |
| `gate` | `preflight` | `preflight` | `runtime`: no preflight check, only a per-command guard; `none`: nothing checks anything |

Presets: `full` (every entry point's default), `classic` (the loop as first
published), `runtime-guard` (how runtime-only LLM agents on autopilots enforce
safety: altitude clamped, a command outside the fence refused at dispatch, then
replan), `no-gate`. Switches compose: `--system full,no-shield`,
`--system full,replan=llm`. `run_mission` defaults to `classic` so library
code keeps its behaviour; pass `system=FULL`.

### Intent: what the mission asks for

`waypoint.intent` makes "does the plan do what was asked?" as deterministic as
"is it safe?".

1. **Extract.** The LLM is asked one narrow question under a strict schema
   (`intent/extract.py`): which targets to photograph, hover at or fly over;
   from how many sides, which sides, from above, at what height; in what order;
   what never to fly over. It names landmarks by copying them from the list it
   is given. `WAYPOINT_INTENT_VOTES=3` asks three times and keeps the reading
   most agree on.
2. **Ground.** A landmark's position, size and height come from the world,
   never from the model, so a hallucinated coordinate cannot leak in. A hill is
   the ground itself, so its slopes are the subject of its photograph, not in
   the way of it.
3. **Assess.** Before planning, `assess()` asks whether the mission can be
   flown at all: a target 1300 m outside the fence, a photograph 200 m up under
   a 120 m ceiling, a route the battery cannot cover even at its shortest. Such
   a mission is refused with the reason (`outcome: infeasible`) instead of
   being flown wrong.
4. **Check.** `check_intent()` runs with every verification. A photograph
   counts from a CAPTURE waypoint within reach of the target (in proportion to
   its size: 20 m for a field corner, about 54 m for a 36 m tomb) with a clear
   line of sight (buildings and terrain in between block it; the target's own
   buildings do not). Views count as distinct sides at least 60 degrees apart
   around it, or, for a large target such as a dam, well apart along it.
   Order is a greedy chain over every waypoint that serves each target.
   Every miss names a fix that is itself flyable: a viewpoint inside the
   envelope with a clear line of sight.

```
[INTENT] The mission asks to photograph 'Bara Gumbad' from 2 distinct sides, but the plan
sees it from only 1 (south); views count as distinct when at least 60 deg apart around it.
Add a CAPTURE from another side, e.g. at (5, 248, 20.7).
```

In the air, the intent shrinks to what is left (`remaining()`, from the
waypoints actually reached). If no plan can both serve everything left and stay
safe, the mission gives way and safety never does: the plan is accepted with
its INTENT findings as warnings, and `intent_relaxed` records what was given up.

### Repair: fix it without asking again

`waypoint.verifier.repair` applies the remedies the violations name: clamp a
speed or an altitude, climb in place before a leg steeper than the airframe can
fly, lift a low leg over the high ground, take a leg round a building, a no-fly
zone or an obstacle by the shortest path on a visibility graph inside the fence
(or over the top, whichever is shorter), pull a waypoint back inside the fence,
put a LAND on usable ground, add the viewpoint the intent check proposed at
the cheapest place in the route. Every round re-runs the full verifier; a plan
comes back only if it passes, so repair can never relax anything.

```bash
python scripts/replay_repairs.py experiments/runs/<stamp>    # every rejected plan of a batch, through repair
```

On the 264 plans the verifier rejected in the published batches, repair
produced a passing plan for 97%, in a median of 5 ms, adding a median 3% to
the route. The LLM's own corrections fixed 75%, in seconds per attempt.

### Replanning in the air

When the monitor fires, the `tiered` policy tries, in order:

1. **repair** the rest of the route from where the aircraft is, against what
   changed (a new obstacle, an off-route position): milliseconds;
2. for a **battery** shortfall, **trim**: give up the phase that saves the
   most distance, then the next, until what is left is flyable; at worst, go
   straight to the landing pad. The observed drain trend is trusted up to 1.5x
   the first time (a passing blip should not cost the mission) and in full when
   the monitor fires again;
3. the **LLM**, with the situation, the intent still to do, and its verify,
   repair and correct loop. With `hold: hover` the aircraft hovers for as long
   as that takes, wind and faults still acting.

### The shield: safety at run time

`waypoint.sim.shield.SafetyShield` filters every velocity command, in the
custom simulator and on PX4 alike. Each hazard near the aircraft is a barrier
function (obstacle keep-outs and tops, buildings and roofs, no-fly zones, the
geofence, the ground, the ceiling), and the command is projected onto the set
that satisfies all of them (Dykstra's algorithm): far from everything nothing
changes, near a boundary the aircraft slides along it. It is braking-aware
(it starts slowing in time to stop), cancels wind the autopilot does not
(a disturbance observer), and compares the GPS fix with the airframe's own
measured motion, so a drifting fix cannot steer it into something.

```bash
python scripts/shield_trial.py experiments/runs/<stamp>          # re-fly a batch's plans, shield off and on
python scripts/shield_trial.py --calm experiments/runs/<stamp>   # no wind, no faults: the verifier's false negatives
```

Re-flying the 169 verified plans of the published batches under their own
faults, with nothing replanning: crashes 4.1% without the shield, 0.6% with
it; keep-out breaches 29.0% and 0.6%. The plans as the LLM first wrote them,
unverified: crashes 25.5% and 1.5%.


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
small custom simulator: pure Python, no binary dependencies, fast enough to fly
a mission in well under a second, and identical on macOS, Linux and Windows.
It is the default backend; PX4 SITL in Gazebo is the other (section 5c), and
it follows the same battery model, fault schedule and crash rules so the two
compare.

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

`write_scene(run, out_dir, px4=True)` writes the same place for PX4 to fly in:
gravity, the sensor systems PX4's airframe reads (IMU, barometer,
magnetometer, GPS), a wind force system, ground the aircraft can stand on, and
a place on Earth for PX4's GPS. The aircraft is left out, because PX4 spawns
its own x500.

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
.venv/bin/python -m pytest          # 434 tests, about 10 seconds
WAYPOINT_PX4_TESTS=1 .venv/bin/python -m pytest tests/test_px4.py   # also fly real PX4 (about 3 minutes)
```

Every test mocks the planner's clients; autouse fixtures replace the real
Gemini and OpenAI constructors with tripwires and clear `OPENAI_API_KEY` and the
model overrides from the environment, so your `.env` cannot steer the suite, so a test that forgets to inject a fake client
fails rather than dialling out. `tests/test_api.py` flies whole missions through
the HTTP API using the bundled plans, so the pipeline, the event stream and the
run files are covered without a key and without a network. `tests/test_px4.py`
flies the PX4 backend against a fake PX4, so the suite needs neither PX4 nor
Gazebo. With `WAYPOINT_PX4_TESTS=1` it also flies the real thing: a square,
and the crane replan.

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
  `sensor_range`, and `run_mission(perception=...)` takes any other.
  `GazeboCameraPerceptionSource` (`gazebo_camera.py`) is the stub for one that
  detects obstacles in the x500_depth's camera frames; its docstring is the
  design. GPS degradation is the obvious next trigger.
- `waypoint/orchestrator/loop.py`: the closed loop. It touches the aircraft only
  through `ExecutorBackend`, and every decision goes through `MissionLog.add`.
  `scripts/explore.py` flies every mission through it; `waypoint/api/missions.py`
  still runs its own (ground-only) replan loop, and moving it onto
  `run_mission` is the next step.
- `waypoint/api/`: the routes are thin. New capability belongs in the packages they call,
  not here; `waypoint/api/missions.py` is the only file with logic of its own, and that
  logic is the stage-by-stage reporting.
- `waypoint/gazebo/`: `scene.py` writes the world (for a replay, or with
  `px4=True` for PX4 to fly in), `launch.py` runs the three processes and flies
  the ticks, `bridge.py` is the only thing that talks to Gazebo in a replay.
- `waypoint/px4/`: `backend.py` flies plans, `link.py` is the MAVLink link,
  `sitl.py` starts and stops PX4 and Gazebo. A new airframe is a name in
  `MODELS` (`sitl.py`) plus any parameters it needs in `PARAMS`
  (`backend.py`). A new PX4-native fault is an entry in `NATIVE_FAILURES`.
  `scripts/px4_check.py` is the end-to-end check after any change here.
