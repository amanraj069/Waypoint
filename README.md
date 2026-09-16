# waypoint

**Safety-Aware Hierarchical LLM for Autonomous UAV Mission Planning and
Adaptation**. The planner and the verifier are real; the rest is still mocked.

```
natural language mission
        -> planner    (LLM hierarchical planner)   [REAL: Gemini, strict JSON response schema]
        -> verifier   (deterministic safety check) [REAL: shapely geometry, no model in the loop]
        -> sim        (UAV executor)               [stub: linear interpolation generator]
        -> feedback   (telemetry / perception)     [stub: never requests a replan]
        -> replan loop back into planner           [planner.replan() ready, not yet wired]
```

## Layout

| Path            | Role |
| --------------- | ---- |
| `planner/`      | `plan()` / `replan()` against Gemini; `Waypoint`, `SubTask`, `Plan`, `Action`, `MissionConstraints` |
| `verifier/`     | `verify(plan, constraints) -> VerificationResult` (`passed`, `violations`); the checks live in `verifier/checks.py` |
| `sim/`          | `MissionState` dataclass, `execute(plan)` generator yielding one state per tick |
| `feedback/`     | `observe(state) -> Feedback` (`replan_required`, `reasons`) |
| `orchestrator/` | `run_mission(mission_text)` and the `python -m orchestrator.run` CLI |
| `tests/`        | pytest suite, one module per component |
| `experiments/`  | scratch scenarios and logged runs |

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
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env                    # then put your Gemini key in it
```

## Run

```bash
.venv/bin/python -m orchestrator.run --mission "survey the north field"
# or: uv run python -m orchestrator.run --mission "survey the north field"
```

Prints the plan, the verification result, and every `MissionState` tick from
launch through touchdown. Exits 0 when the mission ends landed and stationary.
`--quiet` suppresses per-tick output.

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
default — a bad plan never reaches the verifier looking like a good one.

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
logs a warning each time. It does **not** shift on 400/401/403 — a bad request
or a bad key fails identically on every model, so those fail fast on the first
rung. A malformed *plan* does not shift either: that is a planning failure, not
an availability one. If every rung fails, `PlannerAPIError` lists each attempt.

Manual test against the live API (spends tokens):

```bash
python scripts/try_planner.py "survey the north field and return to base"
```

## Verifier

`verify(plan, constraints)` is the gate between the planner and the aircraft. It
is 100% deterministic — plain `shapely` geometry and arithmetic, no model, no
sampling — so the same plan and constraints always produce the same report. It
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

Each `Violation` carries its `type`, the `waypoint_index` it belongs to (0-based
into the flattened plan, `None` for whole-plan findings like battery), a
`severity`, and a `description` written to be handed straight back to the model:

```
waypoint 3 (subtask "survey_north_field") at (80, 150, 30) is outside the
geofence, 55.0 m beyond its boundary. Move this waypoint back inside the
geofence polygon, leaving a few metres of margin.
```

`result.passed` is False when any **blocking** violation (`ERROR` or `CRITICAL`)
was found. `WARNING` findings — a marginal turn, a waypoint hugging the fence —
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
raise `ValueError`. A bad *plan* never raises — it comes back as violations.

Manual check against a real plan (the first form spends tokens):

```bash
python scripts/try_verifier.py "survey the north field and return to base" --save /tmp/plan.json
python scripts/try_verifier.py --plan /tmp/plan.json --shrink 0.3 --hazards
```

It verifies the plan against the normal envelope, then against a geofence
shrunk about its centre (plus, with `--hazards`, a no-fly zone across the route
and an obstacle on its far end) and prints the full violation report.


## Test

```bash
.venv/bin/python -m pytest
```

Every test mocks the Gemini client; an autouse fixture replaces the real
constructor with a tripwire, so a test that forgets to inject a fake client
fails rather than dialling out.

## Where the real pieces go

- `planner/` — done. Tune `planner/prompts.py` (system prompt + example) rather
  than the call site.
- `verifier/` — done. Tune the aircraft model in `VerifierConfig` rather than
  the checks; add a new check as a function in `verifier/checks.py` plus a
  `ViolationType`.
- `sim/executor.py` — replace interpolation with real dynamics, or bridge to
  PX4/Gazebo. Keep the generator interface.
- `feedback/stub_feedback.py` — set `replan_required=True` to trigger the
  adaptation loop. `orchestrator/loop.py` counts these in `MissionReport.replans`;
  calling `planner.replan()` on a failed verification is the next step, and
  `run_mission(planner=...)` already takes an injected planner for it.
# Waypoint
