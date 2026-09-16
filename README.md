# waypoint

**Safety-Aware Hierarchical LLM for Autonomous UAV Mission Planning and
Adaptation**. The planner is real; the rest is still mocked.

```
natural language mission
        -> planner    (LLM hierarchical planner)   [REAL: Gemini, strict JSON response schema]
        -> verifier   (deterministic safety check) [stub: always passes]
        -> sim        (UAV executor)               [stub: linear interpolation generator]
        -> feedback   (telemetry / perception)     [stub: never requests a replan]
        -> replan loop back into planner           [planner.replan() ready, not yet wired]
```

## Layout

| Path            | Role |
| --------------- | ---- |
| `planner/`      | `plan()` / `replan()` against Gemini; `Waypoint`, `SubTask`, `Plan`, `Action`, `MissionConstraints` |
| `verifier/`     | `verify(plan) -> VerificationResult` (`passed`, `violations`) |
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
- `verifier/stub_verifier.py` — geofence, altitude envelope, battery reserve,
  no-fly zones. Populate `violations` and return `passed=False`; the
  orchestrator already aborts on a failed verification.
- `sim/executor.py` — replace interpolation with real dynamics, or bridge to
  PX4/Gazebo. Keep the generator interface.
- `feedback/stub_feedback.py` — set `replan_required=True` to trigger the
  adaptation loop. `orchestrator/loop.py` counts these in `MissionReport.replans`;
  calling `planner.replan()` on a failed verification is the next step, and
  `run_mission(planner=...)` already takes an injected planner for it.
# Waypoint
