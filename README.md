# waypoint

Skeleton for **Safety-Aware Hierarchical LLM for Autonomous UAV Mission Planning
and Adaptation**. Every component is mocked; the point is a provable control
flow, not real behaviour.

```
natural language mission
        -> planner    (LLM hierarchical planner)   [stub: hardcoded 3-waypoint plan]
        -> verifier   (deterministic safety check) [stub: always passes]
        -> sim        (UAV executor)               [stub: linear interpolation generator]
        -> feedback   (telemetry / perception)     [stub: never requests a replan]
        -> replan loop back into planner
```

## Layout

| Path            | Role |
| --------------- | ---- |
| `planner/`      | `plan(mission_text) -> Plan`; pydantic `Waypoint`, `SubTask`, `Plan`, `Action` |
| `verifier/`     | `verify(plan) -> VerificationResult` (`passed`, `violations`) |
| `sim/`          | `MissionState` dataclass, `execute(plan)` generator yielding one state per tick |
| `feedback/`     | `observe(state) -> Feedback` (`replan_required`, `reasons`) |
| `orchestrator/` | `run_mission(mission_text)` and the `python -m orchestrator.run` CLI |
| `tests/`        | pytest suite, one module per component |
| `experiments/`  | scratch scenarios and logged runs |

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/python -m orchestrator.run --mission "survey the north field"
# or: uv run python -m orchestrator.run --mission "survey the north field"
```

Prints the plan, the verification result, and every `MissionState` tick from
launch through touchdown. Exits 0 when the mission ends landed and stationary.
`--quiet` suppresses per-tick output.

## Test

```bash
.venv/bin/python -m pytest
```

## Where the real pieces go

- `planner/stub_planner.py` — swap the hardcoded plan for an LLM call that
  decomposes the mission into subtasks. The `Plan` schema is already the
  structured-output contract.
- `verifier/stub_verifier.py` — geofence, altitude envelope, battery reserve,
  no-fly zones. Populate `violations` and return `passed=False`; the
  orchestrator already aborts on a failed verification.
- `sim/executor.py` — replace interpolation with real dynamics, or bridge to
  PX4/Gazebo. Keep the generator interface.
- `feedback/stub_feedback.py` — set `replan_required=True` to trigger the
  adaptation loop. `orchestrator/loop.py` counts these in `MissionReport.replans`;
  wiring a mid-flight replan back into the planner is the next step.
# Waypoint
