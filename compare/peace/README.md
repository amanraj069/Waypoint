# PEACE, rebuilt

**PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs.**
Uysal, Kehrer, Panichella (University of Bern). arXiv:2606.00104, May 2026.

Rebuilt from the paper and from the authors' release:
`github.com/erdemuysalx/PEACE`, commit `4ceda37` (20 August 2026), which
declares the MIT licence in its `pyproject.toml` (the repository has no licence
file), and the prompt dataset `erdemuysalx/uav-mission-prompts`, MIT, version
1.1.

The paper gives the design in prose and no numbers (no limits, no tool list, no
prompt). Every concrete value below therefore comes from the release, and where
the paper and the release differ the release is followed and the difference is
noted.

## What PEACE is

One LLM call turns a request, the robot's state and a world model into a typed
mission plan: a reasoning string and an ordered list of tool calls. A
deterministic executor runs the calls in order with no LLM in the loop. A
constraint layer checks every motion command before it reaches the autopilot:
an out-of-band altitude is clamped, a target outside the horizontal geofence
aborts the step. A failed step triggers a bounded replan.

## Element by element

| Paper | Release | Here | Match |
| --- | --- | --- | --- |
| III-B: single-pass planning into a `MissionPlan` | `schemas.py` | `schemas.py` | Same fields, same JSON schema |
| III-B: deterministic executor, replan on tool failure only | `nodes/planner_executor.py` | `agent.py` | Same control flow, call for call (see below) |
| III-B: bounded replanning | `agent_max_replan_depth = 2` | `config.py` | Same |
| III-E: static system prompt (tools, rules, schema) | `system_prompts/language.txt` | `prompts/language.txt` | Same text, 96 lines; each em dash written as a hyphen (a rule of this repository). A test restores the dashes and checks the release's SHA-256 |
| III-E: volatile user prompt (state, world model, history) | `system_prompts/__init__.py` | `prompts/__init__.py` | Same formatting functions |
| III-F: altitude band, clamped | `constraint_enforcement_service.py` | `services/constraint_enforcement.py` | Same: 0.5 m to 50 m |
| III-F: horizontal geofence, abort and replan | same file | same file | Same: 500 m in x and in y from home |
| III-A: the fixed tool set | `platforms/robot_tools.py`, `uav_tools.py` | `tools/` | The same 14 tools, geometry, tolerances and result strings |
| III-D: robot-state and world-model services | `services/` | `services/` | Same, including the eight LiDAR sectors |
| III-C: pinhole depth projection, depth-validity flag | `services/depth_service.py` | `services/depth.py` | Same code |
| III-C: detector interface | `backends/detector.py` | `perception.py` | Same interface; a simulated backend (see below) |
| Mission log | `services/mission_logger_service.py` | `services/mission_logger.py` | Same 13 CSV columns |
| IV-A: 108 prompts, 9 categories | Hugging Face dataset | `data/prompts.json` | The published file |
| IV-A: the dataset runner | `evaluation/run_dataset.py` | `evaluation/run_dataset.py` | Same procedure, filters, reset modes and counts |

The control flow of one mission, as released and as rebuilt:

1. Snapshot the robot state and the world model. Build two messages: the
   system prompt, and the query followed by the context (with the last 6 state
   snapshots when memory is on).
2. One LLM call constrained to the `MissionPlan` schema, at temperature 0.7 and
   at most 2000 output tokens. A reply that does not validate, or a plan with
   no steps, fails the mission.
3. Run each step. A result that starts with `ERROR` or contains `failed` is a
   tool failure.
4. On a tool failure, if fewer than 2 replans have been made: one LLM call with
   the original query, the completed steps, the remaining steps, the failure
   reason and a fresh context, asking for a plan "for the remaining objectives
   only". Otherwise the mission fails.
5. Write one row to the mission log; switch to loiter and stop the setpoint
   stream.

## What is different, and why

| Difference | Reason |
| --- | --- |
| ROS 2 and MAVROS are replaced by `compare/common/flight.py` | Neither belongs in this repository. The flight stack offers the same calls (stream a position setpoint, set the mode, arm, read odometry) with PX4's semantics: offboard needs a stream first, loiter one second after the stream stops, disarm on touchdown, disarm after ten idle seconds on the ground, yaw at 45 deg/s |
| The aircraft is Waypoint's simulator, not PX4 SITL in Gazebo | So that PEACE flies the same airframe as Waypoint. Wall-clock sleeps and time-outs become simulated time, so runs are fast and repeatable |
| The LLM is whichever model Waypoint is configured for; the release plans through Ollama (default `qwen2.5:14b`; its `.env.example` names `nemotron-3-super`) | `--model ollama:qwen2.5:14b` uses the release's model through Ollama's own API, with the schema passed as `format`, exactly as the release does. An OpenAI model gets the same two messages with the `MissionPlan` schema as a non-strict `json_schema` (strict mode cannot express its free-form `args`), and no reasoning effort, as the release sets none. A reasoning model takes no temperature and its thinking would count against the 2000-token cap, so on one those two are not sent |
| Detector backends: YOLO and a VLM are not rebuilt | They need rendered camera images, which the simulator does not produce. `SimulatedDetector` sits behind the same interface and reports boxes for the objects a simulated pinhole camera has in view; the release's depth projector then lifts them to world positions |
| The scene of the five object classes is this rebuild's | The paper's Gazebo world is not in the release, so where the person, car, bench, traffic light and fire hydrant stand is not known (`evaluation/scene.py`) |
| While the LLM thinks, the aircraft hovers for as long as the call took | In the release the world keeps running during inference; here time only moves when asked |
| `evaluation/checks.py` | The paper scores nothing ("not a quantitative benchmark"). Three mechanical checks are added and labelled as this rebuild's: the aircraft stayed inside the limits, how a safety prompt was enforced, and whether a waypoint prompt ended at its coordinates |

## What reading the release showed

These are properties of PEACE as released, kept as they are and covered by
tests in `compare/tests/test_peace.py`.

* **The geofence is a square, not a circle.** A target is refused when
  `abs(x) > 500 or abs(y) > 500`. The system prompt says the same ("Geofence:
  +-500 m in x and y from home"). `RELATED_WORK.md` describes it as circular;
  a target at (450, 450), 636 m from home, is accepted.
* **An arrival time-out is not a failure.** Only `ERROR` or `failed` in a
  result fails a step. A waypoint farther than the aircraft can fly in the
  30 s command time-out returns "(arrival timeout)" and the plan carries on
  from wherever the aircraft is.
* **Take-off is not checked against the fence**, only clamped in altitude, and
  a `land(x, y)` whose approach is refused still lands where the aircraft is.
* **A waypoint without a take-off does not arm.** The navigation helper
  switches to offboard but never arms, so the step times out on the ground and
  the mission is reported completed.
* **A detection away from the image centre is placed short.** The projector
  treats the depth image's value as range along the ray; a depth camera
  reports distance along the optical axis. At the edge of the field of view
  the object is placed about 20% too near. With the default inference rate of
  0.2 Hz the estimate is often not refreshed before an approach starts.

## Checked against the release, again

After the head-to-head, every file was compared with the release, line by
line, with docstrings and line wrapping set aside:

| Release | Here | Differences in code |
| --- | --- | --- |
| `platforms/uav_tools.py` | `tools/uav_tools.py` | MAVROS calls become flight-stack calls; wall-clock sleeps become simulated time |
| `platforms/robot_tools.py` | `tools/robot_tools.py` | the same, and log calls |
| `nodes/planner_executor.py` | `agent.py` | ROS plumbing only: the plan call, the executor, the failure test, the replan prompt and the logging are the release's |
| `services/*.py`, `schemas.py`, `system_prompts/__init__.py` | `services/`, `schemas.py`, `prompts/__init__.py` | none |
| `evaluation/run_dataset.py` | `evaluation/run_dataset.py` | ROS and Gazebo resets become the stack's; the checks are added and labelled |

What did differ was how the model was called, and that is fixed
(`agent.release_protocol`):

* **It carried Waypoint's reasoning effort** (`low`). The release sets none.
* **It got bare JSON mode.** The release hands the model the `MissionPlan`
  schema with every planning call; two head-to-head missions had failed on an
  empty plan. The schema is now sent too.

The head-to-head was flown again with this; the first run is kept in
`results/versus/seed0-peace-before-audit`. The 108-prompt run in
`results/peace/full-108` predates the fix (a partial re-run, 17 prompts, is in
`full-108-audited-partial`).

## Run it

```bash
.venv/bin/python -m compare.peace "Take off to 5 meters then move forward 3 meters"
.venv/bin/python -m compare.peace                                   # a prompt, like peace-cli
.venv/bin/python -m compare.peace.evaluation.run_dataset --limit 5
.venv/bin/python -m compare.peace.evaluation.run_dataset --category safety_violation
.venv/bin/python -m pytest compare/tests/test_peace.py
```

Settings are the release's, read from the same environment names
(`AGENT_MAX_REPLAN_DEPTH`, `ACTION_MAX_ALTITUDE`, `ACTION_HORIZONTAL_LIMIT`,
`VISION_INFERENCE_RATE`, ...).
