# DroneServer, rebuilt

**An LLM-Agnostic, MAVLink-Based Drone Command and Control Interface and
Agentic Harness Using the Model Context Protocol.** Ramos Silva and Burke (UC
Irvine). arXiv:2601.15486 (January 2026, v3 September 2026).

Rebuilt from the paper and from the authors' release:
`github.com/PeterJBurke/droneserver`, MIT, commit `06b7d1a` (20 September
2026). The upstream licence is in `LICENSE.upstream`.

## What DroneServer is

An MCP server that gives any tool-calling LLM a drone's command, telemetry,
mission and safety functions as 98 tools. The LLM is treated as an untrusted
commander: every tool call crosses a fixed server-side pipeline before it can
reach the aircraft (authenticate, read state, classify the tier, authorize,
rate-limit, confirmation token, bounds, geofence, preconditions, execute,
audit). A refusal is a structured result with a rule id and a remedy, never an
exception, and the guard fails closed.

The release is about 24,500 lines of Python. The paper's contribution is the
interface and its safety layer, so that is what is rebuilt in full; the list
further down says exactly what is not.

## Element by element

| Paper | Release | Here | Match |
| --- | --- | --- | --- |
| 4.2: criticality tiers, argument-aware escalation | `safety/tiers.py` | `safety/tiers.py` | **Vendored unchanged.** All 98 tools classified: 29 read-only, 62 normal, 6 critical, 1 emergency; 11 escalation rules |
| 4.2: bounds, preconditions, energy-direction failsafe, rate limits | `safety/validation.py` | `safety/validation.py` | **Vendored unchanged** (796 lines), every rule id included |
| 4.2: independent server-side geofence, whole-mission validation | `safety/geofence.py` | `safety/geofence.py` | **Vendored unchanged** |
| 4.2: confirmation-token handshake | `safety/tokens.py` | `safety/tokens.py` | **Vendored unchanged**: single use, bound to client, tool and arguments, 60 s |
| 4.2: API keys and scopes | `safety/auth.py` | `safety/auth.py` | **Vendored unchanged** |
| 4.2: append-only JSONL audit log | `safety/audit.py` | `safety/audit.py` | **Vendored unchanged**, schema `droneserver.audit/1` |
| Limits and switches | `safety/config.py` | `safety/config.py` | Vendored; every default unchanged (120 m, 20 m/s, 1000 m radius, 3 s settling, 60 calls a minute, 6 critical) |
| 4.2: the pipeline, applied at registration | `safety/middleware.py` | `safety/middleware.py` | Same order, same results, same audit record; one layer per session instead of one per process |
| Vehicle state snapshot | `safety/state.py` | `safety/state.py` | Same fields and command history; read from the simulated vehicle |
| 3.3, 3.4: the MCP server, 98 tools | `app.py`, `tools/` | `server.py`, `tools/core.py` | A real MCP server on the official SDK's FastMCP. All 98 tools registered with their released names, parameters and descriptions; the catalogue is about 22,000 tokens, as the paper says. 55 implemented, each ported from its release body (same MavSDK calls in the same order, same checks, same result keys and wording); 43 answer that the vehicle lacks the subsystem |
| 5: the agent loop | `llm/agent.py` | `llm/agent.py` | Same loop, limits (90 turns, 250 tool calls) and client-side rejections; the model is called as the harness calls it (no temperature or reasoning effort set; `parallel_tool_calls` and `tool_choice="auto"` always sent) |
| T4: server-executed missions | `missions/runner.py`, `tools/managed_mission.py` | `managed.py`, `tools/core.py` | The same state machine, item layout, start evidence, auto-actions and completion rule, on the flight clock; no checkpoint or restart recovery |
| 5.1: system prompt and mission prompts (Table 1) | `llm/prompts.py` | `llm/prompts.py` | **Vendored unchanged** |
| 5.2: verdicts from flight evidence | `llm/verdicts.py` | `benchmark/verdicts.py` | Same pass conditions and thresholds for T1 to T5 and T7 to T10 |
| 4.2: the 29-case adversarial suite | `tests/integration/test_adversarial_sitl.py` | `adversarial.py` | The same 29 cases; one is partial (below) |

"Vendored unchanged" means the file is the upstream file with a header, new
import paths, and, in two files, `time.monotonic()` read from the simulation's
clock. `compare/tests/test_droneserver.py` checks the headers and the rule ids.

## What is different, and why

| Difference | Reason |
| --- | --- |
| MavSDK and the autopilot are replaced by `vehicle.py` on Waypoint's simulator | The release needs ArduPilot or PX4 SITL and MavSDK. `Vehicle` offers the same operations in the same frames (WGS84 degrees, metres above sea level or above home, compass headings), with ArduCopter's behaviour where the release depends on it: home moves to wherever the aircraft arms, mission item 0 is the home slot and is never flown, AUTO does not lift off from the ground, DO_CHANGE_SPEED is refused unless flying in GUIDED or AUTO, DO_ORBIT is unsupported, and `WPNAV_SPEED` really is the cruise speed. No camera is fitted unless asked for (plain SITL has none) |
| The safety layer's clock is the flight clock | The aircraft flies in simulated time. A token's 60 s, a minute of rate limit and the 3 s after take-off are measured on that clock, or they would mean nothing |
| While the model thinks, the aircraft keeps flying for as long as the call took (at least 1 s) | In the release the wall clock is the flight clock. This matters: a polling model sees the aircraft move between polls, and an armed aircraft left on the ground for 10 s disarms itself |
| The transport is the SDK's in-memory MCP session, and the API key is presented by the session | The paper's deployment is HTTP on a private network with the key in a header. The network measures of section 4.1 are about that deployment and are not rebuilt |
| The model is reached through this repository's one chat client | The release has an adapter per provider (`llm/providers.py`, 1,100 lines). The messages, the tool catalogue and the request settings the model receives are the same |
| The airframe is Waypoint's simulated quadcopter | Shared by all three systems in the head-to-head. It lands faster than ArduCopter (which descends at 0.5 m/s for its last 10 m), so a model polling a landing sees fewer "landing" answers; no verdict depends on it (see the replay below) |

## Checked against the authors' own trials

`python -m compare.droneserver.replay` takes the two runs in the release that
record every tool call with its arguments, answer, rule and timestamp
(`docs/benchmark_runs/`, copied to `data/upstream_runs/`): GPT-5.2 flying T1 to
T5 and T7 to T9 against the real server on ArduPilot SITL, and Gemini 3.5
Flash-Lite flying T1 and T9. It sends exactly those calls, in order, through
the rebuilt server, with the flight clock pinned to the recorded timestamps,
and compares each answer and each trial's verdict.

| | Before the audit | After |
| --- | --- | --- |
| Trials given the published verdict | 9 of 10 | **10 of 10** |
| Calls answered with the published status and rule | 195 of 243 | **219 of 243** |
| Safety rules fired as published | all | all |

The 24 calls still answered differently are all `monitor_flight` polls: 23 in
a landing's last metres, where ArduCopter is slower to touch down than the
shared airframe, and one in T5 that finds the aircraft already down where the
real one was still hovering. The published answers also carry
fewer keys than the release's code produces: that run (7 August 2026) predates
the release (20 September 2026), whose `monitor_flight` docstring describes
replacing exactly the frozen "LANDING | Descending..." answer the run shows.
`compare/tests/test_droneserver.py` keeps the 10 of 10 as a test.

The audit was prompted by the rebuilt server doing badly in the head-to-head,
and it found real differences from the release, now fixed:

* **The model was called with Waypoint's reasoning effort** (`low`). The
  release's harness never sets one, so the provider's default applies.
* **The server-executed missions were missing.** `start_managed_mission`,
  `get_mission_status` and `control_managed_mission` were stubs; they are the
  path GPT-5.2 used for T4 in the authors' run. They are now `managed.py`.
* **`monitor_flight` waited 30 s while landing**; the release answers at once.
  Its answer now carries the release's fields (`altitude_m`,
  `distance_from_launch_point_m`, `ground_speed_m_s`, `flight_mode`, ...).
* **Return to launch and the emergency stop set a landing latch** the release
  does not set, and the completion radius was 10 m, not the release's 20 m.
* **`reposition` set a destination** that `land` then gated on; the release
  only records the movement. `arm_drone` now clears all four per-flight
  latches.
* **`set_max_speed` had no fallback.** ArduCopter refuses DO_CHANGE_SPEED on the
  ground and the release then writes `WPNAV_SPEED`; that is why the authors'
  T7 found 500 where the default is 1000.
* **Hold, yaw and GUIDED** are now a goto to the current position, as
  released; mission, telemetry and camera answers use the release's wording.
* **The vehicle followed PX4 where ArduCopter differs**: mission item 0, AUTO on
  the ground, the speed command, take-off in the air.
* **The turn limit in the head-to-head was 60**, not the harness's 90.
* **Re-arming after an auto-landing failed.** `monitor_flight` lands the
  aircraft on arrival and leaves it in LAND; the vehicle then undid every arm.
  MavSDK's `arm()` on ArduPilot leaves LAND or RTL for HOLD before arming, and
  the release's own notes describe models re-arming at a destination and
  flying on. Found later, from a gpt-5-mini flight; two head-to-head flights
  had ended on it, and were flown again with the fix (`Comp_res.md` names them).

## Not rebuilt

* 43 of the 98 tools: offboard setpoints, gimbal, calibration, failure
  injection, MAVLink FTP and shell, follow-me, winch and gripper, transponder,
  RTK, raw mission and QGroundControl plan import. They are registered and
  guarded, and say they are unavailable.
* Mission 6 (fly to the nearest hospital), which needs a live Google Maps MCP
  server. The paper scores it apart from the core suite.
* The second half of adversarial case I4 (a QGroundControl plan whose waypoint
  is outside the fence). Its first half runs.
* Real aircraft, ArduPilot and PX4 firmware differences, the network security
  measures, the telemetry recorder and capture tooling, the spend ledger.
* The eleven-model campaign. One model is run here.

## Run it

```bash
.venv/bin/python -m compare.droneserver.adversarial                         # the 29 cases, no LLM
.venv/bin/python -m compare.droneserver.replay                              # the authors' trials, call for call
.venv/bin/python -m compare.droneserver.benchmark --missions T1,T2 --trials 1
.venv/bin/python -m compare.droneserver.benchmark --trials 5                # the paper's N=5
.venv/bin/python -m compare.droneserver.benchmark --missions T8 --trials 10 # geofence-violation trials
.venv/bin/python -m pytest compare/tests/test_droneserver.py
```

```python
from compare.droneserver import DroneServer, fresh_vehicle

server = DroneServer(fresh_vehicle())
async with server.client() as mcp:                       # an MCP client session
    tools = await mcp.list_tools()                       # 98 schemas
    await server.call(mcp, "arm_drone")
    await server.call(mcp, "takeoff", {"takeoff_altitude": 5000})
    # {'status': 'rejected', 'rule': 'bounds.max_altitude', 'remedy': 'Request an altitude of 120 m or less.', ...}
```
