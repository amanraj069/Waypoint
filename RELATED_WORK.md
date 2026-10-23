# Waypoint against the literature

**A comparison of Waypoint (Safety-Aware Hierarchical LLM for Autonomous UAV
Mission Planning and Adaptation) with recent work on LLM planners, plan
verification and in-flight replanning for robots and UAVs, before and after
closing the gaps the comparison found.**

First prepared 22 October 2026; updated 23 October 2026 with the system as it
now stands (branch `close-gaps`). Waypoint numbers come from experiment
batches in `experiments/runs/`. Every number for another paper is the one that
paper reports, with a link in the references.

---

## 1. Summary

- **Where Waypoint sits.** The field has split into three camps:
  1. *LLM to formal spec to solver*: AutoTAMP, SELP, Safety Chip, T3 Planner,
     STL translate-and-repair. These check intent, not only safety, with
     strong guarantees, but in toy 2D worlds.
  2. *LLM agents for real autopilots*: PEACE, DroneServer/MCP, PMR, TypeFly,
     Agentic UAVs. These are realistic stacks, but their safety is usually a
     per-command geofence or altitude clamp.
  3. *LLM plus an optimization or control safety filter*: SwarmGPT, LLM-Land,
     SAGE-LLM. These enforce dynamic feasibility and collision avoidance at
     run time, but for narrow tasks.

  Waypoint now takes the central idea of each camp and puts them in one loop,
  on 3D terrain and buildings, on both a custom simulator and PX4 in Gazebo:
  - a **deterministic intent check** (camp 1's idea): every plan is checked
    against what the mission asked, read by the LLM and grounded in the world;
  - a **whole-plan preflight verifier with deterministic repair** (beyond
    camp 2's per-command clamps);
  - a **control-barrier-function shield** under the controller (camp 3's
    idea), which also catches GPS drift;
  - **millisecond in-flight replanning by repair**, with the LLM as fallback
    (PMR's "call the LLM only when needed").

  Among the papers surveyed, none has all of these.

- **Head to head, on the same benchmark and the same code** (36 missions with
  hand-written ground truth, 2 repeats, gpt-5-mini, identical faults; section 2):

  | | Waypoint `full` | Waypoint `classic` (as first published) | `runtime-guard` (PEACE/DroneServer-style) |
  | --- | --- | --- | --- |
  | Mission accomplished (ground truth) | **82%** (51/62) | 53% (33/62) | 56% (35/62) |
  | Crashed | **0%** (0/62) | 1% (1/68) | 12% (9/72) |
  | Broke a keep-out | **0%** (0/62) | 4% (3/68) | 8% (6/72) |
  | Impossible missions refused | **100%** (10/10) | 0% | 0% |
  | In-flight replan latency, median | **0.005 s** | 14.9 s | 14.6 s |
  | Planner calls for the batch | 174 | 222 | 150 |

  Against `classic`, mission success rises by 29 points (p = 0.001); against
  `runtime-guard`, by 26 points (p = 0.003), with crashes 0 against 9
  (p = 0.004) and breaches 0 against 6 (p = 0.03). The original published
  batch, re-scored on the same ground truth, accomplished 50% of its missions
  and refused none of the impossible ones.

- **What remains open:**
  1. There is no hardware flight.
  2. Camera perception is demonstrated in one PX4 scenario, not across the
     benchmark.
  3. There is one planner model, and 62 scored flights per system.
  4. When a battery fault makes the full route unaffordable, the system gives
     up mission phases to land safely; that is 8 of the 11 missions `full`
     did not accomplish.

---

## 2. Waypoint, before and after

### 2.1 What changed

| Gap the first comparison found | What was built | Where |
| --- | --- | --- |
| Safety was checked, intent was not | The LLM reads the mission into requirements under a strict schema; the world grounds every landmark; a deterministic INTENT check (coverage, line of sight, distinct sides, order, avoid zones, landing) runs with every verification and names a flyable fix; impossible missions are refused before planning | `waypoint/intent/` |
| Corrections cost an LLM call each | Deterministic repair: clamps, climb-over, visibility-graph detours, fence projection, landing moves, added viewpoints; re-verified every round | `waypoint/verifier/repair.py` |
| Replans took 14 to 18 s, hidden by a frozen clock | Tiered replanning (repair, then battery trim, then the LLM); the aircraft hovers, burning battery, while the LLM thinks | `waypoint/orchestrator/loop.py`, `options.py` |
| Verified plans could still crash in wind and drift | A control-barrier-function shield, braking-aware, with a disturbance observer and a GPS drift observer, on both backends | `waypoint/sim/shield.py` |
| The verifier was not sound with respect to the flown path | Exact terrain clearance (no sampling); climbing legs judged on the climb-limited profile; roof clearance for waypoints and legs; landing pads clear of walls | `waypoint/verifier/checks.py`, `world/landing.py` |
| Hierarchy had no job | Corrections name the subtask at fault and the subtasks to keep | `verifier/models.py` |
| Prompt injection through map names | Names sanitised, instruction-like ones withheld, world text quoted as data | `waypoint/planner/sanitize.py` |
| Symbolic perception only | Obstacles from the PX4 x500_depth's depth camera: back-project, subtract the known world, cluster, fit, track | `waypoint/feedback/depth.py`, `gazebo_camera.py` |
| No external baselines, no intent metric, small batch | `runtime-guard` and `no-gate` systems; 36 missions with hand-written ground truth; intent success, refusals, extraction accuracy, repair, shield and soundness metrics | `waypoint/experiments/` |

Every capability is a switch (`--system full,no-repair` and so on), so each
can be measured on its own. 518 tests cover it, including soundness tests
against brute force.

### 2.2 The head-to-head batch

Batches `20261023T071716-compare2` (`full`) and
`20261023T085101-compare2-baselines` (`classic`, `runtime-guard`; re-flown
after a network outage voided the first attempt). All use 36 missions × 2
repeats, seed 0 (identical fault schedules), gpt-5-mini at low reasoning effort,
and the same code, so `classic` here already has the improved verifier. Each
flight is paired with an ungated baseline flying the same first plan.

| Metric | `full` | `classic` | `runtime-guard` |
| --- | --- | --- | --- |
| Mission accomplished, ground truth (95% CI) | **82%** (71 to 90%) | 53% (41 to 65%) | 56% (44 to 68%) |
| Mean share of the mission done | **87%** | 62% | 71% |
| Impossible missions refused | **10/10** | 0/10 | 0/10 |
| Possible missions refused | 0% | 0% | 0% |
| Landed under control | **100%** | 88% | 88% |
| Crashed | **0/62** | 1/68 | 9/72 |
| Broke a keep-out | **0/62** | 3/68 | 6/72 |
| First plans the verifier caught | 39% | 29% | n/a (no preflight) |
| Rejected plans fixed by repair, on the ground | 22/24 (92%) | n/a | n/a |
| In-flight replans resolved | 43/43 by repair, 0 abandoned | 37 by LLM, 10% abandoned | 39 by LLM |
| In-flight replan latency, median | **0.005 s** | 14.9 s | 14.6 s (hovering 7.9 s per flight) |
| Intent read exactly right (vs ground truth) | 83% (target F1 1.00) | n/a | n/a |
| Shield stepped in | 44% of flights | n/a | n/a |
| Planner calls / output tokens per flight pair | 174 / 2,777 | 222 / 5,174 | 150 / 3,315 |
| Paired ungated baseline crashed | 19% | 24% | 14% |

Fisher exact, two-sided:

| Comparison | Counts | p |
| --- | --- | --- |
| Accomplished, `full` vs `classic` | 51/62 vs 33/62 | **0.001** |
| Accomplished, `full` vs `runtime-guard` | 51/62 vs 35/62 | **0.003** |
| Crashed, `full` vs `runtime-guard` | 0/62 vs 9/72 | **0.004** |
| Keep-out, `full` vs `runtime-guard` | 0/62 vs 6/72 | **0.03** |
| Impossible refused, `full` vs either | 10/10 vs 0/10 | **0.00001** |
| Crashed, `full` vs `classic` | 0/62 vs 1/68 | 1.0 (both near zero) |

The cost of the gate, against the ungated baseline flying the same first
plan (13 pairs where it changed the plan): +5.5% distance and +28% time,
median. The time is mostly the shield slowing near boundaries and detours
around appeared obstacles.

**Where `full` still falls short** (11 of 62 feasible flights):
- 8 met a battery fault that made the full route unaffordable. The system
  gave up the phases that cost most and landed safely, which is the intended
  behaviour, but it scores as not accomplished.
- 2 took their photographs from where a drifting GPS put them.
- 2 photographed a tower from 2 of its 3 named sides.

No `full` flight crashed, broke a keep-out, or flew a plan the verifier would
refuse.

### 2.3 The published system, re-scored

The original batches (20 missions × 2 repeats, commit `ad9cbdf`) scored against
the same ground truth (`report --missions experiments/missions.json`):

| Condition | Accomplished | Mean share done | Impossible refused | Crashed |
| --- | --- | --- | --- | --- |
| hierarchical + feedback, gpt-5-mini | 50% (n=36) | 68% | 0/4 | 0% |
| flat + feedback | 58% | 71% | 0/4 | 2.5% |
| hierarchical + feedback, gpt-5-nano | 14% | 28% | 0/4 | 4.2% |

That was the gap: 88 to 95% of flights *landed*, but only 50 to 58% did what
was asked, and no impossible mission was ever refused.

### 2.4 Component measurements (no LLM spent)

- **Repair** (`scripts/replay_repairs.py`): of the 264 plans the verifier
  rejected in the published batches, repair produced a passing plan for 97%,
  in a median of 5 ms, adding a median 3% to the route. The LLM's own
  corrections fixed 75%, at 15 to 30 s per attempt.
- **Shield** (`scripts/shield_trial.py`): re-flying the 169 verified plans of
  the published batches under their own faults, with nothing replanning:

  | Plans | Crashed, shield off | Crashed, shield on | Keep-out, off | Keep-out, on |
  | --- | --- | --- | --- | --- |
  | Verified | 4.1% | **0.6%** | 29.0% | **0.6%** |
  | As the LLM first wrote them | 25.5% | **1.5%** | 27.0% | **0.5%** |

- **Verifier soundness** (`shield_trial.py --calm`, no wind, no faults): two
  verified plans from the published batches crashed in calm air. One was a
  climb steeper than the airframe can fly, which sagged into India Gate; the
  other was a waypoint 0.2 m above a roof, overshot on arrival. Both are
  refused by the current verifier, and `tests/test_soundness.py` keeps them
  refused.
- **Camera** (`scripts/camera_check.py`, live PX4 + Gazebo): the crane known
  to the depth camera alone, shield off. It was seen 23 m from its centre,
  placed 0.0 m off (radius 6.5 m against 6.0 m), and flown round by repair
  with no keep-out breach.

---

## 3. How papers were chosen, and a warning on comparing numbers

The papers were found by searching arXiv, conference proceedings and MDPI for
LLM-based planners that (a) target UAVs, or (b) put a verifier, monitor or
safety filter between the LLM and execution, or (c) replan at run time. The
survey favours 2024 to 2026 work and includes the older papers (AutoTAMP,
Safety Chip, REAL, TypeFly) that newer work treats as baselines.

**The numbers below are not directly comparable.** Each paper defines
"success" and "safety" differently: exact STL match, LTL-satisfying plan,
reaching within 1 m of a goal, avoiding a pedestrian. The tasks range from 2D
grid worlds to swarm choreography. The tables report each paper's own metric
in its own terms. Read them as "what each system is good at", not as a
leaderboard. Where Waypoint now implements a paper's idea, the comparison
says so.

---

## 4. The papers, and what Waypoint took from each

### A. LLM plans checked by formal methods

#### A1. AutoTAMP (Chen et al., MIT, ICRA 2024)

- **Approach.** GPT-4 translates natural language into Signal Temporal Logic,
  and an STL planner produces timed waypoints.
- **Correction loops.** A syntax checker (up to 5 re-prompts) and an
  LLM-judged semantic check (up to 3) wrap the translation.
- **Results (GPT-4 success).** HouseWorld2 rose from 42% (no correction) to
  82% (syntax plus semantic); Chip's Challenge from 42.7% to 87.7%;
  multi-agent Rover from 46% to 79%. LLM-as-planner baselines managed 0 to 49%.
- **Waypoint, then.** It had the same "tell the LLM why" loop, but checked
  only safety.
- **Waypoint, now.** It checks intent too, and deterministically (an LLM
  judge, as in AutoTAMP's semantic step, is not deterministic). It also
  repairs before asking again. Intent success rose from 53% to 82% against
  its own classic loop.

#### A2. Plug in the Safety Chip (Yang, Raman, Shah, Tellex, Brown, ICRA 2024)

- **Approach.** Natural-language rules become LTL and an automaton monitor
  that prunes unsafe actions.
- **Results.** On SELP's drone domain: 84.4% safety rate, 91.8% completion,
  28.1 s planning time.
- **Against Waypoint.** Waypoint verifies the whole plan before take-off and
  every replan, and its intent layer now expresses order ("first ... then")
  and avoid zones ("never fly over"), the temporal constraints Safety Chip
  was built for. It still has no general LTL.

#### A3. SELP (Wu et al., Purdue, ICRA 2025)

- **Approach.**
  - Equivalence voting over many NL-to-LTL translations.
  - Constrained decoding with a Büchi automaton.
  - A fine-tuned 7B planner.
- **Results (drone navigation, 500 tasks).**

  | Method | Safety | Completion |
  | --- | --- | --- |
  | GPT-4 | 37.6% | 78.2% |
  | Safety Chip | 84.4% | 91.8% |
  | SELP | 95.2% | 100% |

  Voting raised LTL translation accuracy from 88.4% to 98.0%.
- **Against Waypoint.**
  - Waypoint now has equivalence voting for intent extraction
    (`WAYPOINT_INTENT_VOTES`). Single-shot extraction already reads 83% of
    missions exactly right, with target F1 1.00.
  - SELP prevents unsafe plans during generation and needs a fine-tuned open
    model. Waypoint rejects and repairs after generation and works with closed
    API models; repair reaches the same end (97% of rejected plans fixed) in
    milliseconds.
  - SELP has no replanning and no faults; Waypoint has both.

#### A4. T3 Planner (Li, Zhao et al., arXiv Oct 2025)

- **Approach.** Three cascaded LLM stages (task, time, trajectory), each
  checked by an STL verifier.
- **Results.** 83.7 to 97.0% success over 1,000 2D tasks, against 50 to 61%
  for AutoTAMP.
- **Against Waypoint.** Waypoint's corrections now name the subtask at fault
  and the subtasks to keep, giving the hierarchy a role. T3's layered
  verification goes further (separate logical, temporal and motion checks).

#### A5. NL to STL with repair (Ping et al., arXiv Mar 2026)

- **Approach.**
  - A 0.5B model translates natural language to STL (98.0% exact).
  - A MILP plans the trajectory.
  - An irreducible infeasible subsystem pinpoints the conflict; the solver
    relaxes it, and safety predicates are never relaxed.
- **Against Waypoint.** Waypoint adopted the same principle in the air: when
  no plan can serve everything left and stay safe, the mission's requirements
  give way and safety never does (`intent_relaxed`). Its `assess()` refuses a
  mission that cannot be flown before planning, much as an infeasible MILP is
  diagnosed.

### B. LLM UAV agents on real autopilot stacks

#### B1. PEACE (Uysal, Kehrer, Panichella, arXiv May 2026)

- **Approach.**
  - One LLM call produces tool calls, dispatched through ROS 2 and MAVLink.
  - A constraint layer clamps altitude and aborts commands outside the
    geofence.
  - Replanning is bounded; perception comes from YOLO or VLM detections.
  - It flies PX4 SITL in Gazebo.
- **Results.** None quantitative.
- **Against Waypoint.** `runtime-guard` reproduces PEACE's enforcement
  (altitude clamped, a command outside the fence refused at dispatch, LLM
  replan). On the same benchmark it accomplished 56% of missions and crashed
  12% of flights. Waypoint `full` accomplished 82% and crashed none. Waypoint
  now also perceives with a camera on PX4 (geometric, not learned).

#### B2. DroneServer (Ramos Silva and Burke, arXiv Jan 2026)

- **Approach.** 98 MAVSDK tools exposed over MCP; the LLM is an untrusted
  commander; every command is validated server-side, with an independent
  geofence.
- **Results.**
  - More than 1,000 SITL flights.
  - 8 of 11 LLMs completed 90 to 100% of six missions.
  - 0 geofence escapes in 110 adversarial trials.
  - 10 of 10 runs on real quadcopters.
- **Against Waypoint.** Same philosophy (the LLM is untrusted), plus
  preflight whole-plan checks over terrain, buildings and battery, and intent.
  DroneServer still leads on breadth (many models, hardware).

#### B3. PMR (Park et al., Korea University, arXiv Jun 2026)

- **Approach.**
  - A local mission runtime handles execution.
  - A remote LLM is called only for recovery, gated by a learned value score.
  - The LLM picks from closed recovery skills, then the choice is verified and
    shielded.
- **Results.** 400 Gazebo/PX4 runs.
  - Hard regime: 95% success (learned gate), 97.5% (rule-based gate), 5%
    (local only).
  - LLM latency about 4.3 s.
- **Against Waypoint.** Waypoint's tiered replanning goes further on
  latency: deterministic repair resolved all 43 in-flight replans in the
  batch, at a median of 5 ms, without calling the LLM, while still allowing
  whole new routes when repair cannot help. PMR's learned gate remains an
  idea worth borrowing.

#### B4. SAGE-LLM (Zhao et al., NWPU and TeleAI, arXiv Feb 2026)

- **Approach.** LLM actions are verified by a fuzzy control barrier function
  over a cone of directions.
- **Results.** 99.73% safe rate and 92% zero-danger rate, in pursuit-evasion
  in Isaac Gym.
- **Against Waypoint.** Waypoint now has a CBF layer of its own, under every
  command and on both backends, with braking awareness, a disturbance observer
  and GPS-drift observation. Re-flying archived plans under their faults, it
  cut crashes of verified plans from 4.1% to 0.6%, and of raw LLM plans from
  25.5% to 1.5%.

### C. LLM plus an optimization safety filter

#### C1. SwarmGPT (Schuck, Zhou, Schoellig et al., TUM, RA-L 2025)

- **Approach.** An MPC safety filter makes minimal corrections to
  LLM-designed swarm trajectories.
- **Results.**
  - Only 21.65% of raw LLM trajectories were collision-free.
  - The filter deviated 8.5 to 23 cm.
  - The LLM took 7.5 s per response.
  - It flew 20 real Crazyflies.
- **Against Waypoint.** Waypoint's repair is the planning-level counterpart
  (minimal geometric fixes, +3% route median), and its shield the
  control-level one.

#### C2. LLM-Land (Cai, Wu, Zhou, arXiv May 2025)

- **Approach.** The LLM only adjusts the constraints; an MPC flies the
  aircraft.
- **Results.**
  - 96% landing success, against 6% when the LLM controlled the aircraft
    directly.
  - 1.45 s on-board latency with a 1B model.
- **Against Waypoint.** Same principle (the LLM never commands the aircraft).
  Waypoint's in-flight latency is now 5 ms for repairable events.

### D. LLM UAV mission generation without a verifier

| Paper | What it does | Reported results |
| --- | --- | --- |
| TypeFly (Chen et al., 2023/2024) | LLM writes MiniSpec, interpreted as a stream | Up to 62% lower response time |
| UAV-VLA (Sautenkov et al., 2025) | Satellite imagery + VLM + GPT to flight plans | 34.22 m mean object error, 6.5× faster than an operator |
| UAV-CodeAgents (Sautenkov et al., 2025) | Multi-agent ReAct on satellite maps | 93% success, 97 s per mission |
| Agentic UAVs (Koubaa and Gabr, 2025) | YOLOv11 + GPT-4 + Gemma-3 agent | Person detection 75% to 91% |
| NeLV (Yuan et al., 2025) | Position paper: five autonomy levels | None |

None checks a plan before it flies. Waypoint's ungated baselines show what that
costs: flying the same first plans without the gate crashed 14 to 24% of the
time.

### E. Adjacent work

- **REAL** (Tagliabue, How et al., MIT, 2023): the LLM adapts control
  in flight. Waypoint keeps the LLM out of the control loop entirely.
- **Mission-level runtime assurance for LLM-assisted ISR swarms** (Kekatos
  et al., 2026): compositional runtime verification catches a prompt-injected
  planner. Waypoint now sanitises map names and quotes world text as data, and
  its deterministic gates are the backstop. It has no multi-vehicle assurance.
- **MultiUAV-Plat** (Zhang et al., 2026): 1,500 tasks with hidden validation.
  Waypoint's benchmark now follows the same pattern (hidden hand-written
  ground truth), at 36 missions.

---

## 5. Feature matrix

✓ = yes, ◐ = partly, ✗ = no, ? = not stated. "Waypoint" is the system now
(`full`); "v1" is the published loop.

| | Waypoint | v1 | AutoTAMP | SELP | T3 | PEACE | DroneServer | PMR | SAGE-LLM | SwarmGPT | LLM-Land |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UAV, 3D terrain and buildings | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ◐ |
| Checks intent, not only safety | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Order and avoid constraints | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Refuses impossible missions up front | ✓ | ✗ | ◐ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Whole-plan preflight verifier | ✓ (exact) | ✓ (sampled) | ✓ | ✓ | ✓ | ◐ | ◐ | ✓ | ✓ | ✓ | ✓ |
| Dynamics in the verifier | ◐ (climb rate, stopping room) | ✗ | ◐ | ✗ | ◐ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Deterministic repair before re-asking | ✓ | ✗ | ✗ | ◐ (decoding) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (filter) | ✗ |
| Runtime safety filter | ✓ (CBF) | ✗ | ✗ | ✗ | ✗ | ◐ (clamp) | ◐ (geofence) | ✓ | ✓ | ✓ | ✓ |
| In-flight replanning without the LLM | ✓ (5 ms) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (local policy) | ◐ | ✗ | ✓ |
| Battery-aware replanning | ✓ | ✓ | ✗ | ✗ | ◐ | ✗ | ? | ◐ | ✗ | ✗ | ✗ |
| GPS-fault tolerance | ◐ (drift observer) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Prompt-injection hardening | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Real perception | ◐ (depth camera, PX4 sim) | ✗ | ✗ | ✗ | ✗ | ✓ | ? | ◐ | ✗ | ✗ | ✓ |
| PX4 / ArduPilot SITL | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Real hardware | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| External baseline in the paper's own tests | ✓ (runtime guard) | ✗ | ✓ | ✓ | ✓ | ✗ | ◐ | ✓ | ✓ | ◐ | ✓ |
| Ground-truth task-success metric | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ◐ | ✓ | ✓ | ◐ | ✓ |
| Flights per condition | 72 | 40 | 100 to 200 | 500 | 300 to 400 | 0 | >1,000 total | 40 per regime | ~20 | 97 | 50 |

---

## 6. The questions each paper answers, with Waypoint's answer now

| Question | Best in the literature | Waypoint now |
| --- | --- | --- |
| How unsafe is a raw LLM plan? | 62% unsafe (SELP, GPT-4); 78% colliding (SwarmGPT) | 38 to 42% would be refused; 14 to 24% crash when flown ungated |
| How much does a gate improve safety? | 37.6% to 95.2% safe (SELP) | Crashes 0/62 with the full system against 9/72 for runtime-only enforcement |
| Does the plan do what was asked? | 95.2% safe and 100% complete (SELP, 2D rooms); 79 to 100% (AutoTAMP); 84 to 97% (T3) | 82% of missions fully accomplished in 3D worlds under faults, 87% of each mission done on average; 100% of impossible missions refused |
| How fast is an in-flight replan? | 1.45 s (LLM-Land), 4.3 s (PMR) | 5 ms by repair (43/43); 13 to 16 s when the LLM is needed |
| What does safety cost? | 8.5 to 23 cm deviation (SwarmGPT); 6.4 s planning (SELP) | +5.5% distance, +28% time against the ungated same plan |

---

## 7. What is still open, plainly

1. **No hardware flight.** Every result is simulation (custom simulator and
   PX4 SITL). PMR, DroneServer, SwarmGPT and LLM-Land all have a hardware run.
2. **Camera perception is demonstrated, not benchmarked.** One PX4 scenario
   passes all six checks; the 36-mission batch uses symbolic perception. The
   depth sensor's 19 m range is short for fast flight.
3. **One planner model, 62 scored flights per system.** The headline
   differences are significant, but the batch should be repeated with other
   models (Gemini, an open model) and more repeats.
4. **Intent extraction is 83% exact.** Voting is implemented
   (`WAYPOINT_INTENT_VOTES=3`) but not yet measured; SELP shows it is worth
   about 10 points.
5. **Battery faults still cost missions.** That is by design, since safety
   comes first, but a smarter trim (keep the targets that matter most) would
   recover some of the 8 flights.
6. **No general temporal logic.** Order and avoid zones cover the common
   cases; "visit A twice", "never return to B" and time windows are not
   expressible.
7. **GPS drift is observed, not corrected in the controller.** The shield
   guards the corrected position; the controller still steers on the raw fix,
   so photographs can be taken from a drifted position (2 of the 11 misses).

---

## 8. Draft related-work paragraph

> LLM planners for robots are increasingly paired with a check between the
> model and the actuator. One line translates instructions into temporal
> logic and plans or verifies against it (AutoTAMP [1], Safety Chip [2], SELP
> [3], T3 [4], STL repair [5]), checking intent but in 2D abstractions. A second
> builds LLM agents on real autopilots and enforces safety per command
> (PEACE [6], DroneServer [7]) or verifies LLM recovery decisions in flight
> (PMR [8]). A third filters LLM outputs through control-level safety layers
> (SAGE-LLM [9], SwarmGPT [10], LLM-Land [11]). Waypoint combines the three in
> one loop over real 3D terrain and buildings: an LLM-read, world-grounded
> mission intent checked deterministically alongside exact safety checks;
> deterministic repair before any re-prompt; millisecond in-flight replanning
> with the LLM as fallback; and a control-barrier-function shield that also
> tolerates GPS drift. On 36 missions with hand-written ground truth it
> accomplishes 82% of feasible missions, against 53% for its own published
> loop and 56% for runtime-only enforcement (p ≤ 0.003), with no crashes
> (against 12% for runtime-only), and it refuses every mission that cannot be
> flown as asked.

---

## 9. References

1. Y. Chen, J. Arkin, C. Dawson, Y. Zhang, N. Roy, C. Fan. *AutoTAMP:
   Autoregressive Task and Motion Planning with LLMs as Translators and
   Checkers.* ICRA 2024. [arXiv:2306.06531](https://arxiv.org/abs/2306.06531)
2. Z. Yang, S. S. Raman, A. Shah, S. Tellex. *Plug in the Safety Chip:
   Enforcing Constraints for LLM-driven Robot Agents.* ICRA 2024.
   [arXiv:2309.09919](https://arxiv.org/abs/2309.09919)
3. Y. Wu, Z. Xiong, Y. Hu, S. S. Iyengar, N. Jiang, A. Bera, L. Tan,
   S. Jagannathan. *SELP: Generating Safe and Efficient Task Plans for Robot
   Agents with Large Language Models.* ICRA 2025.
   [arXiv:2409.19471](https://arxiv.org/abs/2409.19471)
4. J. Li, G. Zhao, et al. *T3 Planner: A Self-Correcting LLM Framework for
   Robotic Motion Planning with Temporal Logic.* arXiv, Oct 2025.
   [arXiv:2510.16767](https://arxiv.org/abs/2510.16767)
5. Y. Ping, H. Ding, T. Liang, et al. *LLM-Enabled Low-Altitude UAV Natural
   Language Navigation via Signal Temporal Logic Specification Translation
   and Repair.* arXiv, Mar 2026. [arXiv:2603.27583](https://arxiv.org/abs/2603.27583)
6. E. Uysal, T. Kehrer, S. Panichella. *PEACE: A Planner-Executor Agent with
   Constraint Enforcement for UAVs.* arXiv, May 2026.
   [arXiv:2606.00104](https://arxiv.org/abs/2606.00104)
7. J. N. Ramos Silva, P. J. Burke. *An LLM-Agnostic, MAVLink-Based Drone
   Command and Control Interface and Agentic Harness Using the Model Context
   Protocol.* arXiv, Jan 2026. [arXiv:2601.15486](https://arxiv.org/abs/2601.15486)
8. T. Park, K. Yoo, S. Yoo, H. Kim. *Selective Agentic Recovery for UAV
   Autonomy with a Persistent Mission Runtime.* arXiv, Jun 2026.
   [arXiv:2606.14219](https://arxiv.org/abs/2606.14219)
9. W. Zhao, Y. Zhao, G. Liu, Z. Jiang, D. Ma, Z. Li, X. Li. *SAGE-LLM.* arXiv,
   Feb 2026. [arXiv:2602.23719](https://arxiv.org/abs/2602.23719)
10. M. Schuck, D. O. Dahanaggamaarachchi, B. Sprenger, V. Vyas, S. Zhou,
    A. P. Schoellig. *SwarmGPT.* IEEE RA-L 2025.
    [arXiv:2412.08428](https://arxiv.org/abs/2412.08428)
11. S. Cai, Y. Wu, L. Zhou. *LLM-Land.* arXiv, May 2025.
    [arXiv:2505.06399](https://arxiv.org/abs/2505.06399)
12. G. Chen, X. Yu, N. Ling, L. Zhong. *TypeFly.* arXiv, 2023/2024.
    [arXiv:2312.14950](https://arxiv.org/abs/2312.14950)
13. O. Sautenkov et al. *UAV-VLA.* 2025. [arXiv:2501.05014](https://arxiv.org/abs/2501.05014)
14. O. Sautenkov et al. *UAV-CodeAgents.* 2025. [arXiv:2505.07236](https://arxiv.org/abs/2505.07236)
15. A. Koubaa, K. Gabr. *Agentic UAVs.* 2025. [arXiv:2509.13352](https://arxiv.org/abs/2509.13352)
16. L. Yuan et al. *Next-Generation LLM for UAV.* Position paper, 2025.
    [arXiv:2510.21739](https://arxiv.org/abs/2510.21739)
17. A. Tagliabue et al. *REAL.* 2023. [arXiv:2311.01403](https://arxiv.org/abs/2311.01403)
18. N. Kekatos et al. *Mission-Level Runtime Assurance for LLM-Assisted ISR
    Swarms over a Verification-Aware Fabric.* arXiv, 2026.
    [arXiv:2607.23532](https://arxiv.org/abs/2607.23532)
19. S. Zhang et al. *MultiUAV-Plat.* arXiv, Jun 2026.
    [arXiv:2606.31073](https://arxiv.org/abs/2606.31073)

### Notes on sources

- Numbers for PEACE, SAGE-LLM, the STL-repair paper, T3, AutoTAMP, SELP,
  SwarmGPT, LLM-Land, DroneServer, MultiUAV-Plat and PMR were read from each
  paper's full text. For TypeFly, UAV-VLA, UAV-CodeAgents, Agentic UAVs, REAL
  and Safety Chip only the abstract was read.
- Several papers contain internal inconsistencies:
  - SAGE-LLM reports both 90% and 80% success.
  - The STL-repair paper claims +5.8 points; its table shows +4.7.
  - LLM-Land reports 96% and 94% for the same scenario.
- Waypoint's result directories are under `experiments/runs/` (gitignored):
  - `20261023T071716-compare2` (`full`);
  - `20261023T085101-compare2-baselines` (`classic`, `runtime-guard`);
  - `20261007T093110-final` and `20261007T120924-cost-LOW` (the published
    batches).

  `python -m waypoint.experiments report <dir> --missions experiments/missions.json`
  reproduces every Waypoint number above from the files alone.
