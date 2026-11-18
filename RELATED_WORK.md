# Waypoint against the closest related work

**Waypoint (Safety-Aware Hierarchical LLM for Autonomous UAV Mission Planning
and Adaptation) compared with the six published systems that match it most
closely, and the research directions that comparison points to.**

27 October 2026. Every number for another paper is the one that paper
reports, with a link in the references. Waypoint's numbers come from its own
experiment batches in `experiments/runs/`.

---

## 1. Summary

Six papers come closest to what Waypoint does: an LLM that plans a UAV
mission, with something deterministic between the LLM and the aircraft.

| # | Paper | Why it is close |
| --- | --- | --- |
| 1 | **PEACE** (2026) | Same shape: an LLM planner, a deterministic executor and a constraint layer, on PX4 in Gazebo |
| 2 | **PMR** (2026) | Verified LLM recovery in flight, on PX4 in Gazebo; calls the LLM only when it must |
| 3 | **DroneServer** (2026) | The LLM as an untrusted commander over MAVLink; every command checked against an independent geofence |
| 4 | **SELP** (ICRA 2025) | An LLM planner with a measured safety rate on a drone domain |
| 5 | **AutoTAMP** (ICRA 2024) | The verifier's errors fed back to the LLM until the plan passes |
| 6 | **SAGE-LLM** (2026) | LLM actions for a UAV checked by a control barrier function |

Waypoint combines the central idea of each:
- whole-plan verification before take-off, with the reasons fed back (as in
  AutoTAMP), and with **deterministic repair** before the LLM is asked again;
- a check that the plan does **what the mission asked** (as in SELP), read by
  the LLM and grounded in the world;
- **in-flight replanning that calls the LLM only when repair cannot help**
  (PMR's principle);
- a **control-barrier-function shield** under every command (SAGE-LLM's idea);
- all of it on **PX4 in Gazebo** (as in PEACE, PMR and DroneServer).

None of the six has all of these.

On 36 missions with hand-written ground truth, Waypoint accomplished
**82%** of feasible missions against **56%** for a PEACE/DroneServer-style
runtime-only guard. It had **no crashes** against 12%, and it refused **all
10** impossible missions against none. The full comparison table is in
[section 6](#6-comparison-table).

---

## 2. Waypoint, as it stands

```
mission in English
  -> intent     the LLM reads what the mission asks for; the world grounds it;
                impossible missions are refused here
  -> planner    the LLM decomposes the mission into named subtasks and waypoints
  -> verifier   deterministic, exact checks: geofence, no-fly zones, altitude,
                speed, turns, battery, separation, buildings, terrain, landing,
                and the mission's intent
  -> repair     deterministic fixes (detours, climbs, clamps, added viewpoints);
                the LLM is asked again only if repair fails
  -> flight     custom simulator, or PX4 SITL in Gazebo
  -> shield     a control barrier function on every velocity command
  -> monitor    new obstacle, battery shortfall, off route (symbolic perception,
                or a depth camera on PX4)
  -> replan     repair first (milliseconds), then battery trim, then the LLM
```

Results on 36 missions × 2 repeats, gpt-5-mini, identical faults:

| Metric | Waypoint (full) | Waypoint as first published | Runtime-only guard |
| --- | --- | --- | --- |
| Mission accomplished, ground truth | **82%** (51/62) | 53% (33/62) | 56% (35/62) |
| Crashed | **0%** (0/62) | 1% (1/68) | 12% (9/72) |
| Broke an obstacle keep-out | **0%** (0/62) | 4% (3/68) | 8% (6/72) |
| Impossible missions refused | **10/10** | 0/10 | 0/10 |
| In-flight replan time, median | **0.005 s** | 14.9 s | 14.6 s |

Mission success is significant against both (p = 0.001 and p = 0.003,
Fisher exact). So are crashes against the runtime-only guard (p = 0.004).

---

## 3. A warning on comparing numbers

Each paper measures something different: an LTL-satisfying plan (SELP),
task success in a 2D world (AutoTAMP), reaching a goal within 1 m (PMR),
pursuit-evasion success (SAGE-LLM). The comparison table therefore reports
each paper's own headline metric in its own terms, beside Waypoint's. Read it
as "what each system can do", not as a single ranking.

---

## 4. The six papers

### 4.1 PEACE: Planner-Executor Agent with Constraint Enforcement for UAVs

Uysal, Kehrer, Panichella (University of Bern), arXiv, May 2026.

- **Approach.**
  - One LLM call turns the request, the robot's state and a world model into
    an ordered list of tool calls.
  - A deterministic executor runs them through ROS 2 and MAVLink.
  - A constraint layer clamps altitude to a band and aborts any command that
    leaves a circular geofence; the failure goes back to a bounded replanner.
  - Perception is a YOLO or VLM detector with depth projection.
  - It flies PX4 SITL in Gazebo.
- **Results.** None quantitative; the paper calls itself "not a quantitative
  benchmark". It releases 108 benchmark prompts.
- **Same as Waypoint.** Planner/executor split, deterministic gate, bounded
  replanning, PX4 in Gazebo, depth-based perception.
- **Different from Waypoint.**
  - PEACE checks each command as it is sent, and only altitude and a
    geofence. Waypoint checks the whole plan before take-off against terrain,
    buildings, battery, turns, separation and landing.
  - PEACE replans only when a command fails, and calls dynamic scenes
    "brittle". Waypoint replans on new obstacles, battery and drift.
  - PEACE never checks that the plan does what was asked.
  - Waypoint's `runtime-guard` system reproduces PEACE-style enforcement. It
    accomplished 56% of missions with 12% crashes, against 82% and 0% for
    Waypoint.

### 4.2 PMR: Selective Agentic Recovery with a Persistent Mission Runtime

Park, Yoo, Yoo, Kim (Korea University), arXiv, June 2026.

- **Approach.**
  - A local runtime flies the mission with a PPO policy.
  - A remote LLM is called only for recovery, when a learned gate (an
    18-feature score) predicts it will help.
  - The LLM chooses from a closed set of recovery skills; its choice is
    parsed, verified and safety-shielded before it can act.
- **Results.** 400 Gazebo/PX4 runs over 8 scenarios.
  - In the hard regime: 95% success with the learned gate, 97.5% with a
    rule-based gate, 5% with no LLM.
  - LLM latency about 4.3 s.
  - A Crazyflie demonstration went 10 of 10.
- **Same as Waypoint.** The LLM's output is never trusted until a local
  verifier and shield accept it; the LLM is a recovery tool, not the pilot.
- **Different from Waypoint.**
  - PMR's LLM picks one skill from a list; Waypoint's LLM writes whole new
    routes, which a full verifier then checks.
  - PMR learns when to call the LLM. Waypoint uses fixed triggers but
    resolves them by deterministic repair first: all 43 in-flight replans in
    the batch were fixed in a median 5 ms without calling the LLM.
  - PMR's tasks are short point-to-point navigation, with no terrain, battery
    or multi-phase missions. PMR has hardware; Waypoint does not.

### 4.3 DroneServer: an LLM-agnostic MAVLink command interface over MCP

Ramos Silva and Burke (UC Irvine), arXiv, January 2026 (v3 September 2026).

- **Approach.**
  - 98 MAVSDK-backed tools exposed to any LLM through the Model Context
    Protocol, on ArduPilot and PX4.
  - The LLM is treated as an untrusted commander: every command is checked
    server-side, with confirmation tiers, an independent geofence and an
    audit log.
- **Results.**
  - More than 1,000 simulated flights.
  - 8 of 11 LLMs completed 90 to 100% of six missions.
  - 0 geofence escapes in 110 adversarial trials.
  - 10 of 10 runs on real quadcopters, with models from five providers.
- **Same as Waypoint.** The LLM is untrusted and a deterministic layer stands
  between it and the aircraft; MAVLink and PX4.
- **Different from Waypoint.**
  - DroneServer checks commands one at a time. It cannot tell that a route
    will run out of battery later, cross a building or miss the mission's
    targets; Waypoint checks all of that before take-off.
  - DroneServer leads on breadth: many models, adversarial testing, hardware.

### 4.4 SELP: Safe and Efficient LLM Planner

Wu et al. (Purdue), ICRA 2025, Best Paper finalist.

- **Approach.**
  - The LLM translates the instruction into LTL many times; the largest group
    of logically equivalent translations wins ("equivalence voting").
  - Plans are decoded under the constraint of an automaton built from that
    LTL, so unsafe actions cannot be generated.
  - A 7B planner is fine-tuned on safe, efficient plans.
- **Results.** Drone navigation in PyBullet, a 12-room building, 500 tasks:

  | Method | Safety rate | Completion |
  | --- | --- | --- |
  | GPT-4 | 37.6% | 78.2% |
  | Safety Chip | 84.4% | 91.8% |
  | SELP | 95.2% | 100% |

  Voting raised LTL translation accuracy from 88.4% to 98.0%.
- **Same as Waypoint.** Both check intent (does the plan do what was asked),
  not only safety; Waypoint also offers voting for its intent extraction.
- **Different from Waypoint.**
  - SELP stops unsafe plans during generation and needs a fine-tuned open
    model. Waypoint rejects and repairs after generation, with any API model.
  - SELP's world is rooms with ordering constraints: no terrain, battery,
    faults or in-flight replanning.
  - SELP has richer temporal logic than Waypoint's order and avoid zones.

### 4.5 AutoTAMP: LLMs as translators and checkers

Chen, Arkin, Dawson, Zhang, Roy, Fan (MIT), ICRA 2024.

- **Approach.**
  - GPT-4 translates the instruction into Signal Temporal Logic, and an STL
    planner produces the waypoints.
  - A syntax checker (up to 5 re-prompts) and an LLM-judged semantic check
    (up to 3) feed errors back to the LLM.
- **Results (GPT-4, success rate).**

  | Domain | LLM as planner | Without corrections | With both corrections |
  | --- | --- | --- | --- |
  | HouseWorld2 | 45 to 49% | 42.0% | 82.0% |
  | Chip's Challenge | 0% | 42.7% | 87.7% |
  | Rover (multi-agent) | 0% | 46.0% | 79.0% |

- **Same as Waypoint.** A verifier's errors go back to the LLM until the plan
  passes; this is the origin of Waypoint's correction loop.
- **Different from Waypoint.**
  - AutoTAMP's semantic check is an LLM judging the plan, which is not
    deterministic. Waypoint's intent check is geometry.
  - Waypoint repairs before re-prompting. On 264 plans its verifier
    rejected, repair fixed 97% in a median 5 ms, against 75% for the LLM's
    own corrections.
  - AutoTAMP's worlds are 2D grids, with only a video demo on a drone.

### 4.6 SAGE-LLM: fuzzy-CBF verification for LLM UAV decisions

Zhao et al. (Northwestern Polytechnical University and China Telecom TeleAI),
arXiv, February 2026.

- **Approach.**
  - An LLM proposes the UAV's actions.
  - A fuzzy control barrier function checks each one, modelled as a cone of
    directions, against the nearest obstacle.
  - Safe actions go to an RL controller; unsafe ones to an LLM safety
    controller.
  - Graph-based retrieval supports both LLM modules.
- **Results (Isaac Gym, pursuit-evasion).**

  | Method | Success | Safe rate | Zero-danger |
  | --- | --- | --- | --- |
  | DDPG | 92% | 90.57% | 52% |
  | MSMAR-RL | 64% | 98.29% | 80% |
  | SAGE-LLM | 90% | 99.73% | 92% |

  The paper's tables are partly inconsistent (90% and 80% success both
  appear).
- **Same as Waypoint.** A barrier function stands between the LLM and the
  motion.
- **Different from Waypoint.**
  - SAGE checks single actions; Waypoint checks whole missions first, then
    filters every command at run time.
  - Waypoint's shield is braking-aware, cancels wind and tracks GPS drift.
    Re-flying past plans under their faults, it cut crashes of verified
    plans from 4.1% to 0.6%.
  - SAGE has one task and no mission planning.

---

## 5. Research directions

What the six papers have that Waypoint lacks, and what none of them has done
yet, ordered from cheapest to largest.

| # | Direction | Why | Builds on | Effort |
| --- | --- | --- | --- | --- |
| 1 | **Learn when to call the LLM** | A learned gate cut PMR's LLM calls by up to 87.5% with no loss of success; Waypoint uses fixed triggers | PMR | Small |
| 2 | **Vote on the intent reading** | Voting took SELP's translation accuracy from 88% to 98%; Waypoint reads 83% of missions exactly right in one shot and has voting built but unmeasured | SELP | Small |
| 3 | **Improve the planner from its own logs** | Every verdict and repair is logged; that data can tune the planner prompt or fine-tune a small model so fewer first plans fail (39% are caught today) | Verification-feedback learning | Small to medium |
| 4 | **Richer mission logic** | "Visit twice", "never return to", time windows: SELP's and AutoTAMP's temporal logic expresses them, Waypoint's order and avoid zones do not | SELP, AutoTAMP | Medium |
| 5 | **Correct GPS drift in the controller, not only the shield** | The shield already estimates the drift; feeding it to the controller would stop photographs being taken from a drifted position (2 of Waypoint's 11 misses) | SAGE-LLM | Medium |
| 6 | **Smarter battery trimming** | 8 of the 11 misses were missions trimmed to land safely; keeping the targets that matter most would recover some | Waypoint's own repair | Medium |
| 7 | **Wider evaluation** | More LLMs (DroneServer used 11), more repeats, adversarial missions, and the camera perception across the whole benchmark | DroneServer | Medium |
| 8 | **Hardware flight** | PMR and DroneServer fly real drones; every Waypoint result is simulated. A small quadcopter running the crane scenario would be the first real result | PMR, DroneServer | Large |
| 9 | **Multi-UAV missions** | None of the six plans for several aircraft with the guarantees Waypoint gives one; allocation plus inter-vehicle separation checks would be new | None of the six | Large |
| 10 | **Formal guarantees for the whole loop** | Waypoint's checks are exact geometry and its shield is a barrier function, but nothing proves the composed loop safe; SELP and AutoTAMP give proofs for simpler worlds | SELP, SAGE-LLM | Large |

Directions 1 to 3 need no new hardware and reuse data Waypoint already
records; they are the quickest to a measurable gain. Directions 8 to 10 are
the ones that would most change what can be claimed in a paper.

---

## 6. Comparison table

✓ = yes, ◐ = partly, ✗ = no, ? = not stated.

| | **Waypoint (ours)** | PEACE | PMR | DroneServer | SELP | AutoTAMP | SAGE-LLM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Year, venue | 2026 | arXiv 2026 | arXiv 2026 | arXiv 2026 | ICRA 2025 | ICRA 2024 | arXiv 2026 |
| Domain | UAV missions in real 3D places | UAV commands | UAV navigation recovery | UAV command and control | Drone room navigation (2D-like) | 2D tasks, multi-agent | UAV pursuit-evasion |
| What the LLM does | Reads intent, plans the whole mission, replans when repair cannot | Plans a sequence of tool calls | Picks a recovery skill | Calls flight tools | Translates to LTL and plans | Translates to STL | Proposes actions |
| Natural-language missions | ✓ | ✓ | ◐ (goal given) | ✓ | ✓ | ✓ | ◐ |
| Checks the plan before flight | ✓ whole plan, exact geometry | ◐ each command | ✓ each decision | ◐ each command | ✓ during decoding | ✓ STL | ✓ each action |
| Checks it does what was asked | ✓ deterministic | ✗ | ✗ | ✗ | ✓ | ✓ (LLM-judged) | ✗ |
| Refuses impossible missions | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ | ✗ |
| 3D terrain and buildings | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Battery feasibility | ✓ | ✗ | ◐ (below 20%) | ? | ✗ | ✗ | ✗ |
| Fixes a rejected plan without the LLM | ✓ (97%, 5 ms) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Errors fed back to the LLM | ✓ by subtask | ◐ failure only | ✓ | ? | ✗ | ✓ | ✗ |
| Replans in flight | ✓ | ◐ on command failure | ✓ | ✗ | ✗ | ✗ | ◐ |
| Replan without the LLM | ✓ (5 ms) | ✗ | ✓ (local policy) | ✗ | ✗ | ✗ | ◐ |
| Runtime safety filter | ✓ barrier function | ◐ clamp | ✓ shield | ◐ geofence | ✗ | ✗ | ✓ barrier function |
| Handles wind, faults, GPS drift | ✓ | ✗ | ◐ | ◐ link loss | ✗ | ✗ | ✗ |
| Real perception | ◐ depth camera in Gazebo | ✓ YOLO/VLM | ◐ summaries | ? | ✗ | ✗ | ✗ |
| Prompt-injection hardening | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Works with closed API LLMs | ✓ | ✓ | ✓ | ✓ | ✗ (fine-tuned 7B) | ✓ | ✓ |
| Simulator / autopilot | Custom sim and PX4 SITL in Gazebo | PX4 SITL, Gazebo | PX4, Gazebo | ArduPilot and PX4 SITL | PyBullet | Custom 2D | Isaac Gym |
| Real hardware | ✗ | ✗ | ✓ Crazyflie | ✓ 3 quadcopters | ✗ | ✗ | ✗ |
| Evaluation size | 216 flights × paired baselines | 0 (qualitative) | 400 runs | >1,000 flights | 500 tasks | ~100 to 200 per domain | ~20 per setting |
| Headline result | 82% missions accomplished, 0% crashes, 10/10 impossible refused | None reported | 95% hard-regime success | 0/110 geofence escapes; 10/10 on hardware | 95.2% safety, 100% completion | 79 to 100% success | 99.73% safe rate |
| Main gap | No hardware; single UAV; one LLM tested | No numbers; geofence-only checks | Closed skill set; simple tasks | Per-command checks only | No replanning; 2D rooms; fine-tuned model | 2D; LLM-judged semantics | One task; no mission planning |

---

## References

1. E. Uysal, T. Kehrer, S. Panichella. *PEACE: A Planner-Executor Agent with
   Constraint Enforcement for UAVs.* arXiv, May 2026.
   [arXiv:2606.00104](https://arxiv.org/abs/2606.00104)
2. T. Park, K. Yoo, S. Yoo, H. Kim. *Selective Agentic Recovery for UAV
   Autonomy with a Persistent Mission Runtime.* arXiv, June 2026.
   [arXiv:2606.14219](https://arxiv.org/abs/2606.14219)
3. J. N. Ramos Silva, P. J. Burke. *An LLM-Agnostic, MAVLink-Based Drone
   Command and Control Interface and Agentic Harness Using the Model Context
   Protocol.* arXiv, January 2026. [arXiv:2601.15486](https://arxiv.org/abs/2601.15486)
4. Y. Wu, Z. Xiong, Y. Hu, S. S. Iyengar, N. Jiang, A. Bera, L. Tan,
   S. Jagannathan. *SELP: Generating Safe and Efficient Task Plans for Robot
   Agents with Large Language Models.* ICRA 2025.
   [arXiv:2409.19471](https://arxiv.org/abs/2409.19471)
5. Y. Chen, J. Arkin, C. Dawson, Y. Zhang, N. Roy, C. Fan. *AutoTAMP:
   Autoregressive Task and Motion Planning with LLMs as Translators and
   Checkers.* ICRA 2024. [arXiv:2306.06531](https://arxiv.org/abs/2306.06531)
6. W. Zhao, Y. Zhao, G. Liu, Z. Jiang, D. Ma, Z. Li, X. Li. *SAGE-LLM: Towards
   Safe and Generalizable LLM Controller with Fuzzy-CBF Verification and
   Graph-Structured Knowledge Retrieval for UAV Decision.* arXiv, February
   2026. [arXiv:2602.23719](https://arxiv.org/abs/2602.23719)

Waypoint's numbers can be reproduced from its result directories with
`python -m waypoint.experiments report <dir> --missions experiments/missions.json`:
`20261023T071716-compare2` (full) and `20261023T085101-compare2-baselines`
(as first published, and the runtime-only guard).
