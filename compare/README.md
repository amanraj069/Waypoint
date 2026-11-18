# compare: three related systems, rebuilt

Three of the six systems in [RELATED_WORK.md](../RELATED_WORK.md) are rebuilt
here from their papers and their authors' released code, so that what each one
does can be run, read and checked next to Waypoint.

| | Paper | What it contributes | Rebuilt in |
| --- | --- | --- | --- |
| 1 | **PEACE** (Uysal, Kehrer, Panichella; arXiv:2606.00104, 2026) | A single-pass LLM planner, a deterministic executor, and a constraint layer on every motion command | [peace/](peace/README.md) |
| 2 | **DroneServer** (Ramos Silva, Burke; arXiv:2601.15486, 2026) | The LLM as an untrusted commander: 98 MCP tools behind a server-side safety pipeline | [droneserver/](droneserver/README.md) |
| 3 | **AutoTAMP** (Chen et al.; ICRA 2024, arXiv:2306.06531) | The LLM as translator to temporal logic and as checker, with a formal planner | [autotamp/](autotamp/README.md) |

Each system's README lists, element by element, what the paper describes, where
it is in the release, where it is here, and how closely it matches; then every
difference and its reason; then what was not rebuilt.

## Why these three

They are the three of the six whose authors published their code under a
permissive licence (all MIT), so a rebuild can be checked against the source
line for line and not only against a paper's prose. That matters most for
PEACE, whose paper names no limit, tool or prompt.

The other three were left out for a stated reason each:

* **PMR** publishes no code, and its paper withholds the pieces a rebuild
  needs: the recovery-skill labels and the 18 gate features are given only in
  "sanitized" form, and the PPO policy and the training logs are not released.
* **SAGE-LLM** publishes no code and depends on a trained RL controller and a
  graph-retrieval corpus in Isaac Gym.
* **SELP** needs its fine-tuned 7B model and constrained decoding over the
  model's logits, which an API model does not expose.

## What "rebuilt" means here

Four levels, named in each README's tables:

* **Vendored unchanged**: the released file, with a header and import paths.
  DroneServer's safety rules are this (about 1,900 lines).
* **Same**: ported by hand, call for call, constant for constant, string for
  string. PEACE's schemas, services, tools and control flow; AutoTAMP's
  prompts, checker, parser and planner encoding.
* **Replaced**: what could not come along, replaced by the nearest thing that
  could, and said so. ROS 2, MAVROS and MavSDK become
  [common/flight.py](common/flight.py), a flight stack with PX4's offboard
  semantics on Waypoint's simulator; Gurobi becomes HiGHS; each paper's LLM
  becomes whichever model Waypoint is configured for.
* **Not rebuilt**: listed, with the reason.

No rebuild changes a system's behaviour to make it look better or worse.
Where the release behaves in a way its paper does not describe, the release is
followed and the behaviour is written down and covered by a test.

## Layout

```
compare/
├── common/
│   ├── llm.py          one chat interface (OpenAI, Gemini, Ollama; a scripted model for tests)
│   ├── flight.py       PX4 as an offboard client sees it, on Waypoint's ExecutorBackend
│   └── sensors.py      a planar LiDAR and a pinhole RGB-D camera, from the world's geometry
├── peace/              PEACE
├── droneserver/        DroneServer
├── autotamp/           AutoTAMP
├── versus/             the head-to-head: shared scenarios, one runner per system, one grading function, the report
├── tests/              96 tests, no network: every test hands the system a scripted model
├── results/            the recorded runs quoted below
└── requirements.txt    scipy (HiGHS) and mcp, beyond Waypoint's own dependencies
```

## Setup and tests

```bash
uv pip install -r compare/requirements.txt       # or: .venv/bin/pip install -r compare/requirements.txt
.venv/bin/python -m pytest compare/tests         # 96 tests, about two minutes (the planner tests solve real programs)
```

Run from the repository root. The LLM is the one in `.env` (OpenAI first, then
Gemini), as for the rest of Waypoint; `--model` on each runner picks another.

## Run each system on its own evaluation

```bash
# PEACE: the 108 published prompts
.venv/bin/python -m compare.peace.evaluation.run_dataset

# DroneServer: the 29 adversarial cases (no LLM), then the mission suite
.venv/bin/python -m compare.droneserver.adversarial
.venv/bin/python -m compare.droneserver.benchmark --trials 5

# AutoTAMP: HouseWorld, the paper's three translation rows
.venv/bin/python -m compare.autotamp.experiment
```

## Head-to-head on Waypoint's missions

`python -m compare.versus run` flies Waypoint, PEACE and DroneServer on the 36
missions of `experiments/missions.json`: same mission text, world, pads, wind,
fault schedule, airframe and LLM, graded by one function against the same
hand-written ground truth. The full results, with the method, the caveats and
each system's score on its own paper's evaluation, are in
**[Comp_res.md](Comp_res.md)**. The headline, one flight per mission, gpt-5-nano:

| | Waypoint | PEACE (rebuilt) | DroneServer (rebuilt) |
| --- | --- | --- | --- |
| Missions accomplished | 61% (19/31) | 39% (12/31) | 0% (0/31) |
| Crashed | 0/36 | 7/36 | 13/36 |
| Impossible missions not flown | 5/5 | 2/5 | 3/5 |

AutoTAMP plans a point in a 2D house and has no photograph, altitude, terrain
or battery, so it is not flown there; its own HouseWorld results are in
`Comp_res.md`.


## Sources

| System | Paper | Code | Licence | Commit |
| --- | --- | --- | --- | --- |
| PEACE | [arXiv:2606.00104](https://arxiv.org/abs/2606.00104) | github.com/erdemuysalx/PEACE, dataset erdemuysalx/uav-mission-prompts | MIT (declared in its `pyproject.toml`; no licence file) | `4ceda37` |
| DroneServer | [arXiv:2601.15486](https://arxiv.org/abs/2601.15486) | github.com/PeterJBurke/droneserver | MIT (`droneserver/LICENSE.upstream`) | `06b7d1a` |
| AutoTAMP | [arXiv:2306.06531](https://arxiv.org/abs/2306.06531) | github.com/yongchao98/AutoTAMP | MIT | `71e5000` |
