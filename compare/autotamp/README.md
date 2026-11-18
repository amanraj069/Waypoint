# AutoTAMP, rebuilt

**AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators
and Checkers.** Chen, Arkin, Dawson, Zhang, Roy, Fan (MIT). ICRA 2024,
arXiv:2306.06531.

Rebuilt from the paper and from the authors' release:
`github.com/yongchao98/AutoTAMP`, MIT, commit `71e5000` (26 September 2024).

## What AutoTAMP is

An LLM does not plan. It translates the instruction into Signal Temporal Logic
(STL), and a formal STL planner turns the formula into a trajectory of timed
waypoints. Two re-prompting loops repair the translation: a rules-based syntax
checker (up to five rounds), and a semantic check in which the LLM is shown
the planner's state sequence and asked whether it matches the instruction (up
to three rounds).

## Element by element

| Paper | Release | Here | Match |
| --- | --- | --- | --- |
| III-C: few-shot translation, instruction and scene to lifted sentence and propositions | `part1` in `env_and_optimize_single_agent.py` | `data/prompts.json`, `prompts.py` | Same text, extracted from the source without retyping; SHA-256 checked by a test |
| III-C: lifted sentence to pre-order STL | `GPT_NL2TL_preorder` in `openai_func.py` | same | Same text, hash checked |
| The chat calls | `GPT_response_first_round`, `_second_round` | `pipeline.py` | Same messages, including the system message "You are a helpful assistant." |
| III-C: syntax checker and its feedback, up to 5 rounds | `check_syntactic_correct`, `func_syntactic_correct_loop` | `syntax.py`, `pipeline.py` | Same rule, same wording |
| III-C: semantic check on the planner's state sequence, up to 3 rounds | inline in `NL2action` | `prompts.py`, `pipeline.py` | Same prompt, same stopping rule (no change in the STL) |
| STL list to planner tree | `test()` | `pipeline.build_spec` | Same parser, token for token |
| III-C: the STL planner of Sun et al. 2022 | `PWLPlan.py` | `stl/pwl_plan.py` | Same encoding and constants (below) |
| IV: HouseWorld, Chip's Challenge layouts | `NL2action` | `envs.py` | Same boxes, same room numbering, same scene text |
| IV: 10 instructions x 10 paraphrases, 2 start and end pairs | `experiment_result/HouseWorld/instr_dir` | `data/instructions/`, `experiment.py` | The released files; starts and ends drawn from the released ranges |
| Table I rows: No Corrections, Syntax, Syntax + Semantics | two switches | `experiment.py --methods` | Same switches |

The planner keeps every encoding of the release (`mu`, `negmu`, `always`,
`eventually`, `until`, `release`), `M = 1000`, `EPS = 0.01`, the fixed 0.1
shrink of an `enter` region, the L1 velocity limit, the final-time objective,
20 segments with a hard end goal for HouseWorld, 30 segments and the first
trajectory found for Chip's Challenge.

## What is different, and why

| Difference | Reason |
| --- | --- |
| The solver is HiGHS (through SciPy), not Gurobi | Gurobi needs a licence. The model is the same mixed-integer program. Gurobi's `abs_` constraint becomes two inequalities where the absolute value is only bounded from above, which is exact |
| The planner's time limit is a setting, 60 s by default in the experiment | The release allows 2000 s. HiGHS usually finds the trajectory in seconds and then spends the rest proving it optimal; a solve that runs out returns the best trajectory it has, as the release's does. Every planner call's status and time are recorded |
| The LLM is whichever model Waypoint is configured for | The paper used GPT-4 and GPT-3 (text-davinci-003) |
| The model's replies are read leniently (`lenient_replies`, on by default) | The release evaluates the whole STL reply as a Python literal and splits the translation at its first colon. GPT-4 in 2023 replied with nothing else; a current model often adds "STL:" or a sentence, and in a first run here every case was lost at that point, before any checking or planning. The lenient reader takes the first bracketed list, and the translation from its last "transformed sentence". `--strict-replies` restores the release's reading |
| Start and end positions are drawn from a seeded generator | The release's are unseeded, so its cases cannot be repeated exactly |
| A token list the parser cannot reduce raises an error | The release would loop for ever |
| `checkers.py`, `reference.py` | The paper's "hard-coded checkers" and HouseWorld2's time limits are not in the release (below) |

## What the release does not contain

* **The success checkers.** `checkers.py` is this rebuild's reading of the ten
  instruction families, written before any model was run. Task success
  numbers here are under these checkers.
* **HouseWorld2's time limits.** The paper: "pre-determined by completing the
  correct trajectory with 0.8 maximum velocity". `reference.py` writes each
  family's meaning as a specification tree, plans its time-optimal trajectory,
  and divides its duration by 0.8. The solver found a reference trajectory for
  17 of the 20 family and start pairs (10 minutes each); the three without one
  (both starts of family 8, one of family 10) are scored without a time limit,
  and the experiment's summary counts them.
* **The multi-agent domains** (Overcooked, Rover, Wall) and the baselines
  (LLM end-to-end, SayCan, task planning with feedback, NL2TL). Only the
  end-to-end prompt is in the release; it is kept in `data/prompts.json` and
  not run. The planner's multi-agent clearance encoding is ported but
  untested.

## What reading the release showed

Kept as released by default, covered by tests in
`compare/tests/test_autotamp.py`.

* **Every `enter(room)` is planned as "eventually enter(room)".** The parser
  wraps an `enter` in "eventually" unless the previous token starts with both
  `glob` and `fina`, which no token does. The consequences:
  * `until not_enter(A) enter(B)`, the form the prompt's own examples use for
    "B before A", becomes "A is not entered until (eventually B)". That holds
    from the first instant if B is ever entered, so **the order is not
    enforced**. A test plans "do not enter the red restroom until the kitchen"
    and the trajectory crosses the restroom first.
  * `finally globally [0,5] enter(B)`, the prompt's form for "stay 5 seconds",
    becomes "for 5 seconds, B is eventually entered". The trajectory waits
    4.6 s elsewhere and enters B at the end. **The stay is not enforced.**

  `--enter-semantics strict` is this rebuild's alternative reading, in which
  an `enter` under `globally`, `finally` or `until` is the bare predicate. It
  is not the release's behaviour; it is there to measure what the wrapping
  costs.
* **Chip's Challenge layout `env2`, the one the release's script runs, has no
  solution.** `python -m compare.autotamp.chip` asks the planner for a
  trajectory that only reaches every goal and avoids every wall, with no key
  or door constraint; any translation of the task asks for at least that.
  For `env1` HiGHS finds one at once. For `env2` and `env3` it finds none in
  600 s, and a second solver (SCIP, run once for this check and not a
  dependency here) proves in 15 s and 20 s that none exists: with the walls
  inflated by the release's 0.18 + 0.22 the passages close. The trajectories
  stored in the release for this domain are
  all the "no solution" placeholder. The paper's 87.7% on Chip's Challenge
  cannot be reproduced from the release as it stands, so the recorded run is
  on HouseWorld only.

## Run it

```bash
.venv/bin/python -m compare.autotamp.experiment --paraphrases 1 --variants a    # 10 cases x 3 methods
.venv/bin/python -m compare.autotamp.experiment                                  # all 200 cases
.venv/bin/python -m compare.autotamp.experiment --methods autotamp --enter-semantics strict
.venv/bin/python -m compare.autotamp.experiment report compare/results/autotamp/<dir>
.venv/bin/python -m compare.autotamp.chip                                        # the Chip's Challenge probe
.venv/bin/python -m compare.autotamp.reference --only-missing --time-limit 900   # HouseWorld2 time limits
.venv/bin/python -m pytest compare/tests/test_autotamp.py
```

```python
from compare.common.llm import default_client
from compare.autotamp import nl_to_trajectory

outcome = nl_to_trajectory(
    default_client(), "Go to the green bedroom, then the yellow restroom.",
    start_position=[1.3, -1.3], end_position=[0.0, 1.3], domain="HouseWorld",
)
outcome.stl          # the final STL, a pre-order token list
outcome.trajectory   # [[x, y, t], ...], or None
```
