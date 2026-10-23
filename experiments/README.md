# Experiments

The harness flies batches of missions through the full closed loop (plan,
verify, fly, replan) and turns what happened into numbers: how often the
verifier catches the planner, how in-flight replans resolve and how long they
take, and what safety costs in path length and time. Ablation flags rerun the
same batch with a flat planner or without the verifier's reasons in the replan
prompt, so the conditions can be compared.

The code is `waypoint/experiments/`; this directory holds the batch file and,
under `runs/` (gitignored), the results.

## Setup

Everything in the main README's setup, plus matplotlib for the plots:

```bash
uv pip install -e ".[dev,experiments]"      # or: .venv/bin/pip install -e ".[dev,experiments]"
cp .env.example .env                         # an OpenAI and/or Gemini key: every mission is planned from scratch
```

Without matplotlib the CSVs and the summary are still written; the plots are
skipped with a note.

## Run

```bash
# the 36 missions in experiments/missions.json, the full system
python -m waypoint.experiments run

# the ablations, one at a time
python -m waypoint.experiments run --flat                    # no decomposition into named phases
python -m waypoint.experiments run --no-violation-feedback   # rejected, but not told why

# or all four combinations in one result directory
python -m waypoint.experiments run --all-conditions

# smaller, for a first look
python -m waypoint.experiments run --limit 5
python -m waypoint.experiments run --only lodhi_bara_gumbad,open_north_field
```

Each run writes `experiments/runs/<stamp>[-<name>]/` and prints the summary when
it finishes. Recompute the metrics and plots from saved files at any time, or
compare separate runs (one per condition, say) in one report:

```bash
python -m waypoint.experiments report experiments/runs/<stamp>
python -m waypoint.experiments report experiments/runs/<a> experiments/runs/<b> --out experiments/runs/compare
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--missions` | `experiments/missions.json` | the batch file |
| `--backend` | `custom` | which `ExecutorBackend` flies it: `custom`, or `px4` for PX4 SITL in Gazebo (one flight at a time, about 20 s to start each; `WAYPOINT_PX4_SPEED=4` runs it faster than real time; see the main README) |
| `--system` | `full` | the loop's capabilities: `full`, `classic` (as first published), `runtime-guard`, `no-gate`, or a preset with switches (`full,no-repair`, `full,replan=llm`); see the main README's "The system" |
| `--systems` | | several systems separated by `;`, each flown under every planner condition, e.g. `"full;classic;runtime-guard"` |
| `--flat` | off | ablation: one flat route instead of named subtasks |
| `--no-violation-feedback` | off | ablation: the replan prompt says a plan was rejected, not why |
| `--all-conditions` | off | all four combinations of the two ablations |
| `--conditions` | | a subset by label, e.g. `hierarchical+feedback,hierarchical+no-feedback` for the feedback ablation alone |
| `--seed` | 0 | seeds the randomized fault schedules |
| `--repeats` | 1 | fly each mission this many times per condition (the LLM is not deterministic) |
| `--no-baseline` | off | skip the unconstrained comparison flights |
| `--no-baseline-replans` | off | the baseline flies its first plan without replanning in flight: no planner calls once airborne |
| `--only`, `--limit` | all | a subset of the batch |
| `--max-attempts`, `--max-replans` | 3, 3 | as in `run_full` |
| `--pause` | 0 | seconds between missions, for rate limits |
| `--out`, `--name` | | where the results go |

**Which LLM.** The planner tries OpenAI first when `OPENAI_API_KEY` is set,
then Gemini (see the main README's Planner section). The manifest records the
provider chain and each ladder; a fallback mid-batch prints a warning, so a
batch that should be all one provider can be checked. To compare providers,
run the batch once with only one key set, then once with only the other.

**Cost.** `COST=LOW` plans with gpt-5-nano instead of gpt-5-mini. gpt-5 models
think at `low` effort (`OPENAI_REASONING_EFFORT` changes it). Thinking is
billed as output and is most of a plan's cost, but `low` is the only setting
that planned well and in time: at `minimal` gpt-5-nano completed 38% of
missions against 80% at `low`, and at `medium` it timed out on 31% of calls. Every flight
records the tokens it spent, per model, and the report prints calls, tokens and
which models answered for each condition, so a batch's cost is measured. The
cheapest useful batch flies only the conditions you are comparing
(`--conditions`), without repeats, with `--no-baseline-replans`.

Every mission is planned from scratch, so a batch spends tokens: one
call for the first plan, one per correction after a rejection, and one or more
per in-flight replan, for the verified run and again for the baseline's
in-flight replans. Thirty-six missions under one condition is typically 90 to 180
calls; `--all-conditions` is four times that.

## The batch file

`missions.json` holds thirty-six missions. The first twenty are the original
batch, texts unchanged so old results can be re-scored; sixteen more ask for
order ("first ... then ..."), avoid zones ("never fly over the lake"), named
sides, low passes and overhead shots, and five of the thirty-six cannot be flown
as asked at all (a point outside the fence, a photograph above the ceiling).
The original twenty are five in open space and three or four in
each bundled place (Lodhi Garden, the Central Vista, Chandni Chowk, Munnar).
Some are ordinary; some ask for something the envelope makes hard (a point
1500 m away, a climb to 200 m, a pass through a grove at 8 m, a run below the
rooftops) so the verifier has something to catch. Each mission:

```jsonc
{"id": "lodhi_glass_house_drop",
 "mission": "take off from the Bonsai Park, photograph the Glass House and the rose garden, and land on the east lawn",
 "world": "lodhi",              // optional: open space if absent
 "start": "the Bonsai Park",    // optional: a landmark or "x,y"; the world origin if absent
 "land": "the east lawn",       // optional: back at the start if absent
 "faults": "random",            // none, a schedule in data/faults (crane_ahead, storm, gps_slip), "random", or a path
 "wind": "3,0,0",               // optional, m/s
 "max_altitude": 150}           // optional
```

Pads are surveyed the same way `scripts/explore.py` does it: a point on a slope
or in a building moves to the nearest flat, clear ground.

**Randomized faults** (`"faults": "random"`) are one to three faults of
different kinds: an unmapped obstacle 25 to 50 m ahead on the route, a 4 to
10 m/s gust, a 0.2 to 1.0 %/s battery drain spike, or a 0.3 to 1.2 m/s GPS
drift, firing in the first 40 s. The schedule is drawn from a seed made of
`--seed`, the mission id and the repeat number, so every condition and every
baseline flies the same faults, and the same seed reproduces them exactly. The
schedule each flight used is in the manifest.

### Ground truth

Every mission has an `expect` entry, written by hand and shown to no planner:
the targets (landmark names, or points relative to the origin), how each must
be served (CAPTURE, HOVER or OVERFLY; views, sides, overhead, height), whether
they are ordered, what to avoid, and `feasible` (false for a mission that
cannot be flown as asked, null where that is not certain). It has the same
shape as the intent extractor's output and is grounded the same way
(`waypoint/experiments/truth.py`). `tests/test_experiments.py` checks that every
landmark it names exists and that its `feasible` agrees with `assess()`.

## What is flown

For each condition, mission and repeat, two flights:

1. **Verified**: the closed loop exactly as `python -m waypoint.orchestrator.run_full`
   flies it, with the condition's planner and system.
2. **Baseline**: the *same first plan*, the same faults and the same backend,
   with the verifier replaced by one that runs every check, records what it
   found, and refuses nothing. In-flight replans still go to the planner, and
   are not verified either. The feedback monitor still runs, so the baseline
   still reacts to obstacles; it just has no gate. This exists only to measure
   what the gate costs and what it prevents; `run_mission` never flies this
   way unless the harness asks it to.

## Results

```
experiments/runs/<stamp>/
├── manifest.json            config (missions file hash, seed, planner providers and ladders, git commit, and `git_dirty` if the checkout had uncommitted changes) and one entry per flight pair
├── per_mission.csv          one row per flight pair
├── replans.csv              one row per in-flight replan trigger
├── summary.json             the aggregates, per condition
├── catch_rate.png           verifier catch rate by condition
├── replan_latency.png       trigger-to-verified-plan latency by fault type
├── replan_outcomes.png      how in-flight replans resolved, by condition
├── tradeoff.png             verified vs baseline path length, one panel per condition
└── <condition>/<mission>[/r<k>]/{verified,baseline}/
    ├── <run>.json           the run file (replay it: python scripts/gazebo_view.py <file>)
    └── <run>.log.jsonl      the mission log
```

The manifest is rewritten after every mission, so an interrupted batch can
still be reported on. A mission that cannot be set up (an unknown landmark, a
pad with no flat ground near it) or that raises is recorded with its `error`
and the batch carries on.

## How each metric is computed

**Mission accomplished.** Scored against the ground truth from the run file
alone: the waypoints the aircraft actually reached (with their actions, where
it reached them) and its track. Accomplished means every target served (from
enough distinct sides, from the named sides, from above, at the asked height),
in order when ordered, no avoid zone entered, the last plan ending with a LAND
on the pad, and the flight landed under control. A touchdown blown off the pad
by wind or a drifting fix is reported apart (`touchdown_error_m`), not counted
as a different mission. The mean share of the mission done is reported beside
it. For missions marked infeasible, the score is the refusal: refusing is right,
flying anyway is not. Wilson 95% intervals throughout.

**Intent extraction.** Where the system read an intent, the reading against the
ground truth: exact (same targets, served the same way, same order flag, same
avoid zones) and the target-set F1.

**Repair, shield, hover.** How many rejected plans repair fixed on the ground
and in the air and how fast; replan latency split by who answered (repair or
the LLM); the share of flights where the shield changed a command; the flight
time spent hovering through LLM replans.

Every number comes from the mission logs and run files alone
(`waypoint/experiments/metrics.py`), so `report` reproduces it from a result
directory without flying anything. "Blocking" means a violation of severity
`ERROR` or `CRITICAL`, the ones that fail verification.

**Verifier catch rate.** Missions whose first plan, before any replan, had at
least one blocking violation, over missions that got a first plan at all
(planning failures are counted separately). This is the first `verification`
entry in the verified run's log. Reported beside it: the share with any
finding at all, warnings included; which violation types were caught; and how
often a caught plan was corrected into one that passed, within
`--max-attempts`.

The violation-feedback ablation only changes what the planner is told after a
rejection; the first plan is asked for with the same prompt either way. So a
catch-rate gap between `+feedback` and `+no-feedback` is sampling noise, not an
effect. Each condition's rate carries a 95% Wilson interval, and the report
also pools it over the conditions that share a first-plan prompt
(`first_plan.pooled_catch_rate` in `summary.json`): compare hierarchical with
flat on the pooled number.

**Replan success.** One row per `replan_triggered` entry in a verified run's
log. Its attempts are the `verification` entries between the trigger and the
next `execution_resumed`, `failsafe` or trigger. It passed on the first try,
the second try, a later try, or was **abandoned**: no attempt passed (the
planner failed, or every correction was rejected), or the flight had already
used its `--max-replans` and landed where it was without trying.

**Replan latency.** For each replan that passed, the wall-clock time between
the trigger's log entry and the passing verification's (millisecond
timestamps), so it is planner time plus verification time. It is broken down
by the fault that caused the trigger: the most recent `FAULT` event in the run
file, at or before the trigger, of a kind that can cause it (an
`OBSTACLE_APPEARS` for an obstacle trigger, a `BATTERY_DRAIN_SPIKE` for a
battery one, a `GPS_DRIFT` or `WIND_GUST` for an off-route one). A trigger no
injected fault explains is `NONE`, for example the battery model alone on a
long route.

**Safety vs optimality.** For each mission where the verifier changed the plan
(the first plan was rejected and a corrected one flew) and both the verified
run and the baseline completed, `(verified - baseline) / baseline` for distance flown
(`mission_finished.distance_m`) and duration (`mission_finished.t`), as a
percentage; the median and mean are reported. Missions whose first plan
passed are left out: both flights flew the same plan, and any difference is
only the LLM answering two in-flight replans differently. For every baseline: whether it
completed, whether it crashed, whether it broke an obstacle's keep-out
(`SEPARATION` events in the run file), and whether it flew a plan the verifier
would have refused (a baseline `verification` entry with a blocking
violation). The verified runs' crash and keep-out rates are reported for
comparison.

**Comparing conditions.** Rerun the same batch with a flag flipped (or use
`--all-conditions`) and `report` the result directories together. With the
same `--seed`, every condition flies the same missions under the same faults;
what differs is the planner's prompt. Because the LLM is not deterministic, use
`--repeats` to see how much of a difference is noise.

## Tests

`tests/test_experiments.py` exercises all of this with scripted planners and
no network: the batch file's twenty missions all set up, random schedules are
reproducible, the ablations change the prompts as described, and a small batch
produces exactly the counts, buckets, attributions and files expected.
