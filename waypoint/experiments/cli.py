"""`python -m waypoint.experiments run ...` and `... report ...`.

    # the batch under the planner as designed (needs a Gemini key)
    python -m waypoint.experiments run

    # one ablation at a time, or all four in one go
    python -m waypoint.experiments run --flat
    python -m waypoint.experiments run --no-violation-feedback
    python -m waypoint.experiments run --all-conditions

    # recompute the metrics and plots from saved runs, or compare several
    python -m waypoint.experiments report experiments/runs/<stamp>
    python -m waypoint.experiments report experiments/runs/<a> experiments/runs/<b>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from waypoint.backends import BACKENDS, BackendError, make_backend
from waypoint.experiments.metrics import load
from waypoint.experiments.report import write_report
from waypoint.experiments.runner import EXPERIMENTS_DIR, RESULTS_DIR, BatchConfig, run_batch
from waypoint.experiments.spec import CONDITIONS, condition_from_label
from waypoint.planner import PlannerOptions
from waypoint.planner.config import API_KEY_VAR, OPENAI_KEY_VAR, get_cost, get_providers, get_reasoning_effort

DEFAULT_MISSIONS = EXPERIMENTS_DIR / "missions.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m waypoint.experiments",
        description="Batch experiments over the closed loop, and the metrics they produce.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Fly a batch of missions, then report on it.")
    run.add_argument("--missions", type=Path, default=DEFAULT_MISSIONS, help="The batch file (JSON).")
    run.add_argument("--backend", choices=BACKENDS, default="custom")
    run.add_argument("--flat", action="store_true", help="Ablation: a flat waypoint list instead of named phases.")
    run.add_argument(
        "--no-violation-feedback", action="store_true",
        help="Ablation: tell the planner a plan was rejected, but not why.",
    )
    run.add_argument(
        "--all-conditions", action="store_true",
        help="Fly the batch under all four combinations of the two ablations.",
    )
    run.add_argument(
        "--conditions",
        help="Comma-separated condition labels to fly, e.g. "
             "hierarchical+feedback,hierarchical+no-feedback (the feedback ablation alone).",
    )
    run.add_argument("--seed", type=int, default=0, help="Seeds the randomized fault schedules.")
    run.add_argument("--repeats", type=int, default=1, help="Fly each mission this many times per condition.")
    run.add_argument("--no-baseline", action="store_true", help="Skip the unconstrained comparison flights.")
    run.add_argument(
        "--no-baseline-replans", action="store_true",
        help="The baseline flies its first plan without replanning in flight (no planner calls once airborne).",
    )
    run.add_argument("--only", help="Comma-separated mission ids to fly, out of the batch.")
    run.add_argument("--limit", type=int, help="Fly only the first N missions of the batch.")
    run.add_argument("--max-attempts", type=int, default=3, help="Corrections per verifier rejection.")
    run.add_argument("--max-replans", type=int, default=3, help="In-flight replans before aborting.")
    run.add_argument("--pause", type=float, default=0.0, help="Seconds between missions (rate limits).")
    run.add_argument("--out", type=Path, help="Result directory (default experiments/runs/<stamp>[-name]).")
    run.add_argument("--name", default="", help="Suffix for the default result directory.")
    run.add_argument("--no-plots", action="store_true")

    report = commands.add_parser("report", help="Metrics, CSVs and plots from one or more result directories.")
    report.add_argument("results", type=Path, nargs="+")
    report.add_argument("--out", type=Path, help="Where to write (default: the result directory, or a new one).")
    report.add_argument("--no-plots", action="store_true")

    args = parser.parse_args(argv)
    try:
        return _run(args) if args.command == "run" else _report(args)
    except (BackendError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args) -> int:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if sum(map(bool, (args.all_conditions, args.conditions, args.flat or args.no_violation_feedback))) > 1:
        raise ValueError("choose one of --all-conditions, --conditions, or --flat/--no-violation-feedback")
    if args.all_conditions:
        conditions = list(CONDITIONS)
    elif args.conditions:
        conditions = [condition_from_label(c.strip()) for c in args.conditions.split(",") if c.strip()]
    else:
        conditions = [PlannerOptions(hierarchical=not args.flat, violation_feedback=not args.no_violation_feedback)]
    make_backend(args.backend)  # an unbuilt backend fails here, not twenty times over
    get_cost(), get_reasoning_effort()  # a mistyped COST or OPENAI_REASONING_EFFORT fails here too
    if not get_providers():
        raise ValueError(
            f"neither {OPENAI_KEY_VAR} nor {API_KEY_VAR} is set; the batch plans every mission "
            "with an LLM (see .env.example)"
        )

    config = BatchConfig(
        missions_path=args.missions,
        conditions=conditions,
        backend=args.backend,
        seed=args.seed,
        repeats=args.repeats,
        baseline=not args.no_baseline,
        baseline_replans=not args.no_baseline_replans,
        only=[m.strip() for m in args.only.split(",") if m.strip()] if args.only else None,
        limit=args.limit,
        max_replan_attempts=args.max_attempts,
        max_total_replans=args.max_replans,
        pause=args.pause,
        out_dir=args.out,
        name=args.name,
    )
    out = run_batch(config)
    print(f"\nResults: {out}\n")
    rows, replans, _ = load(out)
    write_report(rows, replans, out, plots=not args.no_plots)
    return 0


def _report(args) -> int:
    rows, replans = [], []
    for directory in args.results:
        more_rows, more_replans, _ = load(directory)
        rows += more_rows
        replans += more_replans
    if args.out:
        out = args.out
    elif len(args.results) == 1:
        out = args.results[0]
    else:
        out = RESULTS_DIR / f"compare-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    write_report(rows, replans, out, plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
