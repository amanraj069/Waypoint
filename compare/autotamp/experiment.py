"""AutoTAMP's HouseWorld experiment: the paper's Table I rows for "LLMs as Translators".

    python -m compare.autotamp.experiment --paraphrases 2            # a first look: 40 cases per method
    python -m compare.autotamp.experiment                             # all 200 cases, three methods
    python -m compare.autotamp.experiment --methods autotamp --enter-semantics strict
    python -m compare.autotamp.experiment report compare/results/autotamp/<stamp>

The test cases are the release's: ten instruction families, ten paraphrases of
each (`data/instructions/houseworld/`), two start and end pairs per
instruction drawn from the release's ranges, 200 cases in all. HouseWorld1
scores a case by its family's checker; HouseWorld2 also requires the
trajectory to finish inside the case's time limit (`reference.py`). One run of
a case gives both scores, as the trajectory is the same.

Three methods, the paper's ablation: `no-corrections` (translation only),
`syntax` (with syntactic re-prompting), `autotamp` (syntactic and semantic).

Every case's record, with the full LLM transcript and each planner call, is
appended to `cases.jsonl` as it finishes, so an interrupted run resumes where
it stopped (`--out` the same directory).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from compare.autotamp.checkers import check_houseworld, within_time
from compare.autotamp.pipeline import ENTER_SEMANTICS, nl_to_trajectory
from compare.autotamp.reference import time_limits, variant_of

DATA = Path(__file__).parent / "data" / "instructions"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "autotamp"
METHODS: dict[str, tuple[bool, bool]] = {
    "no-corrections": (False, False),
    "syntax": (True, False),
    "autotamp": (True, True),
}
#: Table I of the paper, task success in percent: (HouseWorld1, HouseWorld2).
PAPER = {
    "GPT-4": {"no-corrections": (43.5, 42.0), "syntax": (59.5, 59.0), "autotamp": (82.5, 82.0)},
    "GPT-3": {"no-corrections": (28.0, 27.0), "syntax": (49.0, 47.0), "autotamp": (62.0, 62.0)},
}


def instructions(family: int) -> list[str]:
    """The ten paraphrases of a family, split as the release splits its instruction files."""
    text = (DATA / "houseworld" / f"sent{family}.txt").read_text(encoding="utf-8")
    return [item for item in text.split("\n\n") if len(item) > 5]


def positions(family: int, variant: str, rng: random.Random) -> tuple[list[float], list[float]]:
    """A start and an end drawn from the release's ranges for this family and variant."""
    if family == 9:
        return (
            [-1.35 + rng.uniform(-0.1, 0.1), -1.1 + rng.uniform(-0.1, 0.1)],
            [1.15 + rng.uniform(-0.1, 0.1), 1.15 + rng.uniform(-0.1, 0.1)],
        )
    if variant == "a":
        start = [rng.uniform(1.2, 1.41), rng.uniform(-1.2, -1.41)]
    else:
        start = [rng.uniform(-1.4, -0.6), rng.uniform(-0.1, -0.6)]
    return start, [rng.uniform(-0.2, 0.2), rng.uniform(1.2, 1.41)]


def cases(
    families: list[int] | None = None, paraphrases: int = 10, variants: tuple[str, ...] = ("a", "b"), seed: int = 0
) -> list[dict]:
    """The test cases, each with its instruction, start, end and HouseWorld2 time limit."""
    limits = time_limits()
    out = []
    for family in families or list(range(1, 11)):
        for index, instruction in enumerate(instructions(family)[:paraphrases]):
            for variant in variants:
                rng = random.Random(f"{seed}:{family}:{index}:{variant}")
                start, end = positions(family, variant, rng)
                out.append(
                    {
                        "id": f"f{family:02d}-p{index}-{variant}",
                        "family": family,
                        "paraphrase": index,
                        "variant": variant,
                        "instruction": instruction,
                        "start": [round(v, 4) for v in start],
                        "end": [round(v, 4) for v in end],
                        "time_limit_hw2": limits.get(f"{family}{variant}"),
                    }
                )
    return out


def run_case(case: dict, method: str, llm, *, planner_time_limit: float, enter_semantics: str = "released", lenient_replies: bool = True) -> dict:
    syntactic, semantic = METHODS[method]
    outcome = nl_to_trajectory(
        llm, case["instruction"], start_position=case["start"], end_position=case["end"], domain="HouseWorld",
        syntactic_correct_loop=syntactic, semantic_correct_loop=semantic, time_limit=planner_time_limit,
        enter_semantics=enter_semantics, lenient_replies=lenient_replies,
    )
    correct = check_houseworld(case["family"], outcome.trajectory)
    return {
        **case,
        "method": method,
        "success_hw1": correct,
        "success_hw2": correct and within_time(outcome.trajectory, case["time_limit_hw2"]),
        "duration": None if not outcome.trajectory else round(outcome.trajectory[-1][2], 4),
        "stl": outcome.stl,
        "lifted_stl": outcome.lifted_stl,
        "mark_syntactic": outcome.mark_syntactic,
        "planned": outcome.trajectory is not None,
        "trajectory": None if outcome.trajectory is None else [[round(v, 4) for v in p] for p in outcome.trajectory],
        "syntactic_iterations": outcome.syntactic_iterations,
        "semantic_iterations": outcome.semantic_iterations,
        "llm_calls": outcome.llm_calls,
        "time_total": outcome.time_total,
        "time_planning": round(outcome.time_planning, 3),
        "plans": [{k: v for k, v in p.items() if k != "stl"} for p in outcome.plans],
        "error": outcome.error,
        # The few-shot prompts are fixed and long; only the part that varies is kept with each reply.
        "transcript": [
            {"kind": entry["kind"], "prompt_tail": entry["prompt"][-700:], "reply": entry["reply"]}
            for entry in outcome.transcript
        ],
    }


def _work(job: tuple) -> dict:
    """One case in a worker process, with its own client."""
    case, method, model, planner_time_limit, enter_semantics, lenient_replies = job
    from compare.common.llm import default_client  # noqa: PLC0415

    return run_case(
        case, method, default_client(model), planner_time_limit=planner_time_limit, enter_semantics=enter_semantics,
        lenient_replies=lenient_replies,
    )


def run(
    test_cases: list[dict],
    methods: list[str],
    out_dir: Path,
    *,
    llm=None,
    model: str | None = None,
    workers: int = 1,
    planner_time_limit: float = 60.0,
    enter_semantics: str = "released",
    lenient_replies: bool = True,
    progress=print,
) -> dict:
    """Run every case under every method, appending to `out_dir/cases.jsonl`, and return the summary.

    `llm` runs everything in this process (the tests' scripted model); otherwise
    `workers` processes each build the configured client for `model`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = out_dir / "cases.jsonl"
    done = {(r["id"], r["method"]) for r in load(out_dir)}
    jobs = [(case, method) for method in methods for case in test_cases if (case["id"], method) not in done]
    progress(f"{len(jobs)} case run(s) to do ({len(done)} already in {store.name})")
    began = time.perf_counter()

    def keep(record: dict, count: int) -> None:
        with store.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        progress(
            f"[{count}/{len(jobs)}] {record['method']:14} {record['id']}  "
            f"hw1={'ok ' if record['success_hw1'] else 'no '} hw2={'ok ' if record['success_hw2'] else 'no '} "
            f"calls={record['llm_calls']} plan={record['time_planning']:.0f}s"
            + (f"  ({record['error'][:60]})" if record["error"] else "")
        )

    if llm is not None or workers <= 1:
        if llm is None:
            from compare.common.llm import default_client  # noqa: PLC0415

            llm = default_client(model)
        for count, (case, method) in enumerate(jobs, 1):
            keep(
                run_case(case, method, llm, planner_time_limit=planner_time_limit, enter_semantics=enter_semantics, lenient_replies=lenient_replies),
                count,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_work, (case, method, model, planner_time_limit, enter_semantics, lenient_replies)) for case, method in jobs
            ]
            for count, future in enumerate(as_completed(futures), 1):
                keep(future.result(), count)

    summary = summarise(load(out_dir))
    if llm is None:
        from compare.common.llm import default_client  # noqa: PLC0415

        llm = default_client(model)
    summary["config"] = {
        "model": getattr(llm, "name", None),
        "planner_time_limit": planner_time_limit,
        "enter_semantics": enter_semantics,
        "lenient_replies": lenient_replies,
        "methods": methods,
        "cases": len(test_cases),
        "wall_seconds": round(time.perf_counter() - began, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load(out_dir: Path) -> list[dict]:
    store = Path(out_dir) / "cases.jsonl"
    if not store.exists():
        return []
    return [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(part: int, whole: int, z: float = 1.96) -> tuple[float, float] | None:
    if not whole:
        return None
    p = part / whole
    centre = (p + z * z / (2 * whole)) / (1 + z * z / whole)
    half = z * ((p * (1 - p) / whole + z * z / (4 * whole * whole)) ** 0.5) / (1 + z * z / whole)
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def summarise(records: list[dict]) -> dict:
    summary: dict = {"methods": {}}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        if not rows:
            continue
        n = len(rows)
        entry = {
            "cases": n,
            "cases_without_time_limit": sum(1 for r in rows if r.get("time_limit_hw2") is None),
            "houseworld1": {"success": sum(r["success_hw1"] for r in rows), "rate": round(sum(r["success_hw1"] for r in rows) / n, 4),
                            "wilson95": wilson(sum(r["success_hw1"] for r in rows), n)},
            "houseworld2": {"success": sum(r["success_hw2"] for r in rows), "rate": round(sum(r["success_hw2"] for r in rows) / n, 4),
                            "wilson95": wilson(sum(r["success_hw2"] for r in rows), n)},
            "syntactically_correct": sum(r["mark_syntactic"] for r in rows),
            "planned": sum(r["planned"] for r in rows),
            "pipeline_errors": sum(1 for r in rows if r["error"]),
            "mean_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 2),
            "mean_time_total": round(sum(r["time_total"] for r in rows) / n, 1),
            "mean_time_planning": round(sum(r["time_planning"] for r in rows) / n, 1),
            "by_family": {},
        }
        for family in sorted({r["family"] for r in rows}):
            family_rows = [r for r in rows if r["family"] == family]
            entry["by_family"][str(family)] = {
                "cases": len(family_rows),
                "houseworld1": sum(r["success_hw1"] for r in family_rows),
                "houseworld2": sum(r["success_hw2"] for r in family_rows),
            }
        summary["methods"][method] = entry
    return summary


def format_summary(summary: dict) -> str:
    lines = [
        f"{'method':16} {'cases':>5}  {'HouseWorld1':>20}  {'HouseWorld2':>20}  {'paper GPT-4':>13}  {'planned':>7} {'LLM calls':>9}",
    ]
    for method, entry in summary["methods"].items():
        def cell(key: str) -> str:
            low, high = entry[key]["wilson95"]
            return f"{entry[key]['rate']:6.1%} [{low:.0%}, {high:.0%}]"

        paper = PAPER["GPT-4"][method]
        lines.append(
            f"{method:16} {entry['cases']:>5}  {cell('houseworld1'):>20}  {cell('houseworld2'):>20}  "
            f"{paper[0]:>5.1f}/{paper[1]:<5.1f}  {entry['planned']:>7} {entry['mean_llm_calls']:>9}"
        )
    lines.append("")
    families = sorted({f for entry in summary["methods"].values() for f in entry["by_family"]}, key=int)
    lines.append("successes by instruction family (HouseWorld1 / HouseWorld2 of cases):")
    lines.append(f"{'method':16} " + " ".join(f"{'f' + f:>9}" for f in families))
    for method, entry in summary["methods"].items():
        cells = []
        for f in families:
            row = entry["by_family"].get(f)
            cells.append(f"{row['houseworld1']}/{row['houseworld2']} of {row['cases']}" if row else "-")
        lines.append(f"{method:16} " + " ".join(f"{c:>9}" for c in cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import sys  # noqa: PLC0415

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "report":
        for directory in argv[1:]:
            print(directory)
            print(format_summary(summarise(load(Path(directory)))))
        return

    parser = argparse.ArgumentParser(description="AutoTAMP on HouseWorld: the paper's translation rows of Table I")
    parser.add_argument("--families", type=str, default=None, help="comma separated, e.g. 1,3,4 (default: all ten)")
    parser.add_argument("--paraphrases", type=int, default=10, help="paraphrases per family, 1 to 10")
    parser.add_argument("--variants", type=str, default="a,b", help="start/end variants: a, b, or a,b")
    parser.add_argument("--methods", type=str, default=",".join(METHODS), help="comma separated: " + ", ".join(METHODS))
    parser.add_argument("--model", type=str, default=None, help="e.g. gpt-5-mini (default: the configured ladder)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--planner-time-limit", type=float, default=60.0, help="seconds per planner call (the release: 2000)")
    parser.add_argument("--enter-semantics", choices=ENTER_SEMANTICS, default="released")
    parser.add_argument("--strict-replies", action="store_true", help="read the model's replies exactly as the release does")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in METHODS:
            parser.error(f"unknown method {method!r}")
    families = [int(f) for f in args.families.split(",")] if args.families else None
    test_cases = cases(families, args.paraphrases, tuple(v.strip() for v in args.variants.split(",")), args.seed)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path(args.out) if args.out else RESULTS / (f"{stamp}-{args.name}" if args.name else stamp)
    print(f"{len(test_cases)} case(s) x {len(methods)} method(s), results in {out}")
    summary = run(
        test_cases, methods, out, model=args.model, workers=args.workers,
        planner_time_limit=args.planner_time_limit, enter_semantics=args.enter_semantics,
        lenient_replies=not args.strict_replies,
    )
    print()
    print(format_summary(summary))


if __name__ == "__main__":
    main()
