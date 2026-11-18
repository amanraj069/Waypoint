"""Fly Waypoint's missions with every system, and write the comparison.

    python -m compare.versus run                              # 36 missions x 3 systems
    python -m compare.versus run --limit 3 --systems peace    # a first look
    python -m compare.versus run --model gpt-5-mini           # pin one model for every system
    python -m compare.versus report compare/results/versus/<dir>

A run can be interrupted and started again with the same `--out`: flights that
are already there are not flown twice. `report` writes `compare/Comp_res.md`.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "versus"


def run(args) -> Path:
    from compare.versus.batch import work  # noqa: PLC0415 -- in a module of its own, so worker processes can import it
    from compare.versus.fly import SYSTEMS  # noqa: PLC0415
    from compare.versus.scenario import missions  # noqa: PLC0415
    from waypoint.planner import config  # noqa: PLC0415

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    for system in systems:
        if system not in SYSTEMS:
            raise SystemExit(f"unknown system {system!r}; one of {', '.join(SYSTEMS)}")
    if args.model:
        # One rung, for every system: Waypoint's planner and the rebuilt systems read the same ladder.
        os.environ["OPENAI_MODELS"] = args.model
    specs = [m for m in missions() if not args.only or m.id in args.only.split(",")]
    if args.limit:
        specs = specs[: args.limit]
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path(args.out) if args.out else RESULTS / (f"{stamp}-{args.name}" if args.name else stamp)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "systems": systems, "missions": [m.id for m in specs], "seed": args.seed,
        "providers": list(config.get_providers()), "planner": config.describe_providers(),
        "model_ladder": list(config.get_openai_model_ladder()), "reasoning_effort": config.get_reasoning_effort(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    jobs = [
        (system, spec.id, args.seed, str(out))
        for spec in specs
        for system in systems
        if not (out / system / f"{spec.id}.json").exists()
    ]
    print(f"{len(specs)} mission(s) x {len(systems)} system(s): {len(jobs)} flight(s) to fly, results in {out}", flush=True)
    print(f"LLM: {manifest['planner']}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            g = record.get("grade") or {}
            verdict = (
                f"ERROR {record['error']}" if record["error"]
                else "accomplished" if g.get("accomplished")
                else ("refused" if g.get("refused") else "flew it anyway") if g.get("expected_feasible") is False
                else f"not accomplished ({g.get('why_not')})"
            )
            print(f"[{count}/{len(jobs)}] {record['system']:11} {record['mission_id']:28} {verdict}"
                  + ("  CRASHED" if g.get("crashed") else ""), flush=True)
    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def regrade(directory: Path) -> dict[str, int]:
    """Score the stored flights again, with no flying and no LLM.

    For the rebuilt systems the LAND that scoring reads is put where the
    aircraft believed it touched down (the last tick's GPS position), as
    `fly.run_of` now records it; Waypoint's run files already carry their
    planned LAND. Each record's grade is rewritten; the run files are not.
    """
    import gzip  # noqa: PLC0415

    from compare.versus.scenario import missions, scenario, truths  # noqa: PLC0415
    from compare.versus.score import grade  # noqa: PLC0415

    specs = {m.id: m for m in missions()}
    every = truths()
    changed: dict[str, int] = {}
    for path in sorted(Path(directory).glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        run_path = path.with_name(path.stem + ".run.json.gz")
        if record.get("mission_id") not in specs or not run_path.exists():
            continue
        with gzip.open(run_path, "rt", encoding="utf-8") as handle:
            run = json.load(handle)
        if record["system"] != "waypoint":
            plan = (run.get("plans") or [{}])[-1]
            ticks = [t for t in run.get("ticks") or [] if "reported" in t]
            if plan.get("waypoints") and plan["waypoints"][-1]["action"] == "LAND" and ticks:
                plan["waypoints"][-1]["x"], plan["waypoints"][-1]["y"] = ticks[-1]["reported"][0], ticks[-1]["reported"][1]
        mission_id = record["mission_id"]
        new = grade(run, every[mission_id], scenario(specs[mission_id], record.get("seed", 0)).constraints)
        if new != record["grade"]:
            changed[record["system"]] = changed.get(record["system"], 0) + 1
            record["grade"] = new
            path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Waypoint, PEACE and DroneServer on the same flights")
    commands = parser.add_subparsers(dest="command", required=True)
    fly = commands.add_parser("run", help="fly the missions")
    fly.add_argument("--systems", default="waypoint,peace,droneserver")
    fly.add_argument("--only", default=None, help="comma separated mission ids")
    fly.add_argument("--limit", type=int, default=None)
    fly.add_argument("--seed", type=int, default=0, help="seeds the randomized fault schedules, as in waypoint.experiments")
    fly.add_argument("--workers", type=int, default=4)
    fly.add_argument("--model", default=None, help="one OpenAI model for every system (default: the configured ladder)")
    fly.add_argument("--out", default=None)
    fly.add_argument("--name", default="")
    fly.add_argument("--no-report", action="store_true")
    report = commands.add_parser("report", help="write Comp_res.md from a result directory")
    report.add_argument("directory")
    report.add_argument("--to", default=None, help="where to write (default: compare/Comp_res.md)")
    again = commands.add_parser("regrade", help="score a result directory's stored flights again (no flying, no LLM)")
    again.add_argument("directory")
    args = parser.parse_args(argv)

    from compare.versus.report import write_report  # noqa: PLC0415

    if args.command == "regrade":
        print(f"grades changed: {regrade(Path(args.directory)) or 'none'}")
        return
    if args.command == "run":
        out = run(args)
        if not args.no_report:
            print(f"\nreport: {write_report(out)}")
    else:
        print(f"report: {write_report(Path(args.directory), Path(args.to) if args.to else None)}")


if __name__ == "__main__":
    main()
