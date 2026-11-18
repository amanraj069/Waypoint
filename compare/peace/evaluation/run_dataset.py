"""Drive the 108 evaluation prompts through PEACE (evaluation/run_dataset.py).

The released runner publishes each prompt of the dataset on the agent's query
topic, waits for a terminal status, and counts terminal states; the agent's
mission logger writes one CSV row per mission. This is the same procedure:

    python -m compare.peace.evaluation.run_dataset --limit 5
    python -m compare.peace.evaluation.run_dataset --category safety_violation
    python -m compare.peace.evaluation.run_dataset --reset-between --reset-mode teleport

As released, the aircraft is not reset between prompts unless asked: "Move 5
metres forward" starts wherever the previous prompt left it. `--reset-mode
rtl` returns and lands before each prompt; `teleport` puts a fresh aircraft
on the pad.

The dataset (erdemuysalx/uav-mission-prompts, MIT, version 1.1) is vendored in
`compare/peace/data/prompts.json`. Results go to `compare/results/peace/<stamp>/`:
`agent_log.csv` in the released columns, `missions.jsonl` with every plan and
step, and `summary.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from compare.common.flight import FlightStack, make_stack, sim_config
from compare.common.llm import ChatClient, default_client, usage_since, usage_snapshot
from compare.peace.agent import PeaceAgent
from compare.peace.config import PeaceSettings
from compare.peace.evaluation.checks import check
from compare.peace.evaluation.scene import EVALUATION_SCENE

DATASET = Path(__file__).resolve().parents[1] / "data" / "prompts.json"
RESULTS = Path(__file__).resolve().parents[2] / "results" / "peace"
CATEGORIES = (
    "takeoff_and_landing",
    "relative_motion",
    "altitude_and_attitude_control",
    "waypoint_navigation",
    "geometric_path_following",
    "object_search_and_localisation",
    "open_vocabulary_navigation",
    "composite_multi_step_mission",
    "safety_violation",
)


def load_prompts(source: str | Path = DATASET) -> list[dict]:
    """Prompts from a local prompts.json: a list, or {"prompts": [...]} as the dataset ships it."""
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    prompts = data.get("prompts", data) if isinstance(data, dict) else data
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"No prompts found in {source}")
    return prompts


def filter_prompts(
    prompts: list[dict], category: Optional[str] = None, start_id: Optional[str] = None, limit: Optional[int] = None
) -> list[dict]:
    """Apply category / start-id / limit filters in dataset order."""
    selected = list(prompts)
    if category:
        selected = [p for p in selected if p.get("category") == category]
    if start_id:
        for i, p in enumerate(selected):
            if p.get("id") == start_id:
                selected = selected[i:]
                break
        else:
            raise ValueError(f"start-id '{start_id}' not found in dataset (after filters)")
    if limit is not None:
        selected = selected[:limit]
    return selected


def metadata_for(prompt: dict) -> dict[str, str]:
    """The metadata that rides with each mission into the CSV's metadata_json column."""
    return {"prompt_id": str(prompt.get("id", "")), "task_category": str(prompt.get("category", ""))}


def evaluation_stack() -> FlightStack:
    """A fresh aircraft on the pad in open, flat ground.

    The battery does not run down: PX4 SITL's simulated battery stops at 50%,
    so a hundred prompts on one aircraft never end for want of charge.
    """
    from waypoint.sim.backend import CustomSimBackend  # noqa: PLC0415

    backend = CustomSimBackend(config=sim_config(meters_per_battery_pct=1.0e9, seconds_per_battery_pct=1.0e9))
    return make_stack(backend=backend)


def run(
    prompts: list[dict],
    llm: ChatClient,
    *,
    settings: PeaceSettings | None = None,
    out_dir: Path | None = None,
    reset_between: bool = False,
    reset_mode: str = "rtl",
    scene=EVALUATION_SCENE,
    progress=print,
) -> dict:
    """Send each prompt to the agent, wait for it to finish, and return the summary."""
    settings = settings or PeaceSettings()
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = str(out_dir) if out_dir else None

    stack = evaluation_stack()
    agent = PeaceAgent(stack, llm, settings, objects=scene, log_dir=log_dir)
    counts: Counter = Counter()
    by_category: dict[str, Counter] = {}
    records = []
    spent_before = usage_snapshot()
    trace = (out_dir / "missions.jsonl").open("a", encoding="utf-8") if out_dir else None
    try:
        for i, prompt in enumerate(prompts, start=1):
            pid, category, text = prompt.get("id", ""), prompt.get("category", ""), prompt.get("prompt", "")
            if reset_between:
                if reset_mode == "teleport":
                    agent.close()
                    stack = evaluation_stack()
                    agent = PeaceAgent(stack, llm, settings, objects=scene, log_dir=log_dir)
                else:
                    rtl_and_wait(stack)
            first_state = len(stack.states)
            began = time.perf_counter()
            result = agent.run(text, metadata=metadata_for(prompt))
            terminal = result.state or "no_status"
            counts[terminal] += 1
            by_category.setdefault(category, Counter())[terminal] += 1
            track = [stack.local_of(s.position) for s in stack.states[first_state:]] or [stack.true_position]
            record = {
                "id": pid,
                "category": category,
                "prompt": text,
                "state": terminal,
                "message": result.message,
                "replan_count": result.replan_count,
                "llm_calls": result.llm_calls,
                "llm_seconds": round(result.llm_seconds, 2),
                "flight_seconds": result.flight_seconds,
                "wall_seconds": round(time.perf_counter() - began, 2),
                "steps": result.steps,
                "plans": [
                    {"reasoning": p.reasoning, "steps": [s.model_dump() for s in p.steps], "safety_checks": p.safety_checks}
                    for p in result.plans
                ],
                "final_position": [round(c, 2) for c in stack.true_position],
                "final_yaw_deg": round(stack.yaw_deg % 360.0, 1),
                "armed": stack.armed,
                "max_altitude": round(max(p[2] for p in track), 2),
                "max_horizontal": round(max(max(abs(p[0]), abs(p[1])) for p in track), 2),
                "checks": check(prompt, result, track, stack.true_position, settings),
            }
            records.append(record)
            if trace:
                trace.write(json.dumps(record) + "\n")
                trace.flush()
            progress(
                f"{i}/{len(prompts)}  {pid:9} {terminal:9} replans={result.replan_count} "
                f"steps={len(result.steps)}  {text[:70]}"
            )
            if stack.ended:
                progress(f"  the aircraft is lost ({stack.ended}); a fresh one takes over")
                agent.close()
                stack = evaluation_stack()
                agent = PeaceAgent(stack, llm, settings, objects=scene, log_dir=log_dir)
    finally:
        agent.close()
        if trace:
            trace.close()

    summary = summarise(records, counts, by_category)
    summary["config"] = {
        "model": getattr(llm, "name", ""),
        "settings": settings.model_dump(),
        "reset_between": reset_between,
        "reset_mode": reset_mode if reset_between else None,
        "scene": [vars(o) for o in scene],
        "prompts": len(prompts),
    }
    summary["usage"] = usage_since(spent_before)
    if out_dir:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def rtl_and_wait(stack: FlightStack, rtl_timeout: float = 90.0) -> bool:
    """Engage AUTO.RTL and wait for the aircraft to land and disarm at home."""
    if not stack.armed:
        return True
    if not stack.set_mode("AUTO.RTL"):
        return False
    return stack.advance_until(lambda: not stack.armed, rtl_timeout, poll=0.2)


def summarise(records: list[dict], counts: Counter, by_category: dict[str, Counter]) -> dict:
    """The released runner's terminal-state counts, per category, and this rebuild's mechanical checks."""
    summary: dict = {
        "missions": len(records),
        "terminal_states": dict(counts),
        "by_category": {},
        "llm_calls": sum(r["llm_calls"] for r in records),
        "llm_calls_per_mission": round(sum(r["llm_calls"] for r in records) / len(records), 3) if records else None,
        "replans": sum(r["replan_count"] for r in records),
        "missions_replanned": sum(1 for r in records if r["replan_count"]),
        "safety_aborts": sum(
            1 for r in records for s in r["steps"] if str(s["result"]).startswith("ERROR: Safety violation")
        ),
        "within_limits": sum(1 for r in records if r["checks"]["within_limits"]),
    }
    for category in [c for c in CATEGORIES if c in by_category] + [c for c in by_category if c not in CATEGORIES]:
        rows = [r for r in records if r["category"] == category]
        entry: dict = {"missions": len(rows), **dict(by_category[category])}
        entry["within_limits"] = sum(1 for r in rows if r["checks"]["within_limits"])
        reached = [r["checks"]["reached"] for r in rows if "reached" in r["checks"]]
        if reached:
            entry["reached"] = sum(reached)
        enforced = Counter(r["checks"]["enforced"] for r in rows if "enforced" in r["checks"])
        if enforced:
            entry["enforced"] = dict(enforced)
        summary["by_category"][category] = entry
    return summary


def format_summary(summary: dict) -> str:
    lines = [
        f"{summary['missions']} missions: "
        + ", ".join(f"{state}={n}" for state, n in sorted(summary["terminal_states"].items())),
        f"LLM calls: {summary['llm_calls']} ({summary['llm_calls_per_mission']} per mission); "
        f"{summary['missions_replanned']} mission(s) replanned, {summary['replans']} replan(s); "
        f"{summary['safety_aborts']} step(s) aborted by the constraint layer; "
        f"{summary['within_limits']}/{summary['missions']} flights stayed inside the limits",
        "",
        f"{'category':34} {'n':>3} {'completed':>9} {'failed':>6}  checks",
    ]
    for category, entry in summary["by_category"].items():
        extras = []
        if "reached" in entry:
            extras.append(f"reached {entry['reached']}/{entry['missions']}")
        if "enforced" in entry:
            extras.append("enforced: " + ", ".join(f"{k}={v}" for k, v in entry["enforced"].items()))
        extras.append(f"in limits {entry['within_limits']}/{entry['missions']}")
        lines.append(
            f"{category:34} {entry['missions']:>3} {entry.get('completed', 0):>9} {entry.get('failed', 0):>6}  "
            + "; ".join(extras)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PEACE evaluation prompts through the rebuilt agent")
    parser.add_argument("--prompts", type=str, default=str(DATASET), help="path to a prompts.json (default: the vendored dataset)")
    parser.add_argument("--category", type=str, default=None, help="only run this task category")
    parser.add_argument("--start-id", type=str, default=None, help="resume from this prompt id")
    parser.add_argument("--limit", type=int, default=None, help="stop after N missions")
    parser.add_argument("--reset-between", action="store_true", help="reset the drone before each prompt (see --reset-mode)")
    parser.add_argument("--reset-mode", choices=("rtl", "teleport"), default="rtl")
    parser.add_argument("--model", type=str, default=None, help="e.g. gpt-5-mini, gemini:gemini-2.5-flash, ollama:qwen2.5:14b (the paper's default)")
    parser.add_argument("--out", type=str, default=None, help="result directory (default: compare/results/peace/<stamp>)")
    parser.add_argument("--name", type=str, default="", help="a label added to the result directory's name")
    args = parser.parse_args(argv)

    prompts = filter_prompts(load_prompts(args.prompts), args.category, args.start_id, args.limit)
    llm = default_client(args.model)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path(args.out) if args.out else RESULTS / (f"{stamp}-{args.name}" if args.name else stamp)
    print(f"{len(prompts)} prompt(s), model {llm.name}, results in {out}")
    summary = run(prompts, llm, out_dir=out, reset_between=args.reset_between, reset_mode=args.reset_mode)
    print()
    print(format_summary(summary))


if __name__ == "__main__":
    main()
