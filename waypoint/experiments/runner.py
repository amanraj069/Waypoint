"""Fly a batch: every mission, under every condition, verified and unconstrained.

For each condition, mission and repeat:

1. **verified**: the full closed loop, as `run_full` flies it, with the
   condition's planner. The first plan the planner returns is kept.
2. **baseline** (unless turned off): the same first plan, the same faults and
   the same backend, with the verifier swapped for one that reports what it
   would have found and passes everything. In-flight replans still go to the
   planner, unverified, unless `baseline_replans` is off: then the baseline
   flies its first plan without adapting at all, and costs no planner calls.
   This exists only to measure what safety costs; it is never how a mission
   is flown.

Every flight records the planner tokens it spent (`usage`, per provider and
model), so a batch's cost is measured rather than estimated.

Everything lands in one result directory:

    <out>/manifest.json                     the batch: config, and one entry per flight pair
    <out>/<condition>/<mission>[/r<k>]/verified/<run>.json + .log.jsonl
    <out>/<condition>/<mission>[/r<k>]/baseline/<run>.json + .log.jsonl

The manifest is rewritten after every mission, so an interrupted batch can
still be reported on. Metrics are computed from these files alone
(`experiments.metrics`), never from state held in this process.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from waypoint.backends import make_backend
from waypoint.experiments.spec import Batch, Scenario, prepare
from waypoint.orchestrator.loop import run_mission
from waypoint.paths import ROOT
from waypoint.planner import Plan, PlannerOptions, plan, replan
from waypoint.planner.config import (
    describe_providers,
    get_cost,
    get_model_ladder,
    get_openai_model_ladder,
    get_providers,
    get_reasoning_effort,
)
from waypoint.planner.llm_planner import usage_since, usage_snapshot
from waypoint.verifier import VerificationResult, verify

EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "runs"
MANIFEST = "manifest.json"
MANIFEST_VERSION = 1

#: options -> (planner, replanner), both with run_mission's signatures.
PlannerFactory = Callable[[PlannerOptions], tuple[Callable[..., Plan], Callable[..., Plan]]]


def llm_planners(options: PlannerOptions) -> tuple[Callable[..., Plan], Callable[..., Plan]]:
    """The real planner, asked the way the condition says."""
    return functools.partial(plan, options=options), functools.partial(replan, options=options)


def unverified(candidate: Plan, constraints, *, config=None, world=None) -> VerificationResult:
    """The baseline's gate: runs the real checks, records what they found, refuses nothing."""
    found = verify(candidate, constraints, config=config, world=world)
    return VerificationResult(passed=True, violations=found.violations)


@dataclass
class BatchConfig:
    missions_path: Path
    conditions: list[PlannerOptions]
    backend: str = "custom"
    seed: int = 0
    repeats: int = 1
    baseline: bool = True
    baseline_replans: bool = True
    only: list[str] | None = None
    limit: int | None = None
    max_replan_attempts: int = 3
    max_total_replans: int = 3
    pause: float = 0.0
    out_dir: Path | None = None
    name: str = ""
    extra: dict = field(default_factory=dict)


def run_batch(
    config: BatchConfig,
    *,
    planners: PlannerFactory = llm_planners,
    progress: Callable[[str], None] = print,
) -> Path:
    """Fly the batch and return its result directory."""
    batch = Batch.load(config.missions_path)
    missions = [m for m in batch.missions if not config.only or m.id in config.only]
    if config.only:
        unknown = set(config.only) - {m.id for m in missions}
        if unknown:
            raise ValueError(f"no mission called {', '.join(sorted(unknown))} in {config.missions_path}")
    if config.limit is not None:
        missions = missions[: config.limit]

    out = (config.out_dir or _default_out(config.name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": MANIFEST_VERSION,
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "config": {
            "missions_file": str(config.missions_path),
            "missions_sha256": hashlib.sha256(Path(config.missions_path).read_bytes()).hexdigest(),
            "conditions": [c.label for c in config.conditions],
            "backend": config.backend,
            "seed": config.seed,
            "repeats": config.repeats,
            "baseline": config.baseline,
            "baseline_replans": config.baseline_replans,
            "cost": get_cost(),
            "reasoning_effort": get_reasoning_effort(),
            "max_replan_attempts": config.max_replan_attempts,
            "max_total_replans": config.max_total_replans,
            # Which LLM planned: the providers in the order tried, and each one's ladder.
            # A fallback mid-batch is logged as a warning on the console.
            "planner_providers": list(get_providers()),
            "planner": describe_providers(),
            "model_ladder": {"openai": list(get_openai_model_ladder()), "gemini": list(get_model_ladder())},
            "git_commit": _git_commit(),
            # True when the code that flew is not exactly that commit.
            "git_dirty": _git_dirty(),
            **config.extra,
        },
        "flights": [],
    }
    _write(out, manifest)
    if manifest["config"]["git_dirty"]:
        progress(
            f"note: the checkout has uncommitted changes, so commit {manifest['config']['git_commit']} "
            "is not exactly the code that flies (recorded as git_dirty)"
        )

    total =len(config.conditions) * len(missions) * config.repeats
    done = 0
    for options in config.conditions:
        planner, replanner = planners(options)
        for spec in missions:
            for repeat in range(config.repeats):
                done += 1
                tag = f"[{done}/{total}] {options.label} {spec.id}" + (f" r{repeat}" if config.repeats > 1 else "")
                progress(f"{tag} ...")
                entry = _fly_pair(config, options, spec, repeat, planner, replanner, out)
                manifest["flights"].append(entry)
                _write(out, manifest)
                progress(f"{tag} {_describe(entry)}")
                if config.pause and done < total:
                    time.sleep(config.pause)

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    _write(out, manifest)
    return out


def _fly_pair(config, options, spec, repeat, planner, replanner, out: Path) -> dict:
    where = out / options.label / spec.id
    if config.repeats > 1:
        where = where / f"r{repeat}"
    entry: dict = {
        "condition": options.label,
        "options": {"hierarchical": options.hierarchical, "violation_feedback": options.violation_feedback},
        "mission_id": spec.id,
        "mission": spec.mission,
        "world": spec.world,
        "faults_spec": spec.faults,
        "repeat": repeat,
        "error": None,
        "verified": None,
        "baseline": None,
    }
    try:
        scenario = prepare(spec, seed=config.seed, repeat=repeat)
    except Exception as exc:  # noqa: BLE001 -- one bad mission must not end the batch
        entry["error"] = f"could not set the mission up: {type(exc).__name__}: {exc}"
        return entry
    entry["fault_seed"] = scenario.fault_seed
    entry["faults"] = [e.model_dump(mode="json") for e in scenario.faults.events]

    first: list[Plan] = []

    def keep_first(*args, **kwargs) -> Plan:
        result = planner(*args, **kwargs)
        if not first:
            first.append(result.model_copy(deep=True))
        return result

    try:
        entry["verified"] = _fly(config, scenario, keep_first, replanner, None, where / "verified", out)
        if config.baseline and first:
            reuse = lambda *args, **kwargs: first[0].model_copy(deep=True)  # noqa: E731
            entry["baseline"] = _fly(
                config, scenario, reuse, replanner, unverified, where / "baseline", out,
                adapt=config.baseline_replans,
            )
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _fly(
    config: BatchConfig, scenario: Scenario, planner, replanner, verifier, runs_dir: Path, out: Path,
    *, adapt: bool = True,
) -> dict:
    backend = make_backend(config.backend, world=scenario.world, wind=scenario.wind)
    started = time.perf_counter()
    spent_before = usage_snapshot()
    try:
        log = run_mission(
            scenario.spec.mission,
            scenario.constraints,
            scenario.faults,
            backend,
            world=scenario.world,
            home=scenario.home,
            landing=scenario.landing,
            planner=planner,
            replanner=replanner,
            verifier=verifier,
            adapt=adapt,
            max_replan_attempts=config.max_replan_attempts,
            max_total_replans=config.max_total_replans,
            verbose=False,
            runs_dir=runs_dir,
        )
    finally:
        backend.close()
    # Relative to the result directory, so the directory can be moved or shared.
    return {
        "outcome": log.outcome,
        "wall_s": round(time.perf_counter() - started, 3),
        "usage": usage_since(spent_before),
        "log": str(Path(log.path).relative_to(out)) if log.path else None,
        "run": str(Path(log.run_file).relative_to(out)) if log.run_file else None,
    }


def _describe(entry: dict) -> str:
    if entry["error"]:
        return f"ERROR {entry['error']}"
    verified = entry["verified"] or {}
    text = f"{verified.get('outcome')} ({verified.get('wall_s', 0):.1f}s)"
    if entry["baseline"]:
        text += f"; baseline {entry['baseline']['outcome']}"
    return text


def _default_out(name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return RESULTS_DIR / (f"{stamp}-{name}" if name else stamp)


def _write(out: Path, manifest: dict) -> None:
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out / MANIFEST)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Whether the checkout has uncommitted changes, untracked files included (None outside git)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())
