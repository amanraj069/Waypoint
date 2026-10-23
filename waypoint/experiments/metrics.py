"""Metrics, computed from a result directory's files and nothing else.

`load(result_dir)` reads the manifest, every mission log and every run file,
and returns one `MissionRow` per flight pair and one `ReplanRow` per in-flight
replan trigger. `aggregate(rows, replans)` turns those into the numbers the
README defines, per condition:

* **catch rate**: of the missions that got a first plan, the share whose first
  plan the verifier rejected (at least one blocking violation). The share with
  any finding at all, warnings included, is reported beside it, with a 95%
  Wilson interval, and pooled over the conditions that share a first-plan
  prompt (violation feedback does not change the first plan, only replans).
* **replan success**: of the in-flight replan triggers, the share whose new
  plan passed verification on the first attempt, on the second, later, or
  never (abandoned: the planner failed, every correction was rejected, or the
  flight had already used up its replans).
* **replan latency**: wall-clock seconds from the trigger to a verified new
  plan, overall and by the fault that caused the trigger.
* **intent success**: scored against the batch file's hand-written ground
  truth (`experiments.truth`), never against the system's own reading: the
  share of flights that did everything the mission asked (every target served
  from enough sides, in order, avoid zones untouched, landed on the pad), and
  the mean fraction done. Missions marked infeasible are scored on refusal
  instead: refusing one is right, flying it is not.
* **intent extraction**: where the system read an intent, how it compares with
  the ground truth (exact match, and target-set F1).
* **repair, shield, hover**: how often deterministic repair fixed a rejected
  plan on the ground and in the air (and how fast), how often the safety
  shield had to step in, and the flight time spent hovering through replans.
* **safety vs optimality**: for missions where the verifier changed the plan
  (it rejected the first one and a corrected plan flew instead) and both the
  verified run and the unconstrained baseline completed, how much longer
  (metres and seconds) the verified one was. Where the first plan passed, both
  flights flew the same plan and any difference is only the LLM answering two
  in-flight replans differently, so those pairs are left out. And, for every
  baseline, whether it crashed, broke a keep-out, or flew a plan the verifier
  would have refused.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from waypoint.experiments.runner import MANIFEST
from waypoint.experiments.spec import CONDITION_LABELS, SYSTEM_LABELS, split_label
from waypoint.experiments.truth import Truth, extraction_match, load_truth, score

BLOCKING = frozenset({"ERROR", "CRITICAL"})

#: Which faults can cause which trigger, for attributing a trigger to a fault.
CAUSES = {
    "OBSTACLE_ON_ROUTE": ("OBSTACLE_APPEARS",),
    "BATTERY_INSUFFICIENT": ("BATTERY_DRAIN_SPIKE",),
    "OFF_ROUTE": ("GPS_DRIFT", "WIND_GUST"),
    "COMMAND_REJECTED": (),
}
#: A trigger no injected fault explains (the battery model alone, or a perception source).
NO_FAULT = "NONE"


@dataclass
class MissionRow:
    condition: str
    mission_id: str
    repeat: int
    world: str
    faults_spec: str
    fault_types: str
    error: str = ""
    outcome: str = ""
    planned: bool = False
    initial_blocking: int = 0
    initial_warnings: int = 0
    initial_caught: bool = False
    initial_violation_types: str = ""
    ground_attempts: int = 0
    ground_passed: bool = False
    inflight_triggers: int = 0
    inflight_resumed: int = 0
    planner_calls: int = 0
    distance_m: float | None = None
    duration_s: float | None = None
    battery_pct: float | None = None
    crashed: bool = False
    separation_breaches: int = 0
    wall_s: float | None = None
    baseline_outcome: str = ""
    baseline_distance_m: float | None = None
    baseline_duration_s: float | None = None
    baseline_crashed: bool = False
    baseline_separation_breaches: int = 0
    baseline_unsafe_plans: int = 0
    distance_overhead_pct: float | None = None
    time_overhead_pct: float | None = None
    #: Planner spend for the pair (verified and baseline), from the manifest.
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0
    models: str = ""
    #: The system part of the condition, and whether its first plan was asked for with the intent.
    system: str = ""
    #: Against the ground truth, when the batch file has one for this mission.
    expected_feasible: bool | None = None
    refused: bool = False
    intent_score: float | None = None
    accomplished: bool | None = None
    touchdown_error_m: float | None = None
    baseline_intent_score: float | None = None
    baseline_accomplished: bool | None = None
    extraction_exact: bool | None = None
    extraction_f1: float | None = None
    repairs_ground: int = 0
    repairs_ground_ok: int = 0
    repairs_inflight: int = 0
    repairs_inflight_ok: int = 0
    repair_ms: float | None = None
    intent_relaxed: int = 0
    shield_events: int = 0
    hold_s: float = 0.0

    @property
    def completed(self) -> bool:
        return self.outcome == "completed"


@dataclass
class ReplanRow:
    condition: str
    mission_id: str
    repeat: int
    number: int
    trigger: str
    fault_type: str
    t: float
    attempts: int
    passed: bool
    latency_s: float | None
    resolution: str
    #: "repair" when deterministic repair resolved it without the planner, else "llm".
    tier: str = "llm"

    @property
    def bucket(self) -> str:
        if not self.passed:
            return "abandoned"
        return {1: "first_try", 2: "second_try"}.get(self.attempts, "later")


# ---------------------------------------------------------------------------
# Reading a result directory
# ---------------------------------------------------------------------------


def load(result_dir: Path, missions: Path | None = None) -> tuple[list[MissionRow], list[ReplanRow], dict]:
    """Rows from a result directory. `missions` is the batch file holding the ground truth
    (default: the one the manifest names), so older batches can be scored against it too."""
    result_dir = Path(result_dir)
    manifest = json.loads((result_dir / MANIFEST).read_text())
    truths = load_truth(missions or manifest.get("config", {}).get("missions_file"))
    rows: list[MissionRow] = []
    replans: list[ReplanRow] = []
    for flight in manifest["flights"]:
        row, triggers = analyse_flight(flight, result_dir, truths)
        rows.append(row)
        replans.extend(triggers)
    return rows, replans, manifest


def analyse_flight(flight: dict, base: Path, truths: dict[str, Truth] | None = None) -> tuple[MissionRow, list[ReplanRow]]:
    truth = (truths or {}).get(flight["mission_id"])
    row = MissionRow(
        condition=flight["condition"],
        system=split_label(flight["condition"])[1],
        mission_id=flight["mission_id"],
        repeat=flight.get("repeat", 0),
        world=flight.get("world") or "open",
        faults_spec=flight.get("faults_spec", ""),
        fault_types=";".join(sorted({f["type"] for f in flight.get("faults") or []})),
        error=flight.get("error") or "",
    )
    _add_usage(row, flight)
    verified = flight.get("verified")
    if not verified or not verified.get("log"):
        return row, []

    entries = _read_log(base / verified["log"])
    run = _read_run(base, verified.get("run"))
    row.wall_s = verified.get("wall_s")
    row.outcome = _finished(entries).get("outcome", "")
    row.planner_calls = sum(1 for e in entries if e["kind"] == "plan_proposed" and e.get("source") not in ("repair", "inflight_repair"))
    _fill_flight(row, entries, run)
    _fill_system(row, entries, run, truth)

    ground = _ground_phase(entries)
    verifications = [e for e in ground if e["kind"] == "verification"]
    if verifications:
        first = verifications[0]["violations"]
        row.planned = True
        row.initial_blocking = sum(v["severity"] in BLOCKING for v in first)
        row.initial_warnings = len(first) - row.initial_blocking
        row.initial_caught = not verifications[0]["passed"]
        row.initial_violation_types = ";".join(sorted({v["type"] for v in first if v["severity"] in BLOCKING}))
        row.ground_attempts = len(verifications)
        row.ground_passed = any(v["passed"] for v in verifications)

    triggers = _replans(row, entries, run)
    row.inflight_triggers = len(triggers)
    row.inflight_resumed = sum(t.resolution == "resumed" for t in triggers)

    baseline = flight.get("baseline")
    if baseline and baseline.get("log"):
        b_entries = _read_log(base / baseline["log"])
        b_run = _read_run(base, baseline.get("run"))
        finished = _finished(b_entries)
        row.baseline_outcome = finished.get("outcome", "")
        if truth is not None and truth.feasible is not False and b_run:
            graded = score(truth, b_run)
            row.baseline_intent_score = graded["score"]
            row.baseline_accomplished = graded["accomplished"] and row.baseline_outcome == "completed"
        row.baseline_distance_m = finished.get("distance_m")
        row.baseline_duration_s = finished.get("t")
        row.baseline_crashed = row.baseline_outcome == "crashed" or bool((b_run.get("sim") or {}).get("crashed"))
        row.baseline_separation_breaches = _count_events(b_run, "SEPARATION")
        # Every plan the baseline flew "passed"; these are the ones the real gate would have refused.
        row.baseline_unsafe_plans = sum(
            any(v["severity"] in BLOCKING for v in e["violations"])
            for e in b_entries if e["kind"] == "verification"
        )
        # Only where the gate changed what flew; elsewhere both flew the same plan.
        if row.initial_caught and row.completed and row.baseline_outcome == "completed":
            row.distance_overhead_pct = _overhead(row.distance_m, row.baseline_distance_m)
            row.time_overhead_pct = _overhead(row.duration_s, row.baseline_duration_s)
    return row, triggers


def _add_usage(row: MissionRow, flight: dict) -> None:
    models = set()
    for kind in ("verified", "baseline"):
        for model, spent in ((flight.get(kind) or {}).get("usage") or {}).items():
            models.add(model)
            row.llm_calls += spent.get("calls", 0)
            row.tokens_in += spent.get("input", 0)
            row.tokens_cached += spent.get("cached", 0)
            row.tokens_out += spent.get("output", 0)
            row.tokens_reasoning += spent.get("reasoning", 0)
    row.models = ";".join(sorted(models))


def _fill_flight(row: MissionRow, entries: list[dict], run: dict) -> None:
    finished = _finished(entries)
    if "execution_started" not in {e["kind"] for e in entries}:
        return  # never flew: rejected on the ground, or planning failed
    row.distance_m = finished.get("distance_m")
    row.duration_s = finished.get("t")
    row.battery_pct = finished.get("battery_pct")
    row.crashed = row.outcome == "crashed" or bool((run.get("sim") or {}).get("crashed"))
    row.separation_breaches = _count_events(run, "SEPARATION")


def _fill_system(row: MissionRow, entries: list[dict], run: dict, truth: Truth | None) -> None:
    """Intent against the ground truth, and what repair, the shield and hovering did."""
    row.refused = row.outcome == "infeasible"
    if truth is not None:
        row.expected_feasible = truth.feasible
        if truth.feasible is not False:
            if row.refused or not run:
                row.intent_score, row.accomplished = 0.0, False
            else:
                graded = score(truth, run)
                row.intent_score = graded["score"]
                row.accomplished = graded["accomplished"] and row.completed
                row.touchdown_error_m = graded.get("touchdown_error_m")
        read = next((e for e in entries if e["kind"] == "intent"), None)
        if read is not None:
            match = extraction_match(truth.intent, read)
            row.extraction_exact, row.extraction_f1 = match["exact"], match["target_f1"]
    repairs = [e for e in entries if e["kind"] == "repair"]
    ground = [e for e in repairs if e.get("tier") != "inflight"]
    flight = [e for e in repairs if e.get("tier") == "inflight"]
    row.repairs_ground, row.repairs_ground_ok = len(ground), sum(bool(e.get("repaired")) for e in ground)
    row.repairs_inflight, row.repairs_inflight_ok = len(flight), sum(bool(e.get("repaired")) for e in flight)
    row.repair_ms = _median([e.get("ms") for e in repairs])
    row.intent_relaxed = sum(1 for e in entries if e["kind"] == "intent_relaxed")
    events = run.get("events") or []
    row.shield_events = sum(1 for e in events if e.get("type") == "SHIELD")
    row.hold_s = round(sum((e.get("payload") or {}).get("seconds", 0.0) for e in events if e.get("type") == "HOLD"), 2)


def _ground_phase(entries: list[dict]) -> list[dict]:
    for index, entry in enumerate(entries):
        if entry["kind"] == "execution_started":
            return entries[:index]
    return entries


def _replans(row: MissionRow, entries: list[dict], run: dict) -> list[ReplanRow]:
    faults = [
        (e["t"], (e.get("payload") or {}).get("fault"))
        for e in run.get("events") or []
        if e.get("type") == "FAULT"
    ]
    out = []
    ends = {"execution_resumed", "failsafe", "replan_triggered", "mission_finished"}
    for index, trigger in enumerate(entries):
        if trigger["kind"] != "replan_triggered":
            continue
        segment = []
        resolution = "unresolved"
        for entry in entries[index + 1:]:
            if entry["kind"] in ends:
                resolution = {"execution_resumed": "resumed", "failsafe": "failsafe"}.get(entry["kind"], entry["kind"])
                break
            segment.append(entry)
        checks = [e for e in segment if e["kind"] == "verification"]
        passed = next((e for e in checks if e["passed"]), None)
        tier = "repair" if any(
            e["kind"] == "repair" and e.get("tier") == "inflight" and e.get("repaired") for e in segment
        ) else "llm"
        latency = _seconds(trigger["wall"], passed["wall"]) if passed else None
        out.append(ReplanRow(
            condition=row.condition,
            mission_id=row.mission_id,
            repeat=row.repeat,
            number=trigger.get("number", len(out) + 1),
            trigger=trigger["trigger"],
            fault_type=attribute(trigger["trigger"], trigger["t"], faults),
            t=trigger["t"],
            attempts=len(checks),
            passed=passed is not None and resolution == "resumed",
            latency_s=latency,
            resolution=resolution,
            tier=tier,
        ))
    return out


def attribute(trigger: str, t: float, faults: list[tuple[float, str | None]]) -> str:
    """The most recent injected fault, at or before `t`, that can cause this trigger."""
    causes = CAUSES.get(trigger, ())
    fired = [(when, kind) for when, kind in faults if kind in causes and when <= t + 1e-6]
    return max(fired)[1] if fired else NO_FAULT


def _overhead(value: float | None, reference: float | None) -> float | None:
    if value is None or not reference:
        return None
    return round((value - reference) / reference * 100.0, 2)


def _count_events(run: dict, kind: str) -> int:
    return sum(1 for e in run.get("events") or [] if e.get("type") == kind)


def _finished(entries: list[dict]) -> dict:
    return next((e for e in reversed(entries) if e["kind"] == "mission_finished"), {})


def _seconds(start: str, end: str) -> float:
    return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 3)


def _read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_run(base: Path, relative: str | None) -> dict:
    if not relative:
        return {}
    path = base / relative
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def ordered_conditions(labels) -> list[str]:
    """Conditions in their fixed order (system, then planner), unknown labels after, so colours never move."""

    def key(label: str):
        planner, system = split_label(label)
        return (
            SYSTEM_LABELS.index(system) if system in SYSTEM_LABELS else len(SYSTEM_LABELS),
            CONDITION_LABELS.index(planner) if planner in CONDITION_LABELS else len(CONDITION_LABELS),
            label,
        )

    return sorted(set(labels), key=key)


def first_prompt(condition: str) -> str:
    """The part of a condition that shapes its first plan.

    Violation feedback only changes what the planner hears after a rejection,
    so hierarchical+feedback and hierarchical+no-feedback ask for their first
    plan with the same prompt, and any gap in their catch rates is sampling.
    The intent's requirements are part of that prompt, so systems that read the
    intent are pooled apart from those that do not.
    """
    planner, system = split_label(condition)
    from waypoint.experiments.spec import system_of  # noqa: PLC0415

    options = system_of(condition)
    reads_intent = options.intent if options is not None else False
    return planner.split("+", 1)[0] + ("+intent" if reads_intent else "")


def aggregate(rows: list[MissionRow], replans: list[ReplanRow]) -> dict:
    by_condition: dict[str, dict] = {}
    for condition in ordered_conditions({r.condition for r in rows}):
        mine = [r for r in rows if r.condition == condition]
        triggers = [t for t in replans if t.condition == condition]
        by_condition[condition] = _summarise(mine, triggers)
        same_prompt = [r for r in rows if r.planned and first_prompt(r.condition) == first_prompt(condition)]
        caught = sum(r.initial_caught for r in same_prompt)
        by_condition[condition]["first_plan"] = {
            "prompt": first_prompt(condition),
            "conditions": sorted({r.condition for r in same_prompt}),
            "pooled_catch_rate": _rate(caught, len(same_prompt)),
            "caught": caught,
            "planned": len(same_prompt),
        }
    return by_condition


def wilson(part: int, whole: int, z: float = 1.96) -> tuple[float, float] | None:
    """The 95% Wilson interval for a rate, which stays inside 0..1 at small counts."""
    if not whole:
        return None
    p = part / whole
    centre = (p + z * z / (2 * whole)) / (1 + z * z / whole)
    half = z * (p * (1 - p) / whole + z * z / (4 * whole * whole)) ** 0.5 / (1 + z * z / whole)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def _summarise(rows: list[MissionRow], triggers: list[ReplanRow]) -> dict:
    planned = [r for r in rows if r.planned]
    caught = [r for r in planned if r.initial_caught]
    flown = [r for r in rows if r.distance_m is not None]
    buckets = Counter(t.bucket for t in triggers)
    latencies = [t.latency_s for t in triggers if t.latency_s is not None]
    by_fault: dict[str, list[float]] = defaultdict(list)
    for t in triggers:
        if t.latency_s is not None:
            by_fault[t.fault_type].append(t.latency_s)
    pairs = [r for r in rows if r.distance_overhead_pct is not None]
    baselines = [r for r in rows if r.baseline_outcome]
    scored = [r for r in rows if r.expected_feasible is not False and r.accomplished is not None]
    impossible = [r for r in rows if r.expected_feasible is False and not r.error]
    possible = [r for r in rows if r.expected_feasible is True and not r.error]
    read = [r for r in rows if r.extraction_exact is not None]
    base_scored = [r for r in rows if r.baseline_accomplished is not None]
    by_tier: dict[str, list[float]] = defaultdict(list)
    for t in triggers:
        if t.latency_s is not None:
            by_tier[t.tier].append(t.latency_s)
    flown_rows = [r for r in rows if r.distance_m is not None]
    return {
        "flights": len(rows),
        "errors": sum(bool(r.error) for r in rows),
        "planning_failed": sum(r.outcome == "planning_failed" for r in rows),
        "planned": len(planned),
        "catch_rate": _rate(len(caught), len(planned)),
        "catch_rate_ci95": wilson(len(caught), len(planned)),
        "caught": len(caught),
        "any_violation_rate": _rate(sum(r.initial_blocking + r.initial_warnings > 0 for r in planned), len(planned)),
        "violation_types": dict(Counter(
            kind for r in caught for kind in r.initial_violation_types.split(";") if kind
        ).most_common()),
        "ground_correction_rate": _rate(sum(r.ground_passed for r in caught), len(caught)),
        "mean_ground_attempts_when_caught": _mean([r.ground_attempts for r in caught]),
        "completion_rate": _rate(sum(r.completed for r in planned), len(planned)),
        "completed": sum(r.completed for r in planned),
        "crash_rate": _rate(sum(r.crashed for r in flown), len(flown)),
        "separation_breach_rate": _rate(sum(r.separation_breaches > 0 for r in flown), len(flown)),
        "replans": {
            "triggered": len(triggers),
            **{key: buckets.get(key, 0) for key in ("first_try", "second_try", "later", "abandoned")},
            "first_try_rate": _rate(buckets.get("first_try", 0), len(triggers)),
            "within_two_rate": _rate(buckets.get("first_try", 0) + buckets.get("second_try", 0), len(triggers)),
            "abandoned_rate": _rate(buckets.get("abandoned", 0), len(triggers)),
            "by_trigger": dict(Counter(t.trigger for t in triggers).most_common()),
        },
        "latency_s": {
            "n": len(latencies),
            "median": _median(latencies),
            "mean": _mean(latencies),
            "by_fault": {
                fault: {"n": len(values), "median": _median(values), "mean": _mean(values)}
                for fault, values in sorted(by_fault.items())
            },
        },
        "cost": {
            "llm_calls": sum(r.llm_calls for r in rows),
            "tokens_in": sum(r.tokens_in for r in rows),
            "tokens_cached": sum(r.tokens_cached for r in rows),
            "tokens_out": sum(r.tokens_out for r in rows),
            "tokens_reasoning": sum(r.tokens_reasoning for r in rows),
            "tokens_out_per_flight": _mean([r.tokens_out for r in rows]),
            "models": dict(Counter(m for r in rows for m in r.models.split(";") if m).most_common()),
        },
        "intent": {
            "scored": len(scored),
            "accomplished_rate": _rate(sum(bool(r.accomplished) for r in scored), len(scored)),
            "accomplished_ci95": wilson(sum(bool(r.accomplished) for r in scored), len(scored)),
            "mean_score": _mean([r.intent_score for r in scored]),
            "infeasible_missions": len(impossible),
            "correct_refusal_rate": _rate(sum(r.refused for r in impossible), len(impossible)),
            "false_refusal_rate": _rate(sum(r.refused for r in possible), len(possible)),
            "relaxed_flights": sum(r.intent_relaxed > 0 for r in rows),
            "median_touchdown_error_m": _median([r.touchdown_error_m for r in scored if r.completed]),
            "baseline_accomplished_rate": _rate(sum(bool(r.baseline_accomplished) for r in base_scored), len(base_scored)),
            "baseline_mean_score": _mean([r.baseline_intent_score for r in base_scored]),
        },
        "extraction": {
            "read": len(read),
            "exact_rate": _rate(sum(bool(r.extraction_exact) for r in read), len(read)),
            "mean_target_f1": _mean([r.extraction_f1 for r in read]),
        },
        "repair": {
            "ground_tried": sum(r.repairs_ground for r in rows),
            "ground_fixed": sum(r.repairs_ground_ok for r in rows),
            "ground_fix_rate": _rate(sum(r.repairs_ground_ok for r in rows), sum(r.repairs_ground for r in rows)),
            "inflight_tried": sum(r.repairs_inflight for r in rows),
            "inflight_fixed": sum(r.repairs_inflight_ok for r in rows),
            "inflight_fix_rate": _rate(sum(r.repairs_inflight_ok for r in rows), sum(r.repairs_inflight for r in rows)),
            "median_ms": _median([r.repair_ms for r in rows]),
        },
        "replan_tiers": {
            tier: {"n": len(values), "median_latency_s": _median(values), "mean_latency_s": _mean(values)}
            for tier, values in sorted(by_tier.items())
        },
        "shield": {
            "flights_with_interventions": _rate(sum(r.shield_events > 0 for r in flown_rows), len(flown_rows)),
            "mean_events": _mean([r.shield_events for r in flown_rows]),
        },
        "hover_hold_s": {"total": round(sum(r.hold_s for r in rows), 2), "mean": _mean([r.hold_s for r in flown_rows])},
        "tradeoff": {
            "pairs": len(pairs),
            "median_distance_overhead_pct": _median([r.distance_overhead_pct for r in pairs]),
            "median_time_overhead_pct": _median([r.time_overhead_pct for r in pairs]),
            "mean_distance_overhead_pct": _mean([r.distance_overhead_pct for r in pairs]),
            "mean_time_overhead_pct": _mean([r.time_overhead_pct for r in pairs]),
            "baselines": len(baselines),
            "baseline_completion_rate": _rate(sum(r.baseline_outcome == "completed" for r in baselines), len(baselines)),
            "baseline_crash_rate": _rate(sum(r.baseline_crashed for r in baselines), len(baselines)),
            "baseline_separation_breach_rate": _rate(
                sum(r.baseline_separation_breaches > 0 for r in baselines), len(baselines)
            ),
            "baseline_unsafe_plan_rate": _rate(sum(r.baseline_unsafe_plans > 0 for r in baselines), len(baselines)),
        },
    }


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 3) if values else None


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def row_fields(cls) -> list[str]:
    return [f.name for f in fields(cls)]


def as_dict(row) -> dict:
    return asdict(row)
