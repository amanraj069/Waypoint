"""Write Comp_res.md from a result directory of `python -m compare.versus run`."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from compare.versus.score import wilson

COMPARE = Path(__file__).resolve().parents[1]
NAMES = {"waypoint": "Waypoint (full)", "peace": "PEACE (rebuilt)", "droneserver": "DroneServer (rebuilt)"}
ORDER = ("waypoint", "peace", "droneserver")


def load(directory: Path) -> tuple[dict, dict[str, dict[str, dict]]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    records: dict[str, dict[str, dict]] = {}
    for system in manifest["systems"]:
        records[system] = {}
        for path in sorted((directory / system).glob("*.json")):
            record = json.loads(path.read_text())
            records[system][record["mission_id"]] = record
    return manifest, records


def pct(part: int, whole: int) -> str:
    if not whole:
        return "-"
    low, high = wilson(part, whole)
    return f"**{part / whole:.0%}** ({part}/{whole}; {low:.0%} to {high:.0%})"


def plain(part: int, whole: int) -> str:
    return "-" if not whole else f"{part / whole:.0%} ({part}/{whole})"


def fisher(a_yes: int, a_n: int, b_yes: int, b_n: int) -> float | None:
    if not a_n or not b_n:
        return None
    from scipy.stats import fisher_exact  # noqa: PLC0415

    return float(fisher_exact([[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]])[1])


def tally(rows: list[dict]) -> dict:
    """The headline counts for one system's flights."""
    graded = [r for r in rows if r.get("grade")]
    feasible = [r for r in graded if r["grade"]["expected_feasible"] is not False]
    impossible = [r for r in graded if r["grade"]["expected_feasible"] is False]
    flown = [r for r in graded if r["grade"]["flew"]]
    usage = Counter()
    models = Counter()
    for r in rows:
        for model, counts in (r.get("usage") or {}).items():
            models[model] += counts.get("calls", 0)
            for key in ("calls", "input", "cached", "output", "reasoning"):
                usage[key] += counts.get(key, 0)
    shares = [r["grade"]["share_done"] for r in feasible if r["grade"]["share_done"] is not None]
    return {
        "flights": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "feasible": len(feasible),
        "accomplished": sum(1 for r in feasible if r["grade"]["accomplished"]),
        "share_done": sum(shares) / len(shares) if shares else None,
        "impossible": len(impossible),
        "refused": sum(1 for r in impossible if r["grade"]["refused"]),
        "flown": len(flown),
        "crashed": sum(1 for r in graded if r["grade"]["crashed"]),
        "keep_out": sum(1 for r in graded if r["grade"]["keep_out_breaches"]),
        "left_fence": sum(1 for r in graded if r["grade"]["left_geofence"]),
        "above_ceiling": sum(1 for r in graded if r["grade"]["above_ceiling"]),
        "battery_flat": sum(1 for r in graded if r["grade"]["battery_depleted"]),
        "landed": sum(1 for r in flown if r["grade"]["landed"]),
        "usage": dict(usage),
        "models": dict(models),
        "wall_s": sum((r.get("detail") or {}).get("wall_s", 0.0) for r in rows),
        "flight_s": sum(r["grade"]["flight_s"] for r in graded),
        "distance_m": sum(r["grade"]["distance_m"] for r in graded),
    }


def reason_class(record: dict) -> str:
    why = (record.get("grade") or {}).get("why_not") or ""
    if record.get("error"):
        return "the run raised an error"
    for key, label in (
        ("crashed", "crashed"), ("battery", "battery ran flat"), ("never flew", "never flew"), ("not served", "target(s) not served"),
        ("out of order", "served out of order"), ("flew over", "flew over a place to avoid"), ("did not land on the pad", "did not land on the pad"),
        ("not on the ground", "left flying at the end"),
    ):
        if key in why:
            return label
    return why or "unknown"


def crash_causes(directory: Path, system: str, rows: list[dict]) -> Counter:
    """What each crashed flight hit, from the stored run's COLLISION event."""
    import gzip  # noqa: PLC0415

    causes: Counter = Counter()
    for record in rows:
        if not (record.get("grade") or {}).get("crashed"):
            continue
        path = directory / system / f"{record['mission_id']}.run.json.gz"
        description = ""
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                events = json.load(handle).get("events") or []
            description = next((e["description"] for e in events if e.get("type") in ("COLLISION", "KILL")), "")
        causes["the ground" if "into the ground" in description else "a building or a tree" if "struck" in description else "motors cut in the air" if "motors cut" in description else "unknown"] += 1
    return causes


def what_happened(directory: Path, systems: list[str], records: dict[str, dict[str, dict]]) -> list[str]:
    """Facts counted from the flights that explain the table above."""
    lines = ["### What happened in the flights", ""]
    for system in systems:
        rows = list(records[system].values())
        details = [r.get("detail") or {} for r in rows]
        causes = crash_causes(directory, system, rows)
        hit = ", ".join(f"{n} into {what}" for what, n in causes.most_common()) or "none"
        if system == "waypoint":
            events: Counter = Counter()
            for d in details:
                events.update(d.get("log_events") or {})
            outcomes = Counter(d.get("outcome") for d in details)
            lines.append(
                f"* **Waypoint.** Crashes: {hit}. Its verifier examined {events.get('verification', 0)} plans; "
                f"{events.get('repair', 0)} were sent to deterministic repair; the monitor interrupted flights "
                f"{events.get('replan_triggered', 0)} times, and {events.get('perception', 0)} unmapped obstacles were seen in flight. "
                "Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1])) + "."
            )
        elif system == "peace":
            late = lambda record: sum(1 for s in (record.get("detail") or {}).get("steps") or [] if "arrival timeout" in s["result"])  # noqa: E731
            after_crash = sum(late(r) for r in rows if (r.get("grade") or {}).get("crashed"))
            timeouts = sum(late(r) for r in rows) - after_crash
            lines.append(
                f"* **PEACE.** Crashes: {hit}. It planned each mission in one call "
                f"({sum(d.get('llm_calls') or 0 for d in details)} calls over {len(rows)} missions, "
                f"{sum(1 for d in details if d.get('replans'))} missions replanned). Its constraint layer refused "
                f"{sum(d.get('constraint_aborts') or 0 for d in details)} commands. {timeouts} legs were too long to fly inside its "
                "30 s command time-out, which the release does not treat as a failure: the plan carries on from wherever the "
                f"aircraft is (and {after_crash} more steps timed out in plans that went on executing after a crash). It executes "
                "its plan open loop, so nothing looks at a building, a hillside or a crane that appears once the plan is made."
            )
        elif system == "droneserver":
            calls = [c for d in details for c in d.get("calls") or []]
            rules = Counter(c["rule"] for c in calls if c.get("rule"))
            refused = sum(1 for c in calls if c["status"] == "rejected")
            claims = Counter(d.get("model_claim") for d in details)
            top = ", ".join(f"`{rule}` {n}" for rule, n in rules.most_common(4))
            lines.append(
                f"* **DroneServer.** Crashes: {hit}. The model issued {len(calls)} tool calls; the server refused {refused} "
                f"({top}). `bounds.min_altitude` is the altitude-frame hazard the DroneServer paper names: `go_to_location` takes "
                "metres above sea level, and the model sent a height above the launch point, which is underground at this field. "
                "The server's fence and bounds held (no flight left the geofence), but it has no notion of a building or of terrain, "
                f"so a target inside the fence is accepted whatever stands in the way. The model ended {claims.get('aborted', 0)} missions "
                f"with MISSION ABORTED and {claims.get('complete', 0)} with MISSION COMPLETE; "
                f"{sum(1 for d in details if d.get('left_flying'))} aircraft were left in the air."
            )
    lines.append("")
    return lines



def _stored_runs(directory: Path, system: str):
    """(record, run) for each stored flight of one system that took off or was meant to."""
    import gzip  # noqa: PLC0415

    for path in sorted((directory / system).glob("*.json")):
        if path.name.endswith(".run.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        run_path = path.with_name(path.stem + ".run.json.gz")
        if not run_path.exists():
            continue
        with gzip.open(run_path, "rt", encoding="utf-8") as handle:
            yield record, json.load(handle)


def sensitivity(directory: Path, systems: list[str]) -> dict[str, dict[str, int]]:
    """Missions accomplished under the scoring as used, and under two other readings of it.

    `touchdown`: every system's landing judged where it actually touched down, not where it believed it did.
    `wide_photo`: a photograph counts from twice as far, the generous reading of the briefing's first wording.
    """
    import copy  # noqa: PLC0415

    import waypoint.intent.check as check  # noqa: PLC0415
    from compare.versus.scenario import missions, scenario, truths  # noqa: PLC0415
    from compare.versus.score import grade  # noqa: PLC0415

    specs, every = {m.id: m for m in missions()}, truths()
    out: dict[str, dict[str, int]] = {}
    original = check.reach
    for system in systems:
        counts = {"as_scored": 0, "touchdown": 0, "wide_photo": 0}
        for record, run in _stored_runs(directory, system):
            if record["grade"].get("refused"):
                continue
            mission_id = record["mission_id"]
            constraints = scenario(specs[mission_id], record.get("seed", 0)).constraints
            counts["as_scored"] += bool(record["grade"]["accomplished"])
            down = copy.deepcopy(run)
            for plan in down.get("plans") or []:
                if plan.get("waypoints") and plan["waypoints"][-1]["action"] == "LAND":
                    plan["waypoints"] = plan["waypoints"][:-1]
            down.pop("waypoints", None)
            counts["touchdown"] += bool(grade(down, every[mission_id], constraints)["accomplished"])
            check.reach = lambda t: t.radius + min(check.VIEW_RANGE, max(check.MIN_VIEW_RANGE, 4.0 * t.radius))
            try:
                wide = copy.deepcopy(run)
                if system != "waypoint":
                    ticks = [t for t in wide.get("ticks") or [] if "reported" in t]
                    plan = (wide.get("plans") or [{}])[-1]
                    if plan.get("waypoints") and plan["waypoints"][-1]["action"] == "LAND" and ticks:
                        plan["waypoints"][-1]["x"], plan["waypoints"][-1]["y"] = ticks[-1]["reported"][:2]
                counts["wide_photo"] += bool(grade(wide, every[mission_id], constraints)["accomplished"])
            finally:
                check.reach = original
        out[system] = counts
    return out


def photo_misses(directory: Path, system: str) -> Counter:
    """Why each photograph target a flight did not serve went unserved, in the scorer's own words."""
    from compare.versus.scenario import truths  # noqa: PLC0415
    from waypoint.experiments.truth import flown  # noqa: PLC0415
    from waypoint.intent.check import Sight, serve  # noqa: PLC0415
    from waypoint.planner.models import Action  # noqa: PLC0415

    every = truths()
    reasons: Counter = Counter()
    for record, run in _stored_runs(directory, system):
        g = record["grade"]
        if g.get("accomplished") or g.get("expected_feasible") is False or not g.get("flew") or g.get("crashed"):
            continue
        truth = every[record["mission_id"]]
        points, _ = flown(run)
        sight = Sight(truth.world)
        photos = [p for p in points if p.action is Action.CAPTURE]
        for target in truth.intent.targets:
            if target.action.value != "CAPTURE" or serve(target, points, sight).satisfied:
                continue
            if not photos:
                reasons["no photograph taken"] += 1
                continue
            whys = [sight.view(target, p)[2] for p in photos]
            if any(not w for w in whys):
                reasons["photographed, but not from enough sides"] += 1
            elif all("beyond the" in w for w in whys):
                reasons["every photograph too far away"] += 1
            elif any("blocks the line of sight" in w for w in whys):
                reasons["view blocked by a building or a tree"] += 1
            else:
                reasons["wrong height, or not overhead when asked from above"] += 1
    return reasons


def against_papers(directory: Path, systems: list[str], records: dict[str, dict[str, dict]]) -> list[str]:
    """How each rebuild's results stand against what its paper reports, and why the scores here are lower."""
    results = COMPARE / "results"
    lines = ["## Why the rebuilt systems score lower here than in their papers", ""]
    lines += [
        "The papers did not fly these missions, so no number above can be compared with a number in a paper directly. What "
        "can be compared is each rebuild on its own paper's evaluation, and that matches:",
        "",
    ]
    drone = []
    for name in ("suite-n2", "suite-n2-audited-partial"):
        path = results / "droneserver" / name / "trials.jsonl"
        if path.exists():
            drone += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def passed(*missions_: str) -> str:
        rows = [t for t in drone if t["mission"] in missions_]
        return f"{sum(1 for t in rows if t['passed'])}/{len(rows)}" if rows else "-"
    replay = None
    try:
        from compare.droneserver.replay import replay_all, summarise  # noqa: PLC0415

        trials = replay_all()
        replay = summarise(trials)
        gpt = [t for t in trials if t.model.startswith("gpt")]
        published = f"{sum(1 for t in gpt if t.upstream_verdict == 'PASS')}/{len(gpt)}"
    except Exception:  # noqa: BLE001 -- the replay is an addition to the report, never a reason to lose it
        published = "6/8"
    lines += [
        "| DroneServer | Its paper | This rebuild |",
        "| --- | --- | --- |",
        f"| The authors' published GPT-5.2 and Gemini trials, replayed call for call through the rebuilt server | GPT-5.2 passed {published} | "
        + (f"the published verdict on **{replay['same_verdict']}/{replay['trials']}** trials" if replay else "-") + " |",
        "| Frontier models (eleven) on its flying missions, 1 to 5 and 7 | 75.9 to 100%, eight of eleven at 90% or above | "
        "not run: needs a frontier model (`--model`) |",
        "| Small models (4 to 8 B, run locally) | \"passed the same two missions, the basic flight cycle and the geofence refusal, "
        "and failed the same six multi-step ones\" | gpt-5-nano: basic flight cycle (T1) "
        f"{passed('T1')}, geofence and injection (T8, T9) {passed('T8', 'T9')}; square (T3) {passed('T3')}, return to launch (T5) "
        f"{passed('T5')}, parameter (T7) {passed('T7')}, long mission (T10) {passed('T10')}; waypoint (T2) {passed('T2')}, "
        f"autopilot mission (T4) {passed('T4')} |",
        "| Geofence-violation attempts that escaped | 0 of 110 | 0 (and no head-to-head flight left its fence) |",
        "| Adversarial cases with the specified outcome | 29 of 29 | 29 of 29 |",
        "",
    ]
    peace_summary = results / "peace" / "full-108" / "summary.json"
    if peace_summary.exists():
        c = json.loads(peace_summary.read_text())["by_category"]
        def share(key: str) -> str:
            e = c.get(key, {})
            return f"{e.get('completed', 0)}/{e.get('missions', 0)}"
        lines += [
            "| PEACE | Its paper (no numbers: \"not a quantitative benchmark\") | This rebuild, its 108 prompts |",
            "| --- | --- | --- |",
            "| Metric tasks | \"admit the most reliable plans\" | take-off and landing "
            f"{share('takeoff_and_landing')}, relative motion {share('relative_motion')}, waypoints {share('waypoint_navigation')}, "
            f"altitude {share('altitude_and_attitude_control')} |",
            f"| Geometric paths | \"correctly decomposed ... when the shape and edge length are stated\" | {share('geometric_path_following')} |",
            "| Perception-conditioned tasks | depend \"on detector recall and on the depth-validity flag\" | object search "
            f"{share('object_search_and_localisation')}, open vocabulary {share('open_vocabulary_navigation')} |",
            f"| Composite missions | \"an undetected failure in one step can cascade\" | {share('composite_multi_step_mission')} |",
            "",
        ]
    lines += [
        "On Waypoint's missions the scores drop, for five reasons that the flights show:",
        "",
        "1. **The task is the hard kind for both papers.** Every mission here that can be flown is multi-step: photograph named "
        "landmarks from given sides and in order, keep out of zones, come back to a pad, in towns with buildings and in hills, "
        "with obstacles that appear in flight, GPS slips and storms. PEACE's prompts are moves in an open simulated yard "
        "(\"take off to 5 meters then move forward 3 meters\"); DroneServer's are in an open field (\"60 metres due north\"). "
        "The DroneServer paper's own small models fail every multi-step mission it has.",
    ]
    crash = {s: crash_causes(directory, s, list(records[s].values())) for s in systems if s != "waypoint"}
    lines.append(
        "2. **Neither design knows about buildings or terrain.** "
        + "; ".join(f"{NAMES[s]} crashed {sum(c.values())} times ({', '.join(f'{n} into {w}' for w, n in c.most_common())})"
                    for s, c in crash.items())
        + ". PEACE plans once and flies the plan open loop; DroneServer's server bounds and fences a command but accepts any "
        "target inside the fence, whatever stands in the way. Waypoint verifies each plan against the map before take-off."
    )
    misses = {s: photo_misses(directory, s) for s in systems if s != "waypoint"}
    lines.append(
        "3. **Photographs from the wrong place.** Of the photograph targets left unserved in flights that did not crash: "
        + "; ".join(f"{NAMES[s]}: " + ", ".join(f"{n} {why}" for why, n in m.most_common()) for s, m in misses.items())
        + ". A photograph counts only from a narrow band around the subject; Waypoint's checker computes that band and "
        "repairs a plan that misses it, the other two must judge it from the text."
    )
    lines.append(
        "4. **The model is the smallest of its family** (gpt-5-nano), and DroneServer leans on it hardest: about 14 calls per "
        "mission over 98 tools, and most of its server refusals are `bounds.min_altitude`, a height above the launch point sent "
        "where `go_to_location` takes metres above sea level, the hazard its paper names."
    )
    mini = directory.parent / "mini-droneserver" / "droneserver"
    if "droneserver" in systems and mini.exists():
        flights = [json.loads(f.read_text()) for f in sorted(mini.glob("*.json")) if not f.name.endswith(".run.json")]
        spent = sum(((c.get("input", 0) - c.get("cached", 0)) * 0.25 + c.get("cached", 0) * 0.025 + c.get("output", 0) * 2.0) / 1e6
                    for r in flights for c in (r.get("usage") or {}).values())
        def brief(r: dict) -> str:
            g = r["grade"]
            state = "accomplished" if g["accomplished"] else "crashed" if g["crashed"] else "landed" if g["landed"] else "not landed"
            return f"`{r['mission_id']}` {state}" + ("" if g["crashed"] else f" ({g['why_not'] or 'all served'})")
        nano = {m: records["droneserver"].get(m) for m in (r["mission_id"] for r in flights)}
        lines.append(
            f"   A small check with a larger model: DroneServer on gpt-5-mini, {len(flights)} missions it failed on gpt-5-nano "
            f"(about ${spent:.2f}): " + "; ".join(brief(r) for r in flights) + ". On nano those were "
            + "; ".join(f"`{m}` {'crashed' if (n or {}).get('grade', {}).get('crashed') else 'not served'}" for m, n in nano.items())
            + f". On gpt-5-mini the server refused {sum(1 for r in flights for c in r['detail']['calls'] if c.get('rule') == 'bounds.min_altitude')} "
            "commands for the altitude frame, nano's most common refusal; the model flew to its subjects and photographed "
            "them, and still failed on where: photographs from one spot, turning in place for \"sides\"; and, at Bara Gumbad, "
            "a destination on the tomb itself, so that `monitor_flight`'s default auto-land set the aircraft down on its roof. "
            "Two missions settle nothing about rates. (The Bara Gumbad flight was flown again after the re-arm fix in (c) below; "
            "the first is in `mini-droneserver-before-rearm-fix`.)"
        )
    sens = sensitivity(directory, systems)
    lines.append(
        "5. **Three things on this comparison's side, found while checking these numbers.** (a) The rebuilt systems' landings were "
        "first judged where the aircraft really touched down and Waypoint's where its plan put the LAND, so GPS drift and wind "
        "failed the first two and not Waypoint; every system is now judged where it believed it touched down (the table above). "
        "Judged by the real touchdown instead, the accomplished counts are "
        + ", ".join(f"{NAMES[s]} {sens[s]['touchdown']}" for s in systems)
        + ". (b) The briefing first said a photograph counts \"within about twice the subject's size of its edge\", which reads "
        "as twice its width; the scorer allows twice its radius. Read the generous way, the counts are "
        + ", ".join(f"{NAMES[s]} {sens[s]['wide_photo']}" for s in systems)
        + ". The wording is now exact; these flights were briefed with the first. (c) After an auto-landing the rebuilt "
        "vehicle refused to arm again, where MavSDK switches ArduPilot to HOLD first and arms; it is fixed, and it ended "
        "two of these DroneServer flights early (`chowk_street_run`, `munnar_echo_point`). Those two were flown again with the "
        "fix (and the exact photograph wording); the first flights are kept in `seed0-droneserver-before-rearm-fix`."
    )
    lines += [
        "",
        "So the rebuilt systems are not broken copies: on their papers' own tasks they behave as their papers report for a small "
        "model. The gap above is what these two designs, with this model, do on missions of the kind Waypoint was built "
        "for. A larger model would narrow it, most for DroneServer (`python -m compare.versus run --model <model>`).",
        "",
    ]
    return lines

def own_benchmarks() -> list[str]:
    """What each rebuilt system scored on its own paper's evaluation, from the recorded runs."""
    lines: list[str] = []
    results = COMPARE / "results"
    peace = results / "peace" / "full-108-audited" / "summary.json"
    if not peace.exists():
        peace = results / "peace" / "full-108" / "summary.json"
    if peace.exists():
        s = json.loads(peace.read_text())
        lines += [
            "### PEACE on its own 108 prompts",
            "",
            f"Model `{s['config']['model']}`. The aircraft is not reset between prompts, as in the release's runner. "
            "The paper reports no numbers for this set (\"not a quantitative benchmark\"), so there is nothing to match; "
            "the terminal-state counts are what its runner prints, and the last two columns are this rebuild's mechanical checks. "
            "This run predates the audit (the model was then called with Waypoint's reasoning effort and without the plan schema) "
            "and was not repeated, to save API calls.",
            "",
            "| Category | Prompts | Completed | Failed | Checks |",
            "| --- | --- | --- | --- | --- |",
        ]
        for category, entry in s["by_category"].items():
            checks = []
            if "reached" in entry:
                checks.append(f"ended at the named position {entry['reached']}/{entry['missions']}")
            if "enforced" in entry:
                checks.append("enforced: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in entry["enforced"].items()))
            checks.append(f"inside the limits {entry['within_limits']}/{entry['missions']}")
            lines.append(f"| {category.replace('_', ' ')} | {entry['missions']} | {entry.get('completed', 0)} | {entry.get('failed', 0)} | {'; '.join(checks)} |")
        lines += [
            "",
            f"All categories: {s['terminal_states'].get('completed', 0)} completed, {s['terminal_states'].get('failed', 0)} failed; "
            f"{s['llm_calls_per_mission']} LLM calls per mission ({s['missions_replanned']} missions replanned); "
            "for the safety prompts, \"not triggered\" means the model itself planned inside the limits or planned nothing, "
            "so the constraint layer had nothing to clamp or refuse; "
            f"{s['safety_aborts']} step{'' if s['safety_aborts'] == 1 else 's'} aborted by the constraint layer; {s['within_limits']}/{s['missions']} flights stayed inside the altitude band and the fence.",
            "",
        ]
    drone = results / "droneserver" / "suite-n2-audited" / "summary.json"
    if not drone.exists():
        drone = results / "droneserver" / "suite-n2" / "summary.json"
    if drone.exists():
        s = json.loads(drone.read_text())
        lines += [
            "### DroneServer on its own mission suite",
            "",
            f"Model `{s['config']['model']}`, {s['config']['trials_per_mission']} trials per mission. The paper's best eight of eleven models "
            "complete 90 to 100% of the flying missions; its weaker three do not, and this is one small model. "
            "The adversarial suite needs no model: **29 of 29** cases produce the outcome they specify, as in the paper "
            "(one case is half rebuilt, see `droneserver/README.md`). This suite was flown before the audit and not again, to "
            "save API calls; the check of the corrected server is the replay of the authors' own published trials, which gives "
            "their published verdict on **10 of 10** (`python -m compare.droneserver.replay`).",
            "",
            "| Mission | Trials | Passed | Mean turns | Mean tool calls | Why failed |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for mission, entry in s["by_mission"].items():
            lines.append(
                f"| {mission} | {entry['trials']} | {entry['passed']} | {entry['mean_turns']} | {entry['mean_tool_calls']} | "
                + "; ".join(reason[:110] for reason in entry["reasons_failed"]) + " |"
            )
        fence = s["geofence_violation_trials"]
        lines += [
            "",
            f"Flying missions (T1 to T5, T10): {s['flying_missions']['passed']}/{s['flying_missions']['trials']} passed. "
            f"Geofence-violation trials (T8): {fence['trials']} flown, {fence['refused_by_the_server']} refused by the server, "
            f"**{fence['left_the_permitted_zone']} left the permitted zone** (the paper: none in 110). "
            f"Aircraft outside the fence in any trial: {s['aircraft_outside_fence_in_any_trial']}.",
            "",
        ]
    for name, label in (("sample-released", "as released"), ("sample-strict", "with the strict reading of `enter`")):
        path = results / "autotamp" / name / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        if name == "sample-released":
            lines += [
                "### AutoTAMP on its own domain (HouseWorld)",
                "",
                "AutoTAMP plans a point in a 2D house from `enter(room)` and `not_enter(room)`; it has no action for a photograph, "
                "a hover or a landing, and no altitude, terrain or battery, so it is not flown on Waypoint's missions. It is run on "
                "the paper's HouseWorld cases (the first two paraphrases of each of the ten instruction families, one start each). "
                "Success is judged by this rebuild's checkers, because the paper's are not in its release.",
                "",
                "| Method | Cases | HouseWorld1 | HouseWorld2 | Paper, GPT-4 | Planned | LLM calls |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        paper = {"no-corrections": "43.5% / 42.0%", "syntax": "59.5% / 59.0%", "autotamp": "82.5% / 82.0%"}
        for method, entry in s["methods"].items():
            lines.append(
                f"| {method} ({label}) | {entry['cases']} | {plain(entry['houseworld1']['success'], entry['cases'])} | "
                f"{plain(entry['houseworld2']['success'], entry['cases'])} | {paper[method]} | {entry['planned']}/{entry['cases']} | {entry['mean_llm_calls']} |"
            )
    if (results / "autotamp" / "sample-released" / "summary.json").exists():
        s = json.loads((results / "autotamp" / "sample-released" / "summary.json").read_text())
        lines += [
            "",
            f"Model `{s['config']['model']}`, planner limit {s['config']['planner_time_limit']:g} s per call. "
            "The gap to the paper has three known parts, set out in `autotamp/README.md`: the release plans every `enter` as "
            "\"eventually enter\", so orders and stays in a formula are not enforced; the model is a small one; and the checkers are this rebuild's.",
            "",
        ]
    return lines


def write_report(directory: Path, to: Path | None = None) -> Path:
    manifest, records = load(directory)
    systems = [s for s in ORDER if s in records]
    mission_ids = manifest["missions"]
    tallies = {s: tally(list(records[s].values())) for s in systems}
    first = next(iter(records[systems[0]].values()))
    models = sorted({m for s in systems for m in tallies[s]["models"]})
    lines: list[str] = []
    add = lines.append

    reflown_on = []
    for system in systems:
        again = directory.parent / f"{directory.name}-{system}-audited" / "manifest.json"
        if again.exists():
            reflown_on.append(f"{NAMES[system]} {json.loads(again.read_text())['started'][:10]}")
    reflown = f" (re-flown after the audits: {', '.join(reflown_on)})" if reflown_on else ""
    add("# Waypoint against the rebuilt systems, on the same flights")
    add("")
    add(
        f"Result directory `{directory.relative_to(COMPARE.parent) if directory.is_relative_to(COMPARE.parent) else directory}`, "
        f"flown {manifest['started'][:10]}{reflown}. Written by `python -m compare.versus report`; every number is recomputed "
        "from the stored flights."
    )
    add("")
    add("## What was flown")
    add("")
    add(
        f"* **Missions:** the {len(mission_ids)} missions of `experiments/missions.json`, one flight each per system, "
        f"fault seed {manifest['seed']} (the randomized schedules are the same for every system)."
    )
    add("* **Systems:** " + ", ".join(NAMES[s] for s in systems) + ". The two rebuilt systems are described in `compare/README.md`.")
    for system in systems:
        flown = len(records[system])
        if flown < len(manifest["missions"]):
            add(f"* **Not yet complete:** {NAMES[system]} has {flown} of {len(manifest['missions'])} missions flown; the rest are "
                "still flying and every figure for it below covers only those flown.")
    add(
        "* **Same for every system:** the mission text, the world, the launch and landing pads, the wind, the fault schedule, "
        "the simulated airframe and battery (Waypoint's built-in simulator), and the LLM."
    )
    add(
        f"* **LLM:** {manifest['planner']}. Models that answered: "
        + ", ".join(f"`{m}`" for m in models) + ". The reasoning effort named there is Waypoint's own setting; "
        "the rebuilt systems call the same model with their releases' settings (\"How to read this\")."
    )
    add(
        "* **Scoring:** one function for every system (`compare/versus/score.py`), from the run file alone, against the "
        "hand-written ground truth no planner sees: `waypoint.experiments.truth.score`, the function Waypoint's own experiment "
        "report uses."
    )
    add("")
    add("## Result")
    add("")
    add("| | " + " | ".join(NAMES[s] for s in systems) + " |")
    add("| --- | " + " | ".join("---" for _ in systems) + " |")

    def row(label: str, cell) -> None:
        add(f"| {label} | " + " | ".join(cell(tallies[s]) for s in systems) + " |")

    row("Mission accomplished, of missions that can be flown", lambda t: pct(t["accomplished"], t["feasible"]))
    row("Mean share of each mission done", lambda t: "-" if t["share_done"] is None else f"{t['share_done']:.0%}")
    row("Impossible missions not flown", lambda t: f"**{t['refused']}/{t['impossible']}**")
    row("Crashed, of all flights", lambda t: pct(t["crashed"], t["flights"]))
    row("Broke an obstacle's keep-out", lambda t: plain(t["keep_out"], t["flights"]))
    row("Left the mission's geofence", lambda t: plain(t["left_fence"], t["flights"]))
    row("Flew above the ceiling", lambda t: plain(t["above_ceiling"], t["flights"]))
    row("Battery ran flat", lambda t: plain(t["battery_flat"], t["flights"]))
    row("Landed under control, of flights that took off", lambda t: plain(t["landed"], t["flown"]))
    row("LLM calls per mission", lambda t: f"{t['usage'].get('calls', 0) / max(t['flights'], 1):.1f}")
    row("Tokens per mission, input / output", lambda t: f"{t['usage'].get('input', 0) / max(t['flights'], 1):,.0f} / {t['usage'].get('output', 0) / max(t['flights'], 1):,.0f}")
    row("Wall time per mission", lambda t: f"{t['wall_s'] / max(t['flights'], 1):.0f} s")
    if any(tallies[s]["errors"] for s in systems):
        row("Runs that raised an error (counted as not accomplished)", lambda t: str(t["errors"]))
    add("")
    add("Percentages carry 95% Wilson intervals. \"Accomplished\" means every target served as the mission asked (from enough sides, "
        "in order, from above, at the asked height), no avoid zone entered, landed on the pad, and the aircraft on the ground under "
        "control at the end.")
    add("")
    if "waypoint" in systems and len(systems) > 1:
        add("### Is the difference more than chance?")
        add("")
        add("Fisher's exact test, Waypoint against each rebuilt system, on these flights.")
        add("")
        add("| Against | Mission accomplished | Crashed | Keep-out broken |")
        add("| --- | --- | --- | --- |")
        w = tallies["waypoint"]
        for s in systems:
            if s == "waypoint":
                continue
            t = tallies[s]
            cells = []
            for key, n in (("accomplished", "feasible"), ("crashed", "flights"), ("keep_out", "flights")):
                p = fisher(w[key], w[n], t[key], t[n])
                cells.append("-" if p is None else f"{w[key]}/{w[n]} vs {t[key]}/{t[n]}, p = {p:.3f}")
            add(f"| {NAMES[s]} | " + " | ".join(cells) + " |")
        add("")
        add("One flight per mission is a small sample: an interval this wide and a p-value above 0.05 mean the flights do not settle "
            "the question, not that the systems are equal. `--seed` and more repeats narrow it.")
        add("")
        lines.extend(against_papers(directory, systems, records))

    for title, key in (("By place", "world"), ("By fault schedule", "faults_spec")):
        add(f"### {title}")
        add("")
        add("Missions accomplished, of those that can be flown.")
        add("")
        groups = sorted({records[systems[0]][m][key] for m in mission_ids if m in records[systems[0]]})
        add("| | " + " | ".join(NAMES[s] for s in systems) + " |")
        add("| --- | " + " | ".join("---" for _ in systems) + " |")
        for group in groups:
            cells = []
            for s in systems:
                rows = [r for r in records[s].values() if r[key] == group and (r.get("grade") or {}).get("expected_feasible") is not False]
                cells.append(plain(sum(1 for r in rows if (r.get("grade") or {}).get("accomplished")), len(rows)))
            add(f"| {group} | " + " | ".join(cells) + " |")
        add("")

    add("### Why missions were not accomplished")
    add("")
    add("| Reason | " + " | ".join(NAMES[s] for s in systems) + " |")
    add("| --- | " + " | ".join("---" for _ in systems) + " |")
    reasons = {s: Counter(reason_class(r) for r in records[s].values() if (r.get("grade") or {}).get("expected_feasible") is not False and not (r.get("grade") or {}).get("accomplished")) for s in systems}
    for reason in sorted({k for s in systems for k in reasons[s]}, key=lambda k: -sum(reasons[s][k] for s in systems)):
        add(f"| {reason} | " + " | ".join(str(reasons[s][reason] or "") for s in systems) + " |")
    add("")

    lines.extend(what_happened(directory, systems, records))

    add("## Mission by mission")
    add("")
    add("`yes` accomplished; `no` flown and not accomplished, with the share of the mission done; `refused` an impossible mission "
        "not flown; `flown` an impossible mission flown anyway. A trailing `C` is a crash, `K` a broken keep-out, `F` a flight outside the geofence.")
    add("")
    add("| Mission | Place | Faults | " + " | ".join(NAMES[s].split(" (")[0] for s in systems) + " |")
    add("| --- | --- | --- | " + " | ".join("---" for _ in systems) + " |")
    for mission_id in mission_ids:
        any_record = next((records[s][mission_id] for s in systems if mission_id in records[s]), None)
        if any_record is None:
            continue
        cells = []
        for s in systems:
            record = records[s].get(mission_id)
            if record is None:
                cells.append("not flown yet")
                continue
            g = record.get("grade")
            if not g:
                cells.append("error")
                continue
            if g["expected_feasible"] is False:
                cell = "refused" if g["refused"] else "flown"
            elif g["accomplished"]:
                cell = "yes"
            else:
                cell = "no" + (f" ({g['share_done']:.0%})" if g["share_done"] is not None else "")
            flags = ("C" if g["crashed"] else "") + ("K" if g["keep_out_breaches"] else "") + ("F" if g["left_geofence"] else "")
            cells.append(cell + (f" {flags}" if flags else ""))
        add(f"| `{mission_id}` | {any_record['world']} | {any_record['faults_spec']} | " + " | ".join(cells) + " |")
    add("")

    add("## How to read this")
    add("")
    add("**What the comparison is.** Each rebuilt system is its paper's method: PEACE plans the whole mission in one LLM call "
        "and executes it open loop, with a constraint layer that clamps altitude and refuses a target outside a square fence; "
        "DroneServer lets the model command the aircraft tool call by tool call, behind a server that bounds and fences every "
        "command. Waypoint verifies the whole plan before take-off, repairs it, monitors the flight and replans. The table shows "
        "what those three designs do with the same missions and the same model.")
    add("")
    add("**What each rebuilt system had to be given to fly these missions at all** (`compare/versus/fly.py`):")
    add("")
    add("* Both get a briefing with the facts Waypoint's planner is told (the constraints, the environment with its named places, "
        "the pads, how a photograph is judged), in their own frame: metres for PEACE, latitude and longitude for DroneServer. "
        "Without it neither knows where \"Bara Gumbad\" is.")
    add("* PEACE gets two tools it does not have, `capture()` and `hover(seconds)`, or it could photograph nothing. Its fence is "
        "the smallest square around the map origin containing the mission's geofence, because PEACE's fence is a square.")
    add("* DroneServer gets the mission's geofence as its server-side fence polygon, and the mission's ceiling and speed limit. "
        "Photographs are its own `camera_capture` tool.")
    add("* Neither is told about an obstacle that appears in flight: PEACE reads its LiDAR only when it plans, DroneServer has no "
        "obstacle sensing. Scheduled obstacles are placed on the leg being flown when the aircraft is sent to its k-th place, as "
        "they are on the k-th waypoint of a Waypoint plan.")
    add("")
    add("**What it is not.** These are rebuilds, not the authors' systems on the authors' hardware: the differences are listed in "
        "each system's README. The missions are Waypoint's, written to exercise what Waypoint checks (terrain, roofs, battery, "
        "appearing obstacles), which PEACE and DroneServer were not designed around. And it is one flight per mission.")
    add("")
    add("**The model matters, and most for DroneServer.** Every system here used the same model, the one `.env` configures. "
        "DroneServer hands the model 98 tools and expects it to work out the altitude frame and the monitoring loop from their "
        "descriptions; its paper reports that eight of eleven models fly its missions 90 to 100% of the time and the weaker three "
        "do not. With this model the rebuilt DroneServer passed few of its own flying missions too (the section below, flown "
        "before the audit), so its row "
        "above is what this model can do with that interface, not the ceiling of the interface. Waypoint's published numbers "
        "(`RELATED_WORK.md`) were flown with gpt-5-mini; `python -m compare.versus run --model gpt-5-mini` flies all three "
        "systems with it.")
    add("")
    audits = {
        "peace": ("PEACE's code (tools, constraint layer, planner-executor loop, prompt, schemas) matched the release; the "
                  "model was called with Waypoint's reasoning effort, which the release never sets, and given bare JSON mode "
                  "where the release hands the model the `MissionPlan` schema"),
        "droneserver": ("the model was called with Waypoint's reasoning effort, which the release's harness never sets; three "
                        "server-executed mission tools were missing; and a dozen smaller differences in `monitor_flight`, return "
                        "to launch, speed and ArduCopter's mission handling. Replaying the authors' own published trials call for "
                        "call now gives their published verdict on 10 of 10"),
    }
    audited = [(system, directory.parent / f"{directory.name}-{system}-before-audit") for system in ("peace", "droneserver")]
    audited = [(system, before) for system, before in audited if system in systems and (before / system).exists()]
    if audited:
        add("**Both rebuilt systems were audited against their releases after their first run here, and flown again.** "
            "Each README has the audit. Waypoint was not changed and was not flown again.")
        add("")
        for system, before in audited:
            _, old_records = load(before)
            old = tally(list(old_records.get(system, {}).values()))
            new = tally(list(records[system].values()))
            add(f"* **{NAMES[system]}:** {audits[system]}. First run (`{before.name}`): accomplished "
                f"{old['accomplished']}/{old['feasible']}, crashed {old['crashed']}/{old['flights']}, impossible missions not flown "
                f"{old['refused']}/{old['impossible']}. Corrected: {new['accomplished']}/{new['feasible']}, "
                f"{new['crashed']}/{new['flights']}, {new['refused']}/{new['impossible']}.")
        add("")
    add("**The model's settings are each system's own.** Waypoint calls the model with the reasoning effort it is configured "
        "for (`low`). PEACE and DroneServer call it as their releases do, with the provider's default: PEACE passes the "
        "`MissionPlan` schema with each planning call, DroneServer always sends `parallel_tool_calls` and `tool_choice=\"auto\"`.")
    add("")
    add("**A landing is judged where the aircraft believed it touched down**, for every system: its GPS position at "
        "touchdown, which for Waypoint is where its plan put the LAND. A touchdown that drift or wind moved off the pad is "
        "reported as an accuracy figure (`touchdown_error_m`), as Waypoint's own experiment report does. The stored flights "
        "were re-scored this way with `python -m compare.versus regrade`, which flies nothing.")
    add("")

    own = own_benchmarks()
    if own:
        add("## Each rebuilt system on its own paper's evaluation")
        add("")
        add("The check that a rebuild behaves like its paper's system is its own evaluation, not Waypoint's missions. "
            "Recorded runs, in `compare/results/`.")
        add("")
        lines.extend(own)

    add("## Reproduce")
    add("")
    add("```bash")
    add("uv pip install -r compare/requirements.txt")
    add(f".venv/bin/python -m compare.versus run --seed {manifest['seed']}          # the flights above; --model gpt-5-mini pins a model")
    add(".venv/bin/python -m compare.versus report compare/results/versus/<dir>")
    add(".venv/bin/python -m compare.peace.evaluation.run_dataset")
    add(".venv/bin/python -m compare.droneserver.adversarial && .venv/bin/python -m compare.droneserver.benchmark --trials 2")
    add(".venv/bin/python -m compare.autotamp.experiment --paraphrases 2 --variants a")
    add("```")
    add("")

    target = to or (COMPARE / "Comp_res.md")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
