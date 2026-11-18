"""Replay the authors' published trials through the rebuilt server: is it the same server?

The release ships the record of real LLM trials flown against the real
DroneServer on ArduPilot SITL (`docs/benchmark_runs/`). Two runs record every
tool call with its arguments, the server's answer, the rule that fired, and a
UTC timestamp:

    gpt5.2-final               GPT-5.2, T1 to T5 and T7 to T9, 225 calls, with full transcripts
    gemini-flash-lite-smoke    Gemini 3.5 Flash-Lite, T1 and T9, 18 calls

`replay` sends exactly those calls, in the same order, through the rebuilt MCP
server over a real MCP session, with the simulated clock pinned to the
recorded timestamps (so the aircraft has flown for as long between two calls
as the real one had). It then compares, call by call, the status and the rule
of the rebuilt server's answer with the real server's, the keys of the answer
with the real answer's (where the transcript has it), and the trial's verdict
with the published verdict.

What differs from the real trial on purpose:

* confirmation tokens are random, so a recorded token is replaced by the one
  the rebuilt server issued for the same tool;
* the aircraft starts at the position the transcript shows for that trial,
  with the autopilot parameters the previous trials left (one long-lived
  simulator, as upstream);
* the model is not in the loop: if the rebuilt server answers differently, the
  replay still sends the call the model sent after the real answer. A
  difference therefore shows once, where it happens, and can echo after it.

    python -m compare.droneserver.replay
"""

from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from compare.droneserver.benchmark.missions import DEFAULT_CONTEXT, fresh_vehicle
from compare.droneserver.benchmark.verdicts import Track, judge
from compare.droneserver.llm.agent import CallRecord
from compare.droneserver.server import DroneServer, control_settings
from compare.droneserver.vehicle import DEFAULT_PARAMETERS

RUNS = Path(__file__).parent / "data" / "upstream_runs"
#: Where the authors' SITL stood (Irvine, California): the first position the GPT-5.2 run read.
UPSTREAM_ORIGIN = (33.6485529, -117.8416338, 25.360000610351562)


@dataclass
class CallComparison:
    turn: int
    seq: int
    tool: str
    arguments: dict
    upstream_status: str
    upstream_rule: str
    status: str
    rule: str
    upstream_keys: list[str] | None = None
    keys: list[str] = field(default_factory=list)

    @property
    def same_status(self) -> bool:
        return self.upstream_status == self.status and (self.upstream_rule or "") == (self.rule or "")

    @property
    def same_keys(self) -> bool | None:
        if self.upstream_keys is None:
            return None
        return set(self.upstream_keys) == set(self.keys)


@dataclass
class TrialReplay:
    run: str
    mission: str
    trial: int
    model: str
    upstream_verdict: str
    upstream_reason: str
    verdict: str
    reason: str
    calls: list[CallComparison]
    harness_landed: bool


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _transcript_results(run_dir: Path, mission: str, trial: int) -> list[dict] | None:
    path = run_dir / "transcripts" / f"{mission}_t{trial}.jsonl"
    if not path.exists():
        return None
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("role") == "tool":
            try:
                results.append(json.loads(record["content"]))
            except ValueError:
                results.append({})
    return results


def _start_position(results: list[dict] | None) -> tuple[float, float, float] | None:
    for result in results or []:
        for key in ("position", "home"):
            where = result.get(key) if isinstance(result, dict) else None
            if isinstance(where, dict) and "latitude_deg" in where and "absolute_altitude_m" in where:
                return (where["latitude_deg"], where["longitude_deg"], where["absolute_altitude_m"])
    return None


def replay_run(name: str, run_dir: Path | None = None) -> list[TrialReplay]:
    run_dir = run_dir or RUNS / name
    missions = list(csv.DictReader((run_dir / "missions.csv").open(encoding="utf-8")))
    calls = list(csv.DictReader((run_dir / "tool_calls.csv").open(encoding="utf-8")))
    missions.sort(key=lambda m: m["started_utc"])
    parameters = dict(DEFAULT_PARAMETERS)
    origin = UPSTREAM_ORIGIN
    out = []
    for mission in missions:
        rows = [c for c in calls if c["mission_id"] == mission["mission_id"] and c["trial"] == mission["trial"]]
        rows.sort(key=lambda c: (int(c["turn"]), int(c["seq"])))
        results = _transcript_results(run_dir, mission["mission_id"], int(mission["trial"]))
        origin = _start_position(results) or origin
        replay = asyncio.run(_replay_trial(name, mission, rows, results, origin, parameters))
        out.append(replay)
    return out


async def _replay_trial(name, mission, rows, results, origin, parameters) -> TrialReplay:
    from compare.droneserver.vehicle import Vehicle  # noqa: PLC0415

    template = fresh_vehicle()
    vehicle = Vehicle(template.stack, origin=origin, parameters=parameters)
    param_before = vehicle.parameters.get(DEFAULT_CONTEXT["param_name"])
    settings = control_settings()
    server = DroneServer(vehicle, settings, client_name=mission.get("model"))
    began = _when(mission["started_utc"])
    tokens: dict[str, str] = {}
    comparisons: list[CallComparison] = []
    records: list[CallRecord] = []
    upstream_results = iter(results) if results is not None else None
    async with server.client() as mcp:
        for row in rows:
            arguments = json.loads(row["arguments"] or "{}")
            upstream = next(upstream_results, None) if upstream_results is not None else None
            if row.get("client_side_rejection"):
                continue  # the real harness answered it itself; nothing reached the server
            # The aircraft has flown until the moment the real call was made.
            due = (_when(row["started_utc"]) - began).total_seconds()
            if due > vehicle.stack.now:
                vehicle.sleep(due - vehicle.stack.now)
            if arguments.get("confirm_token") and row["tool"] in tokens:
                arguments["confirm_token"] = tokens[row["tool"]]
            result = await server.call(mcp, row["tool"], arguments)
            if result.get("confirm_token"):
                tokens[row["tool"]] = result["confirm_token"]
            comparisons.append(
                CallComparison(
                    turn=int(row["turn"]), seq=int(row["seq"]), tool=row["tool"], arguments=arguments,
                    upstream_status=row["status"], upstream_rule=row.get("rule") or "",
                    status=str(result.get("status", "")), rule=str(result.get("rule") or ""),
                    upstream_keys=sorted(upstream) if isinstance(upstream, dict) else None, keys=sorted(result),
                )
            )
            records.append(
                CallRecord(
                    turn=int(row["turn"]), seq=int(row["seq"]), tool=row["tool"], arguments=arguments,
                    status=str(result.get("status", "")), rule=result.get("rule"),
                    confirmation_required=result.get("status") == "confirmation_required", result=result,
                )
            )
    # The model's last turn, then the end of the trial.
    end = float(mission.get("duration_s") or 0.0)
    if end > vehicle.stack.now:
        vehicle.sleep(end - vehicle.stack.now)
    track = Track.of(vehicle)
    harness_landed = False
    if vehicle.in_air and not vehicle.stack.ended:
        harness_landed = True
        vehicle.land()
        vehicle.stack.advance_until(lambda: not vehicle.in_air, 300.0, poll=1.0)
    context = dict(DEFAULT_CONTEXT, max_altitude_m=settings.max_altitude_m, geofence_radius_m=settings.geofence_max_radius_m)
    observed = [r.result.get("new_value") for r in records if r.tool == "set_parameter" and r.status == "success"]
    extra = {
        "param_before": param_before,
        "param_after": vehicle.parameters.get(context["param_name"]),
        "param_observed_values": [v for v in observed if v is not None],
        "model_turns": int(mission.get("turns") or 1),
        "model_claim": mission.get("model_claim"),
    }
    verdict = judge(mission["mission_id"], track, records, context, extra)
    parameters.update(vehicle.parameters)  # one long-lived autopilot: what this trial set, the next one reads
    return TrialReplay(
        run=name, mission=mission["mission_id"], trial=int(mission["trial"]), model=mission.get("model", ""),
        upstream_verdict=mission["verdict"], upstream_reason=mission["reason"],
        verdict="PASS" if verdict.passed else "FAIL", reason=verdict.reason, calls=comparisons, harness_landed=harness_landed,
    )


def replay_all() -> list[TrialReplay]:
    return [trial for name in ("gpt5.2-final", "gemini-flash-lite-smoke") for trial in replay_run(name)]


def summarise(trials: list[TrialReplay]) -> dict:
    calls = [c for t in trials for c in t.calls]
    keyed = [c for c in calls if c.same_keys is not None]
    return {
        "trials": len(trials),
        "same_verdict": sum(1 for t in trials if t.verdict == t.upstream_verdict),
        "calls": len(calls),
        "same_status_and_rule": sum(1 for c in calls if c.same_status),
        "answers_compared": len(keyed),
        "same_answer_keys": sum(1 for c in keyed if c.same_keys),
    }


def main() -> None:
    import logging  # noqa: PLC0415

    logging.getLogger("mcp").setLevel(logging.WARNING)
    trials = replay_all()
    print(f"{'trial':16} {'published':10} {'rebuilt':10} {'calls same':>11}  reasons")
    for t in trials:
        same = sum(1 for c in t.calls if c.same_status)
        print(f"{t.model[:8]+' '+t.mission:16} {t.upstream_verdict:10} {t.verdict:10} {same:>5}/{len(t.calls):<5}  "
              f"published: {t.upstream_reason[:60]} | rebuilt: {t.reason[:60]}")
        for c in t.calls:
            if not c.same_status:
                print(f"      turn {c.turn:>2}.{c.seq} {c.tool:18} published {c.upstream_status}{' ('+c.upstream_rule+')' if c.upstream_rule else ''}"
                      f"  rebuilt {c.status}{' ('+c.rule+')' if c.rule else ''}  {json.dumps(c.arguments)[:90]}")
    s = summarise(trials)
    print(f"\nverdicts: {s['same_verdict']}/{s['trials']} the same; calls: {s['same_status_and_rule']}/{s['calls']} with the same status and rule; "
          f"answers: {s['same_answer_keys']}/{s['answers_compared']} with the same keys")
    differing = sorted({(c.tool, tuple(sorted(set(c.upstream_keys) ^ set(c.keys)))) for t in trials for c in t.calls if c.same_keys is False})
    for tool, keys in differing:
        print(f"   keys differ: {tool}: {keys}")


if __name__ == "__main__":
    main()
