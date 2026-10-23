"""The experiment harness: batch specs, ablations, the batch runner, metrics and the report.

Every planner here is scripted or faked; nothing reaches the network.
"""

from __future__ import annotations

import csv
import json
import random

import pytest

from conftest import VALID_PAYLOAD, FakeClient, json_response
from waypoint.experiments import (
    CONDITIONS,
    Batch,
    BatchConfig,
    MissionSpec,
    aggregate,
    load,
    prepare,
    random_schedule,
    run_batch,
    unverified,
    write_report,
)
from waypoint.experiments.cli import DEFAULT_MISSIONS, main
from waypoint.experiments.metrics import NO_FAULT, MissionRow, attribute
from waypoint.experiments.report import format_summary
from waypoint.experiments.spec import mission_seed
from waypoint.planner import DEFAULT_CONSTRAINTS, Plan, PlannerOptions, plan, replan
from waypoint.planner.models import Action, SubTask, Waypoint
from waypoint.planner.prompts import FLAT_SYSTEM_PROMPT, SYSTEM_PROMPT
from waypoint.sim import FaultType


def good_plan(mission="survey the north field") -> Plan:
    return Plan.model_validate({**VALID_PAYLOAD, "mission_text": mission})


def too_high(mission: str) -> Plan:
    """The survey at 150 m, over the 120 m ceiling."""
    p = good_plan(mission)
    for subtask in p.subtasks[:-1]:
        for wp in subtask.waypoints:
            wp.z = 150.0
    return p


class Scripted:
    """Planners per condition: missions with "high" in them get a first plan over the ceiling;
    in the air, the replanner heads straight home."""

    def __init__(self):
        self.options: list[PlannerOptions] = []
        self.calls: list[tuple[str, str]] = []

    def __call__(self, options: PlannerOptions):
        self.options.append(options)

        def planner(mission, constraints, **kwargs):
            self.calls.append((options.label, "plan"))
            return too_high(mission) if "high" in mission else good_plan(mission)

        def replanner(mission, previous, violations, constraints, **kwargs):
            if kwargs.get("situation") is None:
                self.calls.append((options.label, "ground"))
                return good_plan(mission)
            self.calls.append((options.label, "inflight"))
            x, y, _ = kwargs["home"]
            lx, ly, _ = kwargs["landing"]
            return Plan(mission_text=mission, subtasks=[SubTask(name="home", waypoints=[
                Waypoint(x=x, y=y - 20, z=25, speed=4),
                Waypoint(x=lx, y=ly, z=25, speed=4),
                Waypoint(x=lx, y=ly, z=0, speed=2, action=Action.LAND),
            ])])

        return planner, replanner


@pytest.fixture
def batch_file(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"missions": [
        {"id": "clean", "mission": "survey the north field", "faults": "none"},
        {"id": "crane", "mission": "survey the north field again", "faults": "crane_ahead"},
        {"id": "high", "mission": "survey the north field from high up", "faults": "none"},
    ]}))
    return path


@pytest.fixture
def flown(batch_file, tmp_path):
    scripted = Scripted()
    config = BatchConfig(
        missions_path=batch_file,
        conditions=[CONDITIONS[0], CONDITIONS[1]],
        out_dir=tmp_path / "results",
    )
    out = run_batch(config, planners=scripted, progress=lambda _: None)
    return scripted, out


# ---------------------------------------------------------------------------
# Specs and faults
# ---------------------------------------------------------------------------


def test_the_bundled_batch_has_thirty_six_missions_that_all_set_up():
    batch = Batch.load(DEFAULT_MISSIONS)
    assert len(batch.missions) == 36
    for spec in batch.missions:
        scenario = prepare(spec)  # every world loads and every pad resolves
        assert scenario.constraints.geofence


def test_every_mission_has_ground_truth_that_grounds_and_agrees_with_the_assessment():
    """Every landmark the truth names exists, and the truth's feasibility is what assess() finds."""
    from waypoint.intent import assess, ground

    batch = Batch.load(DEFAULT_MISSIONS)
    for spec in batch.missions:
        assert spec.expect is not None, spec.id
        scenario = prepare(spec)
        intent = ground(spec.expect.payload(), scenario.world, landing=scenario.landing)
        assert not [n for n in intent.notes if "not a landmark" in n], spec.id
        reasons = assess(intent, constraints=scenario.constraints, world=scenario.world,
                         home=scenario.home, landing=scenario.landing)
        if spec.expect.feasible is not None:
            assert (not reasons) == spec.expect.feasible, (spec.id, reasons)


def test_a_named_pad_moves_the_start_and_the_landing():
    spec = MissionSpec(id="x", mission="m", world="lodhi", start="the Bonsai Park", land="the east lawn")
    scenario = prepare(spec)
    assert scenario.home[:2] != (0.0, 0.0)
    assert scenario.landing != scenario.home


def test_sample_faults_are_refused_in_a_batch():
    with pytest.raises(ValueError, match="sample"):
        MissionSpec(id="x", mission="m", faults="sample")


def test_mission_ids_must_be_unique():
    with pytest.raises(ValueError, match="twice"):
        Batch.model_validate({"missions": [{"id": "a", "mission": "m"}, {"id": "a", "mission": "n"}]})


def test_random_faults_are_reproducible_and_differ_by_mission():
    spec = MissionSpec(id="tour", mission="m", faults="random")
    first, again = prepare(spec, seed=3), prepare(spec, seed=3)
    assert first.faults == again.faults and first.fault_seed == mission_seed(3, "tour", 0)
    assert prepare(spec, seed=4).faults != first.faults
    assert prepare(spec, seed=3, repeat=1).faults != first.faults


def test_random_schedules_stay_in_their_ranges():
    rng = random.Random(0)
    for _ in range(200):
        schedule = random_schedule(rng)
        kinds = [e.type for e in schedule.events]
        assert 1 <= len(kinds) <= 3 and len(set(kinds)) == len(kinds)
        for event in schedule.events:
            if event.type is FaultType.OBSTACLE_APPEARS:
                assert 25 <= event.ahead <= 50 and event.obstacle.radius >= 5
            else:
                assert 5 <= event.trigger.time <= 40 and event.duration > 0


# ---------------------------------------------------------------------------
# Ablations, in the planner
# ---------------------------------------------------------------------------


def test_the_planner_as_designed_is_unchanged():
    client = FakeClient(json_response(VALID_PAYLOAD))
    plan("survey", DEFAULT_CONSTRAINTS, client=client)
    assert client.last_call["system_instruction"] == SYSTEM_PROMPT
    assert "Decompose this mission" in client.last_prompt


def test_the_flat_planner_asks_for_one_route_and_gets_one():
    client = FakeClient(json_response(VALID_PAYLOAD))  # the model decomposes anyway
    result = plan("survey", DEFAULT_CONSTRAINTS, client=client, options=PlannerOptions(hierarchical=False))
    assert client.last_call["system_instruction"] == FLAT_SYSTEM_PROMPT
    assert "Stage 1" not in FLAT_SYSTEM_PROMPT and "hierarchical" not in FLAT_SYSTEM_PROMPT
    assert 'single subtask named "route"' in client.last_prompt
    assert [s.name for s in result.subtasks] == ["route"]
    assert result.waypoints() == good_plan().waypoints()


def test_the_flat_prompt_keeps_every_safety_rule():
    rules = SYSTEM_PROMPT[SYSTEM_PROMPT.index("Safety is your"):SYSTEM_PROMPT.index("Worked example.")]
    assert rules in FLAT_SYSTEM_PROMPT


def test_without_violation_feedback_the_reasons_are_withheld():
    reasons = ["waypoint 1 is 30 m above the ceiling"]
    told, untold = FakeClient(json_response(VALID_PAYLOAD)), FakeClient(json_response(VALID_PAYLOAD))
    replan("survey", good_plan(), reasons, DEFAULT_CONSTRAINTS, client=told)
    replan("survey", good_plan(), reasons, DEFAULT_CONSTRAINTS, client=untold,
           options=PlannerOptions(violation_feedback=False))
    assert reasons[0] in told.last_prompt
    assert reasons[0] not in untold.last_prompt and "did not say why" in untold.last_prompt


def test_without_violation_feedback_in_flight_rejections_are_withheld_too():
    reasons = ["leg 2 passes 3 m from the crane"]
    client = FakeClient(json_response(VALID_PAYLOAD))
    replan("survey", good_plan(), reasons, DEFAULT_CONSTRAINTS, client=client, situation="a crane appeared",
           home=(0, 40, 25), options=PlannerOptions(violation_feedback=False))
    assert reasons[0] not in client.last_prompt and "a crane appeared" in client.last_prompt


def test_condition_labels_are_distinct():
    assert len({c.label for c in CONDITIONS}) == 4
    assert CONDITIONS[0] == PlannerOptions()


def test_the_baseline_gate_refuses_nothing_but_records_everything():
    result = unverified(too_high("m"), DEFAULT_CONSTRAINTS)
    assert result.passed and result.blocking


# ---------------------------------------------------------------------------
# The batch, end to end
# ---------------------------------------------------------------------------


def test_every_condition_flies_every_mission_with_its_own_planner(flown):
    scripted, out = flown
    manifest = json.loads((out / "manifest.json").read_text())
    assert [o.label for o in scripted.options] == [CONDITIONS[0].label, CONDITIONS[1].label]
    assert len(manifest["flights"]) == 6 and manifest["finished"]
    assert all(f["error"] is None for f in manifest["flights"])
    assert manifest["config"]["conditions"] == [CONDITIONS[0].label, CONDITIONS[1].label]


def test_the_baseline_flies_the_same_first_plan_unverified(flown):
    _, out = flown
    manifest = json.loads((out / "manifest.json").read_text())
    high = next(f for f in manifest["flights"] if f["mission_id"] == "high")
    log = [json.loads(line) for line in (out / high["baseline"]["log"]).read_text().splitlines()]
    assert log[0]["verifier"] == "unverified"
    proposed = [e for e in log if e["kind"] == "plan_proposed"]
    assert proposed[0]["waypoints"][0]["z"] == 150.0  # the rejected plan, flown anyway
    assert high["baseline"]["outcome"] == "completed"


def test_the_metrics_count_what_happened(flown):
    _, out = flown
    rows, replans, _ = load(out)
    summary = aggregate(rows, replans)[CONDITIONS[0].label]

    assert summary["planned"] == 3 and summary["caught"] == 1
    assert summary["catch_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["violation_types"] == {"ALTITUDE": 1}
    assert summary["ground_correction_rate"] == 1.0
    assert summary["completion_rate"] == 1.0

    (trigger,) = [r for r in replans if r.condition == CONDITIONS[0].label]
    assert trigger.trigger == "OBSTACLE_ON_ROUTE" and trigger.fault_type == "OBSTACLE_APPEARS"
    assert trigger.bucket == "first_try" and trigger.latency_s is not None and trigger.latency_s >= 0
    assert summary["replans"]["first_try_rate"] == 1.0

    tradeoff = summary["tradeoff"]
    assert tradeoff["baselines"] == 3
    assert tradeoff["pairs"] == 1  # only "high", where the verifier changed the plan
    assert tradeoff["baseline_unsafe_plan_rate"] == pytest.approx(1 / 3, abs=1e-3)
    high = next(r for r in rows if r.mission_id == "high" and r.condition == CONDITIONS[0].label)
    assert high.baseline_unsafe_plans == 1 and high.distance_overhead_pct is not None


def test_the_catch_rate_is_pooled_over_conditions_that_share_a_first_prompt():
    def row(condition, caught):
        return MissionRow(condition=condition, mission_id="m", repeat=0, world="open", faults_spec="none",
                          fault_types="", planned=True, initial_caught=caught)

    hier_fb, flat_fb, hier_nofb = CONDITIONS[0].label, CONDITIONS[1].label, CONDITIONS[2].label
    rows = [row(hier_fb, True), row(hier_fb, False), row(hier_nofb, True), row(hier_nofb, True), row(flat_fb, False)]
    summary = aggregate(rows, [])
    pooled = summary[hier_fb]["first_plan"]
    assert pooled == summary[hier_nofb]["first_plan"]
    assert (pooled["caught"], pooled["planned"], pooled["prompt"]) == (3, 4, "hierarchical")
    assert summary[flat_fb]["first_plan"]["planned"] == 1  # flat asks differently: not pooled in
    low, high = summary[hier_fb]["catch_rate_ci95"]
    assert 0.0 <= low < 0.5 < high <= 1.0
    text = format_summary(summary)
    assert "pooled, same first prompt" in text and "sampling noise" in text


def test_the_manifest_says_whether_the_checkout_was_dirty(flown):
    _, out = flown
    config = json.loads((out / "manifest.json").read_text())["config"]
    assert "git_dirty" in config and config["git_dirty"] in (True, False, None)


def test_the_report_writes_csvs_a_summary_and_plots(flown, tmp_path):
    pytest.importorskip("matplotlib")
    _, out = flown
    rows, replans, _ = load(out)
    lines: list[str] = []
    write_report(rows, replans, out, echo=lines.append)
    with (out / "per_mission.csv").open() as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 6 and "initial_caught" in records[0]
    assert (out / "replans.csv").exists() and json.loads((out / "summary.json").read_text())
    for name in ("catch_rate.png", "replan_latency.png", "replan_outcomes.png", "tradeoff.png"):
        assert (out / name).stat().st_size > 1000
    assert "verifier catch rate" in lines[0] and "33%" in lines[0]


def test_one_bad_mission_does_not_end_the_batch(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"missions": [
        {"id": "nowhere", "mission": "m", "world": "lodhi", "start": "no such landmark"},
        {"id": "fine", "mission": "survey the north field"},
    ]}))
    out = run_batch(
        BatchConfig(missions_path=path, conditions=[CONDITIONS[0]], baseline=False, out_dir=tmp_path / "r"),
        planners=Scripted(), progress=lambda _: None,
    )
    flights = json.loads((out / "manifest.json").read_text())["flights"]
    assert "no such landmark" in flights[0]["error"] and flights[1]["verified"]["outcome"] == "completed"
    rows, replans, _ = load(out)
    assert aggregate(rows, replans)[CONDITIONS[0].label]["errors"] == 1


def test_repeats_and_selection(batch_file, tmp_path):
    out = run_batch(
        BatchConfig(missions_path=batch_file, conditions=[CONDITIONS[0]], only=["crane"], repeats=2,
                    baseline=False, seed=1, out_dir=tmp_path / "r"),
        planners=Scripted(), progress=lambda _: None,
    )
    flights = json.loads((out / "manifest.json").read_text())["flights"]
    assert [(f["mission_id"], f["repeat"]) for f in flights] == [("crane", 0), ("crane", 1)]
    assert "r1" in flights[1]["verified"]["log"]
    with pytest.raises(ValueError, match="nope"):
        run_batch(BatchConfig(missions_path=batch_file, conditions=[CONDITIONS[0]], only=["nope"],
                              out_dir=tmp_path / "s"), planners=Scripted(), progress=lambda _: None)


# ---------------------------------------------------------------------------
# Metric details
# ---------------------------------------------------------------------------


def test_a_trigger_is_blamed_on_the_latest_fault_that_could_cause_it():
    faults = [(3.0, "GPS_DRIFT"), (9.0, "WIND_GUST"), (12.0, "OBSTACLE_APPEARS"), (30.0, "WIND_GUST")]
    assert attribute("OFF_ROUTE", 20.0, faults) == "WIND_GUST"
    assert attribute("OFF_ROUTE", 5.0, faults) == "GPS_DRIFT"
    assert attribute("OBSTACLE_ON_ROUTE", 12.0, faults) == "OBSTACLE_APPEARS"
    assert attribute("BATTERY_INSUFFICIENT", 20.0, faults) == NO_FAULT


def test_a_replan_with_no_safe_answer_is_abandoned(batch_file, tmp_path):
    def stubborn(options):
        planner, _ = Scripted()(options)

        def replanner(mission, previous, violations, constraints, **kwargs):
            if kwargs.get("situation") is None:
                return good_plan(mission)
            return previous  # straight on through the crane, every time

        return planner, replanner

    out = run_batch(
        BatchConfig(missions_path=batch_file, conditions=[CONDITIONS[0]], only=["crane"], baseline=False,
                    out_dir=tmp_path / "r"),
        planners=stubborn, progress=lambda _: None,
    )
    rows, replans, _ = load(out)
    (trigger,) = replans
    assert trigger.bucket == "abandoned" and trigger.attempts == 4 and trigger.latency_s is None
    assert trigger.resolution == "failsafe" and rows[0].outcome == "aborted"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_report_recomputes_from_saved_files(flown, tmp_path, capsys):
    _, out = flown
    assert main(["report", str(out), "--out", str(tmp_path / "again"), "--no-plots"]) == 0
    assert (tmp_path / "again" / "per_mission.csv").exists()
    assert "verifier catch rate" in capsys.readouterr().out


def test_report_compares_several_result_directories(flown, tmp_path):
    _, out = flown
    assert main(["report", str(out), str(out), "--out", str(tmp_path / "both"), "--no-plots"]) == 0
    with (tmp_path / "both" / "per_mission.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 12


def test_run_needs_a_planner_key(monkeypatch, capsys):
    monkeypatch.setattr("waypoint.experiments.cli.get_providers", lambda: ())
    assert main(["run", "--limit", "1"]) == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err and "GEMINI_API_KEY" in err


def test_an_uninstalled_backend_fails_before_anything_flies(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WAYPOINT_PX4_DIR", str(tmp_path / "no-px4-here"))
    assert main(["run", "--backend", "px4"]) == 2
    assert "make px4_sitl" in capsys.readouterr().err


def test_contradictory_ablation_flags_are_refused(capsys):
    assert main(["run", "--all-conditions", "--flat"]) == 2


# ---------------------------------------------------------------------------
# Spending less
# ---------------------------------------------------------------------------


def test_a_baseline_without_replans_flies_its_first_plan_and_never_asks_again(batch_file, tmp_path):
    scripted = Scripted()
    out = run_batch(
        BatchConfig(missions_path=batch_file, conditions=[CONDITIONS[0]], only=["crane"],
                    baseline_replans=False, out_dir=tmp_path / "r"),
        planners=scripted, progress=lambda _: None,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["baseline_replans"] is False
    flight = manifest["flights"][0]
    log = [json.loads(line) for line in (out / flight["baseline"]["log"]).read_text().splitlines()]
    assert log[0]["adapt"] is False
    assert "replan_triggered" not in [e["kind"] for e in log]  # the crane is flown into, not around
    assert [kind for _, kind in scripted.calls].count("inflight") == 1  # the verified run's replan only


def test_the_cost_of_each_flight_pair_is_recorded_and_summed(flown, monkeypatch):
    _, out = flown
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["flights"][0]["verified"]["usage"] = {
        "openai:gpt-5-mini": {"calls": 2, "input": 4800, "cached": 4000, "output": 1200, "reasoning": 0}
    }
    manifest["flights"][0]["baseline"]["usage"] = {
        "openai:gpt-5-mini": {"calls": 1, "input": 2400, "cached": 0, "output": 600, "reasoning": 0}
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    rows, replans, _ = load(out)
    cost = aggregate(rows, replans)[CONDITIONS[0].label]["cost"]
    assert cost["llm_calls"] == 3 and cost["tokens_in"] == 7200 and cost["tokens_out"] == 1800
    assert cost["models"] == {"openai:gpt-5-mini": 1}


def test_conditions_can_be_picked_by_label(monkeypatch, capsys):
    captured = {}

    class Stop(Exception):
        pass

    def record(config):
        captured["c"] = config
        raise Stop  # the config is all this test is about; fly nothing

    monkeypatch.setattr("waypoint.experiments.cli.get_providers", lambda: ("openai",))
    monkeypatch.setattr("waypoint.experiments.cli.run_batch", record)
    with pytest.raises(Stop):
        main(["run", "--conditions", "hierarchical+feedback,hierarchical+no-feedback", "--no-baseline-replans"])
    config = captured["c"]
    assert [c.label for c in config.conditions] == ["hierarchical+feedback|full", "hierarchical+no-feedback|full"]
    assert config.baseline_replans is False
    assert main(["run", "--conditions", "nonsense"]) == 2
    assert main(["run", "--conditions", "flat+feedback", "--all-conditions"]) == 2

    with pytest.raises(Stop):
        main(["run", "--systems", "full;classic;runtime-guard;full,no-repair"])
    assert [c.label for c in captured["c"].conditions] == [
        "hierarchical+feedback|full",
        "hierarchical+feedback|classic",
        "hierarchical+feedback|runtime-guard",
        "hierarchical+feedback|gate=preflight,intent,no-repair,replan=tiered,hold=hover,shield",
    ]
    assert main(["run", "--system", "warp-drive"]) == 2


def test_a_mistyped_cost_stops_the_batch_before_it_starts(monkeypatch, capsys):
    monkeypatch.setattr("waypoint.experiments.cli.get_providers", lambda: ("openai",))
    monkeypatch.setenv("COST", "MEDIUM")
    assert main(["run", "--limit", "1"]) == 2
    assert "COST='MEDIUM'" in capsys.readouterr().err
