"""The closed loop end to end, with a scripted planner instead of the API.

The scenario: the first plan breaks the altitude ceiling and is rejected; the
correction passes and flies; forty metres up the first leg a crane appears in
the way; the monitor stops the flight, the replanner routes around it from
where the aircraft is, the verifier passes the detour, and the flight resumes
and lands. Every step has to show up in the MissionLog, in order.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator

import pytest

from conftest import VALID_PAYLOAD
from waypoint.backends import BackendError, ExecutorBackend, make_backend
from waypoint.gazebo.scene import write_scene
from waypoint.orchestrator import MissionLog, run_mission
from waypoint.orchestrator.run_full import main
from waypoint.orchestrator.runlog import read_run
from waypoint.planner import DEFAULT_CONSTRAINTS, Obstacle, Plan
from waypoint.planner.models import Action, SubTask, Waypoint
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, Trigger
from waypoint.sim.backend import CustomSimBackend

CRANE = Obstacle(x=0.0, y=0.0, radius=6.0, height=60.0, name="mobile crane")


def good_plan() -> Plan:
    return Plan.model_validate({**VALID_PAYLOAD, "mission_text": "survey the north field"})


def too_high() -> Plan:
    """The survey at 150 m, over the 120 m ceiling: the verifier must refuse it."""
    plan = good_plan()
    for subtask in plan.subtasks[:-1]:
        for wp in subtask.waypoints:
            wp.z = 150.0
    return plan


def crane_ahead() -> FaultSchedule:
    """The crane appears 40 m up the route as the aircraft sets off north."""
    return FaultSchedule.of(
        FaultEvent(
            type=FaultType.OBSTACLE_APPEARS,
            trigger=Trigger(at_waypoint=1),
            ahead=40.0,
            obstacle=CRANE,
            name="crane spotted",
        )
    )


def detour_around(obstacle: Obstacle) -> list[Waypoint]:
    """Out to the east of it, past it, then the rest of the survey and home."""
    ox, oy = obstacle.x, obstacle.y
    return [
        Waypoint(x=ox + 15, y=oy - 20, z=25, speed=4, action=Action.MOVE),
        Waypoint(x=ox + 15, y=oy + 20, z=25, speed=4, action=Action.MOVE),
        Waypoint(x=0, y=80, z=25, speed=4, action=Action.CAPTURE),
        Waypoint(x=40, y=80, z=25, speed=4, action=Action.CAPTURE),
        Waypoint(x=0, y=0, z=0, speed=2, action=Action.LAND),
    ]


def straight_through() -> list[Waypoint]:
    """The rest of the original route, crane or no crane."""
    return [wp.model_copy() for wp in good_plan().waypoints()[1:]]


class Script:
    """A planner and replanner that answer from a script and remember every call.

    `inflight` is what the in-flight replanner returns on successive calls:
    "detour" routes around the crane, "through" ignores it (and so fails
    verification).
    """

    def __init__(self, *, first: Plan | None = None, inflight: tuple[str, ...] = ("detour",)):
        self.first = first or too_high()
        self.inflight = list(inflight)
        self.calls: list[dict] = []

    def planner(self, mission, constraints, **kwargs) -> Plan:
        self.calls.append({"kind": "plan", "constraints": constraints, **kwargs})
        return self.first

    def replanner(self, mission, previous, violations, constraints, **kwargs) -> Plan:
        situation = kwargs.get("situation")
        self.calls.append({
            "kind": "inflight" if situation else "ground",
            "violations": list(violations),
            "constraints": constraints,
            **kwargs,
        })
        if situation is None:
            return good_plan()
        move = self.inflight.pop(0) if len(self.inflight) > 1 else self.inflight[0]
        crane = next(o for o in constraints.obstacles if o.name == CRANE.name)
        route = detour_around(crane) if move == "detour" else straight_through()
        return Plan(mission_text=mission, subtasks=[SubTask(name="rest_of_mission", waypoints=route)])


def fly(script: Script, tmp_path, **kwargs) -> MissionLog:
    return run_mission(
        "survey the north field",
        DEFAULT_CONSTRAINTS,
        kwargs.pop("faults", crane_ahead()),
        kwargs.pop("backend", None) or CustomSimBackend(),
        planner=script.planner,
        replanner=script.replanner,
        verbose=kwargs.pop("verbose", False),
        runs_dir=tmp_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The whole loop
# ---------------------------------------------------------------------------


@pytest.fixture
def flown(tmp_path):
    script = Script()
    return script, fly(script, tmp_path)


def test_the_log_records_every_step_in_order(flown):
    _, log = flown
    assert log.kinds == [
        "mission_started",
        "plan_proposed", "verification",        # too high: rejected
        "plan_proposed", "verification",        # corrected: passes
        "execution_started",
        "replan_triggered",                     # the crane
        "plan_proposed", "verification",        # the detour: passes
        "execution_resumed",
        "mission_finished",
    ]


def test_each_entry_says_what_it_was(flown):
    _, log = flown
    proposed = log.of("plan_proposed")
    verdicts = log.of("verification")

    assert [p["source"] for p in proposed] == ["initial", "verification_replan", "inflight_replan"]
    assert [v["passed"] for v in verdicts] == [False, True, True]
    assert any(v["type"] == "ALTITUDE" for v in verdicts[0]["violations"])
    assert [v["plan_id"] for v in verdicts] == [p["plan_id"] for p in proposed]

    trigger = log.of("replan_triggered")[0]
    assert trigger["trigger"] == "OBSTACLE_ON_ROUTE"
    assert "mobile crane" in trigger["reason"]
    assert trigger["t"] > 0
    assert log.of("execution_resumed")[0]["plan_id"] == proposed[-1]["plan_id"]
    assert proposed[-1]["t"] == pytest.approx(trigger["t"])

    finished = log.of("mission_finished")[0]
    assert finished["outcome"] == "completed"
    assert finished["inflight_replans"] == 1
    assert finished["verification_attempts"] == 3
    assert log.outcome == "completed"


def test_the_backend_is_recorded(flown):
    _, log = flown
    assert log.backend == "custom"
    assert log.of("mission_finished")[0]["backend"] == "custom"
    assert read_run(log.run_file)["backend"]["name"] == "custom"


def test_the_log_is_persisted_as_jsonl(flown):
    _, log = flown
    assert log.path.suffix == ".jsonl"
    assert log.path.name == log.run_file.name.replace(".json", ".log.jsonl")
    lines = log.path.read_text().splitlines()
    assert len(lines) == len(log.entries)
    assert [json.loads(line) for line in lines] == json.loads(json.dumps(log.entries))
    assert MissionLog.read(log.path)[-1]["kind"] == "mission_finished"
    assert [entry["seq"] for entry in MissionLog.read(log.path)] == list(range(len(lines)))


def test_the_replanner_is_told_where_it_is_and_why(flown):
    script, log = flown
    ask = next(call for call in script.calls if call["kind"] == "inflight")
    trigger = log.of("replan_triggered")[0]

    assert ask["home"] == pytest.approx(tuple(trigger["position"]))
    assert ask["landing"] == (0.0, 0.0, 0.0)
    assert "mobile crane" in ask["situation"]
    assert "Current position" in ask["situation"]
    # It plans against the crane, and against the battery that is left, not the full budget.
    assert any(o.name == "mobile crane" for o in ask["constraints"].obstacles)
    assert ask["constraints"].battery_budget_pct < DEFAULT_CONSTRAINTS.battery_budget_pct


def test_the_flight_resumes_rather_than_restarting(flown):
    _, log = flown
    states = log.report.states
    assert [round(b.timestamp - a.timestamp, 6) for a, b in zip(states, states[1:])] == [0.1] * (len(states) - 1)
    assert all(b.battery_pct <= a.battery_pct for a, b in zip(states, states[1:]))
    assert states[0].z == 0.0 and log.report.completed


def test_the_aircraft_goes_around_the_crane(flown):
    _, log = flown
    report = log.report
    crane = next(o for o in report.obstacles if o["name"] == CRANE.name)
    after = [s for s in report.states if s.timestamp >= crane["appeared_at"]]
    closest = min(math.hypot(s.x - crane["x"], s.y - crane["y"]) for s in after)
    assert closest > crane["radius"]
    # ...and the crane really was on the route it would otherwise have flown.
    assert abs(crane["x"]) < 1.0 and 35.0 < crane["y"] < 45.0


def test_the_run_file_holds_every_plan_that_flew(flown):
    _, log = flown
    run = read_run(log.run_file)
    assert [p["source"] for p in run["plans"]] == ["initial", "inflight_replan"]
    assert run["plans"][1]["t"] == pytest.approx(log.of("replan_triggered")[0]["t"])
    assert run["outcome"] == "completed"


def test_gazebo_shows_the_new_route_when_it_took_over(flown, tmp_path):
    _, log = flown
    run = read_run(log.run_file)
    scene = write_scene(run, tmp_path / "scene")
    late = {entry.t: entry.sdf for entry in scene.late}

    assert any('name="reroute_1"' in sdf for sdf in late.values())
    reroute_at = next(t for t, sdf in late.items() if "reroute_1" in sdf)
    assert reroute_at == pytest.approx(run["plans"][1]["t"])


# ---------------------------------------------------------------------------
# When it cannot go on
# ---------------------------------------------------------------------------


def test_a_rejected_detour_is_corrected_in_the_air(tmp_path):
    script = Script(inflight=("through", "detour"))
    log = fly(script, tmp_path)

    inflight = [p for p in log.of("plan_proposed") if p["source"] == "inflight_replan"]
    assert [p["attempt"] for p in inflight] == [0, 1]
    verdicts = log.of("verification")[-2:]
    assert [v["passed"] for v in verdicts] == [False, True]
    rejection = verdicts[0]["violations"]
    assert any(v["type"] == "MIN_SEPARATION" for v in rejection)
    # Read back to the planner in the air, so it must not call the start "home".
    assert all("from home" not in v["description"] for v in rejection)
    assert any("the current position" in v["description"] for v in rejection)
    # The correction was asked for with the verifier's reasons, still in flight.
    retry = [c for c in script.calls if c["kind"] == "inflight"][1]
    assert retry["violations"] and retry["situation"]
    assert log.outcome == "completed"


def test_no_safe_detour_means_landing_where_it_is(tmp_path):
    script = Script(inflight=("through",))
    log = fly(script, tmp_path, max_replan_attempts=1)

    assert log.kinds[-2:] == ["failsafe", "mission_finished"]
    assert log.outcome == "aborted"
    failsafe = log.of("failsafe")[0]
    last = log.report.states[-1]
    assert math.hypot(last.x - failsafe["position"][0], last.y - failsafe["position"][1]) < 2.0
    assert last.z == pytest.approx(last.ground) and last.velocity == 0.0


def test_too_many_interruptions_abort_the_mission(tmp_path):
    log = fly(Script(), tmp_path, max_total_replans=0)
    assert "plan_proposed" not in log.kinds[log.kinds.index("replan_triggered"):]
    assert log.outcome == "aborted"
    assert "in-flight replans" in log.of("mission_finished")[0]["reason"]


def test_a_plan_that_never_passes_never_flies(tmp_path):
    class Stubborn(Script):
        def replanner(self, *args, **kwargs):
            return too_high()

    log = fly(Stubborn(), tmp_path, max_replan_attempts=2)
    assert log.outcome == "rejected"
    assert [v["passed"] for v in log.of("verification")] == [False, False, False]
    assert "execution_started" not in log.kinds
    assert read_run(log.run_file)["ticks"] == []


def test_a_clean_flight_never_replans(tmp_path):
    log = fly(Script(first=good_plan()), tmp_path, faults=FaultSchedule())
    assert "replan_triggered" not in log.kinds
    assert log.outcome == "completed"


# ---------------------------------------------------------------------------
# Backend-agnostic
# ---------------------------------------------------------------------------


class Relay(ExecutorBackend):
    """Exposes nothing but the ExecutorBackend interface, so the loop cannot
    reach past it into the custom simulator."""

    name = "relay"

    def __init__(self):
        self._inner = CustomSimBackend()
        self.plans: list[Plan] = []

    def prepare(self, constraints, faults, home=None) -> None:
        self._inner.prepare(constraints, faults, home=home)

    def run(self, plan) -> Iterator:
        self.plans.append(plan)
        return self._inner.run(plan)

    @property
    def state(self):
        return self._inner.state

    @property
    def obstacles(self):
        return self._inner.obstacles

    def event_records(self):
        return self._inner.event_records()

    def obstacle_records(self):
        return self._inner.obstacle_records()

    @property
    def distance_flown(self):
        return self._inner.distance_flown

    @property
    def crashed(self):
        return self._inner.crashed

    @property
    def battery_depleted(self):
        return self._inner.battery_depleted


def test_the_loop_only_needs_the_backend_interface(tmp_path):
    relay = Relay()
    log = fly(Script(), tmp_path, backend=relay)
    assert log.outcome == "completed"
    assert log.backend == "relay"
    assert len(relay.plans) == 2  # the first plan, then the detour, on the same flight


def test_px4_is_reserved():
    with pytest.raises(BackendError, match="reserved"):
        make_backend("px4")


def test_the_cli_runs_the_whole_loop(monkeypatch, capsys, tmp_path):
    script = Script()
    monkeypatch.setattr("waypoint.orchestrator.loop.default_planner", script.planner)
    monkeypatch.setattr("waypoint.orchestrator.loop.default_replanner", script.replanner)
    faults = tmp_path / "crane.json"
    faults.write_text(crane_ahead().model_dump_json())

    code = main(["--mission", "survey the north field", "--faults", str(faults), "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert code == 0
    phases = [line.split()[1] for line in out.splitlines() if line.startswith("==> ")]
    assert phases[:6] == ["PLANNING", "VERIFYING", "REPLANNING", "RE-VERIFYING", "EXECUTING", "REPLAN"]
    assert "RESUMING" in phases and phases[-1] == "MISSION"
    assert list(tmp_path.glob("*.log.jsonl")) and list(tmp_path.glob("*[0-9].json"))


def test_a_chosen_start_and_landing_are_flown_and_recorded(tmp_path):
    """Take off somewhere other than the origin, and finish somewhere else again."""
    start, finish = (30.0, -20.0, 0.0), (60.0, 10.0, 0.0)
    seen = {}

    def planner(mission, constraints, **kwargs):
        seen.update(kwargs)
        return Plan(mission_text=mission, subtasks=[SubTask(name="hop", waypoints=[
            Waypoint(x=30, y=-20, z=20, speed=3),
            Waypoint(x=60, y=10, z=20, speed=5),
            Waypoint(x=60, y=10, z=0, speed=2, action=Action.LAND),
        ])])

    log = run_mission(
        "hop across", DEFAULT_CONSTRAINTS, FaultSchedule(), CustomSimBackend(),
        home=start, landing=finish, planner=planner, verbose=False, runs_dir=tmp_path,
    )
    states = log.report.states
    assert (seen["home"], seen["landing"]) == (start, finish)
    assert (states[0].x, states[0].y) == (30.0, -20.0)
    assert math.hypot(states[-1].x - 60, states[-1].y - 10) < 1.5 and log.outcome == "completed"
    assert read_run(log.run_file)["sim"]["config"]["home"] == list(start)
