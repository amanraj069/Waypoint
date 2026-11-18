"""The loop as designed now (`FULL`): intent, repair, tiered replanning, hovering through replans.

Scripted planners again; the point is what the loop does *without* them: a
crane on the route is flown around by repair, with no planner call in the air;
a plan through a no-fly zone is fixed on the ground without asking again; a
mission that cannot be flown is refused before anything is planned; a battery
shortfall gives up a phase instead of the aircraft; and a slow planner costs
hover time, not frozen time.
"""

from __future__ import annotations

import time

from conftest import VALID_PAYLOAD
from waypoint.intent import MissionIntent, Target, TargetAction
from waypoint.orchestrator import run_mission
from waypoint.orchestrator.options import CLASSIC, FULL, SystemOptions, system_from
from waypoint.planner import DEFAULT_CONSTRAINTS, Obstacle, Plan
from waypoint.planner.models import GeoPoint
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, Trigger
from waypoint.sim.backend import CustomSimBackend

CRANE = Obstacle(x=0.0, y=0.0, radius=6.0, height=60.0, name="mobile crane")
CORNERS = MissionIntent(targets=[
    Target(name="south-west corner", x=0.0, y=80.0),
    Target(name="south-east corner", x=40.0, y=80.0),
], landing=(0.0, 0.0, 0.0))


def survey() -> Plan:
    return Plan.model_validate({**VALID_PAYLOAD, "mission_text": "survey the north field"})


def crane_ahead() -> FaultSchedule:
    return FaultSchedule.of(FaultEvent(
        type=FaultType.OBSTACLE_APPEARS, trigger=Trigger(at_waypoint=1), ahead=40.0, obstacle=CRANE, name="crane",
    ))


class Planners:
    def __init__(self, first: Plan | None = None, *, slow: float = 0.0):
        self.first = first or survey()
        self.slow = slow
        self.calls: list[str] = []

    def planner(self, mission, constraints, **kwargs) -> Plan:
        self.calls.append("plan")
        return self.first

    def replanner(self, mission, previous, violations, constraints, **kwargs) -> Plan:
        self.calls.append("inflight" if kwargs.get("situation") else "ground")
        time.sleep(self.slow)
        return survey()


def fly(planners: Planners, tmp_path, *, system=FULL, intent=CORNERS, faults=None, constraints=None, backend=None):
    return run_mission(
        "survey the north field",
        constraints or DEFAULT_CONSTRAINTS,
        faults if faults is not None else crane_ahead(),
        backend or CustomSimBackend(),
        planner=planners.planner,
        replanner=planners.replanner,
        system=system,
        intent=intent,
        verbose=False,
        runs_dir=tmp_path,
    )


def test_a_crane_on_the_route_is_flown_around_without_asking_the_planner(tmp_path):
    planners = Planners()
    log = fly(planners, tmp_path)
    assert log.outcome == "completed"
    assert planners.calls == ["plan"]  # nothing asked in the air
    repairs = [e for e in log.of("repair") if e.get("tier") == "inflight"]
    assert repairs and repairs[0]["repaired"]
    assert repairs[0]["ms"] < 2000
    assert "execution_resumed" in log.kinds
    assert not [e for e in log.report.events if e["type"] == "SEPARATION"]
    assert log.report.intent["accomplished"]


def test_classic_asks_the_planner_in_the_air_for_the_same_crane(tmp_path):
    planners = Planners()
    fly(planners, tmp_path, system=CLASSIC, intent=None)
    assert "inflight" in planners.calls


def test_a_plan_through_a_no_fly_zone_is_repaired_on_the_ground(tmp_path):
    zone = [GeoPoint(x=-10, y=30), GeoPoint(x=10, y=30), GeoPoint(x=10, y=50), GeoPoint(x=-10, y=50)]
    constraints = DEFAULT_CONSTRAINTS.model_copy(update={"no_fly_zones": [zone]})
    planners = Planners()
    log = fly(planners, tmp_path, faults=FaultSchedule(), constraints=constraints)
    assert planners.calls == ["plan"]  # repair fixed it; the planner was not asked again
    first, second = log.of("verification")[:2]
    assert not first["passed"] and second["passed"]
    assert log.of("plan_proposed")[1]["source"] == "repair"
    assert log.outcome == "completed"


def test_a_mission_outside_the_geofence_is_refused_before_planning(tmp_path):
    planners = Planners()
    far = MissionIntent(targets=[Target(name="the far point", x=0.0, y=1500.0)], landing=(0.0, 0.0, 0.0))
    log = fly(planners, tmp_path, intent=far, faults=FaultSchedule())
    assert log.outcome == "infeasible"
    assert planners.calls == []
    assert "outside the geofence" in log.of("infeasible")[0]["reasons"][0]


def test_a_failing_cell_is_believed_the_second_time_and_the_aircraft_comes_down_with_charge_left(tmp_path):
    # A cell that keeps failing from take-off, about eleven times the normal drain.
    cell = FaultSchedule.of(FaultEvent(
        type=FaultType.BATTERY_DRAIN_SPIKE, trigger=Trigger(time=3.0), magnitude=1.5, duration=None, name="cell fault",
    ))
    log = fly(Planners(), tmp_path, faults=cell)
    alarms = [e for e in log.of("replan_triggered") if e["trigger"] == "BATTERY_INSUFFICIENT"]
    assert len(alarms) >= 2
    first, second = [e for e in log.of("repair") if e.get("tier") == "inflight"][:2]
    assert first["actions"] == []  # once: could be a blip, nothing given up
    assert any(a.startswith("dropped") for a in second["actions"])  # twice: it is real, the survey goes
    assert not log.report.crashed and log.outcome != "battery_depleted"
    assert log.report.battery_remaining > 10.0


def test_a_passing_spike_does_not_cost_the_mission(tmp_path):
    # A few seconds of fast drain: worth a look, not worth giving up the survey for.
    blip = FaultSchedule.of(FaultEvent(
        type=FaultType.BATTERY_DRAIN_SPIKE, trigger=Trigger(time=3.0), magnitude=1.0, duration=6.0, name="cell blip",
    ))
    log = fly(Planners(), tmp_path, faults=blip)
    assert not log.of("intent_relaxed")
    assert log.outcome == "completed" and log.report.intent["accomplished"]


def test_a_slow_planner_costs_hover_time(tmp_path):
    slow = SystemOptions(replan="llm", hold="hover", intent=False, repair=False, shield=False)
    frozen = SystemOptions(replan="llm", hold="freeze", intent=False, repair=False, shield=False)
    hover_log = fly(Planners(slow=0.6), tmp_path / "a", system=slow, intent=None)
    freeze_log = fly(Planners(slow=0.6), tmp_path / "b", system=frozen, intent=None)
    held = [e for e in hover_log.report.events if e["type"] == "HOLD"]
    assert held and held[0]["payload"]["seconds"] >= 0.5
    assert not [e for e in freeze_log.report.events if e["type"] == "HOLD"]
    assert hover_log.report.battery_remaining < freeze_log.report.battery_remaining


def test_missing_photographs_are_added_by_repair(tmp_path):
    # The plan forgets the second corner; the intent check names a viewpoint and repair adds it.
    plan = survey()
    plan.subtasks[1].waypoints = plan.subtasks[1].waypoints[:1]
    planners = Planners(first=plan)
    log = fly(planners, tmp_path, faults=FaultSchedule())
    assert planners.calls == ["plan"]
    assert any(v["type"] == "INTENT" for v in log.of("verification")[0]["violations"])
    assert log.outcome == "completed"
    assert log.report.intent["accomplished"]


def test_a_rejection_says_which_subtask_is_wrong_and_which_to_keep(tmp_path):
    zone = [GeoPoint(x=-10, y=30), GeoPoint(x=10, y=30), GeoPoint(x=10, y=50), GeoPoint(x=-10, y=50)]
    constraints = DEFAULT_CONSTRAINTS.model_copy(update={"no_fly_zones": [zone]})
    heard = []

    class Listening(Planners):
        def replanner(self, mission, previous, violations, constraints, **kwargs):
            heard.append(list(violations))
            return super().replanner(mission, previous, violations, constraints, **kwargs)

    no_repair = SystemOptions(repair=False, shield=False, name="")
    fly(Listening(), tmp_path, system=no_repair, intent=None, faults=FaultSchedule(), constraints=constraints)
    assert heard and heard[0][0].startswith("[subtask 'survey_north_field']")
    assert "keep them exactly as they are" in heard[0][-1] and "'takeoff'" in heard[0][-1]


def test_systems_by_name():
    assert system_from("full") == FULL
    assert system_from("classic") == CLASSIC
    tweaked = system_from("full,no-shield,replan=llm")
    assert not tweaked.shield and tweaked.replan == "llm" and tweaked.intent
    assert tweaked.label == "gate=preflight,intent,repair,replan=llm,hold=hover,no-shield"
