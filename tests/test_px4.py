"""The PX4 backend, flown against a fake PX4 so the suite needs neither PX4 nor Gazebo.

`FakeLink` stands in for the MAVLink link: it integrates whatever setpoint the
backend streams (a velocity, or a position to hold), lands when told to, and
stamps every position with its own clock, as PX4's lockstep clock would.
`FakeSitl` stands in for the processes and records what the backend asked of
Gazebo. Everything the backend decides (legs, arrival, dwell, landing, faults,
battery, crashes, pausing) is the real code.

`test_a_real_px4_flight` flies PX4 itself, and only runs with WAYPOINT_PX4_TESTS=1.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from waypoint.backends import BackendError
from waypoint.gazebo.scene import write_scene
from waypoint.orchestrator.loop import run_mission
from waypoint.planner import Action, DEFAULT_CONSTRAINTS, GeoPoint, Obstacle, Plan, SubTask, Waypoint
from waypoint.px4 import backend as px4_backend
from waypoint.px4.backend import Px4Backend
from waypoint.px4.link import POSITION_MASK, VELOCITY_MASK, Setpoint, _float_as_int, _int_as_float
from waypoint.sim import FaultEvent, FaultSchedule, FaultType, Trigger
from waypoint.world import Building, World

sys.path.insert(0, str(Path(__file__).parent))
from test_closed_loop import CRANE, Script, crane_ahead  # noqa: E402

STEP_MS = 20


class FakeLink:
    """PX4 as a point that goes where it is told, in NED, on its own clock."""

    def __init__(self, address=None):
        self.t_ms = 5000
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.mode = "idle"
        self.is_armed = False
        self.setpoint: Setpoint | None = None
        self.statustext: list[str] = []
        self.latest: dict = {}
        self.params: dict = {}
        self.killed = False
        FakeLink.last = self
        self._publish()

    def connect(self, timeout=30.0):
        pass

    def close(self):
        pass

    def set_param(self, name, value, *, integer=False, timeout=3.0):
        self.params[name] = value
        return True

    def wait_for(self, kind, timeout, check=lambda message: True):
        if kind == "HOME_POSITION":
            return SimpleNamespace()
        message = self.latest.get(kind)
        return message if message is not None and check(message) else None

    def wait_position(self, after_ms, timeout):
        while self.t_ms < after_ms:
            self._step()
        return self.latest["LOCAL_POSITION_NED"]

    @property
    def armed(self):
        return self.is_armed

    @property
    def on_ground(self):
        return self.mode == "landed"

    def last_words(self, count=4):
        return ""

    def set_offboard(self):
        self.mode = "offboard"
        self._publish()
        return 0

    def arm(self):
        self.is_armed = True
        self._publish()
        return 0

    def land(self):
        self.mode = "land"
        return 0

    def kill(self):
        self.killed, self.is_armed, self.mode = True, False, "dead"
        return 0

    def _step(self):
        dt = STEP_MS / 1000.0
        n, e, d = self.position
        if self.mode == "offboard" and self.setpoint is not None:
            if self.setpoint.mask == VELOCITY_MASK:
                self.velocity = list(self.setpoint.velocity)
            else:
                gap = [p - q for p, q in zip(self.setpoint.position, self.position)]
                size = math.sqrt(sum(g * g for g in gap))
                speed = min(5.0, size / 0.5)
                self.velocity = [g / size * speed for g in gap] if size > 1e-6 else [0.0, 0.0, 0.0]
        elif self.mode == "land":
            self.velocity = [0.0, 0.0, 1.5]
        elif self.mode == "dead":
            self.velocity = [0.0, 0.0, 6.0]
        else:
            self.velocity = [0.0, 0.0, 0.0]
        self.position = [n + self.velocity[0] * dt, e + self.velocity[1] * dt, min(0.0, d + self.velocity[2] * dt)]
        if self.mode == "land" and self.position[2] >= 0.0:
            self.mode, self.is_armed, self.velocity = "landed", False, [0.0, 0.0, 0.0]
        self.t_ms += STEP_MS
        self._publish()

    def _publish(self):
        n, e, d = self.position
        vn, ve, vd = self.velocity
        self.latest["LOCAL_POSITION_NED"] = SimpleNamespace(
            time_boot_ms=self.t_ms, x=n, y=e, z=d, vx=vn, vy=ve, vz=vd
        )
        self.latest["HEARTBEAT"] = SimpleNamespace(
            base_mode=128 if self.is_armed else 0,
            custom_mode=(6 << 16) if self.mode == "offboard" else 0,
        )


class FakeSitl:
    def __init__(self, world, pad, **kwargs):
        self.world, self.pad = world, pad
        self.calls: list = []
        self.running = True
        FakeSitl.last = self

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def pause(self, paused):
        self.calls.append(("pause", paused))

    def push(self, force):
        self.calls.append(("push", force))

    def show(self, sdf):
        self.calls.append(("show", sdf))


class FakeInstall:
    def version(self):
        return "v-test"


@pytest.fixture
def fake_px4(monkeypatch):
    monkeypatch.setattr("waypoint.px4.link.Link", FakeLink)
    monkeypatch.setattr("waypoint.px4.sitl.Sitl", FakeSitl)
    monkeypatch.setattr(px4_backend.time, "sleep", lambda seconds: None)

    def make(**kwargs) -> Px4Backend:
        return Px4Backend(install=FakeInstall(), **kwargs)

    return make


def wp(x, y, z, speed=5.0, action=Action.MOVE) -> Waypoint:
    return Waypoint(x=x, y=y, z=z, speed=speed, action=action)


def square() -> Plan:
    return Plan(mission_text="square", subtasks=[
        SubTask(name="out", waypoints=[wp(0, 0, 15, 3), wp(40, 0, 15, 8), wp(40, 40, 15, 8, Action.HOVER)]),
        SubTask(name="home", waypoints=[wp(0, 0, 15, 6), wp(0, 0, 0, 1.5, Action.LAND)]),
    ])


def fly(backend: Px4Backend, plan: Plan, faults=None, constraints=DEFAULT_CONSTRAINTS, home=None):
    backend.prepare(constraints, faults or FaultSchedule(), home=home)
    return list(backend.run(plan))


# ---------------------------------------------------------------------------
# Flying a plan
# ---------------------------------------------------------------------------


def test_a_plan_is_flown_leg_by_leg_and_landed(fake_px4):
    backend = fake_px4()
    states = fly(backend, square())
    reached = [e["payload"]["waypoint_index"] for e in backend.event_records() if e["type"] == "WAYPOINT"]
    assert reached == [0, 1, 2, 3, 4]
    assert backend.event_records()[-1]["type"] == "LANDED"
    last = states[-1]
    assert last.z == last.ground and last.velocity == 0.0  # what "completed" means to the loop
    assert math.hypot(last.x, last.y) < 2.0
    assert states[0].timestamp == 0.0 and states[0].position == (0.0, 0.0, 0.0)
    gaps = {round(b.timestamp - a.timestamp, 3) for a, b in zip(states, states[1:-1])}
    assert gaps == {0.1}  # one state per tick of simulated time, as the simulator does


def test_it_never_flies_faster_than_the_leg_allows(fake_px4):
    states = fly(fake_px4(), square())
    leg_1 = [s for s in states if s.target_index == 1]
    assert max(s.ground_speed for s in leg_1) <= 8.0 + 1e-6


def test_the_battery_uses_the_simulators_model(fake_px4):
    backend = fake_px4()
    states = fly(backend, square())
    config = backend.config
    expected = 100.0 - backend.distance_flown / config.meters_per_battery_pct - states[-2].timestamp / config.seconds_per_battery_pct
    assert states[-1].battery_pct == pytest.approx(expected, abs=0.05)


def test_the_pad_is_the_frame_origin(fake_px4):
    backend = fake_px4()
    plan = Plan(mission_text="m", subtasks=[SubTask(name="r", waypoints=[
        wp(100, 50, 20), wp(120, 50, 20), wp(100, 50, 0, 1.5, Action.LAND)])])
    states = fly(backend, plan, home=(100.0, 50.0, 0.0))
    assert states[0].position == (100.0, 50.0, 0.0)
    assert max(s.x for s in states) == pytest.approx(120.0, abs=2.0)
    assert FakeSitl.last.pad == (100.0, 50.0, 0.0)


# ---------------------------------------------------------------------------
# Pausing for a replan
# ---------------------------------------------------------------------------


def test_stopping_mid_flight_holds_and_freezes_the_clock(fake_px4):
    backend = fake_px4()
    backend.prepare(DEFAULT_CONSTRAINTS, FaultSchedule())
    flight = backend.run(square())
    for state in flight:
        if state.timestamp >= 5.0:
            break
    flight.close()
    paused_at = backend.state.timestamp
    assert ("pause", True) in FakeSitl.last.calls
    assert FakeLink.last.setpoint.mask == POSITION_MASK  # holding where it is

    rest = list(backend.run(square()))
    assert ("pause", False) in FakeSitl.last.calls
    assert rest[0].timestamp == pytest.approx(paused_at + 0.1, abs=1e-6)  # carries on, not restarts
    assert backend.event_records()[-1]["type"] == "LANDED"


def test_hover_mode_keeps_the_clock_running(fake_px4):
    backend = fake_px4(hold="hover")
    backend.prepare(DEFAULT_CONSTRAINTS, FaultSchedule())
    flight = backend.run(square())
    next(flight), next(flight)
    flight.close()
    assert not any(call[0] == "pause" for call in FakeSitl.last.calls if isinstance(call, tuple))


def test_an_unknown_hold_mode_is_refused():
    with pytest.raises(BackendError, match="hold"):
        Px4Backend(install=FakeInstall(), hold="loiter")


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------


def test_gps_drift_flies_it_off_its_line_believing_it_is_on_it(fake_px4):
    drift = FaultSchedule.of(FaultEvent(
        type=FaultType.GPS_DRIFT, trigger=Trigger(time=3.0), magnitude=1.0, direction=0.0, duration=6.0,
    ))
    backend = fake_px4()
    states = fly(backend, square(), faults=drift)
    last = states[-1]
    assert last.gps_offset[0] == pytest.approx(6.0, abs=0.2)
    # It landed where it believed the pad was: 6 m west of the real one.
    assert last.reported_position[0] == pytest.approx(0.0, abs=1.6)
    assert last.x == pytest.approx(-6.0, abs=1.6)


def test_a_gust_is_a_force_on_the_airframe(fake_px4):
    gust = FaultSchedule.of(FaultEvent(
        type=FaultType.WIND_GUST, trigger=Trigger(time=4.0), magnitude=8.0, direction=90.0, duration=3.0,
    ))
    backend = fake_px4()
    fly(backend, square(), faults=gust)
    pushes = [call[1] for call in FakeSitl.last.calls if isinstance(call, tuple) and call[0] == "push"]
    assert pushes[0][1] == pytest.approx(px4_backend.WIND_DRAG * 64, abs=0.05)  # north, 8 m/s
    assert pushes[-1] == (0.0, 0.0, 0.0)  # and let go when it ends


def test_a_battery_spike_drains_the_shared_model(fake_px4):
    spike = FaultSchedule.of(FaultEvent(
        type=FaultType.BATTERY_DRAIN_SPIKE, trigger=Trigger(time=2.0), magnitude=1.0, duration=10.0,
    ))
    calm, spiked = fake_px4(), fake_px4()
    fly(calm, square())
    calm_left = calm.state.battery_pct
    fly(spiked, square(), faults=spike)
    assert calm_left - spiked.state.battery_pct == pytest.approx(10.0, abs=0.5)


def test_an_obstacle_appears_on_the_route_and_in_gazebo(fake_px4):
    backend = fake_px4()
    fly(backend, square(), faults=crane_ahead())
    crane = next(o for o in backend.obstacle_records() if o["name"] == CRANE.name)
    assert crane["appeared_at"] > 0
    assert crane["x"] == pytest.approx(40.0, abs=1.0) or crane["y"] == pytest.approx(40.0, abs=1.0)
    shown = [call[1] for call in FakeSitl.last.calls if isinstance(call, tuple) and call[0] == "show"]
    assert any("appeared_" in sdf for sdf in shown)


def test_flying_into_a_building_cuts_the_motors(fake_px4):
    block = Building(footprint=[GeoPoint(x=15, y=-10), GeoPoint(x=25, y=-10), GeoPoint(x=25, y=10), GeoPoint(x=15, y=10)],
                     height=30, name="the block")
    backend = fake_px4(world=World(name="w", buildings=[block]))
    fly(backend, square())
    assert backend.crashed
    assert FakeLink.last.killed
    collision = next(e for e in backend.event_records() if e["type"] == "COLLISION")
    assert "the block" in collision["description"]


# ---------------------------------------------------------------------------
# The loop, and what is recorded
# ---------------------------------------------------------------------------


def test_the_loop_replans_around_a_crane_on_px4(fake_px4, tmp_path):
    script = Script()
    backend = fake_px4()
    log = run_mission(
        "survey the north field", DEFAULT_CONSTRAINTS, crane_ahead(), backend,
        planner=script.planner, replanner=script.replanner, verbose=False, runs_dir=tmp_path,
    )
    assert log.outcome == "completed" and log.backend == "px4"
    assert len(log.of("replan_triggered")) == 1
    report = log.report
    crane = next(o for o in report.obstacles if o["name"] == CRANE.name)
    after = [s for s in report.states if s.timestamp >= crane["appeared_at"]]
    assert min(math.hypot(s.x - crane["x"], s.y - crane["y"]) for s in after) > crane["radius"]


def test_describe_records_what_flew_and_how_faults_were_modelled(fake_px4):
    backend = fake_px4()
    backend.prepare(DEFAULT_CONSTRAINTS, FaultSchedule(), home=(5.0, 6.0, 0.0))
    detail = backend.describe()
    assert detail["name"] == "px4" and detail["px4"] == "v-test" and detail["airframe"] == "gz_x500"
    assert detail["config"]["home"] == [5.0, 6.0, 0.0]  # where a replay puts the pad
    assert set(detail["faults"]) == {t.value for t in FaultType}


def test_close_stops_the_simulation(fake_px4):
    backend = fake_px4()
    backend.prepare(DEFAULT_CONSTRAINTS, FaultSchedule())
    sitl = FakeSitl.last
    backend.close()
    backend.close()
    assert sitl.calls.count("stop") == 1


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def test_a_stalled_px4_start_is_tried_again(monkeypatch):
    from waypoint.px4 import sitl

    tries = []

    def flaky(self, timeout):
        tries.append(timeout)
        if len(tries) < 3:
            raise sitl._Stalled("PX4's startup script did not come up")

    monkeypatch.setattr(sitl.Sitl, "_start", flaky)
    session = sitl.Sitl(None, (0.0, 0.0, 0.0), install=FakeInstall())
    session.start()
    assert len(tries) == 3

    tries.clear()
    monkeypatch.setattr(sitl.Sitl, "_start", lambda self, timeout: (_ for _ in ()).throw(sitl._Stalled("stuck")))
    with pytest.raises(BackendError, match="gave up after 3 attempts"):
        session.start()


def test_integer_parameters_travel_as_their_own_bytes():
    for value in (0, 4, 21196, -1):
        assert _float_as_int(_int_as_float(value)) == value
    assert _int_as_float(4) != 4.0


def test_the_px4_world_has_gravity_sensors_and_ground_but_no_aircraft(tmp_path):
    world = World(name="w", buildings=[Building(
        footprint=[GeoPoint(x=0, y=0), GeoPoint(x=10, y=0), GeoPoint(x=10, y=10), GeoPoint(x=0, y=10)], height=20,
    )])
    sdf = write_scene({"world": world.model_dump(mode="json")}, tmp_path / "px4", px4=True).sdf.read_text()
    assert "<gravity>0 0 -9.8</gravity>" in sdf
    for system in ("Imu", "NavSat", "Magnetometer", "AirPressure", "ApplyLinkWrench"):
        assert f"systems::{system}" in sdf
    assert '<collision name="ground">' in sdf and "<plane>" in sdf
    assert "spherical_coordinates" in sdf  # PX4's GPS needs a place on Earth
    assert 'model name="uav"' not in sdf  # PX4 spawns its own

    replay = write_scene({"world": world.model_dump(mode="json")}, tmp_path / "replay").sdf.read_text()
    assert "<gravity>0 0 0</gravity>" in replay and 'model name="uav"' in replay


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("WAYPOINT_PX4_TESTS") != "1", reason="flies PX4 SITL; set WAYPOINT_PX4_TESTS=1")
def test_a_real_px4_flight(tmp_path):
    from waypoint.backends import make_backend

    backend = make_backend("px4")
    try:
        log = run_mission("square", DEFAULT_CONSTRAINTS, None, backend,
                          planner=lambda *args, **kwargs: square(), verbose=False, runs_dir=tmp_path)
    finally:
        backend.close()
    assert log.outcome == "completed"
    last = log.report.states[-1]
    assert math.hypot(last.x, last.y) < 3.0
