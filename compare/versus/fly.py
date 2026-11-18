"""One runner per system: fly one of Waypoint's missions and return the flight as a Waypoint run.

    waypoint      the full loop, exactly as `python -m waypoint.experiments run --system full` flies it
    peace         the rebuilt PEACE agent
    droneserver   the rebuilt DroneServer, commanded by the model through its MCP tools

What the two rebuilt systems need in order to be flown on these missions at
all, and nothing more:

PEACE
    * its tool set gains `capture()` and `hover(seconds)`. PEACE has no camera
      action, and without one it could not photograph anything, so every
      mission would score zero for a reason that says nothing about it;
    * its system prompt lists those two tools and states the mission's limits
      in place of the release's defaults (50 m, 500 m);
    * its context carries the briefing of `scenario.py` (PEACE's own world
      model has detections only, and the missions name places);
    * its constraint layer gets the mission's ceiling, and for the fence the
      smallest square around the map origin that contains the mission's
      geofence: PEACE's fence is a square by construction, so it cannot be the
      mission's polygon. Whether it left the polygon is measured.

DroneServer
    * the operator's request is the mission text followed by the briefing, in
      latitude and longitude. Photographs are its own `camera_capture` tool;
    * its safety settings get the mission's geofence as its server-side fence
      polygon, the mission's ceiling and speed limit. Nothing else is changed.

Neither is told about obstacles that appear in flight, because neither has a
way to be: PEACE reads its LiDAR only when it plans, and DroneServer has no
obstacle sensing. Waypoint's monitor does, which is part of what is compared.
"""

from __future__ import annotations

import asyncio
import functools
import math
import time
from pathlib import Path

from compare.common.flight import FlightStack, sim_config
from compare.common.llm import ChatClient
from compare.versus.scenario import briefing
from waypoint.experiments.spec import Scenario
from waypoint.planner.models import Action, Plan, SubTask, Waypoint
from waypoint.verifier.models import VerificationResult

SYSTEMS = ("waypoint", "peace", "droneserver")


# -- the record every runner returns ------------------------------------------------


def stack_for(scn: Scenario, **options) -> FlightStack:
    """A flight stack on Waypoint's simulator, in the mission's world, wind and faults, on its launch pad."""
    from waypoint.sim.backend import CustomSimBackend  # noqa: PLC0415

    backend = CustomSimBackend(world=scn.world, wind=scn.wind, config=sim_config())
    return FlightStack(
        backend, scn.constraints, scn.faults, home=scn.home, world=scn.world,
        cruise_speed=scn.constraints.max_velocity, count_legs=True, **options,
    )


def run_of(stack: FlightStack, scn: Scenario) -> dict:
    """The flight as a Waypoint run file (schema 2), so Waypoint's own scoring reads it."""
    from waypoint.orchestrator.loop import MissionReport  # noqa: PLC0415
    from waypoint.orchestrator.runlog import build_run  # noqa: PLC0415

    events = stack.events()
    waypoints = []
    for event in events:
        if event["type"] == "WAYPOINT":
            x, y, z = event["payload"]["position"]
            waypoints.append(Waypoint(x=x, y=y, z=z, speed=stack.cruise_speed, action=Action(event["payload"]["action"])))
    last = stack.state
    flew = any(s.z > s.ground + 1.0 for s in stack.states)
    crashed = stack.backend.crashed or stack.killed_in_air
    landed = flew and stack.on_ground and not crashed
    if landed:
        # Where it believes it set down (its GPS position at touchdown) is where its plan ended: the LAND the
        # scoring looks for. That is how Waypoint is judged too: by the LAND its plan put on the pad, which its
        # executor flies to by the same GPS. A touchdown that drift or wind moved off the pad is an accuracy
        # figure (touchdown_error_m) for every system alike, not a missed pad for some and not others.
        rx, ry, _ = last.reported_position
        waypoints.append(Waypoint(x=rx, y=ry, z=last.ground, speed=1.5, action=Action.LAND))
    plan = Plan(mission_text=scn.spec.mission, subtasks=[SubTask(name="flown", waypoints=waypoints)] if waypoints else [])
    report = MissionReport(scn.spec.mission, plan, VerificationResult(passed=True), scn.constraints, faults=scn.faults, world=scn.world)
    report.states = list(stack.states)
    report.events = events
    report.obstacles = stack.backend.obstacle_records()
    report.distance_flown = stack.backend.distance_flown
    report.crashed = crashed
    report.backend = stack.backend.describe()
    report.plans = [
        {
            "plan_id": 0, "source": "initial", "t": 0.0, "start": [round(c, 2) for c in stack.home], "reason": None,
            "waypoints": [{"x": w.x, "y": w.y, "z": w.z, "speed": w.speed, "action": w.action.value} for w in waypoints],
        }
    ]
    report.outcome = (
        "crashed" if crashed
        else "battery_depleted" if stack.backend.battery_depleted
        else "completed" if landed
        else "not_flown" if not flew
        else "incomplete"
    )
    return build_run(report)


# -- Waypoint -----------------------------------------------------------------------


def fly_waypoint(scn: Scenario, runs_dir: Path) -> tuple[dict | None, dict]:
    """Waypoint's full system, flown the way its experiment harness flies it."""
    import json  # noqa: PLC0415

    from waypoint.backends import make_backend  # noqa: PLC0415
    from waypoint.orchestrator import run_mission  # noqa: PLC0415
    from waypoint.orchestrator.options import FULL  # noqa: PLC0415
    from waypoint.planner import plan, replan  # noqa: PLC0415
    from waypoint.planner.prompts import DEFAULT_OPTIONS  # noqa: PLC0415

    backend = make_backend("custom", world=scn.world, wind=scn.wind)
    try:
        log = run_mission(
            scn.spec.mission, scn.constraints, scn.faults, backend,
            world=scn.world, home=scn.home, landing=scn.landing,
            planner=functools.partial(plan, options=DEFAULT_OPTIONS),
            replanner=functools.partial(replan, options=DEFAULT_OPTIONS),
            system=FULL, max_replan_attempts=3, max_total_replans=3, verbose=False, runs_dir=runs_dir,
        )
    finally:
        backend.close()
    run = json.loads(Path(log.run_file).read_text()) if log.run_file else None
    kinds = [entry["kind"] for entry in log.entries]
    detail = {
        "outcome": log.outcome,
        "inflight_replans": getattr(log.report, "replans", 0) if getattr(log, "report", None) else 0,
        "log_events": {kind: kinds.count(kind) for kind in sorted(set(kinds))},
    }
    return run, detail


# -- PEACE --------------------------------------------------------------------------


def peace_prompt(max_altitude: float, min_altitude: float, limit: float) -> str:
    """PEACE's system prompt with the two mission tools and this mission's limits."""
    from compare.peace.prompts import LANGUAGE_SYSTEM_PROMPT  # noqa: PLC0415

    edits = [
        (
            "- land(x: float | None = None, y: float | None = None) - land at given position; None = current position\n",
            "- land(x: float | None = None, y: float | None = None) - land at given position; None = current position\n"
            "- capture() - take a photograph from the current position and heading\n"
            "- hover(seconds: float = 5.0) - hold the current position for seconds\n",
        ),
        ("Local ENU (East-North-Up), relative to home", "Local ENU (East-North-Up): x and y from the map origin, z from the take-off point"),
        ("- Maximum altitude: 50 m\n", f"- Maximum altitude: {max_altitude:g} m\n"),
        ("- Minimum altitude: 0.5 m\n", f"- Minimum altitude: {min_altitude:g} m\n"),
        ("- Geofence: ±500 m in x and y from home\n", f"- Geofence: ±{limit:g} m in x and y from the map origin\n"),
    ]
    prompt = LANGUAGE_SYSTEM_PROMPT
    for old, new in edits:
        if old not in prompt:
            raise RuntimeError(f"PEACE's system prompt no longer contains {old!r}")
        prompt = prompt.replace(old, new)
    return prompt


def fly_peace(scn: Scenario, llm: ChatClient) -> tuple[dict, dict]:
    from compare.peace.agent import PeaceAgent  # noqa: PLC0415
    from compare.peace.config import PeaceSettings  # noqa: PLC0415
    from compare.peace.tools.robot_tools import CONTROL_DISABLED, tool  # noqa: PLC0415
    from compare.peace.tools.uav_tools import UAVTools  # noqa: PLC0415

    class MissionUAVTools(UAVTools):
        """PEACE's tool set, with the two actions these missions ask for and PEACE lacks."""

        @tool
        def capture(self) -> str:
            """Take a photograph from the current position and heading."""
            if not self._settings.action_enable_control:
                return CONTROL_DISABLED
            ego = self._ego_state_svc.get_ego_state()
            self._stack.mark_waypoint(Action.CAPTURE, "photograph")
            return f"Photograph taken at ({ego.position_x:.1f},{ego.position_y:.1f},{ego.position_z:.1f})m, yaw={ego.orientation_yaw:.0f}°"

        @tool
        def hover(self, seconds: float = 5.0) -> str:
            """Hold the current position for seconds."""
            if not self._settings.action_enable_control:
                return CONTROL_DISABLED
            ego = self._ego_state_svc.get_ego_state()
            self._hold_position(ego.position_x, ego.position_y, ego.position_z, math.radians(ego.orientation_yaw))
            self._sleep(max(0.0, min(float(seconds), 60.0)))
            self._stack.mark_waypoint(Action.HOVER, "hover")
            return f"Hovered {seconds:g}s at ({ego.position_x:.1f},{ego.position_y:.1f},{ego.position_z:.1f})m"

    datum = scn.home[2]
    stack = stack_for(scn, origin=(0.0, 0.0, datum))
    limit = math.ceil(max(max(abs(p.x), abs(p.y)) for p in scn.constraints.geofence))
    ceiling = scn.constraints.max_altitude - datum
    floor = max(0.5, scn.constraints.min_altitude - datum)
    settings = PeaceSettings(
        action_max_altitude=ceiling, action_min_altitude=floor, action_horizontal_limit=float(limit), log_enable=False,
    )
    text = briefing(
        scn, lambda x, y: f"({x:.0f}, {y:.0f})", "measured from the take-off point (your z)",
        extra="Positions are x east and y north in metres, in the same frame as your own position. "
        "Use capture() to take a photograph and hover(seconds) to hold a position.",
    )
    agent = PeaceAgent(
        stack, llm, settings, tools_class=MissionUAVTools,
        system_prompt=peace_prompt(ceiling, floor, limit), extra_context=lambda: text,
    )
    result = agent.run(scn.spec.mission)
    agent.close()
    detail = {
        "outcome": result.state,
        "message": result.message[:300],
        "llm_calls": result.llm_calls,
        "replans": result.replan_count,
        "steps": [{"tool": s["tool"], "args": s["args"], "result": str(s["result"])[:160], "plan": s["plan"]} for s in result.steps],
        "constraint_aborts": len(result.safety_aborts),
        "fence_half_side_m": limit,
    }
    return run_of(stack, scn), detail


# -- DroneServer --------------------------------------------------------------------


def fly_droneserver(scn: Scenario, llm: ChatClient, max_turns: int = 90) -> tuple[dict, dict]:
    from compare.droneserver.llm.agent import Limits, run_agent  # noqa: PLC0415
    from compare.droneserver.llm.prompts import SYSTEM_PROMPT  # noqa: PLC0415
    from compare.droneserver.server import DroneServer, control_settings  # noqa: PLC0415
    from compare.droneserver.vehicle import DEFAULT_ORIGIN, Vehicle  # noqa: PLC0415

    datum = scn.home[2]
    stack = stack_for(scn)
    world = scn.world
    origin = (
        (world.origin_lat, world.origin_lon, DEFAULT_ORIGIN[2])
        if world is not None and world.origin_lat is not None and world.origin_lon is not None
        else DEFAULT_ORIGIN
    )
    # A camera is fitted: these missions photograph places, and plain SITL (the paper's) has none.
    vehicle = Vehicle(stack, origin=origin, parameters={"WPNAV_SPEED": scn.constraints.max_velocity * 100.0}, camera=True)

    def point(x: float, y: float) -> str:
        lat, lon = vehicle.to_latlon(x, y)
        return f"(latitude {lat:.6f}, longitude {lon:.6f})"

    fence = ";".join("{:.7f},{:.7f}".format(*vehicle.to_latlon(p.x, p.y)) for p in scn.constraints.geofence)
    ceiling = scn.constraints.max_altitude - datum
    settings = control_settings(
        geofence_polygon=fence, geofence_max_radius_m=0.0, geofence_max_altitude_m=ceiling, max_altitude_m=ceiling,
        max_speed_m_s=scn.constraints.max_velocity,
    )
    launch_amsl = vehicle.amsl(datum)
    text = briefing(
        scn, point, "measured from the launch point",
        extra=f"The launch point is {launch_amsl:.1f} m above mean sea level. Take a photograph with the camera tools.",
    )
    server = DroneServer(vehicle, settings, client_name=getattr(llm, "name", None))

    async def trial():
        async with server.client() as mcp:
            return await run_agent(
                llm, server, mcp, SYSTEM_PROMPT, f"{scn.spec.mission}\n\n{text}",
                Limits(max_turns=max_turns),
            )

    run = asyncio.run(trial())
    left_flying = vehicle.in_air and not stack.ended
    record = run_of(stack, scn)  # as the model left it: a harness landing is not the system's doing
    detail = {
        "outcome": run.stop_reason,
        "model_claim": run.model_claim,
        "llm_calls": len(run.turns),
        "tool_calls": len(run.calls),
        "refusals": run.rejections,
        "rules_fired": sorted({c.rule for c in run.calls if c.rule}),
        "confirmations_demanded": run.confirmations_demanded,
        "left_flying": left_flying,
        "error": run.error,
        "calls": [{"tool": c.tool, "arguments": c.arguments, "status": c.status, "rule": c.rule, "t": c.flight_time_s} for c in run.calls],
        "final_text": run.final_text[:500],
    }
    return record, detail


# -- one job ------------------------------------------------------------------------


def compact(run: dict | None) -> dict | None:
    """A run small enough to keep: without the embedded world, and with one tick in five."""
    if run is None:
        return None
    kept = {k: v for k, v in run.items() if k not in ("world", "ticks")}
    ticks = run.get("ticks") or []
    kept["ticks"] = ticks[::5] + (ticks[-1:] if len(ticks) % 5 != 1 else [])
    return kept


def fly(system: str, scn: Scenario, llm: ChatClient | None, scratch: Path) -> tuple[dict | None, dict]:
    began = time.perf_counter()
    if system == "waypoint":
        run, detail = fly_waypoint(scn, scratch)
    elif system == "peace":
        run, detail = fly_peace(scn, llm)
    elif system == "droneserver":
        run, detail = fly_droneserver(scn, llm)
    else:
        raise ValueError(f"unknown system {system!r}; one of {', '.join(SYSTEMS)}")
    detail["wall_s"] = round(time.perf_counter() - began, 1)
    return run, detail
