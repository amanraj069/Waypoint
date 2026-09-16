"""Stub executor.

Stands in for the UAV simulator. Walks the plan's waypoints with straight-line
linear interpolation and yields a MissionState per tick. No dynamics, no wind,
no collision model -- just enough motion to prove the control loop.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from planner.models import Action, Plan, Waypoint
from sim.state import MissionState

TICK_SECONDS = 1.0
BATTERY_DRAIN_PER_TICK = 0.25
HOVER_TICKS = 2


def initial_state(battery_pct: float = 100.0) -> MissionState:
    """State at the launch pad: on the ground, stationary, full battery."""
    return MissionState(
        x=0.0, y=0.0, z=0.0, velocity=0.0, heading=0.0,
        battery_pct=battery_pct, timestamp=0.0,
    )


def _ticks_for(state: MissionState, waypoint: Waypoint) -> int:
    """Number of ticks needed to fly the leg at the commanded speed."""
    if waypoint.action == Action.HOVER:
        return HOVER_TICKS
    distance = math.dist(state.position, (waypoint.x, waypoint.y, waypoint.z))
    return max(1, math.ceil(distance / (waypoint.speed * TICK_SECONDS)))


def _heading_to(state: MissionState, waypoint: Waypoint) -> float:
    """Bearing in degrees [0, 360) toward the waypoint; hold heading if in place."""
    dx, dy = waypoint.x - state.x, waypoint.y - state.y
    if math.isclose(dx, 0.0) and math.isclose(dy, 0.0):
        return state.heading
    return math.degrees(math.atan2(dy, dx)) % 360.0


def execute(plan: Plan, start: MissionState | None = None) -> Iterator[MissionState]:
    """Yield a MissionState for every tick of the plan, ending landed."""
    state = start or initial_state()

    for waypoint in plan.waypoints():
        leg_start = (state.x, state.y, state.z)
        ticks = _ticks_for(state, waypoint)
        heading = _heading_to(state, waypoint)
        distance = math.dist(leg_start, (waypoint.x, waypoint.y, waypoint.z))

        for tick in range(1, ticks + 1):
            fraction = tick / ticks
            state = MissionState(
                x=leg_start[0] + (waypoint.x - leg_start[0]) * fraction,
                y=leg_start[1] + (waypoint.y - leg_start[1]) * fraction,
                z=leg_start[2] + (waypoint.z - leg_start[2]) * fraction,
                velocity=distance / (ticks * TICK_SECONDS),
                heading=heading,
                battery_pct=max(0.0, state.battery_pct - BATTERY_DRAIN_PER_TICK),
                timestamp=state.timestamp + TICK_SECONDS,
            )
            yield state

    # Touchdown: guarantee the mission ends on the ground and stationary.
    if state.z != 0.0 or state.velocity != 0.0:
        state = MissionState(
            x=state.x, y=state.y, z=0.0, velocity=0.0, heading=state.heading,
            battery_pct=max(0.0, state.battery_pct - BATTERY_DRAIN_PER_TICK),
            timestamp=state.timestamp + TICK_SECONDS,
        )
        yield state
