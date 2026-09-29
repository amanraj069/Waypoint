"""The in-flight monitor: should the aircraft keep flying this plan?

`needs_replan(snapshot)` looks at one moment of telemetry and returns a
`ReplanTrigger` when the plan in hand is no longer safe to finish, or None:

* **an obstacle on the route**: something appeared that the plan was not
  verified against, and the rest of the route passes through its keep-out
  cylinder (its radius plus the mission's separation, up to its height);
* **not enough battery**: what the rest of the route needs, plus the reserve,
  is more than is left.

`summarize_for_llm(snapshot, trigger)` turns the same moment into the words the
planner is given when it is asked for a new plan from here.
"""

from __future__ import annotations

import math

from waypoint.feedback.models import FeedbackConfig, ReplanTrigger, TelemetrySnapshot, TriggerKind
from waypoint.planner.models import Obstacle

DEFAULT_CONFIG = FeedbackConfig()


def needs_replan(
    snapshot: TelemetrySnapshot,
    config: FeedbackConfig | None = None,
    *,
    last_battery_replan: float | None = None,
) -> ReplanTrigger | None:
    """The first reason to replan right now, or None to carry on."""
    config = config or DEFAULT_CONFIG
    return _obstacle_on_route(snapshot) or _battery_short(snapshot, config, last_battery_replan)


def remaining_route(snapshot: TelemetrySnapshot) -> list[tuple[float, float, float]]:
    """Where the aircraft still has to go: from here, through every waypoint left."""
    return [snapshot.position, *((wp.x, wp.y, wp.z) for wp in snapshot.remaining)]


def estimated_battery_need(snapshot: TelemetrySnapshot, config: FeedbackConfig | None = None) -> float:
    """Percent of battery the rest of the route will use, distance and time together."""
    config = config or DEFAULT_CONFIG
    route = remaining_route(snapshot)
    metres = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
    seconds = metres / config.cruise_speed
    return metres / config.meters_per_battery_pct + seconds / config.seconds_per_battery_pct


def _obstacle_on_route(snapshot: TelemetrySnapshot) -> ReplanTrigger | None:
    route = remaining_route(snapshot)
    for obstacle in snapshot.new_obstacles:
        keep_out = obstacle.radius + snapshot.min_separation
        for leg, (a, b) in enumerate(zip(route, route[1:])):
            gap = _gap_to_segment((obstacle.x, obstacle.y), a, b)
            if gap >= keep_out or not _below_top(obstacle, a, b):
                continue
            return ReplanTrigger(
                kind=TriggerKind.OBSTACLE_ON_ROUTE,
                t=snapshot.t,
                reason=(
                    f"'{obstacle.name}' appeared at ({obstacle.x:g}, {obstacle.y:g}), "
                    f"radius {obstacle.radius:g} m"
                    + (f", {obstacle.height:g} m tall" if obstacle.height is not None else "")
                    + f"; the route passes {gap:.1f} m from it, inside its {keep_out:g} m keep-out"
                ),
                detail={"obstacle": obstacle.model_dump(), "gap": round(gap, 2), "leg": leg},
            )
    return None


def _battery_short(
    snapshot: TelemetrySnapshot, config: FeedbackConfig, last_battery_replan: float | None
) -> ReplanTrigger | None:
    if last_battery_replan is not None and snapshot.t - last_battery_replan < config.battery_cooldown_s:
        return None
    if not snapshot.remaining:
        return None
    need = estimated_battery_need(snapshot, config)
    if snapshot.battery_pct - need >= config.battery_reserve_pct:
        return None
    return ReplanTrigger(
        kind=TriggerKind.BATTERY_INSUFFICIENT,
        t=snapshot.t,
        reason=(
            f"battery is at {snapshot.battery_pct:.1f}% and the rest of the route needs about "
            f"{need:.1f}%, which would land with less than the {config.battery_reserve_pct:g}% reserve"
        ),
        detail={"battery_pct": round(snapshot.battery_pct, 2), "estimated_need_pct": round(need, 2)},
    )


def summarize_for_llm(snapshot: TelemetrySnapshot, trigger: ReplanTrigger) -> str:
    """The situation, as the planner needs it to plan the rest of the mission."""
    x, y, z = snapshot.position
    flown = snapshot.plan_waypoints[: snapshot.target_index]
    lines = [
        f"Time into the flight: {snapshot.t:.1f} s.",
        f"Current position: ({x:.1f}, {y:.1f}, {z:.1f}), "
        f"{z - snapshot.ground:.1f} m above the ground below; heading {snapshot.heading:.0f} deg; "
        f"speed {math.dist((0, 0, 0), snapshot.velocity):.1f} m/s.",
        f"Battery: {snapshot.battery_pct:.1f}% left.",
        f"Why the plan has to change: {trigger.reason}.",
    ]
    if snapshot.active_faults:
        lines.append(f"Faults in effect now: {', '.join(snapshot.active_faults)}.")
    if flown:
        done = ", ".join(f"({wp.x:g}, {wp.y:g}, {wp.z:g}) {wp.action.value}" for wp in flown)
        lines.append(f"Waypoints already reached (do not repeat them): {done}.")
    lines.append(f"Was flying toward: {snapshot.subtask or 'the next waypoint'}.")
    if snapshot.remaining:
        left = ", ".join(f"({wp.x:g}, {wp.y:g}, {wp.z:g}) {wp.action.value}" for wp in snapshot.remaining)
        lines.append(f"Waypoints not yet flown: {left}.")
    new = snapshot.new_obstacles
    if new:
        lines.append("Obstacles detected in flight: " + "; ".join(_describe(o) for o in new) + ".")
    return "\n".join(lines)


def _describe(obstacle: Obstacle) -> str:
    top = f", {obstacle.height:g} m tall" if obstacle.height is not None else ""
    return f"'{obstacle.name}' at ({obstacle.x:g}, {obstacle.y:g}), radius {obstacle.radius:g} m{top}"


def _gap_to_segment(point, a, b) -> float:
    """Horizontal distance from a point to the segment a-b."""
    px, py = point
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    f = 0.0 if span == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
    return math.hypot(px - (ax + f * dx), py - (ay + f * dy))


def _below_top(obstacle: Obstacle, a, b) -> bool:
    """Whether any part of the leg is low enough to meet the obstacle."""
    return obstacle.height is None or min(a[2], b[2]) < obstacle.height
