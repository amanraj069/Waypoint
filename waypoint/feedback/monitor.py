"""The in-flight monitor: should the aircraft keep flying this plan?

`needs_replan(history)` looks at the recent telemetry of the plan being flown
(oldest first, newest last; a single snapshot also works) and returns a
`ReplanTrigger` when the plan in hand is no longer safe to finish, or None:

* **an obstacle on the route**: something was perceived that the plan was not
  verified against, and the rest of the route passes through its keep-out
  cylinder (its radius plus the mission's separation, up to its height);
* **not enough battery**: what the rest of the route needs, plus the reserve,
  is more than is left. The need is the energy model's estimate, scaled up by
  the battery's recent trend when it has been draining faster than the model
  says it should (a failing cell, a heavy headwind);
* **off route**: the aircraft has been further than the threshold from the leg
  it should be on, continuously, for longer than the hold time. This is
  measured on the reported position, so a GPS drift shows only as far as it
  pushes the reported track off the leg (the controller heads straight for the
  next waypoint rather than back onto the line).

`summarize_for_llm(snapshot, trigger)` turns the same moment into the words the
planner is given when it is asked for a new plan from here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from waypoint.feedback.models import (
    FeedbackConfig,
    ReplanTrigger,
    TelemetrySnapshot,
    TriggerKind,
    horizontal_gap,
)
from waypoint.feedback.perception import PerceptionSeverity
from waypoint.planner.models import Obstacle

DEFAULT_CONFIG = FeedbackConfig()


def needs_replan(
    history: TelemetrySnapshot | Sequence[TelemetrySnapshot],
    config: FeedbackConfig | None = None,
    *,
    last_battery_replan: float | None = None,
) -> ReplanTrigger | None:
    """The first reason to replan right now, or None to carry on.

    `history` is the plan in hand's telemetry, oldest first; the newest
    snapshot is "now", and carries the plan itself. Battery trends and
    sustained deviation need more than one snapshot; with just one, only the
    instantaneous checks can fire.
    """
    config = config or DEFAULT_CONFIG
    history = (history,) if isinstance(history, TelemetrySnapshot) else tuple(history)
    if not history:
        return None
    now = history[-1]
    return (
        _obstacle_on_route(now)
        or _battery_short(history, config, last_battery_replan)
        or _off_route(history, config)
    )


def path_deviation(snapshot: TelemetrySnapshot, threshold: float) -> bool:
    """Whether the aircraft is more than `threshold` metres off the leg the plan has it on."""
    return snapshot.deviation > threshold


def battery_drain_ratio(
    history: Sequence[TelemetrySnapshot], config: FeedbackConfig | None = None
) -> float | None:
    """Battery actually used over the recent window, over what the model says it should have.

    1.0 is a battery behaving as modelled. None when there is too little
    flight in the window to judge.
    """
    config = config or DEFAULT_CONFIG
    if not history:
        return None
    now = history[-1]
    window = [s for s in history if s.t >= now.t - config.battery_trend_window_s]
    seconds = now.t - window[0].t
    if seconds < max(config.battery_trend_min_s, 1e-9):
        return None
    metres = sum(math.dist(a.position, b.position) for a, b in zip(window, window[1:]))
    expected = metres / config.meters_per_battery_pct + seconds / config.seconds_per_battery_pct
    used = window[0].battery_pct - now.battery_pct
    return used / expected if expected > 0 else None


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
            gap = horizontal_gap((obstacle.x, obstacle.y), a, b)
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
    history: Sequence[TelemetrySnapshot], config: FeedbackConfig, last_battery_replan: float | None
) -> ReplanTrigger | None:
    snapshot = history[-1]
    if last_battery_replan is not None and snapshot.t - last_battery_replan < config.battery_cooldown_s:
        return None
    if not snapshot.remaining:
        return None
    need = estimated_battery_need(snapshot, config)
    ratio = battery_drain_ratio(history, config)
    trending = ratio is not None and ratio > config.battery_trend_tolerance
    if trending:
        need *= ratio
    if snapshot.battery_pct - need >= config.battery_reserve_pct:
        return None
    rate = (
        f" is draining {ratio:.1f}x faster than expected over the last "
        f"{min(config.battery_trend_window_s, snapshot.t - history[0].t):.0f} s; at that rate"
        if trending else ""
    )
    detail = {"battery_pct": round(snapshot.battery_pct, 2), "estimated_need_pct": round(need, 2)}
    if ratio is not None:
        detail["drain_ratio"] = round(ratio, 2)
    return ReplanTrigger(
        kind=TriggerKind.BATTERY_INSUFFICIENT,
        t=snapshot.t,
        reason=(
            f"battery is at {snapshot.battery_pct:.1f}%{rate} and the rest of the route needs about "
            f"{need:.1f}%, which would land with less than the {config.battery_reserve_pct:g}% reserve"
        ),
        detail=detail,
    )


def _off_route(history: Sequence[TelemetrySnapshot], config: FeedbackConfig) -> ReplanTrigger | None:
    now = history[-1]
    threshold = config.deviation_threshold_m
    if not path_deviation(now, threshold):
        return None
    since = now.t - config.deviation_hold_s
    if history[0].t > since:
        return None  # not watched for long enough to call it sustained
    held = [s for s in history if s.t >= since]
    if not all(path_deviation(s, threshold) for s in held):
        return None
    (ax, ay, az), (bx, by, bz) = now.leg
    return ReplanTrigger(
        kind=TriggerKind.OFF_ROUTE,
        t=now.t,
        reason=(
            f"the aircraft is {now.deviation:.1f} m off its planned leg from ({ax:g}, {ay:g}, {az:g}) "
            f"to ({bx:g}, {by:g}, {bz:g}), and has been more than {threshold:g} m off it "
            f"for {config.deviation_hold_s:g} s"
        ),
        detail={"deviation": round(now.deviation, 2), "threshold": threshold, "leg": now.target_index},
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
    if snapshot.deviation >= 1.0:
        lines.append(f"Off the planned leg by {snapshot.deviation:.1f} m (horizontal).")
    new = snapshot.new_obstacles
    if new:
        lines.append("Obstacles detected in flight: " + "; ".join(_describe(o) for o in new) + ".")
    nearby = [e for e in snapshot.perception if e.severity is not PerceptionSeverity.INFO]
    if nearby:
        lines.append(
            "Perception reports nearby: "
            + "; ".join(f"[{e.severity.value}] {e.description}" for e in nearby)
            + "."
        )
    return "\n".join(lines)


def _describe(obstacle: Obstacle) -> str:
    top = f", {obstacle.height:g} m tall" if obstacle.height is not None else ""
    return f"'{obstacle.name}' at ({obstacle.x:g}, {obstacle.y:g}), radius {obstacle.radius:g} m{top}"


def _below_top(obstacle: Obstacle, a, b) -> bool:
    """Whether any part of the leg is low enough to meet the obstacle."""
    return obstacle.height is None or min(a[2], b[2]) < obstacle.height
