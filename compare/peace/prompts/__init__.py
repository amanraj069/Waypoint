"""PEACE's prompt templates (peace/system_prompts/__init__.py).

`language.txt` is the released planner system prompt: the tool set, the
coordinate frame, the depth-validity rule, the output schema, the safety
limits, the mission rules and the one worked example. Its only difference
from the release is typographic: each em dash is written as a hyphen, a rule
of this repository.

`format_world_context` builds the volatile half of the prompt, the paper's
section III-E: "the current robot state, and the current world model, with a
short history of past states for temporal reasoning."
"""

from __future__ import annotations

from pathlib import Path

from compare.peace.schemas import RobotState, WorldModel

_PROMPTS_DIR = Path(__file__).parent

LANGUAGE_SYSTEM_PROMPT = (_PROMPTS_DIR / "language.txt").read_text(encoding="utf-8")


def _format_detections(detections: list) -> str:
    if not detections:
        return "No objects detected"
    lines = []
    for i, det in enumerate(detections[:10], 1):
        depth_status = (
            "valid position" if det.get("depth_valid", False) else "position UNCERTAIN (depth sensor failure)"
        )
        lines.append(
            f"{i}. {det.get('label', 'unknown')} at "
            f"({det.get('world_x', 0):.1f}m, {det.get('world_y', 0):.1f}m) - "
            f"{depth_status}, confidence={det.get('confidence', 0):.0%}"
        )
    return "\n".join(lines)


def _format_lidar(lidar_ranges: dict) -> str:
    if not lidar_ranges:
        return "No LiDAR data available"
    min_range = lidar_ranges.get("min_range", float("inf"))
    lines = [
        (f"Closest obstacle: {min_range:.1f}m" if min_range != float("inf") else "No obstacles detected"),
        f"- Front: {lidar_ranges.get('front', float('inf')):.1f}m",
        f"- Front-Right: {lidar_ranges.get('front_right', float('inf')):.1f}m",
        f"- Right: {lidar_ranges.get('right', float('inf')):.1f}m",
        f"- Back-Right: {lidar_ranges.get('back_right', float('inf')):.1f}m",
        f"- Back: {lidar_ranges.get('back', float('inf')):.1f}m",
        f"- Back-Left: {lidar_ranges.get('back_left', float('inf')):.1f}m",
        f"- Left: {lidar_ranges.get('left', float('inf')):.1f}m",
        f"- Front-Left: {lidar_ranges.get('front_left', float('inf')):.1f}m",
    ]
    return "\n".join(lines)


def _format_memory_block(history: list[RobotState]) -> str:
    """Past ego state snapshots as a compact temporal context block."""
    if not history:
        return ""
    lines = ["--- Ego State History (oldest → newest) ---"]
    for i, snap in enumerate(history):
        lines.append(
            f"t-{len(history) - i}  "
            f"pos=({snap.position_x:.1f},{snap.position_y:.1f},{snap.position_z:.1f})m  "
            f"yaw={snap.orientation_yaw:.0f}°  "
            f"mode={snap.flight_mode or '?'}  "
            f"armed={snap.armed}"
        )
    lines.append("--- End of History ---")
    return "\n".join(lines)


def _format_ego_block(ego_state: RobotState) -> str:
    return (
        f"Position: ({ego_state.position_x:.2f}m, {ego_state.position_y:.2f}m, {ego_state.position_z:.2f}m), "
        f"Yaw={ego_state.orientation_yaw:.1f}°\n"
        f"Flight Mode: {ego_state.flight_mode or 'UNKNOWN'}, Armed: {ego_state.armed}"
    )


def _format_world_block(world_model: WorldModel) -> str:
    detections_summary = _format_detections(world_model.detections)
    lidar_summary = _format_lidar(world_model.lidar_ranges or {})
    return f"Detected Objects:\n{detections_summary}\n\nObstacle Distances (LiDAR):\n{lidar_summary}"


def format_world_context(
    ego_state: RobotState, world_model: WorldModel, history: list[RobotState] | None = None
) -> str:
    """The current ego state and world model, and optionally the state history, as the LLM's context."""
    context = f"{_format_ego_block(ego_state)}\n\n{_format_world_block(world_model)}"
    if history:
        return f"{context}\n\n{_format_memory_block(history)}"
    return context
