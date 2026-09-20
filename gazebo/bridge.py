"""The one process that speaks gz-transport: JSON lines in, Gazebo service calls out.

Gazebo's Python bindings are built for Homebrew's Python, not the project's, so
this file runs under that interpreter (see `gazebo.launch.gz_python`) and must
import nothing from the project. The launcher writes one command per line:

    {"op": "pose", "x": 0, "y": 0, "z": 10, "yaw": 1.57}
    {"op": "trail", "points": [[0, 0, 0], [1, 2, 3]]}
    {"op": "spawn", "sdf": "<sdf>...</sdf>"}
    {"op": "follow"}      (chase camera; held until the GUI is up)

and reads back a single "ready" once the world's services are up.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.cameratrack_pb2 import CameraTrack
from gz.msgs.entity_factory_pb2 import EntityFactory
from gz.msgs.marker_pb2 import Marker
from gz.msgs.pose_pb2 import Pose
from gz.transport import Node

TIMEOUT_MS = 500
TRAIL_COLOUR = (1.0, 0.12, 0.82)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wait", type=float, default=60.0)
    args = parser.parse_args()

    node = Node()
    set_pose = f"/world/{args.world}/set_pose"
    create = f"/world/{args.world}/create"

    deadline = time.monotonic() + args.wait
    while set_pose not in node.service_list():
        if time.monotonic() > deadline:
            print(f"error: {set_pose} never appeared; is gz sim running?", flush=True)
            return 1
        time.sleep(0.2)
    print("ready", flush=True)

    # The trail and the camera belong to the GUI; with no window (or before it
    # is up) their endpoints do not exist, and asking would block every frame.
    known: set[str] = set()
    checked = 0.0

    def offered(name: str) -> bool:
        nonlocal known, checked
        if name not in known and time.monotonic() - checked > 2.0:
            known = set(node.service_list()) | set(node.topic_list())
            checked = time.monotonic()
        return name in known

    camera = node.advertise("/gui/track", CameraTrack)
    follow = 0  # frames left to keep asking the camera to follow
    for line in sys.stdin:
        if not line.strip():
            continue
        command = json.loads(line)
        op = command["op"]
        if op == "follow":
            follow = -1
        # Held until the window is up (it starts well after the server), then
        # repeated for a moment, because a new subscriber misses the first few.
        if follow and offered("/gui/track"):
            follow = 30 if follow < 0 else follow - 1
            camera.publish(_follow(args.model))
        if op == "pose":
            node.request(set_pose, _pose(args.model, command), Pose, Boolean, TIMEOUT_MS)
        elif op == "trail" and offered("/marker"):
            node.request("/marker", _trail(command["points"]), Marker, Boolean, TIMEOUT_MS)
        elif op == "spawn":
            factory = EntityFactory(sdf=command["sdf"], allow_renaming=True)
            node.request(create, factory, EntityFactory, Boolean, TIMEOUT_MS * 4)
    return 0


def _pose(model: str, command: dict) -> Pose:
    pose = Pose(name=model)
    pose.position.x, pose.position.y, pose.position.z = command["x"], command["y"], command["z"]
    half = command.get("yaw", 0.0) / 2
    pose.orientation.w, pose.orientation.z = math.cos(half), math.sin(half)
    return pose


def _follow(model: str) -> CameraTrack:
    """Chase from behind and above, looking at the aircraft."""
    track = CameraTrack(track_mode=CameraTrack.FOLLOW_LOOK_AT, follow_pgain=0.08, track_pgain=0.2)
    track.follow_target.name = track.track_target.name = model
    track.follow_offset.x, track.follow_offset.y, track.follow_offset.z = -10.0, -10.0, 6.0
    return track


def _trail(points: list[list[float]]) -> Marker:
    marker = Marker(ns="trail", id=1, action=Marker.ADD_MODIFY, type=Marker.LINE_STRIP)
    r, g, b = TRAIL_COLOUR
    for colour in (marker.material.ambient, marker.material.diffuse, marker.material.emissive):
        colour.r, colour.g, colour.b, colour.a = r, g, b, 1.0
    for x, y, z in points:
        point = marker.point.add()
        point.x, point.y, point.z = x, y, z
    return marker


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # the launcher stops us with SIGINT on the way out
        raise SystemExit(0) from None
