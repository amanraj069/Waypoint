"""The depth camera, out of gz-transport: the newest frame and the camera's pose, to one file.

Runs under Gazebo's Python (see `gazebo.launch.gz_python`) and imports nothing
from the project. It subscribes to the depth image and to the world's poses,
and at most `--rate` times a second writes

    {"stamp": <sim s>, "width": W, "height": H, "hfov": <rad>, "pose": [x, y, z, qw, qx, qy, qz]}\\n
    <W * H little-endian float32 depths, row by row>

to `--out`, through a temporary file and a rename, so a reader never sees half
a frame. The pose is the aircraft model's at the moment the frame arrives;
the reader adds the camera's mount on the airframe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from gz.msgs.image_pb2 import Image
from gz.msgs.pose_v_pb2 import Pose_V
from gz.transport import Node


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topic", default="/depth_camera")
    parser.add_argument("--hfov", type=float, default=1.274)
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    node = Node()
    lock = threading.Lock()
    pose: dict = {}
    written = [0.0]

    def on_pose(message: Pose_V) -> None:
        for p in message.pose:
            if p.name == args.model:
                with lock:
                    pose["value"] = [
                        p.position.x, p.position.y, p.position.z,
                        p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z,
                    ]
                return

    def on_depth(message: Image) -> None:
        now = time.monotonic()
        if now - written[0] < 1.0 / args.rate:
            return
        with lock:
            here = pose.get("value")
        if here is None:
            return
        stamp = message.header.stamp.sec + message.header.stamp.nsec * 1e-9
        head = {"stamp": stamp, "width": message.width, "height": message.height, "hfov": args.hfov, "pose": here}
        tmp = args.out + ".tmp"
        with open(tmp, "wb") as f:
            f.write(json.dumps(head).encode() + b"\n")
            f.write(message.data)
        os.replace(tmp, args.out)
        written[0] = now

    if not node.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_pose):
        print(f"error: cannot subscribe to /world/{args.world}/pose/info", flush=True)
        return 1
    if not node.subscribe(Image, args.topic, on_depth):
        print(f"error: cannot subscribe to {args.topic}", flush=True)
        return 1
    print("ready", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
