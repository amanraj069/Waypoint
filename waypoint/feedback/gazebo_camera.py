"""`GazeboCameraPerceptionSource`: obstacles seen by the PX4 airframe's depth camera.

This is where perception stops being symbolic. The `px4` backend can fly
`x500_depth` (`WAYPOINT_PX4_MODEL=x500_depth`), PX4's x500 carrying an OAK-D
Lite, and Gazebo then publishes, headless or with the window:

    /depth_camera     640 x 480 depth (R_FLOAT32), 1.274 rad across, 0.2 to 19.1 m

How it works:

1. **Frames.** Gazebo's Python bindings are built for Homebrew's Python, not
   the project's, so `waypoint/gazebo/depth_bridge.py` runs under `.venv-gz`:
   it subscribes to the depth topic and to the world's poses, and writes the
   newest frame, with the camera's full pose at that moment (attitude
   included: a multirotor tilts as it accelerates), to one file, replaced
   atomically. This source reads that file when it changes and never blocks
   the loop; a missed frame is simply no news.
2. **Detection** is geometric, not learned (`feedback.depth.detect`): the
   frame is back-projected into the world, everything the map already
   explains is taken away (ground, buildings, known obstacles, the fence's
   and no-fly zones' drawn walls), and what is left is clustered and fitted.
   Anything solid the map does not have is an obstacle, whatever it looks like.
3. **Tracking.** A cluster seen in two frames within `merge` metres of the
   same place is confirmed; once confirmed, its estimate is frozen and
   reported every tick as one obstacle, so the loop is told once, as it is by
   the symbolic source. A later, clearly bigger estimate (the camera has come
   round to see more of it) is reported as a further obstacle.

The range is the camera's: about 19 m. That is short, which is why the
in-flight repair (milliseconds) and the shield matter when this source is on.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waypoint.feedback.depth import DepthFrame, Scene, detect, rotation_from_quaternion
from waypoint.feedback.perception import PerceptionEvent, PerceptionKind, PerceptionSeverity, PerceptionSource
from waypoint.planner.models import MissionConstraints, Obstacle
from waypoint.sim.state import UAVState
from waypoint.world.models import World

#: The OAK-D Lite on PX4's x500_depth, as Gazebo publishes it.
RGB_TOPIC = "/world/waypoint/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
DEPTH_TOPIC = "/depth_camera"
#: The depth sensor's field of view, and where it sits on the airframe (x500_depth's
#: camera joint plus the sensor's own offset on the OAK-D, metres, body frame).
DEPTH_HFOV = 1.274
CAMERA_OFFSET = (0.12 + 0.01233, 0.03 - 0.03, 0.242 + 0.01878)


@dataclass
class _Track:
    obstacle: Obstacle
    hits: int
    seen: float
    reported: Obstacle | None = None


class FileFrames:
    """Frames the depth bridge writes: one JSON line (stamp, size, pose), then raw float32 depth."""

    def __init__(self, path: Path, *, offset: tuple[float, float, float] = CAMERA_OFFSET, hfov: float = DEPTH_HFOV):
        self.path = Path(path)
        self.offset = np.asarray(offset)
        self.hfov = hfov
        self._mtime = 0.0

    def latest(self) -> DepthFrame | None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return None
        if mtime == self._mtime:
            return None
        self._mtime = mtime
        raw = self.path.read_bytes()
        newline = raw.index(b"\n")
        head = json.loads(raw[:newline])
        depth = np.frombuffer(raw[newline + 1:], dtype=np.float32).reshape(head["height"], head["width"])
        x, y, z, qw, qx, qy, qz = head["pose"]
        body = rotation_from_quaternion(qw, qx, qy, qz)
        position = np.array([x, y, z]) + body @ self.offset
        return DepthFrame(depth=depth, hfov=head.get("hfov", self.hfov), position=tuple(position),
                          rotation=body, stamp=head.get("stamp", 0.0))


class GazeboCameraPerceptionSource(PerceptionSource):
    """Obstacles detected in the x500_depth's depth frames."""

    name = "gazebo_camera"

    #: Edge distances, in metres, under which an event is CRITICAL or WARNING (as the symbolic source).
    critical_within = 10.0
    warning_within = 40.0

    def __init__(
        self,
        *,
        frames: FileFrames | Callable[[], DepthFrame | None] | None = None,
        world: World | None = None,
        constraints: MissionConstraints | None = None,
        rgb_topic: str = RGB_TOPIC,
        depth_topic: str = DEPTH_TOPIC,
        max_range: float = 18.0,
        merge: float = 4.0,
        confirm: int = 2,
    ):
        self.frames = frames
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.max_range = max_range
        self.merge = merge
        self.confirm = confirm
        self.scene = Scene(world, constraints, list(constraints.obstacles) if constraints else [])
        self.tracks: list[_Track] = []
        self.frames_seen = 0
        self._bridge: subprocess.Popen | None = None
        self._backend = None

    # -- the loop's interface ------------------------------------------------------

    def get_events(self, current_state: UAVState) -> list[PerceptionEvent]:
        if self.frames is None and self._backend is not None:
            self._start()  # the backend is flying by now, so its world and its Gazebo are up
        if self.frames is None:
            raise RuntimeError(
                "camera perception has no frames: use GazeboCameraPerceptionSource.on(backend) (or for_px4), "
                "or pass frames="
            )
        latest = self.frames.latest() if hasattr(self.frames, "latest") else self.frames()
        if latest is not None:
            self.frames_seen += 1
            self._update(latest)
        return [self._event(track.reported, current_state) for track in self.tracks if track.reported is not None]

    # -- tracking --------------------------------------------------------------------

    def _update(self, frame: DepthFrame) -> None:
        for found in detect(frame, self.scene, max_range=self.max_range):
            estimate = found.obstacle
            track = next(
                (t for t in self.tracks if math.hypot(t.obstacle.x - estimate.x, t.obstacle.y - estimate.y)
                 < self.merge + max(t.obstacle.radius, estimate.radius)),
                None,
            )
            if track is None:
                self.tracks.append(_Track(obstacle=estimate, hits=1, seen=frame.stamp))
                continue
            track.hits += 1
            track.seen = frame.stamp
            merged = _merge(track.obstacle, estimate)
            track.obstacle = merged
            if track.reported is None and track.hits >= self.confirm:
                track.reported = merged.model_copy(update={"name": f"perceived obstacle {self._count()}"})
            elif track.reported is not None and _grown(track.reported, merged):
                # Seen more of it now: report the bigger estimate as well (the old one stays known).
                track.reported = merged.model_copy(update={"name": f"perceived obstacle {self._count()}"})

    def _count(self) -> int:
        return sum(t.reported is not None for t in self.tracks) + 1

    def _event(self, obstacle: Obstacle, state: UAVState) -> PerceptionEvent:
        gap = max(0.0, math.hypot(obstacle.x - state.x, obstacle.y - state.y) - obstacle.radius)
        severity = (
            PerceptionSeverity.CRITICAL if gap <= self.critical_within
            else PerceptionSeverity.WARNING if gap <= self.warning_within
            else PerceptionSeverity.INFO
        )
        top = "of unknown height" if obstacle.height is None else f"{obstacle.height:g} m tall"
        return PerceptionEvent(
            kind=PerceptionKind.OBSTACLE,
            t=state.timestamp,
            position=(obstacle.x, obstacle.y, 0.0),
            severity=severity,
            description=(
                f"the depth camera sees something not on the map at ({obstacle.x:g}, {obstacle.y:g}), "
                f"about {obstacle.radius:g} m in radius, {top}, {gap:.0f} m from the aircraft"
            ),
            source=self.name,
            distance=gap,
            obstacle=obstacle,
        )

    # -- live, with PX4 ------------------------------------------------------------------

    @classmethod
    def on(cls, backend) -> "GazeboCameraPerceptionSource":
        """Camera perception for a PX4 backend flying x500_depth, started on its first call.

        `run_mission` prepares the backend itself, so the bridge cannot start
        before the flight does; it starts the first time the loop asks, and
        until frames arrive the source reports nothing.
        """
        if getattr(backend, "model", None) != "x500_depth":
            raise RuntimeError("camera perception needs the x500_depth airframe (WAYPOINT_PX4_MODEL=x500_depth)")
        source = cls()
        source._backend = backend
        return source

    @classmethod
    def for_px4(cls, backend, *, frames_path: Path | None = None) -> "GazeboCameraPerceptionSource":
        """Start the depth bridge now, against a backend that is already prepared."""
        source = cls.on(backend)
        source._start(frames_path)
        return source

    def _start(self, frames_path: Path | None = None) -> None:
        from waypoint.gazebo.launch import gz_python  # noqa: PLC0415
        from waypoint.px4.sitl import WORLD_NAME, model_name  # noqa: PLC0415

        backend = self._backend
        path = frames_path or Path(backend._sitl.dir) / "depth.frame"
        self.scene = Scene(backend.world, backend.constraints, list(backend.constraints.obstacles))
        self.frames = FileFrames(path)
        bridge = Path(__file__).resolve().parents[1] / "gazebo" / "depth_bridge.py"
        self._bridge = subprocess.Popen(
            [gz_python(), str(bridge), "--world", WORLD_NAME, "--model", model_name(backend.model),
             "--out", str(path), "--hfov", str(DEPTH_HFOV)],
            env=dict(os.environ, GZ_IP="127.0.0.1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        time.sleep(0.5)
        if self._bridge.poll() is not None:
            raise RuntimeError(f"the depth bridge stopped: {self._bridge.stdout.read()[-400:]}")

    def close(self) -> None:
        if self._bridge is not None and self._bridge.poll() is None:
            self._bridge.terminate()
            try:
                self._bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._bridge.kill()


def _merge(a: Obstacle, b: Obstacle) -> Obstacle:
    """Two sightings of one thing: the centre between them, the bigger size."""
    height = None if a.height is None or b.height is None else max(a.height, b.height)
    return a.model_copy(update={
        "x": round((a.x + b.x) / 2, 1), "y": round((a.y + b.y) / 2, 1),
        "radius": max(a.radius, b.radius), "height": height,
    })


def _grown(old: Obstacle, new: Obstacle) -> bool:
    taller = old.height is not None and (new.height is None or new.height > old.height + 2.0)
    return new.radius > old.radius + 1.5 or taller
