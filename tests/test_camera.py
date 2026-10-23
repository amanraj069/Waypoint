"""Depth-camera perception, against frames ray-cast here: no Gazebo needed."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from waypoint.feedback import GazeboCameraPerceptionSource, PerceptionSource
from waypoint.feedback.depth import DepthFrame, Scene, back_project, detect, rotation_from_quaternion
from waypoint.feedback.gazebo_camera import FileFrames
from waypoint.planner import DEFAULT_CONSTRAINTS
from waypoint.planner.models import GeoPoint
from waypoint.sim.state import UAVState
from waypoint.world.models import Building, World

HFOV = 1.274


def rotation(yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    """Camera axes in the world: yaw about z (0 = looking east), then pitch about y (positive = nose down)."""
    cz, sz, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ ry


def render(position, rot, cylinders=(), boxes=(), *, w=160, h=120, hfov=HFOV, far=19.1) -> DepthFrame:
    """Depth along the optical axis for a ground plane at z = 0, vertical cylinders and axis-aligned boxes."""
    f = (w / 2) / math.tan(hfov / 2)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([np.ones_like(u, float), -(u - w / 2) / f, -(v - h / 2) / f], axis=-1) @ rot.T
    px, py, pz = position
    depth = np.full((h, w), np.inf)
    dz = rays[..., 2]
    t_ground = np.where(dz < -1e-9, -pz / np.where(dz < -1e-9, dz, -1), np.inf)
    depth = np.minimum(depth, t_ground)
    for cx, cy, r, top in cylinders:
        ox, oy = px - cx, py - cy
        a = rays[..., 0] ** 2 + rays[..., 1] ** 2
        b = 2 * (ox * rays[..., 0] + oy * rays[..., 1])
        c = ox * ox + oy * oy - r * r
        disc = b * b - 4 * a * c
        ok = (disc >= 0) & (a > 1e-12)
        t = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * np.where(ok, a, 1)), np.inf)
        z = pz + t * rays[..., 2]
        t = np.where((t > 0) & (z >= 0) & (z <= top), t, np.inf)
        depth = np.minimum(depth, t)
    for x0, y0, x1, y1, top in boxes:
        lo, hi = np.array([x0, y0, 0.0]), np.array([x1, y1, top])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - np.array(position)) / rays
            t2 = (hi - np.array(position)) / rays
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= tmin) & (tmax > 0)
        depth = np.minimum(depth, np.where(hit, np.maximum(tmin, 0), np.inf))
    depth[depth > far] = np.inf
    return DepthFrame(depth=depth.astype(np.float32), hfov=hfov, position=tuple(position), rotation=rot)


def test_ground_alone_is_nothing():
    frame = render((0, 0, 15), rotation(pitch=0.3))
    assert detect(frame, Scene(None, DEFAULT_CONSTRAINTS)) == []


def test_back_projection_lands_ground_pixels_on_the_ground():
    frame = render((0, 0, 15), rotation(yaw=0.7, pitch=0.4))
    pts, _ = back_project(frame, stride=2)
    assert len(pts) > 100
    assert np.abs(pts[:, 2]).max() < 0.05


def test_a_crane_ahead_is_found_where_it_stands():
    frame = render((0, 0, 20), rotation(), cylinders=[(15, 2, 3, 60)])
    found = detect(frame, Scene(None, DEFAULT_CONSTRAINTS))
    assert len(found) == 1
    crane = found[0].obstacle
    assert math.hypot(crane.x - 15, crane.y - 2) < 1.0
    assert 2.5 <= crane.radius <= 4.5
    assert crane.height is None  # it runs off the top of the frame: assume it goes all the way up


def test_a_short_obstacle_gets_its_height():
    frame = render((0, 0, 20), rotation(pitch=0.5), cylinders=[(14, -1, 2, 8)])
    found = detect(frame, Scene(None, DEFAULT_CONSTRAINTS))
    assert len(found) == 1 and found[0].obstacle.height == pytest.approx(9.0, abs=1.0)


def test_what_the_map_already_has_is_not_news():
    hall = Building(footprint=[GeoPoint(x=10, y=-8), GeoPoint(x=14, y=-8), GeoPoint(x=14, y=-4), GeoPoint(x=10, y=-4)],
                    height=30, name="the hall")
    world = World(buildings=[hall])
    frame = render((0, 0, 20), rotation(), cylinders=[(15, 4, 2, 60)], boxes=[(10, -8, 14, -4, 30)])
    found = detect(frame, Scene(world, DEFAULT_CONSTRAINTS))
    assert len(found) == 1 and found[0].obstacle.y > 2  # the crane, not the hall


def test_a_tilted_airframe_still_places_it_right():
    """Accelerating, the x500 pitches nose down: the attitude goes into the projection."""
    frame = render((0, 0, 20), rotation(yaw=math.pi / 2, pitch=0.25), cylinders=[(1, 16, 3, 60)])
    crane = detect(frame, Scene(None, DEFAULT_CONSTRAINTS))[0].obstacle
    assert math.hypot(crane.x - 1, crane.y - 16) < 1.0


def test_tracking_confirms_once_and_keeps_it_still():
    frames = iter([render((0, 0, 20), rotation(), cylinders=[(15, 2, 3, 60)]) for _ in range(4)])
    source = GazeboCameraPerceptionSource(frames=lambda: next(frames, None), constraints=DEFAULT_CONSTRAINTS)
    assert isinstance(source, PerceptionSource) and source.name == "gazebo_camera"
    state = UAVState(x=0.0, y=0.0, z=20.0)
    assert source.get_events(state) == []  # seen once: not yet
    first = source.get_events(state)
    assert len(first) == 1 and first[0].source == "gazebo_camera"
    again = source.get_events(state)
    assert again[0].obstacle == first[0].obstacle  # the loop is told once
    assert "not on the map" in first[0].description


def test_frames_written_by_the_bridge_are_read_back(tmp_path):
    frame = render((0, 0, 20), rotation(), cylinders=[(15, 2, 3, 60)])
    h, w = frame.depth.shape
    head = {"stamp": 3.5, "width": w, "height": h, "hfov": HFOV, "pose": [0, 0, 20 - 0.26078, 1, 0, 0, 0]}
    path = tmp_path / "depth.frame"
    path.write_bytes(json.dumps(head).encode() + b"\n" + frame.depth.astype("<f4").tobytes())
    read = FileFrames(path).latest()
    assert read is not None and read.stamp == 3.5
    assert read.position[2] == pytest.approx(20.0, abs=1e-6)
    assert np.array_equal(read.depth, frame.depth)
    assert FileFrames(path).latest() is not None and FileFrames(path).latest().depth.shape == (h, w)


def test_quaternion_and_rotation_agree():
    yaw = 0.8
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    assert np.allclose(rotation_from_quaternion(*q), rotation(yaw=yaw))
