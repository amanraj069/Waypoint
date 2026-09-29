"""The Gazebo scene: a run file becomes an SDF world and a terrain mesh.

Only the generated files are tested here; launching Gazebo needs a display and
the gz binaries, which CI does not have.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from waypoint.gazebo.launch import _yaw
from waypoint.gazebo.scene import UAV_NAME, WORLD_NAME, write_scene
from waypoint.world.generate import build


def _run(world=None, **extra) -> dict:
    return {
        "world": world.model_dump(mode="json") if world else None,
        "constraints": {
            "geofence": [{"x": -100, "y": -100}, {"x": 100, "y": -100}, {"x": 100, "y": 100}],
            "max_altitude": 120.0,
            "no_fly_zones": [[{"x": 10, "y": 10}, {"x": 20, "y": 10}, {"x": 20, "y": 20}]],
        },
        "waypoints": [
            {"x": 0, "y": 0, "z": 20, "action": "MOVE"},
            {"x": 50, "y": 0, "z": 20, "action": "CAPTURE"},
        ],
        "ticks": [
            {"timestamp": 0.0, "x": 0, "y": 0, "z": 0, "heading": 0},
            {"timestamp": 1.0, "x": 60, "y": 0, "z": 0, "heading": 0},
        ],
        **extra,
    }


def _models(sdf_path) -> dict[str, ET.Element]:
    world = ET.parse(sdf_path).getroot().find("world")
    assert world.get("name") == WORLD_NAME
    return {model.get("name"): model for model in world.findall("model")}


def test_empty_world_still_has_ground_route_and_aircraft(tmp_path):
    scene = write_scene(_run(), tmp_path)
    models = _models(scene.sdf)
    assert {"terrain", "route", "envelope", "launch_pad", "landing_pad", UAV_NAME} <= set(models)
    assert (tmp_path / "terrain.obj").stat().st_size > 0
    assert "newmtl lawn" in (tmp_path / "terrain.mtl").read_text()


def test_buildings_are_named_models_and_trees_share_one(tmp_path):
    world = build("lodhi")
    scene = write_scene(_run(world), tmp_path)
    models = _models(scene.sdf)
    solid = [b for b in world.buildings if b.kind != "tree"]
    trees = [b for b in world.buildings if b.kind == "tree"]
    extruded = [
        name for name, model in models.items()
        if name != "envelope" and model.find("link/visual/geometry/polyline") is not None
    ]
    assert len(extruded) == len(solid)
    assert len(models["trees"].findall("link/visual")) == 2 * len(trees)
    assert "bara_gumbad" in models


def test_ground_cover_becomes_terrain_materials(tmp_path):
    write_scene(_run(build("lodhi")), tmp_path)
    used = {line.split()[1] for line in (tmp_path / "terrain.obj").read_text().splitlines()
            if line.startswith("usemtl")}
    assert {"road", "water"} <= used


def test_obstacles_that_appear_mid_flight_are_held_back(tmp_path):
    obstacles = [
        {"x": 30, "y": 30, "radius": 5, "height": 40, "name": "mast", "appeared_at": 0.0},
        {"x": 40, "y": 40, "radius": 10, "height": 40, "name": "crane", "appeared_at": 55.6},
    ]
    scene = write_scene(_run(obstacles=obstacles), tmp_path)
    assert "obstacle_0" in _models(scene.sdf)
    assert [late.t for late in scene.late] == [55.6]
    assert ET.fromstring(scene.late[0].sdf).find("model").get("name") == "obstacle_1"


def test_yaw_turns_the_short_way_round():
    # Halfway from 350 to 10 degrees is due east, not due west.
    yaw = _yaw({"heading": 350.0}, {"heading": 10.0}, 0.5)
    assert (math.cos(yaw), math.sin(yaw)) == pytest.approx((1.0, 0.0), abs=1e-9)


@pytest.mark.skipif(shutil.which("gz") is None, reason="Gazebo is not installed")
def test_gazebo_accepts_the_world(tmp_path):
    scene = write_scene(_run(build("chowk")), tmp_path)
    result = subprocess.run(["gz", "sdf", "-k", str(scene.sdf)], capture_output=True, text=True)
    assert "Valid" in result.stdout, result.stdout + result.stderr
