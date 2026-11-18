"""The sensors the rebuilt systems read, simulated from Waypoint's world.

PEACE's airframe carries a planar LiDAR and a forward RGB-D camera; its world
model is built from them. Waypoint's simulator has no rendered images, so both
are computed from the geometry the simulator already has:

    lidar_scan(...)     a 360 degree planar scan at the aircraft's altitude:
                        ranges to building walls, obstacles and scene objects
    Camera.observe(...) what a pinhole RGB-D camera would see of a set of
                        scene objects: a bounding box for each one in view,
                        and a depth image with the object's depth in its box

They produce the raw inputs (a LaserScan's ranges, boxes and a depth image),
not the conclusions, so a rebuilt system's own processing still runs on them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point

from waypoint.planner.models import Obstacle
from waypoint.world.models import World

Vector = tuple[float, float, float]


@dataclass(frozen=True)
class SceneObject:
    """Something a detector can find: a person, a car, a bench. An upright box, in the world frame."""

    label: str
    x: float
    y: float
    width: float = 0.6
    height: float = 1.8
    base: float = 0.0
    #: What a detector reports for it when it is in clear view, 0 to 100.
    confidence: float = 90.0

    @property
    def top(self) -> float:
        return self.base + self.height


def lidar_scan(
    world: World | None,
    obstacles: list[Obstacle],
    position: Vector,
    yaw: float,
    *,
    objects: list[SceneObject] | tuple[SceneObject, ...] = (),
    samples: int = 360,
    range_min: float = 0.1,
    range_max: float = 30.0,
) -> tuple[list[float], float, float]:
    """A planar scan from `position` (world frame): (ranges, angle_min, angle_increment).

    Angles are relative to the aircraft's heading, 0 ahead and counter-clockwise
    positive, from -pi; a ray that hits nothing within `range_max` reads inf,
    as a LaserScan does.
    """
    x, y, z = position
    here = Point(x, y)
    solids = []
    if world is not None:
        for building in world.obstructions(here.buffer(range_max)):
            if building.base <= z <= building.roof:
                solids.append(building.polygon)
    for obstacle in obstacles:
        if obstacle.radius > 0 and (obstacle.height is None or z <= obstacle.height):
            solids.append(Point(obstacle.x, obstacle.y).buffer(obstacle.radius, 16))
    for item in objects:
        if item.base <= z <= item.top:
            solids.append(Point(item.x, item.y).buffer(max(item.width / 2.0, 0.05), 8))
    angle_min = -math.pi
    increment = 2.0 * math.pi / samples
    ranges = [math.inf] * samples
    if not solids:
        return ranges, angle_min, increment
    boundaries = [solid.boundary for solid in solids]
    inside = any(solid.contains(here) for solid in solids)
    for index in range(samples):
        bearing = yaw + angle_min + index * increment
        ray = LineString([(x, y), (x + range_max * math.cos(bearing), y + range_max * math.sin(bearing))])
        nearest = math.inf
        for boundary in boundaries:
            hit = ray.intersection(boundary)
            if hit.is_empty:
                continue
            nearest = min(nearest, here.distance(hit))
        if inside and nearest is math.inf:
            nearest = range_min
        if range_min <= nearest <= range_max:
            ranges[index] = nearest
    return ranges, angle_min, increment


@dataclass
class Sighting:
    """One object as a camera frame shows it."""

    item: SceneObject
    bbox: list[int]
    depth: float


@dataclass
class Camera:
    """A forward-facing pinhole RGB-D camera, level with the horizon.

    The defaults are the OAK-D Lite on PX4's x500_depth, the airframe PEACE
    flies: 640 x 480, 1.274 rad horizontal field of view.

    Image convention: a point to the left of the optical axis lands to the
    right of the image centre (u = cx + fx * left / forward). That is the
    convention PEACE's released depth projector assumes when it turns a pixel
    back into a world position, so its output is the object's true position.
    """

    width: int = 640
    height: int = 480
    hfov: float = 1.274
    #: How far away a detector can still pick an object out of the image.
    detect_range: float = 40.0
    #: How far the depth sensor reads; beyond it an object is seen but its depth is not.
    depth_range: float = 19.0
    min_range: float = 0.3
    fx: float = field(init=False)
    fy: float = field(init=False)
    cx: float = field(init=False)
    cy: float = field(init=False)

    def __post_init__(self) -> None:
        self.fx = self.fy = (self.width / 2.0) / math.tan(self.hfov / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def observe(
        self,
        objects: list[SceneObject] | tuple[SceneObject, ...],
        position: Vector,
        yaw: float,
        world: World | None = None,
    ) -> tuple[list[Sighting], np.ndarray]:
        """The objects in view from `position` (world frame) facing `yaw`, and the depth image in metres.

        A depth of 0 means the sensor read nothing there (background, or an
        object beyond `depth_range`).
        """
        depth_image = np.zeros((self.height, self.width), dtype=np.float32)
        px, py, pz = position
        sightings: list[Sighting] = []
        for item in objects:
            dx, dy = item.x - px, item.y - py
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            left = -dx * math.sin(yaw) + dy * math.cos(yaw)
            up = (item.base + item.height / 2.0) - pz
            distance = math.sqrt(dx * dx + dy * dy + up * up)
            if forward <= self.min_range or distance > self.detect_range:
                continue
            if world is not None and _blocked(world, (px, py, pz), (item.x, item.y, item.base + item.height / 2.0)):
                continue
            u = self.cx + self.fx * left / forward
            v = self.cy - self.fy * up / forward
            half_w = max(0.5, self.fx * (item.width / 2.0) / forward)
            half_h = max(0.5, self.fy * (item.height / 2.0) / forward)
            x1, x2 = int(round(u - half_w)), int(round(u + half_w))
            y1, y2 = int(round(v - half_h)), int(round(v + half_h))
            if x2 <= 0 or y2 <= 0 or x1 >= self.width - 1 or y1 >= self.height - 1:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.width - 1, max(x2, x1 + 1)), min(self.height - 1, max(y2, y1 + 1))
            if forward <= self.depth_range:
                depth_image[y1 : y2 + 1, x1 : x2 + 1] = forward
            sightings.append(Sighting(item=item, bbox=[x1, y1, x2, y2], depth=forward))
        # Nearer objects are drawn last, so they cover the ones behind them.
        for sighting in sorted(sightings, key=lambda s: -s.depth):
            if sighting.depth <= self.depth_range:
                x1, y1, x2, y2 = sighting.bbox
                depth_image[y1 : y2 + 1, x1 : x2 + 1] = sighting.depth
        return sightings, depth_image


def _blocked(world: World, a: Vector, b: Vector) -> bool:
    """Whether a building stands in the straight line of sight from a to b."""
    line = LineString([(a[0], a[1]), (b[0], b[1])])
    length = line.length
    for building in world.obstructions(line):
        crossing = line.intersection(building.polygon)
        if crossing.is_empty or length <= 0:
            continue
        # The sight line's height where it passes over the footprint.
        for x, y in _ends(crossing):
            share = math.hypot(x - a[0], y - a[1]) / length
            if a[2] + (b[2] - a[2]) * share <= building.roof:
                return True
    return False


def _ends(geometry) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for part in getattr(geometry, "geoms", [geometry]):
        coords = list(getattr(part, "coords", []))
        if coords:
            points.extend([coords[0], coords[-1]])
    return points
