"""Obstacles from a depth image: back-project, subtract what is already known, cluster, fit.

Pure numpy and shapely, no Gazebo: the frames come from anywhere (the Gazebo
bridge, a test that ray-casts its own). One frame is

    depth     H x W metres along the camera's optical axis (inf or 0: nothing seen)
    hfov      horizontal field of view, radians (square pixels)
    position  the camera's position in world ENU metres
    rotation  its orientation, 3x3, camera axes in the world: x forward, y left, z up
              (Gazebo's camera convention)

and `detect(frame, scene)` turns it into the clusters that nothing known
explains, each as an `Obstacle` estimate:

1. **back-project** every `stride`-th pixel with a valid depth into world ENU;
2. **subtract the known world**: the ground (anything within `ground_margin`
   of the terrain), every building of the map (its footprint grown by a
   margin, up to its roof), the obstacles already known, the geofence's walls
   and the no-fly volumes (drawn in the simulator, never solid) -- what is
   left is something the map does not have;
3. **cluster** what is left on a 1 m grid, 8-connected, and drop clusters too
   small to be a structure (route markers, a stray pixel);
4. **fit** a circle to each cluster's points as seen from above. A camera only
   sees the near face of a mast or a crane, so the circle through that arc,
   not the arc's own middle, is where the thing stands. Its height is the top
   of what was seen, or None (unbounded, the safe assumption) when the cluster
   runs off the top of the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Point

from waypoint.planner.models import MissionConstraints, Obstacle
from waypoint.verifier.checks import polygon_of
from waypoint.world.models import World


@dataclass
class DepthFrame:
    depth: np.ndarray
    hfov: float
    position: tuple[float, float, float]
    rotation: np.ndarray
    stamp: float = 0.0


@dataclass
class Detection:
    obstacle: Obstacle
    points: int
    distance: float  # horizontal metres from the camera to the obstacle's edge


@dataclass
class Scene:
    """What is already known, ready for fast vectorised subtraction."""

    world: World | None
    constraints: MissionConstraints | None
    known: list[Obstacle] = field(default_factory=list)
    ground_margin: float = 1.2
    building_margin: float = 1.5
    _tree: STRtree | None = field(default=None, init=False, repr=False)
    _roofs: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.world is not None and self.world.buildings:
            grown = [b.polygon.buffer(self.building_margin) for b in self.world.buildings]
            self._tree = STRtree(grown)
            self._roofs = np.array([b.roof for b in self.world.buildings])
        self.fence = polygon_of(self.constraints.geofence, "geofence").buffer(-2.0) if self.constraints else None
        self.zones = [polygon_of(z, "no-fly zone").buffer(2.0) for z in self.constraints.no_fly_zones] if self.constraints else []

    def ground(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Terrain elevation under many points at once, bilinear like `Terrain.elevation_at`."""
        terrain = self.world.terrain if self.world is not None else None
        if terrain is None:
            return np.zeros_like(x)
        heights = np.asarray(terrain.heights, dtype=float)
        col = np.clip((x - terrain.origin_x) / terrain.spacing, 0.0, terrain.cols - 1.0)
        row = np.clip((y - terrain.origin_y) / terrain.spacing, 0.0, terrain.rows - 1.0)
        c0 = np.floor(col).astype(int)
        r0 = np.floor(row).astype(int)
        c1 = np.minimum(c0 + 1, terrain.cols - 1)
        r1 = np.minimum(r0 + 1, terrain.rows - 1)
        fc, fr = col - c0, row - r0
        top = heights[r0, c0] * (1 - fc) + heights[r0, c1] * fc
        bottom = heights[r1, c0] * (1 - fc) + heights[r1, c1] * fc
        return top * (1 - fr) + bottom * fr

    def unexplained(self, pts: np.ndarray) -> np.ndarray:
        """A mask over N x 3 world points: True for those nothing known accounts for."""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        keep = z > self.ground(x, y) + self.ground_margin
        if self.fence is not None and keep.any():
            keep &= shapely.contains_xy(self.fence, x, y)
        for zone in self.zones:
            if keep.any():
                keep &= ~shapely.contains_xy(zone, x, y)
        for o in self.known:
            near = np.hypot(x - o.x, y - o.y) < o.radius + 1.5
            if o.height is not None:
                near &= z < o.height + 1.5
            keep &= ~near
        if self._tree is not None and keep.any():
            index = np.flatnonzero(keep)
            hits = self._tree.query(shapely.points(x[index], y[index]), predicate="intersects")
            if hits.size:
                point_at, building_at = hits
                inside = z[index[point_at]] <= self._roofs[building_at] + self.building_margin
                keep[index[point_at[inside]]] = False
        return keep


def back_project(frame: DepthFrame, *, stride: int = 4, max_range: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    """World points for every `stride`-th valid pixel, and each one's image row (to tell a cut-off top)."""
    depth = frame.depth[::stride, ::stride]
    h, w = frame.depth.shape
    rows = np.arange(0, h, stride)[:, None] * np.ones((1, depth.shape[1]))
    cols = np.ones((depth.shape[0], 1)) * np.arange(0, w, stride)[None, :]
    valid = np.isfinite(depth) & (depth > 0.5) & (depth < max_range)
    d = depth[valid]
    u, v = cols[valid], rows[valid]
    f = (w / 2.0) / math.tan(frame.hfov / 2.0)
    cam = np.stack([d, -(u - w / 2.0) * d / f, -(v - h / 2.0) * d / f], axis=1)
    world = cam @ np.asarray(frame.rotation).T + np.asarray(frame.position)[None, :]
    return world, v


def clusters(pts: np.ndarray, cell: float = 1.0) -> list[np.ndarray]:
    """Indices of points grouped by 8-connected occupied cells of an xy grid."""
    if not len(pts):
        return []
    keys = np.floor(pts[:, :2] / cell).astype(int)
    cells: dict[tuple[int, int], list[int]] = {}
    for index, (cx, cy) in enumerate(map(tuple, keys)):
        cells.setdefault((cx, cy), []).append(index)
    seen: set[tuple[int, int]] = set()
    groups = []
    for start in cells:
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            cx, cy = stack.pop()
            members += cells[(cx, cy)]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        groups.append(np.array(members))
    return groups


def fit_circle(xy: np.ndarray, toward: tuple[float, float]) -> tuple[float, float, float]:
    """Centre and radius of the circle through an arc of points (algebraic least squares).

    Falls back to the points' bounding circle, pushed away from the camera by
    its radius, when the arc is too short or too straight to fit.
    """
    x, y = xy[:, 0], xy[:, 1]
    extent = max(np.ptp(x), np.ptp(y), 0.5)
    try:
        a = np.stack([x, y, np.ones_like(x)], axis=1)
        b = -(x * x + y * y)
        (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = -d / 2, -e / 2
        r = math.sqrt(max(cx * cx + cy * cy - f, 0.0))
        if 0.3 < r < 3.0 * extent + 2.0:
            return float(cx), float(cy), float(r)
    except np.linalg.LinAlgError:
        pass
    mx, my = float(x.mean()), float(y.mean())
    r = extent / 2.0
    away = math.atan2(my - toward[1], mx - toward[0])
    return mx + r * math.cos(away), my + r * math.sin(away), r


def detect(
    frame: DepthFrame,
    scene: Scene,
    *,
    stride: int = 4,
    max_range: float = 18.0,
    min_points: int = 12,
    min_extent: float = 1.5,
) -> list[Detection]:
    """Everything in the frame that the known world does not explain, as obstacle estimates."""
    pts, rows = back_project(frame, stride=stride, max_range=max_range)
    if not len(pts):
        return []
    keep = scene.unexplained(pts)
    pts, rows = pts[keep], rows[keep]
    cam = frame.position
    found = []
    for members in clusters(pts):
        if len(members) < min_points:
            continue
        group = pts[members]
        if max(np.ptp(group[:, 0]), np.ptp(group[:, 1]), np.ptp(group[:, 2])) < min_extent:
            continue  # a marker, a bird, noise: nothing to plan around
        cx, cy, r = fit_circle(group[:, :2], (cam[0], cam[1]))
        cut_off = rows[members].min() <= stride  # it runs off the top of the image
        top = None if cut_off else float(group[:, 2].max()) + 1.0
        gap = max(0.0, math.hypot(cx - cam[0], cy - cam[1]) - r)
        found.append(Detection(
            obstacle=Obstacle(x=round(cx, 1), y=round(cy, 1), radius=round(r + 0.5, 1),
                              height=None if top is None else round(top, 1), name="perceived obstacle"),
            points=len(members),
            distance=gap,
        ))
    return found


def rotation_from_quaternion(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
