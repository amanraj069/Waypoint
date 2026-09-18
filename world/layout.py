"""Laying out the things that make a place look lived in.

The hand-authored worlds name the buildings a mission is *about*. Everything
between them -- the roads, the lawns, the houses nobody will ask to photograph,
the trees along every avenue -- used to be left out, which made each place a
handful of monuments on an empty plate. This module fills that in, with two
rules:

* Nothing lands where it would change what the world is for. Every placement
  goes through a `Lot`, which knows the launch pad, the corridors a place is
  flown down and every building already standing, and refuses anything that
  would touch them.
* Everything is seeded. The same builder gives the same world on every machine.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable

from shapely import prepared
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from planner.models import GeoPoint
from world.models import Area, Building

Ground = Callable[[float, float], float]


def points(geometry: BaseGeometry) -> list[GeoPoint]:
    """A shapely polygon's outline as footprint points, closing point dropped."""
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda g: g.area)
    coords = list(geometry.exterior.coords)[:-1]
    return [GeoPoint(x=round(x, 2), y=round(y, 2)) for x, y in coords]


def area(geometry: BaseGeometry, kind: str, name: str = "", level: float | None = None) -> Area:
    return Area(footprint=points(geometry), kind=kind, name=name, level=level)


def strip(path: Iterable[tuple[float, float]], width: float) -> Polygon:
    """A road, a canal or a walk: a polyline with a width and square ends."""
    return LineString(list(path)).buffer(width / 2.0, cap_style="flat", join_style="round",
                                         quad_segs=4)


def blob(rng: random.Random, cx: float, cy: float, radius: float,
         wobble: float = 0.18, sides: int = 28, stretch: float = 1.0,
         rot: float = 0.0) -> Polygon:
    """An irregular round patch: a lake, a grove's floor, an estate on a hillside."""
    phases = [rng.uniform(0, math.tau) for _ in range(3)]
    angle = math.radians(rot)
    ring = []
    for i in range(sides):
        t = math.tau * i / sides
        r = radius * (1.0 + wobble * (0.6 * math.sin(2 * t + phases[0])
                                      + 0.3 * math.sin(3 * t + phases[1])
                                      + 0.2 * math.sin(5 * t + phases[2])))
        dx, dy = r * math.cos(t) * stretch, r * math.sin(t)
        ring.append((cx + dx * math.cos(angle) - dy * math.sin(angle),
                     cy + dx * math.sin(angle) + dy * math.cos(angle)))
    return Polygon(ring)


def ring(cx: float, cy: float, radius: float, width: float, sides: int = 48,
         rot: float = 0.0) -> Polygon:
    """A ring road: an annulus, as a polygon with a hole."""
    def loop(r: float) -> list[tuple[float, float]]:
        start = math.radians(rot)
        return [(cx + r * math.cos(start + math.tau * i / sides),
                 cy + r * math.sin(start + math.tau * i / sides)) for i in range(sides)]
    return Polygon(loop(radius + width / 2.0), [loop(radius - width / 2.0)])


def rect(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def disc(x: float, y: float, radius: float) -> Polygon:
    return Point(x, y).buffer(radius, quad_segs=12)


def rotated_box(cx: float, cy: float, width: float, depth: float, rot: float = 0.0) -> Polygon:
    hw, hd = width / 2.0, depth / 2.0
    angle = math.radians(rot)
    cos, sin = math.cos(angle), math.sin(angle)
    return Polygon([(cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)
                    for dx, dy in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))])


class Lot:
    """Where there is still room to put something.

    Seeded with the places that must stay open -- the pad, the street a world
    is flown down, a lawn a plan skims at 10 m -- and with every building that
    already stands. Each thing placed is claimed, so nothing lands on top of
    anything else.
    """

    def __init__(self, keep_clear: Iterable[BaseGeometry], standing: Iterable[Building] = (),
                 margin: float = 4.0) -> None:
        blocked = [g for g in keep_clear if not g.is_empty]
        self._blocked = prepared.prep(unary_union(blocked)) if blocked else None
        self._cell = 40.0
        self._grid: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
        for building in standing:
            self.claim(building.polygon, margin)

    def _cells(self, bounds: tuple[float, float, float, float]):
        minx, miny, maxx, maxy = bounds
        for i in range(math.floor(minx / self._cell), math.floor(maxx / self._cell) + 1):
            for j in range(math.floor(miny / self._cell), math.floor(maxy / self._cell) + 1):
                yield (i, j)

    def claim(self, geometry: BaseGeometry, margin: float = 0.0) -> None:
        minx, miny, maxx, maxy = geometry.bounds
        grown = (minx - margin, miny - margin, maxx + margin, maxy + margin)
        for cell in self._cells(grown):
            self._grid.setdefault(cell, []).append(grown)

    def fits(self, geometry: BaseGeometry, gap: float = 0.0) -> bool:
        minx, miny, maxx, maxy = geometry.bounds
        wanted = (minx - gap, miny - gap, maxx + gap, maxy + gap)
        for cell in self._cells(wanted):
            for other in self._grid.get(cell, ()):
                if (wanted[0] < other[2] and wanted[2] > other[0]
                        and wanted[1] < other[3] and wanted[3] > other[1]):
                    return False
        if self._blocked is not None and self._blocked.intersects(geometry):
            return False
        return True

    def take(self, geometry: BaseGeometry, gap: float = 0.0) -> bool:
        """Claim the spot if it is free; say whether it was."""
        if not self.fits(geometry, gap):
            return False
        self.claim(geometry)
        return True


def tree(x: float, y: float, height: float, spread: float, base: float, name: str) -> Building:
    """A tree as the simulator sees it: a box the size of its canopy."""
    return Building(
        footprint=[GeoPoint(x=round(x - spread, 2), y=round(y - spread, 2)),
                   GeoPoint(x=round(x + spread, 2), y=round(y - spread, 2)),
                   GeoPoint(x=round(x + spread, 2), y=round(y + spread, 2)),
                   GeoPoint(x=round(x - spread, 2), y=round(y + spread, 2))],
        height=round(height, 1),
        base=round(base, 2),
        name=name,
        kind="tree",
    )


def plant(
    rng: random.Random,
    lot: Lot,
    ground: Ground,
    region: BaseGeometry,
    count: int,
    species: tuple[str, ...],
    heights: tuple[float, float],
    spreads: tuple[float, float],
    gap: float = 1.0,
    tries: int = 12,
) -> list[Building]:
    """Scatter up to `count` trees over a region, wherever there is room."""
    minx, miny, maxx, maxy = region.bounds
    inside = prepared.prep(region)
    trees: list[Building] = []
    for _ in range(count * tries):
        if len(trees) >= count:
            break
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not inside.contains(Point(x, y)):
            continue
        spread = rng.uniform(*spreads)
        crown = box(x - spread, y - spread, x + spread, y + spread)
        if not lot.take(crown, gap):
            continue
        trees.append(tree(x, y, rng.uniform(*heights), spread, ground(x, y),
                          f"{species[len(trees) % len(species)]} {len(trees) + 1}"))
    return trees


def line_trees(
    rng: random.Random,
    lot: Lot,
    ground: Ground,
    path: list[tuple[float, float]],
    offset: float,
    step: float,
    species: str,
    heights: tuple[float, float],
    spreads: tuple[float, float],
    sides: tuple[int, ...] = (-1, 1),
) -> list[Building]:
    """Trees at a steady spacing either side of a road: an avenue."""
    line = LineString(path)
    trees: list[Building] = []
    distance = step / 2.0
    while distance < line.length:
        here = line.interpolate(distance)
        ahead = line.interpolate(min(distance + 0.5, line.length))
        behind = line.interpolate(max(distance - 0.5, 0.0))
        dx, dy = ahead.x - behind.x, ahead.y - behind.y
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        for side in sides:
            x = here.x + side * offset * nx + rng.uniform(-0.8, 0.8)
            y = here.y + side * offset * ny + rng.uniform(-0.8, 0.8)
            spread = rng.uniform(*spreads)
            crown = box(x - spread, y - spread, x + spread, y + spread)
            if lot.take(crown, 0.5):
                trees.append(tree(x, y, rng.uniform(*heights), spread, ground(x, y),
                                  f"{species} {len(trees) + 1}"))
        distance += step
    return trees


def build_on(
    lot: Lot,
    ground: Ground,
    footprint: Polygon,
    height: float,
    name: str,
    kind: str = "building",
    gap: float = 0.0,
) -> Building | None:
    """A building on the given plot, if the plot is free."""
    if not lot.take(footprint, gap):
        return None
    # On a slope, stand it on the lowest corner: a house dug into a hillside
    # looks right and over-states its volume, which is the safe way to be wrong.
    corners = [ground(x, y) for x, y in list(footprint.exterior.coords)[:-1]]
    return Building(footprint=points(footprint), height=round(height, 1),
                    base=round(min(corners), 2), name=name, kind=kind)


def nearest_name(x: float, y: float, named: list[tuple[str, BaseGeometry]], fallback: str) -> str:
    """The name of whichever named line or area is closest: 'a bungalow on Ashoka Road'."""
    here = Point(x, y)
    best = min(named, key=lambda item: item[1].distance(here), default=None)
    return best[0] if best else fallback


def bearing_of(x: float, y: float, lines: list[BaseGeometry]) -> float:
    """Which way the nearest road runs, in degrees, so a house can face it."""
    here = Point(x, y)
    line = min(lines, key=lambda g: g.distance(here))
    at = line.project(here)
    a = line.interpolate(max(at - 1.0, 0.0))
    b = line.interpolate(min(at + 1.0, line.length))
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def flatten(heights: list[list[float]], origin: tuple[float, float], spacing: float,
            cx: float, cy: float, radius: float, blend: float, level: float) -> list[list[float]]:
    """Press a basin into a height grid: `level` inside `radius`, easing out over `blend`.

    Only ever lowers the ground, so a lake never ends up perched on a slope and
    nothing that was clear of the ground before is buried by it.
    """
    ox, oy = origin
    out = []
    for row, line in enumerate(heights):
        new = []
        for col, h in enumerate(line):
            d = math.hypot(ox + col * spacing - cx, oy + row * spacing - cy)
            if d <= radius:
                t = 1.0
            elif d >= radius + blend:
                t = 0.0
            else:
                u = (d - radius) / blend
                t = 1.0 - u * u * (3.0 - 2.0 * u)
            new.append(round(min(h, h * (1 - t) + level * t), 2))
        out.append(new)
    return out
