"""The environment the aircraft flies through: terrain relief and buildings.

Until now the world was empty space above a flat plane, so a plan only had to
be right in x, y and z. A `World` gives the mission somewhere to be: ground that
rises and falls, and buildings that occupy volume and must be flown around or
over.

Altitude convention: a waypoint's `z` is height above the local datum (z = 0),
*not* above whatever is underneath it. Height above ground is derived --
`world.agl(x, y, z)` -- so a plan that holds 30 m over a 40 m hill is correctly
seen as underground. With no world, or a flat one, the two are identical, which
is why everything written before this still means what it meant.
"""

from __future__ import annotations

import math
from functools import cached_property

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from shapely import STRtree
from shapely.geometry import Point, Polygon

from planner.models import GeoPoint


class Terrain(BaseModel):
    """Ground elevation as a regular grid, sampled bilinearly between posts.

    `heights[row][col]` is the elevation at
    (origin_x + col * spacing, origin_y + row * spacing).
    """

    origin_x: float = 0.0
    origin_y: float = 0.0
    spacing: float = Field(default=25.0, gt=0, description="Metres between grid posts.")
    heights: list[list[float]] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_grid(self) -> "Terrain":
        widths = {len(row) for row in self.heights}
        if len(widths) != 1:
            raise ValueError("every terrain row must have the same number of columns")
        if widths.pop() < 2:
            raise ValueError("terrain needs at least 2 columns")
        return self

    @property
    def rows(self) -> int:
        return len(self.heights)

    @property
    def cols(self) -> int:
        return len(self.heights[0])

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + (self.cols - 1) * self.spacing,
            self.origin_y + (self.rows - 1) * self.spacing,
        )

    @property
    def relief(self) -> tuple[float, float]:
        """Lowest and highest points, for framing and for prompts."""
        flat = [h for row in self.heights for h in row]
        return (min(flat), max(flat))

    def levelled_at(self, x: float = 0.0, y: float = 0.0) -> "Terrain":
        """The same relief, shifted so (x, y) sits at elevation 0.

        The launch pad is the datum everything else is measured from, so a
        waypoint at z = 30 is 30 m above the pad however the ground rolls.
        """
        offset = self.elevation_at(x, y)
        return self.model_copy(
            update={"heights": [[round(h - offset, 2) for h in row] for row in self.heights]}
        )

    def elevation_at(self, x: float, y: float) -> float:
        """Ground elevation under (x, y), clamped to the edge outside the grid."""
        col = (x - self.origin_x) / self.spacing
        row = (y - self.origin_y) / self.spacing
        col = min(max(col, 0.0), self.cols - 1.0)
        row = min(max(row, 0.0), self.rows - 1.0)

        c0, r0 = int(math.floor(col)), int(math.floor(row))
        c1, r1 = min(c0 + 1, self.cols - 1), min(r0 + 1, self.rows - 1)
        fc, fr = col - c0, row - r0

        top = self.heights[r0][c0] * (1 - fc) + self.heights[r0][c1] * fc
        bottom = self.heights[r1][c0] * (1 - fc) + self.heights[r1][c1] * fc
        return top * (1 - fr) + bottom * fr


class Building(BaseModel):
    """Anything solid the aircraft must not fly into.

    A footprint extruded upward from `base`. `kind` is only about how it looks
    and reads -- a tree stops a drone exactly as well as a wall does.
    """

    footprint: list[GeoPoint] = Field(min_length=3)
    height: float = Field(gt=0, description="Metres from its base to its roof.")
    base: float = Field(default=0.0, description="Ground elevation at the footprint.")
    name: str = ""
    kind: str = Field(default="building", description="building, tree, mast, pavilion...")

    @property
    def roof(self) -> float:
        """Absolute altitude of the roof, in the same frame as a waypoint's z."""
        return self.base + self.height

    @cached_property
    def polygon(self) -> Polygon:
        return Polygon([(p.x, p.y) for p in self.footprint])

    @property
    def centre(self) -> tuple[float, float]:
        point = self.polygon.centroid
        return (point.x, point.y)

    def covers(self, x: float, y: float) -> bool:
        return bool(self.polygon.covers(Point(x, y)))

    def contains(self, x: float, y: float, z: float) -> bool:
        """True if the point is inside the building's volume."""
        return z <= self.roof and self.covers(x, y)

    def label(self) -> str:
        cx, cy = self.centre
        return f"'{self.name or 'building'}' at ({cx:.0f}, {cy:.0f}), {self.height:g} m tall"


class Landmark(BaseModel):
    """Something in the world worth naming.

    Landmarks are what make a mission sayable: the planner is told where Bara
    Gumbad is, so "photograph Bara Gumbad" becomes waypoints. They are
    descriptive, not solid -- anything the aircraft can hit is a `Building`.
    """

    name: str
    x: float
    y: float
    radius: float = Field(default=5.0, gt=0, description="Rough extent in metres.")
    height: float = Field(default=0.0, ge=0, description="How far it stands up, if at all.")
    kind: str = Field(default="feature", description="water, pavilion, monument, grove, path...")
    note: str = Field(default="", description="Anything else the planner should know.")

    def describe(self) -> str:
        parts = [f"'{self.name}' ({self.kind}) at ({self.x:.0f}, {self.y:.0f})",
                 f"about {self.radius * 2:.0f} m across"]
        if self.height:
            parts.append(f"{self.height:.0f} m tall")
        if self.note:
            parts.append(self.note)
        return ", ".join(parts)


class Area(BaseModel):
    """What the ground *is* somewhere: a lawn, a road, a lake, a tea terrace.

    Ground cover is how a place reads from the air -- the thing that makes a
    park look like a park and a bazaar look like a bazaar -- but it is flat and
    it is not in the way. Nothing here is solid: the verifier and the simulator
    never look at it, and anything the aircraft can hit is a `Building`.
    """

    footprint: list[GeoPoint] = Field(min_length=3)
    kind: str = Field(
        default="lawn",
        description="lawn, field, tea, forest, road, path, plaza, water, rail, sand, built...",
    )
    name: str = ""
    level: float | None = Field(
        default=None, description="For water: the surface's elevation, when it is known."
    )


class World(BaseModel):
    """Somewhere to fly: named terrain plus the buildings standing on it."""

    name: str = "flatland"
    source: str = Field(default="procedural", description="procedural, osm, or hand-made.")
    origin_lat: float | None = None
    origin_lon: float | None = None
    terrain: Terrain | None = None
    buildings: list[Building] = Field(default_factory=list)
    landmarks: list[Landmark] = Field(default_factory=list)
    ground: list[Area] = Field(
        default_factory=list,
        description="Ground cover, drawn but never collided with. Later entries draw on top.",
    )

    _tree: STRtree | None = PrivateAttr(default=None)

    model_config = {"ignore_types": (cached_property,)}

    # -- queries ------------------------------------------------------------

    def elevation_at(self, x: float, y: float) -> float:
        """Ground elevation under a point; 0 when the world has no terrain."""
        return self.terrain.elevation_at(x, y) if self.terrain else 0.0

    def agl(self, x: float, y: float, z: float) -> float:
        """Height above whatever is directly underneath: ground or roof."""
        return z - self.surface_at(x, y)

    def surface_at(self, x: float, y: float) -> float:
        """The altitude of the highest thing at (x, y): a roof, or the ground."""
        ground = self.elevation_at(x, y)
        roofs = [b.roof for b in self.buildings_at(x, y)]
        return max([ground, *roofs])

    def buildings_at(self, x: float, y: float) -> list[Building]:
        """Every building whose footprint covers the point."""
        if not self.buildings:
            return []
        hits = self._index().query(Point(x, y))
        return [self.buildings[i] for i in hits if self.buildings[i].covers(x, y)]

    def obstructions(self, geometry) -> list[Building]:
        """Buildings whose footprint the given shapely geometry touches."""
        if not self.buildings:
            return []
        hits = self._index().query(geometry)
        return [self.buildings[i] for i in hits if self.buildings[i].polygon.intersects(geometry)]

    def collides(self, x: float, y: float, z: float) -> Building | None:
        """The building the point is inside, if any."""
        for building in self.buildings_at(x, y):
            if z <= building.roof:
                return building
        return None

    def underground(self, x: float, y: float, z: float) -> bool:
        return z < self.elevation_at(x, y)

    # -- summary ------------------------------------------------------------

    @property
    def tallest(self) -> Building | None:
        return max(self.buildings, key=lambda b: b.roof, default=None)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) over terrain and buildings."""
        boxes = [b.polygon.bounds for b in self.buildings]
        if self.terrain:
            boxes.append(self.terrain.extent)
        if not boxes:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def find(self, name: str) -> Landmark | None:
        """Look a landmark up by name, case and punctuation insensitively."""
        wanted = "".join(c for c in name.lower() if c.isalnum())
        for landmark in self.landmarks:
            if "".join(c for c in landmark.name.lower() if c.isalnum()) == wanted:
                return landmark
        return None

    def describe(self) -> str:
        """Prompt-ready summary: enough for a planner to fly it safely."""
        if not self.buildings and not self.terrain and not self.landmarks:
            return f"- Environment '{self.name}': open, flat ground at elevation 0."

        lines = [f"- Environment '{self.name}':"]
        if self.terrain:
            low, high = self.terrain.relief
            lines.append(
                f"  - Ground is not flat: elevation runs from {low:.0f} m to {high:.0f} m. "
                "A waypoint's z is height above the zero datum, NOT above the ground "
                "below it, so add the ground elevation to the clearance you want."
            )
        if self.buildings:
            tallest = self.tallest
            lines.append(
                f"  - {len(self.buildings)} buildings stand in the area; the tallest roof is at "
                f"{tallest.roof:.0f} m ({tallest.label()}). Do not fly through one: either "
                f"cross above every roof, or route around them at street level."
            )
            listed = sorted(self.buildings, key=lambda b: b.roof, reverse=True)[:8]
            for building in listed:
                cx, cy = building.centre
                lines.append(
                    f"    - {building.name or 'building'}: centre ({cx:.0f}, {cy:.0f}), "
                    f"roof at {building.roof:.0f} m"
                )
            if len(self.buildings) > len(listed):
                lines.append(f"    - ...and {len(self.buildings) - len(listed)} lower ones")
        if self.landmarks:
            lines.append(
                "  - Named places here. The mission may refer to any of them by name; "
                "fly to the coordinates given, and stand off far enough to see the whole thing:"
            )
            for landmark in self.landmarks:
                lines.append(f"    - {landmark.describe()}")
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def summary(self) -> dict:
        """Compact header for a run file or a listing."""
        low, high = self.terrain.relief if self.terrain else (0.0, 0.0)
        return {
            "name": self.name,
            "source": self.source,
            "buildings": len(self.buildings),
            "trees": sum(1 for b in self.buildings if b.kind == "tree"),
            "landmarks": [lm.name for lm in self.landmarks],
            "elevation_min": round(low, 1),
            "elevation_max": round(high, 1),
            "tallest_roof": round(self.tallest.roof, 1) if self.tallest else 0.0,
        }

    def _index(self) -> STRtree:
        if self._tree is None:
            self._tree = STRtree([b.polygon for b in self.buildings])
        return self._tree


EMPTY_WORLD = World()
