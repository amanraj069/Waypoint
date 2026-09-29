"""Is this a place you could actually put an aircraft down?

A launch pad or a landing site is not just a coordinate: it is a patch of
ground with a size. A point can be perfectly clear and still be useless if the
ground under the skids falls away across the width of the aircraft -- that is
how a multirotor tips over on touchdown.

So a site is *surveyed* rather than assumed. The ground is sampled across the
footprint the aircraft actually occupies, and the answer carries the numbers
that decided it, so a refusal can say why.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from waypoint.world.models import World

#: The aircraft's footprint: how far from the centre the skids reach.
PAD_RADIUS = 3.0
#: Steepest ground a multirotor should be asked to touch down on.
MAX_SLOPE_DEG = 8.0
#: And the most the ground may rise or fall across that footprint.
MAX_DROP = 0.9


class Site(BaseModel):
    """A surveyed patch of ground, and whether it will do."""

    x: float
    y: float
    elevation: float = Field(description="Ground height at the centre, in metres.")
    slope_deg: float = Field(description="Steepest gradient across the footprint.")
    drop: float = Field(description="Highest minus lowest point across the footprint.")
    obstructed: str | None = Field(
        default=None, description="Name of whatever is standing here, if anything."
    )
    moved: float = Field(default=0.0, description="Metres from the point that was asked for.")

    @property
    def usable(self) -> bool:
        return self.obstructed is None and self.slope_deg <= MAX_SLOPE_DEG and self.drop <= MAX_DROP

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.elevation)

    def why_not(self) -> str:
        """Why this patch will not do, in words worth showing somebody."""
        if self.obstructed:
            return f"'{self.obstructed}' is standing there"
        if self.drop > MAX_DROP:
            return (
                f"the ground falls {self.drop:.1f} m across the {PAD_RADIUS * 2:.0f} m pad "
                f"({self.slope_deg:.0f}°), too uneven to set down on"
            )
        if self.slope_deg > MAX_SLOPE_DEG:
            return f"the ground slopes {self.slope_deg:.0f}°, steeper than the {MAX_SLOPE_DEG:.0f}° limit"
        return "it is fine"

    def describe(self) -> str:
        where = f"({self.x:.0f}, {self.y:.0f})"
        if not self.usable:
            return f"{where}: {self.why_not()}"
        ground = f"ground at {self.elevation:.1f} m, {self.slope_deg:.0f}° slope"
        if self.moved:
            return f"{where}: {ground}, moved {self.moved:.0f} m to find it"
        return f"{where}: {ground}"


def survey(world: World | None, x: float, y: float, radius: float = PAD_RADIUS) -> Site:
    """Measure the ground across one pad-sized patch.

    Samples the centre and a ring around it, which is enough to catch a slope
    or a step without pretending to a precision the terrain grid does not have.
    """
    if world is None:
        return Site(x=x, y=y, elevation=0.0, slope_deg=0.0, drop=0.0)

    centre = world.elevation_at(x, y)
    heights = [centre]
    slopes = [0.0]
    for step in range(8):
        angle = step * math.tau / 8
        px, py = x + radius * math.cos(angle), y + radius * math.sin(angle)
        height = world.elevation_at(px, py)
        heights.append(height)
        slopes.append(math.degrees(math.atan2(abs(height - centre), radius)))

    building = world.collides(x, y, centre + 0.5)
    return Site(
        x=round(x, 2),
        y=round(y, 2),
        elevation=round(centre, 2),
        slope_deg=round(max(slopes), 1),
        drop=round(max(heights) - min(heights), 2),
        obstructed=(building.name or "something") if building else None,
    )


def find(
    world: World | None,
    x: float,
    y: float,
    radius: float = PAD_RADIUS,
    search: float = 120.0,
    step: float = 8.0,
) -> Site:
    """The nearest patch of ground you could land on, starting where asked.

    Returns the requested point when it will do. Otherwise it spirals outwards
    and returns the first spot that will, with `moved` set -- and if nothing
    within `search` works, it returns the original survey, still unusable, so
    the caller can say what was wrong rather than silently landing somewhere
    surprising.
    """
    here = survey(world, x, y, radius)
    if here.usable or world is None:
        return here

    rings = int(search / step)
    for ring in range(1, rings + 1):
        distance = ring * step
        points = max(8, ring * 8)
        for index in range(points):
            angle = index * math.tau / points
            site = survey(world, x + distance * math.cos(angle), y + distance * math.sin(angle), radius)
            if site.usable:
                return site.model_copy(update={"moved": round(distance, 1)})
    return here
