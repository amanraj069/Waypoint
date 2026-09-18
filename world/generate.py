"""Build a world to fly in, without needing the internet.

Four real places in India, hand-authored from their published dimensions rather
than downloaded, so they are identical on every machine and in every run --
a scenario you can keep re-flying while you change the planner or the
controller.

| Builder                       | Place                                |
| ----------------------------- | ------------------------------------ |
| `lutyens_delhi()` / `city()`  | The Central Vista, New Delhi         |
| `lodhi_garden()` / `park()`   | Lodhi Garden, New Delhi              |
| `chandni_chowk()` / `canyon()`| The Chandni Chowk bazaar, Old Delhi  |
| `munnar()` / `hills()`        | The Kannan Devan hills, Kerala       |

Every building carries the name of the thing it is, because a name is what
makes a mission sayable: "photograph the Jama Masjid minarets from the east"
has to land on something. Heights are the real ones. Distances are *not* always
the real ones -- the Central Vista is 2.6 km end to end and the aircraft has
about 3.2 km of battery, so the vista is laid out at roughly a third scale and
Munnar's 1100 m of relief is compressed to 235 m. Each builder says what it did.
"""

from __future__ import annotations

import math
import random

from planner.models import GeoPoint
from world import dressing
from world.models import Building, Landmark, Terrain, World

# The launch pad sits at the origin, so keep a plaza clear around it rather
# than starting the aircraft inside somebody's lobby.
PAD_CLEARANCE = 45.0


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[GeoPoint]:
    return [
        GeoPoint(x=x0, y=y0),
        GeoPoint(x=x1, y=y0),
        GeoPoint(x=x1, y=y1),
        GeoPoint(x=x0, y=y1),
    ]


def _box(cx: float, cy: float, width: float, depth: float, rot: float = 0.0) -> list[GeoPoint]:
    """A rectangle `width` (east-west) by `depth` (north-south) about its centre.

    `rot` turns it anticlockwise in degrees, which is what a building on a
    radial street needs.
    """
    hw, hd = width / 2.0, depth / 2.0
    angle = math.radians(rot)
    cos, sin = math.cos(angle), math.sin(angle)
    corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
    return [
        GeoPoint(x=round(cx + dx * cos - dy * sin, 2), y=round(cy + dx * sin + dy * cos, 2))
        for dx, dy in corners
    ]


def _polygon(cx: float, cy: float, radius: float, sides: int = 24, rot: float = 0.0) -> list[GeoPoint]:
    """A regular polygon: 24 sides for a dome, 8 for a Lodhi-era octagonal tomb."""
    start = math.radians(rot)
    return [
        GeoPoint(
            x=round(cx + radius * math.cos(start + math.tau * i / sides), 2),
            y=round(cy + radius * math.sin(start + math.tau * i / sides), 2),
        )
        for i in range(sides)
    ]


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def rolling_terrain(
    half_extent: float = 400.0,
    spacing: float = 25.0,
    amplitude: float = 18.0,
    seed: int = 7,
) -> Terrain:
    """Gentle relief from a couple of sine waves -- deterministic, no noise library."""
    rng = random.Random(seed)
    phase_x, phase_y = rng.uniform(0, math.tau), rng.uniform(0, math.tau)
    lambda_x = half_extent / rng.uniform(1.1, 1.8)
    lambda_y = half_extent / rng.uniform(1.1, 1.8)

    steps = int((half_extent * 2) / spacing) + 1
    heights = []
    for row in range(steps):
        y = -half_extent + row * spacing
        line = []
        for col in range(steps):
            x = -half_extent + col * spacing
            h = (
                math.sin(x / lambda_x + phase_x) * math.cos(y / lambda_y + phase_y)
                + 0.45 * math.sin((x + y) / (lambda_x * 1.7) + phase_y)
            )
            line.append(round(amplitude * (h + 1.0) / 2.0, 2))
        heights.append(line)
    terrain = Terrain(
        origin_x=-half_extent, origin_y=-half_extent, spacing=spacing, heights=heights
    )
    # The pad is the datum: the ground at (0, 0) is elevation 0, and everything
    # else rises or falls relative to where the aircraft took off.
    return terrain.levelled_at(0.0, 0.0)


def _sampled_terrain(elevation, half_extent: float, spacing: float) -> Terrain:
    """Turn an `elevation(x, y) -> metres` function into a height grid."""
    steps = int((half_extent * 2) / spacing) + 1
    heights = [
        [
            round(elevation(-half_extent + col * spacing, -half_extent + row * spacing), 2)
            for col in range(steps)
        ]
        for row in range(steps)
    ]
    return Terrain(
        origin_x=-half_extent, origin_y=-half_extent, spacing=spacing, heights=heights
    ).levelled_at(0.0, 0.0)


def _smoothstep(value: float, low: float, high: float) -> float:
    """0 below `low`, 1 above `high`, with no corner at either end."""
    if high == low:
        return 0.0 if value < low else 1.0
    t = min(max((value - low) / (high - low), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bump(x: float, y: float, cx: float, cy: float, amplitude: float, sigma: float) -> float:
    """One hill (or, with a negative amplitude, one basin) as a Gaussian."""
    return amplitude * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma))


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


def _tree(x: float, y: float, height: float, spread: float, base: float, name: str) -> Building:
    """A tree as the simulator sees it: a box the size of its canopy.

    Square rather than round on purpose -- it over-estimates the canopy a
    little, and an obstacle you clip the corner of should count as a hit.
    """
    return Building(
        footprint=_rect(x - spread, y - spread, x + spread, y + spread),
        height=round(height, 1),
        base=round(base, 2),
        name=name,
        kind="tree",
    )


def _grove(
    rng: random.Random,
    ground,
    centre: tuple[float, float],
    radius: float,
    count: int,
    species: tuple[str, ...],
    heights: tuple[float, float],
    spreads: tuple[float, float],
) -> list[Building]:
    """A stand of trees scattered in a disc, each named for what it is."""
    cx, cy = centre
    trees = []
    for i in range(count):
        angle = rng.uniform(0, math.tau)
        distance = radius * math.sqrt(rng.random())
        x, y = cx + distance * math.cos(angle), cy + distance * math.sin(angle)
        trees.append(
            _tree(
                x, y,
                rng.uniform(*heights),
                rng.uniform(*spreads),
                ground(x, y),
                f"{species[i % len(species)]} {i + 1}",
            )
        )
    return trees


def _avenue(
    rng: random.Random,
    ground,
    species: str,
    offsets: tuple[float, float],
    along: tuple[float, float, float],
    heights: tuple[float, float],
    spreads: tuple[float, float],
    axis: str = "y",
    sides: tuple[str, str] = ("west", "east"),
) -> list[Building]:
    """Two rows of trees flanking a straight walk, `offsets` metres either side."""
    start, stop, step = along
    trees = []
    for index, distance in enumerate(_steps(start, stop, step), 1):
        for side, offset in zip(sides, offsets):
            if axis == "y":
                x, y = offset + rng.uniform(-1.5, 1.5), distance + rng.uniform(-2.0, 2.0)
            else:
                x, y = distance + rng.uniform(-2.0, 2.0), offset + rng.uniform(-1.5, 1.5)
            trees.append(
                _tree(
                    x, y,
                    rng.uniform(*heights),
                    rng.uniform(*spreads),
                    ground(x, y),
                    f"{species} {side} {index}",
                )
            )
    return trees


def _steps(start: float, stop: float, step: float) -> list[float]:
    values, value = [], start
    while value <= stop + 1e-9:
        values.append(value)
        value += step
    return values


# ---------------------------------------------------------------------------
# The Central Vista, New Delhi
# ---------------------------------------------------------------------------

# (name, x, y, width, depth, height, rotation)
_VISTA_BLOCKS: tuple[tuple[str, float, float, float, float, float, float], ...] = (
    ("India Gate",                      120.0,    0.0,  26.0,  26.0, 42.0, 0.0),
    ("the Canopy",                      190.0,    0.0,  14.0,  14.0, 22.0, 0.0),
    ("Jaipur House",                    148.0,  -96.0,  56.0,  56.0, 20.0, 0.0),
    ("Patiala House",                   148.0,   96.0,  56.0,  56.0, 16.0, 0.0),
    ("Hyderabad House",                  55.0,  102.0,  74.0,  46.0, 17.0, 0.0),
    ("Baroda House",                     55.0, -102.0,  74.0,  46.0, 17.0, 0.0),
    ("the National Stadium",            238.0, -178.0, 165.0, 105.0, 24.0, 0.0),
    ("the National Archives",           -78.0,   62.0,  72.0,  50.0, 20.0, 0.0),
    ("the National Museum",            -102.0, -152.0, 106.0,  80.0, 24.0, 0.0),
    ("Vigyan Bhavan",                   -96.0,  156.0,  80.0,  55.0, 18.0, 0.0),
    ("Rail Bhavan",                    -196.0,  102.0,  96.0,  58.0, 22.0, 0.0),
    ("Udyog Bhavan",                   -196.0,  192.0, 126.0,  58.0, 26.0, 0.0),
    ("Krishi Bhavan",                  -196.0, -102.0,  96.0,  58.0, 22.0, 0.0),
    ("Shastri Bhavan",                 -196.0, -192.0, 126.0,  58.0, 26.0, 0.0),
    ("Nirman Bhavan",                   -96.0, -244.0, 110.0,  55.0, 24.0, 0.0),
    ("the North Block",                -396.0,  110.0, 196.0,  70.0, 42.0, 0.0),
    ("the South Block",                -396.0, -110.0, 196.0,  70.0, 42.0, 0.0),
    ("Rashtrapati Bhavan",             -548.0,    0.0, 168.0, 126.0, 55.0, 0.0),
    ("the Jaipur Column",              -468.0,    0.0,   7.0,   7.0, 44.0, 0.0),
    ("the Vice-President's House",     -552.0,  208.0,  78.0,  58.0, 18.0, 0.0),
)


def lutyens_delhi(name: str = "central vista", seed: int = 7) -> World:
    """The Central Vista, New Delhi: India Gate to Rashtrapati Bhavan.

    Laid out at roughly a third of true scale -- the real avenue is 2.6 km end
    to end and the aircraft has about 3.2 km of battery -- but the buildings
    keep their real heights, so a 42 m India Gate and a 55 m Rashtrapati Bhavan
    on a 26 m hill are the numbers the planner has to fly around.

    The pad is on the lawns at the east end, looking west up Kartavya Path. The
    ground is flat there and climbs Raisina Hill towards the Secretariat, which
    is the point of flying here: a constant altitude is not a constant
    clearance, and the two Secretariat roofs are 68 m above the pad even though
    the buildings themselves are only 42 m tall.
    """
    rng = random.Random(seed)

    def elevation(x: float, y: float) -> float:
        # Raisina Hill: flat lawns east of Janpath, climbing to the Secretariat,
        # and falling away to north and south rather than running on as a wall.
        ramp = 26.0 * _smoothstep(-x, 120.0, 500.0)
        shoulder = math.exp(-((y / 420.0) ** 2))
        return ramp * shoulder + 0.7 * math.sin(x / 260.0) * math.cos(y / 300.0)

    terrain = _sampled_terrain(elevation, half_extent=660.0, spacing=20.0)
    ground = terrain.elevation_at

    buildings = [
        Building(
            footprint=_box(x, y, width, depth, rot),
            height=height,
            base=round(ground(x, y), 2),
            name=block_name,
            kind="monument" if height > 40 and width < 30 else "building",
        )
        for block_name, x, y, width, depth, height, rot in _VISTA_BLOCKS
    ]

    # Sansad Bhavan is a circle 173 m across; drawing it as a box would lose the
    # one thing everybody recognises it by.
    buildings.append(
        Building(
            footprint=_polygon(-462.0, 228.0, 56.0, sides=28),
            height=27.0,
            base=round(ground(-462.0, 228.0), 2),
            name="Sansad Bhavan",
        )
    )
    buildings.append(
        Building(
            footprint=_polygon(-352.0, 248.0, 66.0, sides=3, rot=90.0),
            height=39.0,
            base=round(ground(-352.0, 248.0), 2),
            name="the new Parliament building",
        )
    )

    # The jamun trees along Kartavya Path, and the ones ringing the India Gate
    # hexagon. They are 12-15 m and the canals are 60 m apart, so a low run up
    # the avenue has to mean it.
    buildings += _avenue(
        rng, ground, "Kartavya Path jamun", (-44.0, 44.0), (-290.0, 96.0, 26.0),
        heights=(11.0, 15.0), spreads=(3.5, 5.5), axis="x", sides=("south", "north"),
    )
    buildings += _grove(
        rng, ground, (120.0, 0.0), 86.0, 14,
        ("India Gate hexagon arjun", "India Gate hexagon neem"),
        heights=(9.0, 14.0), spreads=(3.5, 6.0),
    )

    landmarks = [
        Landmark(name="the Central Vista lawns", x=0, y=0, radius=90, kind="open",
                 note="where the aircraft takes off, halfway along the avenue"),
        Landmark(name="Kartavya Path", x=-100, y=0, radius=34, height=15, kind="path",
                 note="the ceremonial avenue running east-west, 60 m clear between "
                      "the two canals, with rows of jamun trees outside them"),
        Landmark(name="India Gate", x=120, y=0, radius=32, height=42, kind="monument",
                 note="a 42 m memorial arch standing alone on the lawns at the east end"),
        Landmark(name="the north canal", x=-60, y=32, radius=16, kind="water",
                 note="a long strip of water beside the avenue"),
        Landmark(name="the south canal", x=-60, y=-32, radius=16, kind="water",
                 note="a long strip of water beside the avenue"),
        Landmark(name="the National Museum", x=-102, y=-152, radius=56, height=24, kind="building"),
        Landmark(name="Vijay Chowk", x=-292, y=0, radius=58, kind="open",
                 note="the open square at the foot of Raisina Hill, between the two "
                      "Secretariat blocks -- clear of buildings, but the ground is "
                      "already 14 m above the pad here"),
        Landmark(name="Raisina Hill", x=-460, y=0, radius=150, height=26, kind="hill",
                 note="the ground rises 26 m from Vijay Chowk to the forecourt, so "
                      "altitudes that clear the lawns do not clear this"),
        Landmark(name="the North Block", x=-396, y=110, radius=64, height=42, kind="building",
                 note="42 m tall on ground already 24 m up: its roof is 66 m above the pad"),
        Landmark(name="the South Block", x=-396, y=-110, radius=64, height=42, kind="building",
                 note="the mirror of the North Block, across Vijay Chowk"),
        Landmark(name="Rashtrapati Bhavan", x=-548, y=0, radius=92, height=55, kind="monument",
                 note="the dome at the head of the vista, on the highest ground here"),
        Landmark(name="Sansad Bhavan", x=-462, y=228, radius=60, height=27, kind="monument",
                 note="circular, 112 m across, north of the Secretariat"),
        Landmark(name="the National Stadium", x=238, y=-178, radius=92, height=24, kind="building"),
    ]

    # Lutyens' Delhi around it: the avenues, their trees, the bungalows.
    fill, ground_cover = dressing.dress_vista(random.Random(seed + 101), terrain, buildings)
    return World(
        name=name, source="procedural", terrain=terrain,
        buildings=[*buildings, *fill], landmarks=landmarks, ground=ground_cover,
    )


# ---------------------------------------------------------------------------
# Lodhi Garden, New Delhi
# ---------------------------------------------------------------------------


def lodhi_garden(name: str = "lodhi garden", half_extent: float = 470.0, seed: int = 5) -> World:
    """Lodhi Garden, New Delhi: six centuries of tombs standing in a park.

    Built for *saying* things to the aircraft. Bara Gumbad, Sheesh Gumbad,
    Mohammed Shah's Tomb, Sikander Lodi's Tomb and the Athpula bridge are all
    named, positioned and the height they really are, and Safdarjung's Tomb
    stands 47 m tall just beyond the west wall. The trees are solid and the
    species are the ones actually planted there, so a route down the ashoka
    avenue at 8 m has to mean it.
    """
    rng = random.Random(seed)
    terrain = rolling_terrain(half_extent=half_extent, spacing=20.0, amplitude=8.0, seed=seed)
    terrain, lake_level = dressing.lodhi_lake(terrain)
    ground = terrain.elevation_at

    def at(footprint: list[GeoPoint], x: float, y: float, height: float,
           label: str, kind: str = "building") -> Building:
        return Building(footprint=footprint, height=height, base=round(ground(x, y), 2),
                        name=label, kind=kind)

    monuments = [
        # The Sayyid and Lodi tombs, in the order you meet them walking north.
        at(_polygon(-130, 95, 17, sides=8, rot=22.5), -130, 95, 20.0,
           "Mohammed Shah's Tomb", "monument"),
        at(_box(5, 215, 21, 21), 5, 215, 22.0, "Bara Gumbad", "monument"),
        at(_box(-24, 215, 26, 15), -24, 215, 16.0, "the Bara Gumbad mosque", "monument"),
        at(_box(5, 186, 27, 12), 5, 186, 9.0, "the Mehman Khana", "monument"),
        at(_box(62, 258, 22, 22), 62, 258, 20.0, "Sheesh Gumbad", "monument"),
        at(_polygon(-75, 410, 16, sides=8, rot=22.5), -75, 410, 19.0,
           "Sikander Lodi's Tomb", "monument"),
        at(_box(150, 445, 42, 9), 150, 445, 6.0, "Athpula", "monument"),
        # The working buildings.
        at(_box(185, 130, 34, 16), 185, 130, 8.0, "the Glass House", "pavilion"),
        at(_box(120, 60, 14, 10), 120, 60, 4.0, "the Bonsai Park pavilion", "pavilion"),
        at(_box(-45, -38, 11, 8), -45, -38, 5.0, "the Lodhi Road gate lodge", "pavilion"),
        at(_box(252, -42, 56, 36), 252, -42, 17.0, "the India International Centre"),
        # Safdarjung's Tomb, over the west wall: the tallest thing in sight.
        at(_polygon(-372, 216, 27, sides=8, rot=22.5), -372, 216, 47.0,
           "Safdarjung's Tomb", "monument"),
    ]
    # The four corner minarets of Safdarjung's enclosure.
    for index, (dx, dy) in enumerate(((-46, -46), (46, -46), (46, 46), (-46, 46)), 1):
        monuments.append(
            at(_box(-372 + dx, 216 + dy, 9, 9), -372 + dx, 216 + dy, 22.0,
               f"the Safdarjung corner minaret {index}", "monument")
        )
    # Sikander Lodi's walled enclosure: four low walls you can fly over but not
    # through, which is what makes the tomb inside a two-sided problem.
    for side, (dx, dy, w, d) in (
        ("north", (0, 46, 96, 5)), ("south", (0, -46, 96, 5)),
        ("east", (46, 0, 5, 96)), ("west", (-46, 0, 5, 96)),
    ):
        monuments.append(
            at(_box(-75 + dx, 410 + dy, w, d), -75 + dx, 410 + dy, 7.0,
               f"the Sikander Lodi enclosure wall, {side}")
        )

    trees: list[Building] = []
    # The ashoka avenue: the main walk north from the Lodhi Road gate.
    trees += _avenue(
        rng, ground, "avenue ashoka", (-13.0, 13.0), (-18.0, 176.0, 22.0),
        heights=(12.0, 16.0), spreads=(3.0, 4.5),
    )
    # The old grove in the north-west: neem, peepal and banyan, dense and wide.
    trees += _grove(
        rng, ground, (-182.0, 300.0), 62.0, 22,
        ("old grove neem", "old grove peepal", "old grove banyan"),
        heights=(11.0, 21.0), spreads=(5.0, 9.5),
    )
    # Specimens on the lawns: far enough apart to fly between, if you are careful.
    trees += _grove(
        rng, ground, (170.0, 230.0), 110.0, 12,
        ("lawn gulmohar", "lawn amaltas", "lawn jamun", "lawn semal"),
        heights=(8.0, 15.0), spreads=(3.0, 6.0),
    )
    trees += _grove(
        rng, ground, (-180.0, -60.0), 95.0, 9,
        ("west walk neem", "west walk jamun"),
        heights=(9.0, 14.0), spreads=(3.5, 5.5),
    )

    landmarks = [
        Landmark(name="the Lodhi Road gate", x=0, y=-30, radius=15, kind="path",
                 note="where the aircraft takes off, at the south end of the avenue"),
        Landmark(name="the ashoka avenue", x=0, y=90, radius=17, height=16, kind="path",
                 note="a straight run of ashoka trees; the two rows stand 26 m apart, leaving roughly 17 m of clear air down the middle"),
        Landmark(name="Bara Gumbad", x=5, y=215, radius=18, height=22, kind="monument",
                 note="a 15th-century domed tomb with its mosque attached to the west"),
        Landmark(name="Sheesh Gumbad", x=62, y=258, radius=16, height=20, kind="monument",
                 note="the tiled tomb north-east of Bara Gumbad"),
        Landmark(name="Mohammed Shah's Tomb", x=-130, y=95, radius=21, height=20, kind="monument",
                 note="octagonal, standing on its own in the south-west of the garden"),
        Landmark(name="Sikander Lodi's Tomb", x=-75, y=410, radius=46, height=19, kind="monument",
                 note="inside a square walled enclosure 96 m across, at the north end"),
        Landmark(name="Athpula", x=150, y=445, radius=26, height=6, kind="monument",
                 note="the low eight-pier bridge over the stream"),
        Landmark(name="the lake", x=115, y=360, radius=52, kind="water",
                 note="open water, nothing to hit, good for a low pass"),
        Landmark(name="the old grove", x=-182, y=300, radius=64, height=21, kind="grove",
                 note="dense neem, peepal and banyan -- fly over it, not through it"),
        Landmark(name="the rose garden", x=175, y=205, radius=48, kind="open",
                 note="open beds with a few specimen trees around them"),
        Landmark(name="the Glass House", x=185, y=130, radius=22, height=8, kind="pavilion"),
        Landmark(name="the Bonsai Park", x=120, y=60, radius=30, kind="open"),
        Landmark(name="Safdarjung's Tomb", x=-372, y=216, radius=50, height=47, kind="monument",
                 note="over the west wall, and the tallest thing anywhere near: "
                      "a 47 m dome inside four 22 m corner minarets"),
        Landmark(name="the east lawn", x=250, y=300, radius=72, kind="open",
                 note="open grass, clear of trees"),
    ]

    # The city the garden sits in, and the garden's own walks, lawns and water.
    standing = [*trees, *monuments]
    fill, ground_cover = dressing.dress_lodhi(random.Random(seed + 101), terrain, standing,
                                              lake_level)
    return World(
        name=name, source="procedural", terrain=terrain,
        buildings=[*standing, *fill], landmarks=landmarks, ground=ground_cover,
    )


# ---------------------------------------------------------------------------
# Chandni Chowk, Old Delhi
# ---------------------------------------------------------------------------

_CHOWK_NORTH = (
    "the Fatehpuri Bazaar shops", "Katra Neel", "Ballimaran", "the Nai Sarak corner",
    "Moti Bazaar", "Kinari Bazaar", "Dariba Kalan", "Bhagirath Palace",
)
_CHOWK_SOUTH = (
    "the Khari Baoli godowns", "Naughara galli", "Paranthe Wali Gali",
    "the Chawri Bazaar approach", "Katra Ashrafi", "Lal Kuan",
    "the Esplanade Road block", "Sunehri Katra",
)


def chandni_chowk(name: str = "chandni chowk", street: float = 30.0, seed: int = 3) -> World:
    """Chandni Chowk, Old Delhi: the bazaar street from the Red Fort westward.

    A deliberately hard place to fly. The street is 30 m wide with unbroken
    shopfronts 11-19 m tall on both sides, so at street level the only safe
    route is straight down the middle -- which is exactly where a drifting
    position estimate or a crosswind shows up. Above the roofs there are the
    Jama Masjid's 41 m minarets to the south, the 33 m Lahori Gate to the east
    and the Old Delhi station clock tower to the north.

    The ground in Old Delhi is flat, so this world has no terrain: everything
    here is a problem in x and y.
    """
    rng = random.Random(seed)
    edge = street / 2.0
    buildings: list[Building] = []

    def block(label: str, x: float, y: float, w: float, d: float, h: float,
              rot: float = 0.0, kind: str = "building") -> None:
        buildings.append(Building(footprint=_box(x, y, w, d, rot), height=h, name=label, kind=kind))

    # The shopfront terraces. They are continuous, which is the whole point:
    # there is no gap to duck into between the Red Fort and Fatehpuri Masjid --
    # except where one of the monuments below fronts the street itself.
    frontages = {
        "north": ((-176.0, -154.0), (-63.0, 13.0), (13.0, 57.0),
                  (99.0, 125.0), (217.0, 243.0), (243.0, 281.0)),
        "south": ((61.0, 89.0),),
    }
    for side, sign, names in (("north", 1, _CHOWK_NORTH), ("south", -1, _CHOWK_SOUTH)):
        x = -300.0
        index = 0
        while x < 270.0:
            width = rng.uniform(46.0, 64.0)
            depth = rng.uniform(30.0, 44.0)
            if not any(x < stop and x + width > start for start, stop in frontages[side]):
                block(
                    f"{names[index % len(names)]}, {side} side",
                    x + width / 2.0,
                    sign * (edge + depth / 2.0),
                    width - 3.0,
                    depth,
                    round(rng.uniform(11.0, 19.0), 1),
                )
                index += 1
            x += width

    # The landmarks people navigate by, at their real heights.
    block("the Lahori Gate of the Red Fort", 330, 0, 74, 42, 33.0, kind="monument")
    block("the Red Fort wall, north", 368, 132, 14, 250, 20.0, kind="monument")
    block("the Red Fort wall, south", 368, -132, 14, 250, 20.0, kind="monument")
    block("the Digambar Jain Lal Mandir", 262, 54, 34, 30, 22.0, kind="monument")
    block("the Gauri Shankar Temple", 230, 56, 22, 20, 18.0, kind="monument")
    block("the Central Baptist Church", 112, 38, 22, 18, 19.0, kind="monument")
    block("the Sunehri Masjid", 75, -34, 24, 20, 21.0, kind="monument")
    block("Gurudwara Sis Ganj Sahib", 35, 36, 40, 34, 26.0, kind="monument")
    block("the Town Hall", -25, 60, 72, 34, 22.0)
    block("Mirza Ghalib ki Haveli", -165, 50, 18, 16, 11.0)
    block("the Khari Baoli spice market", -332, 58, 92, 42, 15.0)
    block("Fatehpuri Masjid", -286, 0, 62, 46, 26.0, kind="monument")
    block("the Fatehpuri Masjid minaret, north", -316, 22, 8, 8, 32.0, kind="monument")
    block("the Fatehpuri Masjid minaret, south", -316, -22, 8, 8, 32.0, kind="monument")

    # The Jama Masjid, 330 m south: the tallest thing for a kilometre.
    block("the Jama Masjid prayer hall", 140, -330, 68, 32, 31.0, kind="monument")
    block("the Jama Masjid minaret, west", 108, -330, 13, 13, 41.0, kind="monument")
    block("the Jama Masjid minaret, east", 172, -330, 13, 13, 41.0, kind="monument")
    block("the Jama Masjid eastern gate", 140, -272, 30, 22, 18.0, kind="monument")

    # Old Delhi station, 300 m north.
    block("the Old Delhi railway station", -120, 302, 152, 40, 18.0)
    block("the Old Delhi station clock tower", -186, 302, 12, 12, 30.0)

    landmarks = [
        Landmark(name="Chandni Chowk", x=0, y=0, radius=22, height=19, kind="path",
                 note=f"the bazaar street, running east-west with only {street:g} m clear "
                      "between the shopfronts: at street level this is the only through route, "
                      "and the pad is on the median outside the Town Hall"),
        Landmark(name="the Lahori Gate", x=330, y=0, radius=46, height=33, kind="monument",
                 note="the Red Fort's main gate, closing the east end of the street"),
        Landmark(name="the Red Fort", x=400, y=0, radius=180, height=33, kind="monument",
                 note="sandstone walls 20 m tall enclosing the ground east of the gate"),
        Landmark(name="the Jama Masjid", x=140, y=-330, radius=62, height=41, kind="monument",
                 note="330 m south of the street; the two minarets reach 41 m, "
                      "well above every roof around them"),
        Landmark(name="Gurudwara Sis Ganj Sahib", x=35, y=36, radius=24, height=26, kind="monument"),
        Landmark(name="the Town Hall", x=-25, y=60, radius=42, height=22, kind="building"),
        Landmark(name="the Sunehri Masjid", x=75, y=-34, radius=15, height=21, kind="monument"),
        Landmark(name="Dariba Kalan", x=185, y=42, radius=26, height=17, kind="path",
                 note="the jewellers' lane running north off the street"),
        Landmark(name="Kinari Bazaar", x=120, y=44, radius=26, height=17, kind="path"),
        Landmark(name="Fatehpuri Masjid", x=-286, y=0, radius=38, height=32, kind="monument",
                 note="closes the west end of the street"),
        Landmark(name="Khari Baoli", x=-332, y=58, radius=52, height=15, kind="path",
                 note="the spice market, beyond Fatehpuri Masjid"),
        Landmark(name="the Old Delhi railway station", x=-120, y=302, radius=92, height=30,
                 kind="building", note="300 m north, with a 30 m clock tower at its west end"),
    ]

    # The old city behind the shopfronts, the fort, the mosque's courtyard.
    fill, ground_cover = dressing.dress_chowk(random.Random(seed + 101), buildings)
    return World(
        name=name, source="procedural", terrain=None,
        buildings=[*buildings, *fill], landmarks=landmarks, ground=ground_cover,
    )


# ---------------------------------------------------------------------------
# The Kannan Devan hills, Munnar
# ---------------------------------------------------------------------------

# (name, x, y, amplitude in metres, sigma in metres)
_MUNNAR_PEAKS = (
    ("Anamudi", 300.0, 340.0, 1.00, 190.0),
    ("Meesapulimala", 330.0, -290.0, 0.83, 170.0),
    ("Chokramudi", 70.0, 250.0, 0.55, 115.0),
    ("the Pothamedu ridge", -230.0, -60.0, 0.36, 130.0),
)


def munnar(
    name: str = "munnar",
    half_extent: float = 540.0,
    spacing: float = 20.0,
    relief: float = 235.0,
    seed: int = 12,
    masts: int = 2,
) -> World:
    """The Kannan Devan hills above Munnar, Kerala: tea estates and peaks.

    Open country, where a constant altitude is emphatically not a constant
    clearance. Anamudi really stands 1100 m above the town; here it stands
    `relief` metres above the pad, because otherwise a mission would need a
    3 km climb. The shape of the ground is the real one -- Anamudi north-east,
    Meesapulimala south-east, Chokramudi between them, the Pothamedu ridge
    behind the town -- with the Mattupetty reservoir in the basin between.

    `masts` is how many relay towers stand on the ridges; they are what an
    otherwise empty landscape gives the aircraft to hit.
    """
    rng = random.Random(seed)

    def elevation(x: float, y: float) -> float:
        height = sum(
            _bump(x, y, cx, cy, relief * weight, sigma)
            for _, cx, cy, weight, sigma in _MUNNAR_PEAKS
        )
        # The reservoir basin -- a real hollow, or the "lake" would sit on a
        # slope -- and the small folds a hillside always has.
        height += _bump(x, y, 170.0, 130.0, -0.30 * relief, 125.0)
        height += 0.035 * relief * math.sin(x / 130.0) * math.cos(y / 155.0)
        return height

    terrain = _sampled_terrain(elevation, half_extent=half_extent, spacing=spacing)
    # The dam holds the water two metres over the floor of the hollow; the
    # reservoir is dug and banked to that line before anything is built on it.
    water_line = terrain.elevation_at(170.0, 130.0) + 2.0
    terrain = dressing.munnar_reservoir(terrain, water_line)
    ground = terrain.elevation_at

    def at(label: str, x: float, y: float, w: float, d: float, h: float,
           rot: float = 0.0, kind: str = "building") -> Building:
        return Building(footprint=_box(x, y, w, d, rot), height=h,
                        base=round(ground(x, y), 2), name=label, kind=kind)

    buildings = [
        at("the KDHP tea factory at Nallathanni", -165, -130, 62, 30, 15.0),
        at("the Tata Tea Museum", -246, 26, 46, 26, 11.0),
        at("Christ Church, Munnar", -300, -216, 26, 14, 17.0, kind="monument"),
        at("the High Range Club", -268, -272, 42, 20, 10.0),
        at("the Mattupetty dairy farm sheds", 118, 176, 42, 18, 7.0),
        at("the Echo Point shelter", 215, 196, 15, 10, 5.0, kind="pavilion"),
        at("the Top Station viewing deck", 336, 322, 16, 10, 6.0, kind="pavilion"),
        at("the Eravikulam park checkpost", 256, 300, 12, 8, 5.0, kind="pavilion"),
        at("the Lockhart estate bungalow", -120, 232, 28, 16, 9.0),
        at("the Sevenmallay estate shed", -40, -260, 30, 14, 7.0),
    ]

    # The dam is what makes the reservoir a reservoir: a wall across the neck of
    # the valley below it, standing a few metres proud of the water it holds.
    # Deriving its height from the terrain means it still reaches the water line
    # when `relief` is turned up or down.
    dam_x, dam_y = 168.0, 66.0
    dam_height = round(max(10.0, water_line + 3.0 - ground(dam_x, dam_y)), 1)
    buildings.append(at("the Mattupetty dam wall", dam_x, dam_y, 178, 12, dam_height))

    named_masts = (
        ("the KSEB relay mast on Chokramudi", 70.0, 250.0, 42.0),
        ("the BSNL tower above Top Station", 330.0, 300.0, 36.0),
        ("the estate repeater on Pothamedu", -230.0, -60.0, 28.0),
        ("the survey mast on Meesapulimala", 330.0, -290.0, 24.0),
    )
    for mast_name, x, y, height in named_masts[: max(0, masts)]:
        buildings.append(at(mast_name, x, y, 7, 7, height, kind="mast"))

    # Shola forest in the folds, and the silver-oak windbreaks the estates plant
    # along the contours. Tea itself is knee-high and not worth modelling.
    buildings += _grove(
        rng, ground, (-100.0, 330.0), 72.0, 18,
        ("shola forest tree",), heights=(13.0, 24.0), spreads=(4.5, 8.0),
    )
    buildings += _avenue(
        rng, ground, "estate silver oak", (-150.0, -110.0), (-300.0, -40.0, 24.0),
        heights=(12.0, 18.0), spreads=(3.0, 5.0), axis="x", sides=("lower", "upper"),
    )

    landmarks = [
        Landmark(name="Munnar town", x=0, y=0, radius=72, kind="open",
                 note="the valley floor, where the aircraft takes off"),
        Landmark(name="Anamudi", x=300, y=340, radius=175, height=round(relief, 0), kind="hill",
                 note=f"the high ground to the north-east, about {relief:.0f} m above the pad"),
        Landmark(name="Meesapulimala", x=330, y=-290, radius=155,
                 height=round(relief * 0.83, 0), kind="hill",
                 note="the south-eastern peak, across the valley from Anamudi"),
        Landmark(name="Chokramudi", x=70, y=250, radius=112,
                 height=round(relief * 0.55, 0), kind="hill",
                 note="the nearer hill north of the town, with a 42 m relay mast on top"),
        Landmark(name="the Pothamedu ridge", x=-230, y=-60, radius=130,
                 height=round(relief * 0.36, 0), kind="hill",
                 note="the ridge behind the town, looking back over the estates"),
        Landmark(name="Mattupetty Lake", x=170, y=130, radius=86, kind="water",
                 note=f"the reservoir held back by the dam, in the hollow between "
                      f"Chokramudi and Anamudi; the water sits about "
                      f"{water_line:.0f} m above the pad and there is nothing to hit over it"),
        Landmark(name="the Mattupetty dam", x=dam_x, y=dam_y, radius=94,
                 height=round(dam_height, 0), kind="building",
                 note=f"a 178 m wall across the neck of the valley, "
                      f"{dam_height:.0f} m from its footing to its parapet"),
        Landmark(name="Echo Point", x=215, y=196, radius=42, kind="open",
                 note="the shoulder above the lake"),
        Landmark(name="Top Station", x=336, y=322, radius=42, height=6, kind="open",
                 note="the viewpoint on Anamudi's western slope, the highest thing you can land on"),
        Landmark(name="the Kanan Devan tea estate", x=-180, y=-120, radius=125, kind="open",
                 note="terraced tea, with the factory at Nallathanni in the middle of it"),
        Landmark(name="the Tata Tea Museum", x=-246, y=26, radius=28, height=11, kind="building"),
        Landmark(name="the shola forest", x=-100, y=330, radius=74, height=24, kind="grove",
                 note="a dense patch of native trees in the fold -- fly over it, not through it"),
        Landmark(name="the Eravikulam park gate", x=256, y=300, radius=32, kind="open"),
    ]

    # The town, the estates and their lines, the shola, the roads.
    fill, ground_cover = dressing.dress_munnar(random.Random(seed + 101), terrain, buildings,
                                               water_line)
    return World(
        name=name, source="procedural", terrain=terrain,
        buildings=[*buildings, *fill], landmarks=landmarks, ground=ground_cover,
    )


# ---------------------------------------------------------------------------
# The names the rest of the project knows these by
# ---------------------------------------------------------------------------


def city(name: str = "central vista", seed: int = 7, **kwargs) -> World:
    """The built-up world: the Central Vista. See `lutyens_delhi`."""
    return lutyens_delhi(name=name, seed=seed)


def park(name: str = "lodhi garden", seed: int = 5, **kwargs) -> World:
    """The green world: Lodhi Garden. See `lodhi_garden`."""
    return lodhi_garden(name=name, seed=seed)


def canyon(name: str = "chandni chowk", **kwargs) -> World:
    """The street-canyon world: Chandni Chowk. See `chandni_chowk`."""
    return chandni_chowk(name=name, **{k: v for k, v in kwargs.items() if k in ("street", "seed")})


def hills(name: str = "munnar", **kwargs) -> World:
    """The open-country world: the hills above Munnar. See `munnar`."""
    return munnar(name=name, **kwargs)


#: Everything a caller can ask for by name -- the CLI, the API and the picker
#: all resolve through this, so there is one list of built-in worlds.
BUILDERS: dict[str, tuple[str, str]] = {
    "delhi": ("lutyens_delhi", "The Central Vista, New Delhi — India Gate to Rashtrapati Bhavan"),
    "lodhi": ("lodhi_garden", "Lodhi Garden, New Delhi — tombs, groves and a lake"),
    "chowk": ("chandni_chowk", "Chandni Chowk, Old Delhi — a 30 m street between solid shopfronts"),
    "munnar": ("munnar", "The Kannan Devan hills, Kerala — tea estates under Anamudi"),
    # The names the project used before these places had names.
    "city": ("lutyens_delhi", "alias for delhi"),
    "park": ("lodhi_garden", "alias for lodhi"),
    "canyon": ("chandni_chowk", "alias for chowk"),
    "hills": ("munnar", "alias for munnar"),
}

#: Just the four, in the order a picker should offer them.
BUILT_IN = ("delhi", "lodhi", "chowk", "munnar")


def build(spec: str) -> World:
    """Build a built-in world by name, e.g. `build("delhi")`."""
    try:
        builder, _ = BUILDERS[spec]
    except KeyError:
        raise ValueError(
            f"unknown world '{spec}'; try one of: {', '.join(BUILT_IN)}"
        ) from None
    return globals()[builder]()
