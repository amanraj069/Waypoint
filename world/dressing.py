"""Everything around the landmarks: the city, the lawns, the trees, the roads.

`world.generate` places what a mission is about, at its real height. These
functions fill in the rest of each place so that it reads as the place it is --
Lutyens' bungalows along tree-lined avenues, Old Delhi's solid fabric of
havelis behind the bazaar, a walled garden inside a city, tea terraces
under a peak -- without changing what the world is for:

* The launch pad, and the corridors each world is flown down, stay open. Every
  placement goes through a `Lot` that refuses anything touching them.
* Named buildings are never moved, resized or crowded; fill keeps its distance.
* Ground cover (`Area`) is flat and is never an obstacle.

Each `dress_*` takes the world as `generate` built it and returns the extra
buildings plus the ground cover.
"""

from __future__ import annotations

import math
import random

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from world import layout as L
from world.models import Area, Building, Terrain

PAD_KEEP_CLEAR = 45.0


def _roads(specs, ground_kind: str = "road"):
    """(name, path, width) -> shapely strips, Areas and named centre lines."""
    strips, areas, named = [], [], []
    for name, path, width in specs:
        strip = L.strip(path, width)
        strips.append(strip)
        areas.append(L.area(strip, ground_kind, name))
        named.append((name, L.LineString(path)))
    return strips, areas, named


# ---------------------------------------------------------------------------
# The Central Vista
# ---------------------------------------------------------------------------

_VISTA_ROADS = (
    ("Kartavya Path", [(-300.0, 0.0), (60.0, 0.0)], 14.0),
    ("Janpath", [(-20.0, -660.0), (-20.0, 660.0)], 16.0),
    ("Rafi Marg", [(-272.0, 58.0), (-272.0, 660.0)], 14.0),
    ("Rajendra Prasad Road", [(-272.0, -58.0), (-272.0, -660.0)], 14.0),
    ("the Raisina Road", [(-350.0, 0.0), (-458.0, 0.0)], 18.0),
    ("Ashoka Road", [(-272.0, 300.0), (660.0, 300.0)], 14.0),
    ("Prithviraj Road", [(-272.0, -300.0), (660.0, -300.0)], 14.0),
    ("Church Road", [(-660.0, 340.0), (-272.0, 340.0)], 12.0),
    ("South Avenue", [(-660.0, -300.0), (-272.0, -300.0)], 12.0),
    ("Baba Kharak Singh Marg", [(-660.0, 490.0), (660.0, 490.0)], 12.0),
    ("Motilal Nehru Marg", [(-660.0, -490.0), (660.0, -490.0)], 12.0),
    ("Mathura Road", [(380.0, -660.0), (380.0, 660.0)], 18.0),
    ("Bhagwan Das Road", [(106.0, 140.0), (106.0, 660.0)], 12.0),
    ("Tilak Marg", [(106.0, -140.0), (106.0, -660.0)], 12.0),
    ("Copernicus Marg", [(179.0, 34.0), (660.0, 312.0)], 14.0),
    ("Purana Qila Road", [(179.0, -34.0), (660.0, -312.0)], 14.0),
)

# Lutyens planted each avenue with one species, and it is still how they read.
_AVENUE_SPECIES = ("jamun", "neem", "arjun", "imli", "peepal", "sheesham")


def dress_vista(rng: random.Random, terrain: Terrain, standing: list[Building]):
    ground = terrain.elevation_at
    half = 650.0

    strips, road_areas, named = _roads(_VISTA_ROADS)
    hexagon = Polygon([(120 + 62 * math.cos(math.radians(30 + 60 * i)),
                        62 * math.sin(math.radians(30 + 60 * i))) for i in range(6)])
    hex_ring = L.ring(120.0, 0.0, 62.0, 12.0, sides=6, rot=30.0)
    strips.append(hex_ring)

    lawns = L.rect(-300, -64, 66, 64)
    estate = L.rect(-660, -150, -440, 150)
    canal_n = L.rect(-110, 28, 40, 36)
    canal_s = L.rect(-110, -36, 40, -28)
    vijay_chowk = L.disc(-292, 0, 58)
    forecourt = L.rect(-470, -46, -400, 46)

    areas: list[Area] = [
        L.area(estate, "lawn", "the Rashtrapati Bhavan estate"),
        L.area(lawns, "lawn", "the Kartavya Path lawns"),
        L.area(hexagon, "lawn", "the India Gate lawns"),
        L.area(L.rect(-300, 38, 60, 42), "path", "the north canal walk"),
        L.area(L.rect(-300, -42, 60, -38), "path", "the south canal walk"),
        L.area(canal_n, "water", "the north canal", level=round(ground(-35, 32) + 0.3, 2)),
        L.area(canal_s, "water", "the south canal", level=round(ground(-35, -32) + 0.3, 2)),
        *road_areas,
        L.area(hex_ring, "road", "the C-Hexagon"),
        L.area(L.rect(96, -24, 144, 24), "plaza", "the India Gate plinth"),
        L.area(vijay_chowk, "plaza", "Vijay Chowk"),
        L.area(forecourt, "plaza", "the Rashtrapati Bhavan forecourt"),
        L.area(L.rect(-494, -150, -298, -70), "plaza", "the South Block court"),
        L.area(L.rect(-494, 70, -298, 150), "plaza", "the North Block court"),
    ]
    # The Mughal Gardens behind the house: a grid of lawns and water channels.
    for gx in (-652, -644):
        areas.append(L.area(L.rect(gx, -120, gx + 3, 120), "stream", "a Mughal Gardens channel"))
    for gy in (-110, -70, 70, 110):
        areas.append(L.area(L.rect(-660, gy - 1.5, -440, gy + 1.5), "path", "a Mughal Gardens walk"))

    keep = [
        L.disc(0, 0, 60), lawns, L.disc(120, 0, 72), vijay_chowk.buffer(4),
        L.rect(-470, -60, -300, 60),
        *(s.buffer(3) for s in strips),
    ]
    lot = L.Lot(keep, standing, margin=8.0)
    added: list[Building] = []

    # The avenue trees, one species to a road.
    for index, (name, path, width) in enumerate(_VISTA_ROADS[1:]):
        added += L.line_trees(rng, lot, ground, path, width / 2 + 5, 34.0,
                              f"{name} {_AVENUE_SPECIES[index % len(_AVENUE_SPECIES)]}",
                              heights=(10.0, 16.0), spreads=(3.0, 4.5))

    # The bungalow zone: a house on every plot the plan left free, set back
    # behind its lawn and its trees.
    lines = [line for _, line in named]
    step = 84.0
    y = -half + step / 2
    count = 0
    while y < half:
        x = -half + step / 2
        while x < half:
            cx, cy = x + rng.uniform(-7, 7), y + rng.uniform(-7, 7)
            x += step
            if estate.contains(Point(cx, cy)):
                continue
            angle = L.bearing_of(cx, cy, lines)
            plot = L.rotated_box(cx, cy, 46, 46, angle)
            if not lot.fits(plot):
                continue
            road = L.nearest_name(cx, cy, named, "Lutyens' Delhi")
            count += 1
            if count % 7 == 0:
                house = L.build_on(lot, ground, L.rotated_box(cx, cy, 36, 22, angle),
                                   rng.uniform(14.0, 18.0), f"a ministry annexe on {road}")
            else:
                house = L.build_on(lot, ground,
                                   L.rotated_box(cx, cy, rng.uniform(18, 26), rng.uniform(12, 17), angle),
                                   rng.uniform(7.0, 10.0), f"a bungalow on {road}", kind="house")
            if house is None:
                continue
            added.append(house)
            areas.append(L.area(plot, "garden", f"the garden of {house.name}"))
            added += L.plant(rng, lot, ground, plot, rng.randint(1, 2),
                             (f"{road} garden neem", f"{road} garden mango"),
                             heights=(8.0, 14.0), spreads=(2.8, 4.8), gap=1.0)
        y += step

    # Trees in whatever is left: the Secretariat lawns, the verges, the estate.
    everywhere = L.rect(-half, -half, half, half)
    added += L.plant(rng, lot, ground, estate, 18, ("Mughal Gardens cypress", "estate maulsari"),
                     heights=(8.0, 13.0), spreads=(2.2, 3.6), gap=4.0)
    added += L.plant(rng, lot, ground, everywhere, 70, ("neem", "jamun", "arjun", "peepal"),
                     heights=(9.0, 16.0), spreads=(3.0, 5.5), gap=3.0)
    return added, areas


# ---------------------------------------------------------------------------
# Lodhi Garden
# ---------------------------------------------------------------------------

_LODHI_ROADS = (
    ("Lodhi Road", [(-470.0, -72.0), (470.0, -72.0)], 24.0),
    ("Max Mueller Marg", [(334.0, -84.0), (334.0, 470.0)], 18.0),
    ("Aurobindo Marg", [(-452.0, -470.0), (-452.0, 470.0)], 22.0),
    ("the Lodhi Colony road", [(-440.0, -170.0), (470.0, -170.0)], 10.0),
    ("the Lodhi Colony road", [(-440.0, -290.0), (470.0, -290.0)], 10.0),
    ("the Lodhi Colony road", [(-440.0, -410.0), (470.0, -410.0)], 10.0),
    ("the Meherchand Market road", [(-240.0, -84.0), (-240.0, -470.0)], 10.0),
    ("the Lodhi Colony road", [(0.0, -84.0), (0.0, -470.0)], 10.0),
    ("the Lodhi Colony road", [(220.0, -84.0), (220.0, -470.0)], 10.0),
    ("the Lodhi Estate road", [(343.0, 120.0), (470.0, 120.0)], 9.0),
    ("the Lodhi Estate road", [(343.0, 290.0), (470.0, 290.0)], 9.0),
    ("the Jor Bagh road", [(-441.0, 60.0), (-262.0, 60.0)], 9.0),
    ("the Jor Bagh road", [(-360.0, -60.0), (-360.0, 150.0)], 9.0),
    ("the Jor Bagh road", [(-441.0, 330.0), (-262.0, 330.0)], 9.0),
)

_LODHI_WALKS = (
    ("the ashoka avenue walk", [(0.0, -46.0), (0.0, 196.0)], 6.0),
    ("the ring walk", [(0.0, -30.0), (-90.0, 10.0), (-170.0, 34.0), (-228.0, 110.0),
                       (-240.0, 240.0), (-232.0, 372.0), (-160.0, 452.0), (-40.0, 462.0),
                       (80.0, 452.0), (200.0, 462.0), (292.0, 420.0), (304.0, 300.0),
                       (292.0, 180.0), (262.0, 60.0), (200.0, -12.0), (80.0, -30.0),
                       (0.0, -30.0)], 4.5),
    ("the tomb walk", [(0.0, 150.0), (-60.0, 120.0), (-130.0, 95.0), (-228.0, 110.0)], 3.5),
    ("the tomb walk", [(0.0, 196.0), (62.0, 258.0), (70.0, 300.0)], 3.5),
    ("the tomb walk", [(-75.0, 356.0), (-75.0, 300.0), (-20.0, 250.0), (0.0, 236.0)], 3.5),
    ("the tomb walk", [(62.0, 258.0), (140.0, 250.0), (185.0, 145.0), (262.0, 60.0)], 3.5),
    ("the tomb walk", [(0.0, 60.0), (120.0, 60.0), (185.0, 115.0)], 3.5),
    ("the tomb walk", [(150.0, 420.0), (150.0, 452.0)], 3.5),
)


def lodhi_lake(terrain: Terrain) -> tuple[Terrain, float]:
    """Sink the lake into the ground, so it is a lake and not a tilted disc."""
    rim = min(terrain.elevation_at(115 + 58 * math.cos(a), 360 + 58 * math.sin(a))
              for a in (math.tau * i / 24 for i in range(24)))
    level = round(rim - 0.6, 2)
    heights = L.flatten(terrain.heights, (terrain.origin_x, terrain.origin_y), terrain.spacing,
                        115.0, 360.0, 62.0, 30.0, level - 1.4)
    return terrain.model_copy(update={"heights": heights}), level


def dress_lodhi(rng: random.Random, terrain: Terrain, standing: list[Building], lake_level: float):
    ground = terrain.elevation_at
    half = 470.0

    garden = Polygon([(-262, -48), (212, -48), (212, -14), (318, -14), (318, 478),
                      (-262, 478)])
    lake = L.blob(rng, 115, 360, 50, wobble=0.12, stretch=1.1, rot=-12)
    safdarjung = L.rect(-432, 156, -312, 276)
    strips, road_areas, named = _roads(_LODHI_ROADS)
    walks, walk_areas, _ = _roads(_LODHI_WALKS, "path")

    areas: list[Area] = [
        L.area(L.rect(-half, -half, half, half), "built", "New Delhi"),
        L.area(garden, "lawn", "Lodhi Garden"),
        L.area(safdarjung, "lawn", "the Safdarjung garden"),
        L.area(L.rect(-374, 156, -370, 276), "stream", "the Safdarjung channel"),
        L.area(L.rect(-432, 214, -312, 218), "stream", "the Safdarjung channel"),
        L.area(L.rect(-400, 188, -344, 244), "plaza", "the Safdarjung platform"),
        *road_areas,
        *walk_areas,
        L.area(L.rect(-121, 41, -29, 45), "path", "the Sikander Lodi enclosure walk"),
        L.area(L.rect(-77, 364, -73, 456), "path", "the Sikander Lodi enclosure walk"),
        L.area(L.rect(-121, 408, -29, 412), "path", "the Sikander Lodi enclosure walk"),
        L.area(L.rect(-40, 172, 30, 238), "plaza", "the Bara Gumbad court"),
        L.area(L.disc(-130, 95, 25), "plaza", "the Mohammed Shah plinth"),
        L.area(L.disc(62, 258, 19), "plaza", "the Sheesh Gumbad plinth"),
        L.area(L.disc(120, 60, 24), "plaza", "the Bonsai Park"),
        L.area(L.strip([(150, 404), (150, 452), (176, 478)], 7), "stream", "the stream"),
        L.area(lake, "water", "the lake", level=lake_level),
    ]
    for row in range(4):
        for col in range(3):
            bed = L.rect(150 + col * 17, 186 + row * 11, 162 + col * 17, 191 + row * 11)
            areas.append(L.area(bed, "flowerbed", "the rose garden"))

    keep = [
        L.disc(0, 0, PAD_KEEP_CLEAR), L.rect(-9, -60, 9, 200), lake.buffer(6),
        # The lawns a tour flies low over: south of the lake and along its west shore.
        L.strip([(40, 226), (70, 348), (158, 366)], 36),
        L.strip([(158, 366), (-60, 294)], 30),
        L.disc(250, 300, 72), L.disc(175, 205, 44), L.disc(120, 60, 30),
        safdarjung.buffer(2),
        *(s.buffer(2) for s in strips), *(w.buffer(1.5) for w in walks),
    ]
    lot = L.Lot(keep, standing, margin=6.0)
    added: list[Building] = []

    # The garden is a wood with lawns in it, not a lawn with trees.
    added += L.plant(rng, lot, ground, L.disc(-182, 300, 70), 12,
                     ("old grove neem", "old grove peepal", "old grove banyan"),
                     heights=(11.0, 21.0), spreads=(4.5, 8.0), gap=0.5)
    added += L.plant(rng, lot, ground, garden, 140,
                     ("garden neem", "garden jamun", "garden arjun", "garden peepal",
                      "garden amaltas", "garden silk cotton"),
                     heights=(8.0, 19.0), spreads=(3.0, 7.0), gap=2.0)

    # Lodhi Colony: rows of four-storey flats south of Lodhi Road.
    for index, y in enumerate(range(-112, -470, -46)):
        shift = 0 if index % 2 == 0 else 22
        for x in range(-440 + shift, 470, 76):
            block = L.build_on(lot, ground, L.rotated_box(x, y, 38, 11), rng.uniform(11.0, 14.0),
                               f"Lodhi Colony block {len(added) % 97 + 1}", gap=2.0)
            if block:
                added.append(block)
    added += L.plant(rng, lot, ground, L.rect(-half, -half, half, -90), 50,
                     ("Lodhi Colony neem", "Lodhi Colony jamun"),
                     heights=(8.0, 15.0), spreads=(3.0, 5.0), gap=1.5)

    # Lodhi Estate to the east and Jor Bagh to the west: houses in gardens.
    for region, place, size, height in (
        (L.rect(343, -60, half, half), "Lodhi Estate", (20, 14), (8.0, 11.0)),
        (L.rect(-441, -60, -266, half), "Jor Bagh", (16, 12), (9.0, 13.0)),
    ):
        minx, miny, maxx, maxy = region.bounds
        y = miny + 20
        while y < maxy:
            x = minx + 18
            while x < maxx:
                house = L.build_on(lot, ground,
                                   L.rotated_box(x + rng.uniform(-4, 4), y + rng.uniform(-4, 4),
                                                 size[0], size[1]),
                                   rng.uniform(*height), f"a house in {place}", kind="house", gap=4.0)
                if house:
                    added.append(house)
                x += 54
            y += 50
        added += L.plant(rng, lot, ground, region, 20, (f"{place} neem", f"{place} jamun"),
                         heights=(8.0, 15.0), spreads=(3.0, 5.5), gap=1.0)

    for name, path, width in _LODHI_ROADS[:3]:
        added += L.line_trees(rng, lot, ground, path, width / 2 + 4, 32.0, f"{name} neem",
                              heights=(10.0, 15.0), spreads=(3.0, 4.5))
    return added, areas


# ---------------------------------------------------------------------------
# Chandni Chowk
# ---------------------------------------------------------------------------

_CHOWK_LANES_NORTH = (("Katra Neel", -230.0), ("Ballimaran", -150.0),
                      ("Haveli Haider Quli", -60.0), ("Kinari Bazaar", 120.0),
                      ("Dariba Kalan", 185.0))
_CHOWK_LANES_SOUTH = (("Naughara galli", -210.0), ("Paranthe Wali Gali", -110.0),
                      ("Nai Sarak", -40.0), ("the Chawri Bazaar lane", 30.0),
                      ("Kucha Mahajani", 230.0))
_CHOWK_KINDS = ("haveli", "katra", "shop-house", "godown", "haveli", "shop-house")


def dress_chowk(rng: random.Random, standing: list[Building]):
    ground = lambda x, y: 0.0  # noqa: E731 -- Old Delhi is flat

    road_specs = [
        ("Chandni Chowk", [(-255.0, 0.0), (293.0, 0.0)], 30.0),
        ("Netaji Subhash Marg", [(286.0, -470.0), (286.0, 450.0)], 12.0),
        ("Esplanade Road", [(-470.0, -236.0), (286.0, -236.0)], 10.0),
        ("Lothian Road", [(-470.0, 270.0), (286.0, 270.0)], 10.0),
        ("the Khari Baoli road", [(-470.0, 20.0), (-380.0, 20.0)], 10.0),
    ]
    lanes = ([(name, [(x, 15.0), (x, 265.0)], 5.0) for name, x in _CHOWK_LANES_NORTH]
             + [(name, [(x, -15.0), (x, -231.0)], 5.0) for name, x in _CHOWK_LANES_SOUTH])
    strips, road_areas, named = _roads(road_specs)
    lane_strips, lane_areas, lane_named = _roads(lanes, "path")

    jama_court = L.rect(76, -352, 204, -256)
    maidan = L.rect(292, -250, 361, 250)
    fort = L.rect(375, -257, 580, 257)
    jama_park = L.rect(212, -440, 280, -250)

    areas: list[Area] = [
        L.area(L.rect(-470, -470, 600, 450), "built", "Old Delhi"),
        L.area(maidan, "lawn", "the Red Fort maidan"),
        L.area(fort, "lawn", "the Red Fort gardens"),
        L.area(jama_park, "lawn", "the Jama Masjid park"),
        *road_areas,
        *lane_areas,
        L.area(jama_court, "plaza", "the Jama Masjid courtyard"),
        L.area(L.rect(137, -262, 143, -236), "plaza", "the Jama Masjid steps"),
        L.area(L.rect(375, -4, 580, 4), "path", "the Red Fort axis"),
        L.area(L.rect(476, -257, 484, 257), "path", "the Red Fort cross walk"),
        L.area(L.rect(520, 140, 560, 180), "stream", "the Hayat Bakhsh tank"),
        L.area(L.rect(-470, 330, 286, 384), "rail", "the Old Delhi railway yard"),
    ]

    keep = [
        L.disc(0, 0, PAD_KEEP_CLEAR), L.rect(-260, -62, 300, 62), jama_court.buffer(2),
        *(s.buffer(2) for s in strips), *(s.buffer(1) for s in lane_strips),
    ]
    lot = L.Lot(keep, standing, margin=3.0)
    added: list[Building] = []

    def block(label, x, y, w, d, h, kind="monument", rot=0.0):
        made = L.build_on(lot, ground, L.rotated_box(x, y, w, d, rot), h, label, kind)
        if made:
            added.append(made)

    # The Jama Masjid's cloisters, closing the courtyard on three sides.
    block("the Jama Masjid cloister, west", 73, -304, 6, 96, 9.0)
    block("the Jama Masjid cloister, east", 207, -304, 6, 96, 9.0)
    block("the Jama Masjid cloister, north-west", 100, -253, 50, 6, 9.0)
    block("the Jama Masjid cloister, north-east", 180, -253, 50, 6, 9.0)
    # Inside the Red Fort.
    block("Chatta Chowk", 400, 0, 46, 16, 9.0)
    block("the Naubat Khana", 446, 0, 26, 22, 16.0)
    block("the Diwan-i-Aam", 510, 0, 58, 26, 12.0)
    block("the Rang Mahal", 556, -70, 22, 50, 9.0)
    block("the Khas Mahal", 556, 60, 22, 34, 9.0)
    block("the Red Fort wall, east", 588, 0, 14, 530, 20.0)
    block("the Red Fort wall, far north", 482, 264, 200, 14, 20.0)
    block("the Red Fort wall, far south", 482, -264, 200, 14, 20.0)
    # The station's platforms.
    for index, y in enumerate((336.0, 352.0, 368.0), 1):
        block(f"Old Delhi station platform {index} canopy", -120, y, 220, 7, 6.0, "pavilion")

    # The old city: solid blocks of havelis and katras, split by gallis just
    # wide enough to walk down, behind the shopfronts on both sides.
    all_lanes = named + lane_named
    for region in (L.rect(-470, 62, 280, 262), L.rect(-470, -228, 280, -62),
                   L.rect(-470, 62, -300, 330)):
        minx, miny, maxx, maxy = region.bounds
        y = miny + 11
        while y < maxy - 6:
            x = minx + 12
            while x < maxx - 6:
                w, d = rng.uniform(17, 25), rng.uniform(15, 21)
                if rng.random() < 0.3:          # a courtyard, or a small square
                    x += w + 4
                    continue
                cx, cy = x + w / 2 + rng.uniform(-1.5, 1.5), y + rng.uniform(-1.5, 1.5)
                lane = L.nearest_name(cx, cy, all_lanes, "Old Delhi")
                kind_name = _CHOWK_KINDS[int(rng.random() * len(_CHOWK_KINDS))]
                height = rng.triangular(8.0, 19.0, 12.0)
                made = L.build_on(lot, ground, L.rotated_box(cx, cy, w, d), height,
                                  f"a {kind_name} off {lane}", "house", gap=1.2)
                if made:
                    added.append(made)
                x += w + rng.uniform(3.5, 6.0)
            y += rng.uniform(21, 26)

    added += L.plant(rng, lot, ground, jama_park, 14, ("Jama Masjid park neem", "Jama Masjid park peepal"),
                     heights=(9.0, 15.0), spreads=(3.5, 6.0), gap=1.5)
    added += L.plant(rng, lot, ground, fort, 30, ("Red Fort garden mango", "Red Fort garden neem"),
                     heights=(8.0, 14.0), spreads=(3.0, 5.0), gap=2.0)
    added += L.plant(rng, lot, ground, maidan, 10, ("Red Fort maidan neem",),
                     heights=(9.0, 14.0), spreads=(3.0, 5.0), gap=6.0)
    added += L.line_trees(rng, lot, ground, road_specs[1][1], 10.0, 30.0,
                          "Netaji Subhash Marg peepal", heights=(10.0, 15.0), spreads=(3.0, 4.5))
    return added, areas


# ---------------------------------------------------------------------------
# Munnar
# ---------------------------------------------------------------------------

_MUNNAR_ROADS = (
    ("the Mattupetty road", [(0.0, 0.0), (60.0, 16.0), (120.0, 34.0), (180.0, 46.0),
                             (270.0, 62.0), (292.0, 130.0), (262.0, 200.0), (290.0, 270.0),
                             (336.0, 322.0), (420.0, 400.0), (540.0, 450.0)], 8.0),
    ("the Kochi road", [(0.0, 0.0), (-70.0, -30.0), (-150.0, -104.0), (-230.0, -200.0),
                        (-300.0, -226.0), (-380.0, -300.0), (-540.0, -380.0)], 8.0),
    ("the museum road", [(-70.0, -30.0), (-160.0, -6.0), (-246.0, 10.0), (-340.0, 60.0),
                         (-540.0, 100.0)], 7.0),
    ("the Devikulam road", [(0.0, 0.0), (40.0, -80.0), (60.0, -200.0), (140.0, -300.0),
                            (230.0, -400.0), (300.0, -540.0)], 7.0),
    ("the Lockhart road", [(-70.0, -30.0), (-60.0, 90.0), (-120.0, 214.0),
                           (-180.0, 330.0), (-240.0, 540.0)], 6.0),
)


def munnar_reservoir(terrain: Terrain, water_line: float) -> Terrain:
    """Give the Mattupetty dam a reservoir to hold.

    The hollow the dam closes is an ellipse on the hillside above it. Inside,
    the ground is dug down below the water; around the uphill sides it is
    banked up to the water line, so the lake has a shore instead of spilling
    down the valley. The south side is left alone -- that is the dam's job.
    """
    cx, cy, a, b = 168.0, 112.0, 88.0, 44.0
    heights = []
    for row, line in enumerate(terrain.heights):
        y = terrain.origin_y + row * terrain.spacing
        new = []
        for col, h in enumerate(line):
            x = terrain.origin_x + col * terrain.spacing
            e = math.sqrt(((x - cx) / a) ** 2 + ((y - cy) / b) ** 2)
            if e <= 1.0:
                h = min(h, water_line - 4.0)
            elif y > 66.0 and e < 2.2:
                bank = water_line + 1.5 - max(0.0, e - 1.3) * 25.0
                h = max(h, bank)
            new.append(round(h, 2))
        heights.append(new)
    return terrain.model_copy(update={"heights": heights})


def munnar_lake_outline() -> Polygon:
    # A little wider than the dug-out hollow: the edge runs into the bank, so
    # the shore is wherever the ground comes up through the water.
    return Polygon([(168 + 106 * math.cos(t), 112 + 53 * math.sin(t))
                    for t in (math.tau * i / 40 for i in range(40))])


def dress_munnar(rng: random.Random, terrain: Terrain, standing: list[Building], water_line: float):
    ground = terrain.elevation_at
    half = 540.0
    strips, road_areas, named = _roads(_MUNNAR_ROADS)
    lake = munnar_lake_outline()

    tea = [L.blob(rng, x, y, r, wobble=0.22, rot=rng.uniform(0, 180)) for x, y, r in (
        (-180, -120, 140), (-330, 130, 120), (-70, -300, 120), (-20, 170, 80),
        (70, -140, 80), (-420, -310, 110), (-430, 330, 100), (-250, 410, 90),
        (180, -170, 70), (460, -40, 90), (-470, -60, 70), (60, 520, 80))]
    shola = [L.blob(rng, x, y, r, wobble=0.3) for x, y, r in (
        (-100, 330, 84), (120, 400, 60), (480, 150, 70), (200, -470, 70), (-330, -470, 60))]
    grassland = [L.blob(rng, 300, 340, 210, wobble=0.12), L.blob(rng, 330, -290, 180, wobble=0.12),
                 L.blob(rng, 70, 250, 90, wobble=0.15)]
    eucalyptus = [L.blob(rng, -440, 220, 60), L.blob(rng, -130, -470, 55)]
    town = L.disc(0, 0, 150)

    areas: list[Area] = [
        *(L.area(g, "field", "montane grassland") for g in grassland),
        *(L.area(t.difference(town), "tea", "a tea estate") for t in tea),
        *(L.area(s, "forest", "shola forest") for s in shola),
        *(L.area(e, "forest", "a eucalyptus plantation") for e in eucalyptus),
        L.area(L.disc(0, 0, 120), "built", "Munnar town"),
        L.area(L.strip([(168, 56), (110, 25), (50, -12), (-20, -44), (-90, -150),
                        (-120, -300), (-150, -540)], 9), "stream", "the Muthirapuzha"),
        *road_areas,
        L.area(L.disc(0, 0, 22), "plaza", "the town square"),
        L.area(lake, "water", "Mattupetty Lake", level=round(water_line, 2)),
    ]

    keep = [
        L.disc(0, 0, PAD_KEEP_CLEAR), lake.buffer(4),
        L.disc(215, 196, 42), L.disc(336, 322, 42), L.disc(256, 300, 32),
        *(s.buffer(2) for s in strips),
        *(L.disc(*b.centre, 16) for b in standing if b.kind == "mast"),
    ]
    lot = L.Lot(keep, standing, margin=8.0)
    added: list[Building] = []

    # Munnar town: shops and houses packed along the roads in the valley floor.
    lines = [line for _, line in named]
    for _ in range(900):
        if sum(1 for b in added if b.kind == "house") >= 70:
            break
        angle = rng.uniform(0, math.tau)
        distance = math.sqrt(rng.uniform((PAD_KEEP_CLEAR + 8) ** 2, 175 ** 2))
        cx, cy = distance * math.cos(angle), distance * math.sin(angle)
        near = min(line.distance(Point(cx, cy)) for line in lines)
        if near > 34 and rng.random() < 0.8:
            continue
        facing = L.bearing_of(cx, cy, lines)
        road = L.nearest_name(cx, cy, named, "Munnar")
        house = L.build_on(lot, ground,
                           L.rotated_box(cx, cy, rng.uniform(8, 15), rng.uniform(6, 10), facing),
                           rng.uniform(4.0, 9.0), f"a house on {road}", kind="house", gap=2.0)
        if house:
            added.append(house)
    for label, x, y, w, d, h in (("the Munnar bus stand", -54, 58, 30, 14, 7.0),
                                 ("the Munnar market hall", 56, -54, 28, 18, 9.0),
                                 ("the Mount Carmel church", -96, 70, 22, 12, 14.0)):
        made = L.build_on(lot, ground, L.rotated_box(x, y, w, d, L.bearing_of(x, y, lines)), h, label)
        if made:
            added.append(made)

    # Estate lines: the long low rows of workers' houses on the estates.
    for index in range(8):
        estate = tea[index % 4]
        minx, miny, maxx, maxy = estate.bounds
        for _ in range(30):
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if estate.contains(Point(x, y)):
                made = L.build_on(lot, ground, L.rotated_box(x, y, 32, 7, rng.uniform(0, 180)),
                                  4.5, f"estate lines {index + 1}", kind="house", gap=6.0)
                if made:
                    added.append(made)
                    break

    # Silver oak shading the tea, the shola in the folds, gums on the edges.
    added += L.plant(rng, lot, ground, unary_union(tea), 110, ("tea estate silver oak",),
                     heights=(12.0, 19.0), spreads=(2.4, 3.6), gap=18.0)
    added += L.plant(rng, lot, ground, unary_union(shola), 80, ("shola forest tree",),
                     heights=(13.0, 24.0), spreads=(4.5, 8.0), gap=0.3)
    added += L.plant(rng, lot, ground, unary_union(eucalyptus), 30, ("eucalyptus",),
                     heights=(22.0, 30.0), spreads=(2.4, 3.4), gap=1.0)
    added += L.plant(rng, lot, ground, town, 20, ("town jacaranda", "town rain tree"),
                     heights=(8.0, 14.0), spreads=(3.0, 5.0), gap=2.0)
    return added, areas
