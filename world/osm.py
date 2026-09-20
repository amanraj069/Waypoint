"""Import a real place from OpenStreetMap.

Overpass gives building footprints in latitude and longitude; this converts them
to the local ENU metres the rest of the project works in, with the point you
asked for as the origin -- which is also where the aircraft takes off.

Only the standard library is used for the request, so there is nothing to
install. It is the one part of the project that needs the internet, and it fails
with a clear message rather than a stack trace when Overpass is busy.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge, split, unary_union

from planner.models import GeoPoint
from world.models import Area, Building, Landmark, Terrain, World

# Overpass is a free service that is often busy; try the mirrors in turn
# rather than failing the first time one of them is overloaded.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
METRES_PER_DEGREE_LAT = 110_540.0
METRES_PER_DEGREE_LON = 111_320.0

# A storey is about this tall when a building only tells us how many it has.
STOREY = 3.2
DEFAULT_HEIGHT = 9.0


class OSMError(RuntimeError):
    """Overpass was unreachable, slow, or returned something unusable."""


# What the ground is, beyond the buildings: roads, parks, water, the coast.
# Asked for alongside the buildings so a place reads as a place rather than as
# footprints floating on a grey sheet.
_GROUND_FILTERS = (
    'way["highway"]', 'way["landuse"]', 'way["leisure"]', 'way["natural"]',
    'way["amenity"="parking"]', 'way["man_made"~"pier|breakwater|quay"]',
    'way["railway"="rail"]', 'way["waterway"]', 'node["natural"="tree"]',
)


def query(lat: float, lon: float, radius: float, timeout: float = 60.0,
          ground: bool = True) -> dict:
    """Ask Overpass for every building within `radius` metres of a point.

    With `ground`, the roads, land use, water and street trees come too.
    """
    around = f"(around:{radius:.0f},{lat:.6f},{lon:.6f});"
    wanted = ['way["building"]', *(_GROUND_FILTERS if ground else ())]
    overpass_ql = (
        f"[out:json][timeout:{int(timeout)}];("
        + "".join(f"{selector}{around}" for selector in wanted)
        + ");(._;>;);out body;"
    )
    body = urllib.parse.urlencode({"data": overpass_ql}).encode()
    attempts: list[str] = []

    for endpoint in OVERPASS_MIRRORS:
        request = urllib.request.Request(
            endpoint, data=body, headers={"User-Agent": "waypoint-uav-sim/0.1 (research project)"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            attempts.append(f"{_host(endpoint)}: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError) as exc:
            attempts.append(f"{_host(endpoint)}: {exc}")
        except json.JSONDecodeError as exc:
            attempts.append(f"{_host(endpoint)}: not JSON ({exc})")

    raise OSMError(
        "Every Overpass mirror refused or timed out:\n  "
        + "\n  ".join(attempts)
        + "\nOverpass is free and often busy. Wait a minute, use a smaller "
        "--radius, or fly a procedural world instead (--city)."
    )


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc


def height_of(tags: dict) -> float:
    """Metres, from whichever tag the mapper happened to use."""
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        try:
            return max(2.0, float(str(raw).split()[0].replace(",", ".")))
        except ValueError:
            pass
    levels = tags.get("building:levels") or tags.get("levels")
    if levels:
        try:
            return max(2.0, float(str(levels).split(";")[0]) * STOREY)
        except ValueError:
            pass
    return DEFAULT_HEIGHT


# `building=yes` is what a mapper types when the building is just a building,
# so it is a tag value rather than a name; the same goes for the rest of these.
_GENERIC = {"yes", "true", "1", "building", "roof", "construction", "part"}

# A named OSM building this big or this tall is worth telling the planner about
# by name, because it is what somebody would say to describe where to fly.
LANDMARK_MIN_AREA = 600.0
LANDMARK_MIN_HEIGHT = 40.0
LANDMARK_LIMIT = 24


def name_of(tags: dict, index: int) -> str:
    """The best name this building has, and never an empty one.

    Most OSM buildings are anonymous, so the fallbacks walk down from the name
    the mapper gave it, through its address, to what kind of thing it is --
    because a mission that says "the block at 12 Baker Street" is still a
    mission, and "building" repeated 400 times is not.
    """
    for key in ("name", "official_name", "alt_name", "addr:housename", "operator", "brand"):
        value = (tags.get(key) or "").strip()
        if value:
            return value

    number = (tags.get("addr:housenumber") or "").strip()
    street = (tags.get("addr:street") or "").strip()
    if number and street:
        return f"{number} {street}"
    if street:
        return f"a block on {street}"

    kind = (tags.get("building") or "").strip().lower()
    if kind and kind not in _GENERIC:
        return f"{kind.replace('_', ' ')} {index}"
    return f"unnamed block {index}"


def named(building: Building) -> bool:
    """True when the building carries a real name rather than a made-up one."""
    return not building.name.startswith(("unnamed block ",))


def to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular projection about the origin: good to centimetres over a city."""
    x = (lon - lon0) * METRES_PER_DEGREE_LON * math.cos(math.radians(lat0))
    y = (lat - lat0) * METRES_PER_DEGREE_LAT
    return (x, y)


def build_world(
    payload: dict,
    lat: float,
    lon: float,
    name: str,
    limit: int = 400,
    terrain: Terrain | None = None,
) -> World:
    """Turn an Overpass response into a World centred on (lat, lon)."""
    nodes = {
        element["id"]: (element["lat"], element["lon"])
        for element in payload.get("elements", ())
        if element.get("type") == "node"
    }

    buildings: list[Building] = []
    for element in payload.get("elements", ()):
        if element.get("type") != "way" or _is_ground(element.get("tags", {})):
            continue
        ring = [nodes[ref] for ref in element.get("nodes", ()) if ref in nodes]
        if len(ring) < 4:
            continue  # not a closed footprint we can use
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        points = [GeoPoint(x=round(x, 2), y=round(y, 2))
                  for x, y in (to_local(plat, plon, lat, lon) for plat, plon in ring)]
        if len(points) < 3:
            continue

        tags = element.get("tags", {})
        base = terrain.elevation_at(*_centroid(points)) if terrain else 0.0
        try:
            buildings.append(
                Building(
                    footprint=points,
                    height=round(height_of(tags), 1),
                    base=round(base, 2),
                    name=name_of(tags, len(buildings) + 1),
                )
            )
        except ValueError:
            continue  # degenerate footprint

    if not buildings:
        raise OSMError(
            "No buildings came back for that point. Try a larger --radius, or a "
            "spot that is actually built up."
        )

    buildings.sort(key=lambda b: b.polygon.area, reverse=True)
    kept = buildings[:limit]
    minx = min(p.x for b in kept for p in b.footprint)
    maxx = max(p.x for b in kept for p in b.footprint)
    miny = min(p.y for b in kept for p in b.footprint)
    maxy = max(p.y for b in kept for p in b.footprint)
    frame = box(minx - 60, miny - 60, maxx + 60, maxy + 60)
    return World(
        name=name,
        source="osm",
        origin_lat=lat,
        origin_lon=lon,
        terrain=terrain,
        buildings=kept + trees_from(payload, lat, lon, kept, frame, terrain),
        landmarks=landmarks_from(kept),
        ground=ground_from(payload, lat, lon, frame),
    )


# -- ground cover -----------------------------------------------------------

_GROUND_KEYS = ("highway", "landuse", "leisure", "natural", "amenity", "man_made",
                "railway", "waterway")


def _is_ground(tags: dict) -> bool:
    """A way that describes the ground rather than a building on it.

    Anything tagged `building` is a building whatever else it says; so is an
    untagged way, which is what the building-only query used to return.
    """
    return "building" not in tags and any(key in tags for key in _GROUND_KEYS)

# Carriageway widths, roughly: enough to tell an arterial from a lane.
_ROAD_WIDTH = {
    "motorway": 22.0, "trunk": 18.0, "primary": 16.0, "secondary": 14.0, "tertiary": 12.0,
    "motorway_link": 9.0, "trunk_link": 9.0, "primary_link": 9.0, "secondary_link": 8.0,
    "tertiary_link": 8.0, "residential": 9.0, "unclassified": 8.0, "living_street": 7.0,
    "service": 6.0, "pedestrian": 8.0, "road": 8.0,
}
_PATH_WIDTH = {"footway": 3.0, "path": 3.0, "cycleway": 3.0, "steps": 3.0, "track": 4.0,
               "bridleway": 3.0}

# (tag, value) -> kind, in the order they should be drawn: land use first,
# then what is on it, then water.
_AREA_KIND = {
    ("landuse", "residential"): "built", ("landuse", "commercial"): "built",
    ("landuse", "retail"): "built", ("landuse", "institutional"): "built",
    ("landuse", "government"): "built", ("landuse", "military"): "built",
    ("landuse", "industrial"): "built", ("landuse", "railway"): "rail",
    ("landuse", "construction"): "sand", ("landuse", "farmland"): "field",
    ("landuse", "meadow"): "field", ("natural", "grassland"): "field",
    ("natural", "scrub"): "field", ("landuse", "grass"): "lawn",
    ("landuse", "recreation_ground"): "lawn", ("landuse", "village_green"): "lawn",
    ("leisure", "park"): "lawn", ("leisure", "garden"): "lawn",
    ("leisure", "recreation_ground"): "lawn", ("landuse", "forest"): "forest",
    ("natural", "wood"): "forest", ("leisure", "pitch"): "pitch",
    ("leisure", "stadium"): "pitch", ("leisure", "sports_centre"): "plaza",
    ("leisure", "playground"): "plaza", ("leisure", "schoolyard"): "plaza",
    ("amenity", "parking"): "parking", ("natural", "beach"): "sand",
    ("natural", "sand"): "sand", ("leisure", "swimming_pool"): "water",
    ("natural", "water"): "water", ("landuse", "reservoir"): "water",
    ("landuse", "basin"): "water",
}
_AREA_ORDER = ("built", "rail", "field", "sand", "forest", "lawn", "pitch", "plaza",
               "parking", "water")


def _way_points(element: dict, nodes: dict, lat0: float, lon0: float) -> list[tuple[float, float]]:
    return [to_local(*nodes[ref], lat0, lon0) for ref in element.get("nodes", ()) if ref in nodes]


def _as_area(geometry, kind: str, name: str = "") -> Area | None:
    if geometry.is_empty:
        return None
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda g: g.area)
    if geometry.geom_type != "Polygon" or geometry.area < 4.0:
        return None
    geometry = geometry.simplify(0.3)
    coords = list(geometry.exterior.coords)[:-1]
    if len(coords) < 3:
        return None
    return Area(
        footprint=[GeoPoint(x=round(x, 2), y=round(y, 2)) for x, y in coords],
        kind=kind,
        name=name,
    )


def sea_from(coastlines: list[list[tuple[float, float]]], frame: Polygon) -> list[Polygon]:
    """The sea inside the frame, from OSM coastline ways.

    OSM draws every coastline with the land on its left, so cutting the frame
    along the coast and keeping the pieces on the right-hand side gives the
    water -- without needing the global land polygons.
    """
    if not coastlines:
        return []
    merged = linemerge([LineString(c) for c in coastlines if len(c) >= 2])
    lines = list(getattr(merged, "geoms", [merged]))
    pieces = [frame]
    for line in lines:
        pieces = [part for piece in pieces for part in split(piece, line).geoms]
    if len(pieces) < 2:
        return []

    score = [0] * len(pieces)
    for line in lines:
        coords = list(line.coords)
        for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 1e-6:
                continue
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            nx, ny = (y1 - y0) / length, -(x1 - x0) / length    # to the right
            for sign in (1, -1):
                probe = Point(mx + sign * 1.5 * nx, my + sign * 1.5 * ny)
                for index, piece in enumerate(pieces):
                    if piece.contains(probe):
                        score[index] += sign
                        break
    return [piece for piece, votes in zip(pieces, score) if votes > 0]


def ground_from(payload: dict, lat: float, lon: float, frame: Polygon) -> list[Area]:
    """Roads, land use, parks and water as ground cover, clipped to the frame."""
    nodes = {
        element["id"]: (element["lat"], element["lon"])
        for element in payload.get("elements", ())
        if element.get("type") == "node"
    }
    by_kind: dict[str, list[Area]] = {kind: [] for kind in _AREA_ORDER}
    roads: list[tuple[float, Area]] = []
    coastlines: list[list[tuple[float, float]]] = []

    for element in payload.get("elements", ()):
        if element.get("type") != "way":
            continue
        tags = element.get("tags", {})
        if not _is_ground(tags):
            continue
        coords = _way_points(element, nodes, lat, lon)
        if len(coords) < 2:
            continue
        name = tags.get("name", "")

        if tags.get("natural") == "coastline":
            coastlines.append(coords)
            continue

        highway = tags.get("highway")
        width = _ROAD_WIDTH.get(highway) or _PATH_WIDTH.get(highway)
        waterway = tags.get("waterway")
        linear = None
        if width and not (tags.get("area") == "yes" and len(coords) >= 4):
            linear = (width, "road" if highway in _ROAD_WIDTH else "path")
        elif tags.get("railway") == "rail":
            linear = (4.0, "rail")
        elif waterway in ("river", "canal", "stream", "drain", "ditch"):
            linear = ({"river": 14.0, "canal": 10.0}.get(waterway, 4.0), "stream")
        elif tags.get("man_made") in ("pier", "breakwater", "quay") and coords[0] != coords[-1]:
            linear = (8.0, "plaza")

        if linear:
            width, kind = linear
            shape = LineString(coords).buffer(width / 2, cap_style="flat", join_style="round",
                                              quad_segs=3).intersection(frame)
            area = _as_area(shape, kind, name)
            if area:
                roads.append((width, area))
            continue

        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        kind = next((_AREA_KIND[(key, tags[key])] for key in
                     ("natural", "leisure", "amenity", "landuse")
                     if (key, tags.get(key)) in _AREA_KIND), None)
        if tags.get("highway") == "pedestrian" or tags.get("man_made") in ("pier", "quay"):
            kind = "plaza"
        if kind is None:
            continue
        try:
            shape = Polygon(coords).buffer(0).intersection(frame)
        except ValueError:
            continue
        area = _as_area(shape, kind, name)
        if area:
            by_kind[kind].append(area)

    sea = [area for area in (_as_area(p, "water", "the sea") for p in sea_from(coastlines, frame))
           if area]
    # Narrow first, so the arterials are drawn over the lanes that join them.
    roads.sort(key=lambda item: item[0])
    return [*sea, *(area for kind in _AREA_ORDER for area in by_kind[kind]),
            *(area for _, area in roads)]


def trees_from(payload: dict, lat: float, lon: float, standing: list[Building],
               frame: Polygon, terrain: Terrain | None) -> list[Building]:
    """Mapped street trees, as the solid things they are."""
    solid = unary_union([b.polygon for b in standing]) if standing else None
    trees: list[Building] = []
    for element in payload.get("elements", ()):
        tags = element.get("tags", {})
        if element.get("type") != "node" or tags.get("natural") != "tree":
            continue
        x, y = to_local(element["lat"], element["lon"], lat, lon)
        if not frame.contains(Point(x, y)):
            continue
        height = height_of(tags) if tags.get("height") else 10.0
        spread = 3.0
        crown = box(x - spread, y - spread, x + spread, y + spread)
        if solid is not None and solid.intersects(crown):
            continue
        species = tags.get("species:en") or tags.get("species") or tags.get("genus") or "street tree"
        trees.append(Building(
            footprint=[GeoPoint(x=round(px, 2), y=round(py, 2))
                       for px, py in list(crown.exterior.coords)[:-1]],
            height=round(height, 1),
            base=round(terrain.elevation_at(x, y), 2) if terrain else 0.0,
            name=f"{species} {len(trees) + 1}",
            kind="tree",
        ))
    return trees


def landmarks_from(buildings: list[Building]) -> list[Landmark]:
    """Promote the named, notable buildings to places a mission can refer to.

    Without this the planner is handed four hundred footprints and no way to
    say which one you meant. With it, "photograph the cathedral" works on a
    world nobody hand-authored.
    """
    notable = [
        b for b in buildings
        if named(b) and (b.polygon.area >= LANDMARK_MIN_AREA or b.height >= LANDMARK_MIN_HEIGHT)
    ]
    notable.sort(key=lambda b: (b.roof, b.polygon.area), reverse=True)

    landmarks = []
    for building in notable[:LANDMARK_LIMIT]:
        cx, cy = building.centre
        minx, miny, maxx, maxy = building.polygon.bounds
        landmarks.append(
            Landmark(
                name=building.name,
                x=round(cx, 1),
                y=round(cy, 1),
                radius=round(max(8.0, max(maxx - minx, maxy - miny) / 2.0), 1),
                height=round(building.height, 1),
                kind="building",
                note=f"roof at {building.roof:.0f} m",
            )
        )
    return landmarks


def _centroid(points: list[GeoPoint]) -> tuple[float, float]:
    return (
        sum(p.x for p in points) / len(points),
        sum(p.y for p in points) / len(points),
    )


def fetch(
    lat: float,
    lon: float,
    radius: float = 400.0,
    name: str | None = None,
    limit: int = 400,
    terrain: Terrain | None = None,
) -> World:
    """Download a real place and return it as a World."""
    payload = query(lat, lon, radius)
    return build_world(
        payload,
        lat,
        lon,
        name or f"osm_{lat:.4f}_{lon:.4f}",
        limit=limit,
        terrain=terrain,
    )
