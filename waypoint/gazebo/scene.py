"""Turn a run into a Gazebo world: terrain, buildings, trees, the envelope and the route.

Gazebo's frame is ours already (x east, y north, z up, metres), so nothing is
reprojected. The ground is written as a mesh (`terrain.obj`) coloured by what
covers it, because an SDF heightmap cannot carry ground cover and wants a
power-of-two image; everything else is plain SDF geometry in `world.sdf`.

Nothing in the scene is a physics body except the aircraft, and the world has
no gravity: the aircraft is posed every frame from the simulator's ticks, so
the flight you watch is the flight that was verified and flown, not a second
simulation that could disagree with it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import shapely
from shapely.geometry import Polygon

from waypoint.world.models import World

WORLD_NAME = "waypoint"
UAV_NAME = "uav"

# The daylight palette the scene is drawn in.
GROUND = {
    "base": "#dcd6c8",
    "valley": "#a9bf8c",
    "slope": "#93ad78",
    "summit": "#c3c393",
    "built": "#d9d3c7",
    "lawn": "#b3c796",
    "garden": "#bccd9f",
    "field": "#c7c998",
    "tea": "#8fb070",
    "forest": "#86a377",
    "pitch": "#a4c287",
    "flowerbed": "#cfaea3",
    "sand": "#e4d7b6",
    "plaza": "#e6ded0",
    "parking": "#cdc8be",
    "path": "#dfceac",
    "road": "#b7b4ad",
    "rail": "#aaa59c",
    "stream": "#94b9c9",
    "water": "#8fb3c4",
}
WALL = {
    "building": "#a9b3bb",
    "house": "#e6dfd1",
    "monument": "#c4a184",
    "pavilion": "#b5bec6",
    "mast": "#9ea6ad",
}
CANOPY, TRUNK = "#5f8f5a", "#6b5a48"
PLANNED = "#8899aa"
REROUTE = "#22d3ee"  # a plan that took over mid-flight: the route it replaced stays grey
ACTION = {"MOVE": "#4c78a8", "HOVER": "#f58518", "CAPTURE": "#54a24b", "LAND": "#e45756"}
FENCE, NOFLY, OBSTACLE = "#45d08a", "#ff5c5c", "#ff7a5c"
PAD, LANDING = "#45d08a", "#ffb020"

# Finer than the terrain posts so a road or a stream still shows, but capped so
# a city-sized world stays a mesh Gazebo loads in a second or two.
CELL = 4.0
MAX_CELLS = 320
MARGIN = 40.0


@dataclass
class LateObstacle:
    """Something that joins the scene mid-flight, spawned when the playback reaches it:
    an obstacle that appeared, or the route of a plan that took over."""

    t: float
    sdf: str


@dataclass
class Scene:
    sdf: Path
    late: list[LateObstacle] = field(default_factory=list)


def write_scene(run: dict, out_dir: Path, *, px4: bool = False) -> Scene:
    """Write `world.sdf` and `terrain.obj`/`.mtl` for a run file's contents.

    `px4=True` writes the same scene as a world PX4 SITL can fly in: gravity,
    real physics, the sensor systems PX4's airframe needs, and ground the
    aircraft can stand on. The aircraft itself is left out, because PX4 spawns
    its own.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    world = World.model_validate(run["world"]) if run.get("world") else World()
    constraints = run.get("constraints") or {}
    # `pads` is set when a world is shown before anything has flown, so the
    # chosen launch and landing points are on the map; a flown run has the
    # launch pad in its settings and the landing where its last tick is.
    pads = run.get("pads") or {}
    home = tuple(pads.get("start") or (run.get("sim") or {}).get("config", {}).get("home", (0.0, 0.0, 0.0)))

    write_terrain(world, out_dir, _extent(world, constraints, run.get("waypoints") or []))
    models = [
        _terrain_model(collide=px4, flat=world.terrain is None),
        *_buildings(world),
        _trees(world),
        *_pads(world, home, run.get("ticks") or [], pads.get("landing")),
        _envelope(world, constraints),
        _route(world, _on_ground(world, home), run.get("waypoints") or []),
        "" if px4 else _uav(world, home),
    ]
    late = []
    for index, record in enumerate(run.get("obstacles") or []):
        model = _obstacle(world, record, constraints, f"obstacle_{index}")
        if record.get("appeared_at", 0.0) > 0:
            late.append(LateObstacle(t=record["appeared_at"], sdf=_sdf_document(model)))
        else:
            models.append(model)
    # Each replacement plan appears when it took over, so the reroute shows up
    # at the moment it happened, next to the route it replaced.
    for number, segment in enumerate((run.get("plans") or [])[1:], 1):
        if segment.get("source") == "failsafe":
            continue
        model = _route(
            world, tuple(segment["start"]), segment["waypoints"],
            name=f"reroute_{number}", colour=REROUTE,
        )
        if model:
            late.append(LateObstacle(t=segment["t"], sdf=_sdf_document(model)))

    sdf = out_dir / "world.sdf"
    sdf.write_text(_world_document(world, [m for m in models if m], px4=px4))
    return Scene(sdf=sdf, late=late)


# -- terrain ------------------------------------------------------------------


def _extent(world: World, constraints: dict, waypoints: list[dict]) -> tuple[float, ...]:
    """Ground wide enough for the world, the fence and the route, whichever is biggest."""
    xs, ys = [-100.0, 100.0], [-100.0, 100.0]
    if world.bounds != (0.0, 0.0, 0.0, 0.0):
        xs += [world.bounds[0], world.bounds[2]]
        ys += [world.bounds[1], world.bounds[3]]
    for point in [*constraints.get("geofence", []), *waypoints]:
        xs.append(point["x"])
        ys.append(point["y"])
    return (min(xs) - MARGIN, min(ys) - MARGIN, max(xs) + MARGIN, max(ys) + MARGIN)


def write_terrain(world: World, out_dir: Path, extent: tuple[float, ...]) -> None:
    """A triangulated ground mesh, each face coloured by the cover under it."""
    minx, miny, maxx, maxy = extent
    cell = max(CELL, (maxx - minx) / MAX_CELLS, (maxy - miny) / MAX_CELLS)
    xs = np.linspace(minx, maxx, int(math.ceil((maxx - minx) / cell)) + 1)
    ys = np.linspace(miny, maxy, int(math.ceil((maxy - miny) / cell)) + 1)
    heights = _sample(world, xs, ys)

    # Normals from the height field's slope, so hills are lit like hills.
    gy, gx = np.gradient(heights, ys, xs)
    normals = np.dstack([-gx, -gy, np.ones_like(heights)])
    normals /= np.linalg.norm(normals, axis=2, keepdims=True)

    # One material per cell, sampled at its centre; later areas draw on top.
    cx = (xs[:-1] + xs[1:]) / 2
    cy = (ys[:-1] + ys[1:]) / 2
    gcx, gcy = np.meshgrid(cx, cy)
    kinds = np.full(gcx.shape, _relief_kind(world), dtype=object)
    if world.terrain:
        low, high = world.terrain.relief
        if high - low > 15:
            centre = (heights[:-1, :-1] + heights[1:, 1:]) / 2
            level = (centre - low) / (high - low)
            kinds[:] = np.where(level < 0.3, "valley", np.where(level < 0.7, "slope", "summit"))
    for area in world.ground:
        if area.kind not in GROUND:
            continue
        polygon = Polygon([(p.x, p.y) for p in area.footprint])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        kinds[shapely.contains_xy(polygon, gcx, gcy)] = area.kind

    cols = len(xs)
    grid_x = np.repeat(xs[None, :], len(ys), axis=0)
    grid_y = np.repeat(ys[:, None], len(xs), axis=1)
    parts = ["mtllib terrain.mtl\n"]
    parts.append(_verts(np.stack([grid_x, grid_y, heights], axis=-1).reshape(-1, 3)))
    parts.append(_normals(normals.reshape(-1, 3)))
    # One quad per cell rather than two triangles: the mesh Gazebo ends up with
    # is the same (it triangulates on load) from half as many lines of OBJ.
    for kind in sorted(set(kinds.flat)):
        parts.append(f"usemtl {kind}\n")
        j, i = np.nonzero(kinds == kind)
        a = j * cols + i + 1  # OBJ indices are 1-based
        corners = np.stack([a, a + 1, a + cols + 1, a + cols], axis=-1)
        parts.append(_faces(corners))
    (out_dir / "terrain.obj").write_text("".join(parts))

    materials = []
    for kind, colour in GROUND.items():
        r, g, b, _ = _rgba(colour)
        materials += [f"newmtl {kind}", f"Ka {r} {g} {b}", f"Kd {r} {g} {b}", "Ks 0 0 0", ""]
    (out_dir / "terrain.mtl").write_text("\n".join(materials))


def _sample(world: World, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """`world.elevation_at` over a whole grid at once.

    The same bilinear read as `Terrain.elevation_at`, done in numpy: a city
    grid is ~90,000 posts, and one Python call each is most of a scene build.
    """
    terrain = world.terrain
    if terrain is None:
        return np.zeros((len(ys), len(xs)))

    posts = np.asarray(terrain.heights)
    cols = np.clip((xs - terrain.origin_x) / terrain.spacing, 0.0, terrain.cols - 1.0)
    rows = np.clip((ys - terrain.origin_y) / terrain.spacing, 0.0, terrain.rows - 1.0)
    c0 = np.floor(cols).astype(int)
    r0 = np.floor(rows).astype(int)
    c1 = np.minimum(c0 + 1, terrain.cols - 1)
    r1 = np.minimum(r0 + 1, terrain.rows - 1)
    fc = (cols - c0)[None, :]
    fr = (rows - r0)[:, None]

    top = posts[r0[:, None], c0[None, :]] * (1 - fc) + posts[r0[:, None], c1[None, :]] * fc
    bottom = posts[r1[:, None], c0[None, :]] * (1 - fc) + posts[r1[:, None], c1[None, :]] * fc
    return top * (1 - fr) + bottom * fr


def _verts(values: np.ndarray) -> str:
    """`v` lines for an (n, 3) array. `tolist()` first: formatting numpy
    scalars one at a time costs more than the whole conversion does."""
    return "".join([f"v {x:.2f} {y:.2f} {z:.2f}\n" for x, y, z in values.tolist()])


def _normals(values: np.ndarray) -> str:
    return "".join([f"vn {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in values.tolist()])


def _faces(corners: np.ndarray) -> str:
    """`f a//a b//b ...` lines: the vertex and its normal always share an index."""
    return "".join(
        [f"f {a}//{a} {b}//{b} {c}//{c} {d}//{d}\n" for a, b, c, d in corners.tolist()]
    )


def _relief_kind(world: World) -> str:
    return "base" if world.buildings or world.ground else "lawn"


def _terrain_model(*, collide: bool = False, flat: bool = True) -> str:
    """The ground mesh; with `collide`, also something to stand on.

    Flat ground collides as a plane, which physics handles best; relief
    collides as the same mesh that is drawn.
    """
    collision = ""
    if collide:
        shape = (
            "<plane><normal>0 0 1</normal><size>10000 10000</size></plane>"
            if flat else "<mesh><uri>terrain.obj</uri></mesh>"
        )
        collision = f"""
        <collision name="ground">
          <geometry>{shape}</geometry>
        </collision>"""
    return f"""
    <model name="terrain">
      <static>true</static>
      <link name="ground">
        <visual name="mesh">
          <cast_shadows>false</cast_shadows>
          <geometry><mesh><uri>terrain.obj</uri></mesh></geometry>
        </visual>{collision}
      </link>
    </model>"""


# -- what stands on it -----------------------------------------------------------


def _buildings(world: World) -> list[str]:
    """One named model per building, so the entity tree reads like the world does."""
    models = []
    used: set[str] = set()
    for index, building in enumerate(world.buildings):
        if building.kind == "tree":
            continue
        points = [(p.x, p.y) for p in building.footprint]
        if len(points) > 3 and points[0] == points[-1]:
            points = points[:-1]
        name = _unique(_slug(building.name) or f"{building.kind}_{index}", used)
        # Sunk two metres, so a building on a slope has no gap on its low side.
        polyline = "".join(f"<point>{x:.2f} {y:.2f}</point>" for x, y in points)
        geometry = f"<polyline>{polyline}<height>{building.height + 2:.2f}</height></polyline>"
        colour = WALL.get(building.kind, WALL["building"])
        models.append(f"""
    <model name="{name}">
      <static>true</static>
      <pose>0 0 {building.base - 2:.2f} 0 0 0</pose>
      <link name="body">
        <visual name="walls">
          <geometry>{geometry}</geometry>
          {_material(colour)}
        </visual>
      </link>
    </model>""")
    return models


def _trees(world: World) -> str:
    """Every tree as a trunk and an ellipsoid canopy, in one model."""
    visuals = []
    for index, tree in enumerate(b for b in world.buildings if b.kind == "tree"):
        x, y = tree.centre
        radius = max(1.0, math.sqrt(tree.polygon.area / math.pi))
        trunk = tree.height * 0.35
        canopy = tree.height - trunk
        visuals.append(f"""
        <visual name="trunk_{index}">
          <pose>{x:.2f} {y:.2f} {tree.base + trunk / 2:.2f} 0 0 0</pose>
          <geometry><cylinder><radius>{max(0.2, radius * 0.15):.2f}</radius><length>{trunk:.2f}</length></cylinder></geometry>
          {_material(TRUNK)}
        </visual>
        <visual name="canopy_{index}">
          <pose>{x:.2f} {y:.2f} {tree.base + trunk + canopy / 2:.2f} 0 0 0</pose>
          <geometry><ellipsoid><radii>{radius:.2f} {radius:.2f} {canopy / 2:.2f}</radii></ellipsoid></geometry>
          {_material(CANOPY)}
        </visual>""")
    if not visuals:
        return ""
    return f"""
    <model name="trees">
      <static>true</static>
      <link name="canopy">{"".join(visuals)}
      </link>
    </model>"""


def _pads(world: World, home: tuple, ticks: list[dict], landing: list | None = None) -> list[str]:
    """The launch pad (green), and a landing pad (amber) when it is somewhere else."""
    pads = [_pad("launch_pad", home[0], home[1], world, PAD)]
    end = landing or ([ticks[-1]["x"], ticks[-1]["y"]] if ticks else None)
    if end and math.hypot(end[0] - home[0], end[1] - home[1]) > 5:
        pads.append(_pad("landing_pad", end[0], end[1], world, LANDING))
    return pads


def _pad(name: str, x: float, y: float, world: World, colour: str) -> str:
    ground = world.elevation_at(x, y)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.2f} {y:.2f} {ground + 0.1:.2f} 0 0 0</pose>
      <link name="disc">
        <visual name="disc">
          <geometry><cylinder><radius>3</radius><length>0.2</length></cylinder></geometry>
          {_material(colour, emissive=True)}
        </visual>
        <visual name="beam">
          <pose>0 0 15 0 0 0</pose>
          <geometry><cylinder><radius>0.15</radius><length>30</length></cylinder></geometry>
          {_material(colour, alpha=0.5, emissive=True)}
        </visual>
      </link>
    </model>"""


# -- the safety envelope ------------------------------------------------------------


def _envelope(world: World, constraints: dict) -> str:
    """Geofence walls and no-fly volumes, drawn as glass: seen, never in the way."""
    if not constraints:
        return ""
    floor = world.terrain.relief[0] if world.terrain else 0.0
    ceiling = constraints.get("max_altitude", 120.0)
    height = ceiling - floor
    visuals = []
    fence = [(p["x"], p["y"]) for p in constraints.get("geofence", [])]
    for index, (a, b) in enumerate(zip(fence, fence[1:] + fence[:1])):
        length = math.dist(a, b)
        if length < 1e-6:
            continue
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        visuals.append(f"""
        <visual name="fence_{index}">
          <pose>{mx:.2f} {my:.2f} {floor + height / 2:.2f} 0 0 {yaw:.4f}</pose>
          <geometry><box><size>{length:.2f} 0.3 {height:.2f}</size></box></geometry>
          {_material(FENCE, alpha=0.12)}
        </visual>""")
    for index, zone in enumerate(constraints.get("no_fly_zones", [])):
        polyline = "".join(f"<point>{p['x']:.2f} {p['y']:.2f}</point>" for p in zone)
        visuals.append(f"""
        <visual name="nofly_{index}">
          <pose>0 0 {floor:.2f} 0 0 0</pose>
          <geometry><polyline>{polyline}<height>{height:.2f}</height></polyline></geometry>
          {_material(NOFLY, alpha=0.3)}
        </visual>""")
    if not visuals:
        return ""
    return f"""
    <model name="envelope">
      <static>true</static>
      <link name="glass">{"".join(visuals)}
      </link>
    </model>"""


def _obstacle(world: World, record: dict, constraints: dict, name: str) -> str:
    """A known hazard: a cylinder from the ground to its top (or the ceiling)."""
    ground = world.elevation_at(record["x"], record["y"])
    top = record.get("height")
    height = top if top is not None else constraints.get("max_altitude", 120.0) - ground
    radius = max(record.get("radius", 0.0), 1.0)
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{record['x']:.2f} {record['y']:.2f} {ground + height / 2:.2f} 0 0 0</pose>
      <link name="body">
        <visual name="body">
          <geometry><cylinder><radius>{radius:.2f}</radius><length>{height:.2f}</length></cylinder></geometry>
          {_material(OBSTACLE, alpha=0.35)}
        </visual>
      </link>
    </model>"""


# -- the route and the aircraft -------------------------------------------------------


def _on_ground(world: World, point: tuple) -> tuple:
    return (point[0], point[1], world.elevation_at(point[0], point[1]))


def _route(
    world: World,
    start: tuple,
    waypoints: list[dict],
    *,
    name: str = "route",
    colour: str = PLANNED,
) -> str:
    """A planned path from `start`: a bead per waypoint, coloured by action, strung together."""
    if not waypoints:
        return ""
    points = [start] + [(wp["x"], wp["y"], wp["z"]) for wp in waypoints]
    visuals = []
    for index, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-3:
            continue
        # Cylinders stand on z; tip that axis onto the segment.
        pitch = math.atan2(math.hypot(dx, dy), dz)
        yaw = math.atan2(dy, dx)
        mid = [(a[k] + b[k]) / 2 for k in range(3)]
        visuals.append(f"""
        <visual name="leg_{index}">
          <cast_shadows>false</cast_shadows>
          <pose>{mid[0]:.2f} {mid[1]:.2f} {mid[2]:.2f} 0 {pitch:.4f} {yaw:.4f}</pose>
          <geometry><cylinder><radius>0.25</radius><length>{length:.2f}</length></cylinder></geometry>
          {_material(colour, alpha=0.4 if colour == PLANNED else 0.7, emissive=colour != PLANNED)}
        </visual>""")
    for index, wp in enumerate(waypoints):
        visuals.append(f"""
        <visual name="waypoint_{index}">
          <cast_shadows>false</cast_shadows>
          <pose>{wp['x']:.2f} {wp['y']:.2f} {wp['z']:.2f} 0 0 0</pose>
          <geometry><sphere><radius>1.2</radius></sphere></geometry>
          {_material(ACTION.get(wp.get('action', 'MOVE'), ACTION['MOVE']), emissive=True)}
        </visual>""")
    return f"""
    <model name="{name}">
      <static>true</static>
      <link name="plan">{"".join(visuals)}
      </link>
    </model>"""


def _uav(world: World, home: tuple) -> str:
    """A quadrotor about three metres across: oversized, so it can be found from across a city.

    It is the one non-static model; with gravity off and no collisions it goes
    exactly where the playback puts it and nowhere else.
    """
    z = world.elevation_at(home[0], home[1]) + 0.5
    arms, rotors = [], []
    for index, angle in enumerate((45, 135, 225, 315)):
        rad = math.radians(angle)
        x, y = 1.5 * math.cos(rad), 1.5 * math.sin(rad)
        arms.append(f"""
        <visual name="arm_{index}">
          <pose>{x / 2:.3f} {y / 2:.3f} 0 0 0 {rad:.4f}</pose>
          <geometry><box><size>1.6 0.16 0.12</size></box></geometry>
          {_material("#2a2f36")}
        </visual>""")
        rotors.append(f"""
        <visual name="rotor_{index}">
          <pose>{x:.3f} {y:.3f} 0.15 0 0 0</pose>
          <geometry><cylinder><radius>0.65</radius><length>0.04</length></cylinder></geometry>
          {_material("#4cc2ff", alpha=0.45, emissive=True)}
        </visual>""")
    return f"""
    <model name="{UAV_NAME}">
      <pose>{home[0]:.2f} {home[1]:.2f} {z:.2f} 0 0 0</pose>
      <link name="base_link">
        <gravity>false</gravity>
        <visual name="body">
          <geometry><box><size>1.0 0.8 0.35</size></box></geometry>
          {_material("#1e2329")}
        </visual>
        <visual name="nose">
          <pose>0.6 0 0 0 0 0</pose>
          <geometry><box><size>0.3 0.3 0.2</size></box></geometry>
          {_material("#ff1fd1", emissive=True)}
        </visual>{"".join(arms)}{"".join(rotors)}
      </link>
    </model>"""


# -- the document ---------------------------------------------------------------------


#: Where PX4's own worlds put the origin, for a world that is not anywhere real.
#: PX4 needs a latitude and longitude, because its GPS is simulated from them.
PX4_ORIGIN = (47.397971057728974, 8.546163739800146)

#: The systems PX4's airframe needs: its gz_bridge reads the IMU, barometer,
#: magnetometer and GPS, and ApplyLinkWrench is how wind reaches the aircraft.
#: (PX4's own server.config also loads optical-flow and camera plugins, which
#: nothing here uses.)
PX4_SYSTEMS = """
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-air-speed-system" name="gz::sim::systems::AirSpeed"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>"""


def _world_document(world: World, models: list[str], *, px4: bool = False) -> str:
    geo = ""
    origin = (world.origin_lat, world.origin_lon)
    if px4 and None in origin:
        origin = PX4_ORIGIN
    if None not in origin:
        geo = f"""
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>{origin[0]}</latitude_deg>
      <longitude_deg>{origin[1]}</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>"""
    if px4:
        physics = f"""
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <physics name="4ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>{PX4_SYSTEMS}"""
    else:
        physics = """
    <gravity>0 0 0</gravity>
    <physics name="playback" type="ignored">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>"""
    return f"""<?xml version="1.0"?>
<!-- {escape(world.name)}: written by gazebo/scene.py, regenerated on every launch. -->
<sdf version="1.9">
  <world name="{WORLD_NAME}">
{physics}
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.74 0.82 0.9 1</background>
      <shadows>true</shadows>
    </scene>{geo}
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 500 0 0 0</pose>
      <diffuse>0.9 0.88 0.82 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.85</direction>
    </light>{"".join(models)}
  </world>
</sdf>
"""


def _sdf_document(model: str) -> str:
    return f'<?xml version="1.0"?>\n<sdf version="1.9">{model}\n</sdf>\n'


def _material(colour: str, alpha: float = 1.0, emissive: bool = False) -> str:
    r, g, b, _ = _rgba(colour)
    rgba = f"{r} {g} {b} {alpha}"
    glow = f"<emissive>{r * 0.6:.3f} {g * 0.6:.3f} {b * 0.6:.3f} 1</emissive>" if emissive else ""
    transparency = f"<transparency>{1 - alpha:.2f}</transparency>" if alpha < 1 else ""
    return (
        f"<material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>"
        f"<specular>0.05 0.05 0.05 1</specular>{glow}</material>{transparency}"
    )


def _rgba(colour: str) -> tuple[float, float, float, float]:
    value = colour.lstrip("#")
    r, g, b = (round(int(value[i : i + 2], 16) / 255, 3) for i in (0, 2, 4))
    return (r, g, b, 1.0)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _unique(name: str, used: set[str]) -> str:
    candidate, n = name, 2
    while candidate in used or candidate in {"terrain", "trees", "route", "envelope", UAV_NAME}:
        candidate, n = f"{name}_{n}", n + 1
    used.add(candidate)
    return candidate
