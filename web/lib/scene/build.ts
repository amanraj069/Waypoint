/**
 * Turning a run file into three.js objects.
 *
 * Everything here is imperative and framework-free: give it the JSON, get back
 * a `THREE.Group`. React mounts the result and drives it, which keeps the part
 * that has to be fast out of the render tree.
 *
 * Frames: the simulator works in ENU — x east, y north, z up — and three.js
 * has y up with z toward the viewer, so every position goes through `toScene`.
 */

import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type {
  Building,
  Constraints,
  GeoPoint,
  Obstacle,
  PlannedWaypoint,
  Run,
  Terrain,
  World,
} from "@/lib/types";
import {
  ACTION_COLOR,
  CANOPY_DARK,
  CANOPY_LIGHT,
  COLORS,
  finishFor,
  isLight,
  sceneColors,
  surfaceFor,
} from "./palette";
import {
  frameFor,
  groundSlab,
  hasWater,
  hash,
  paintGround,
  terrainSkirt,
  waterSurfaces,
} from "./ground";

/**
 * Which palette the objects being built are being coloured for.
 *
 * Materials bake their colour in at build time, so the theme has to be known
 * while building rather than looked up later. `buildScene` sets this once from
 * its argument, which is what makes the theme an honest dependency of the
 * memo that calls it instead of an invisible read of the DOM.
 */
let building_light = false;

export const toScene = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);

/** Bilinear sample of the height grid — the same one the Python side uses, so
 *  a label sits on the ground the verifier checked rather than above it. */
export function elevationAt(terrain: Terrain | null | undefined, x: number, y: number): number {
  if (!terrain) return 0;
  const rows = terrain.heights.length;
  const cols = terrain.heights[0].length;
  let col = (x - terrain.origin_x) / terrain.spacing;
  let row = (y - terrain.origin_y) / terrain.spacing;
  col = Math.min(Math.max(col, 0), cols - 1);
  row = Math.min(Math.max(row, 0), rows - 1);
  const c0 = Math.floor(col);
  const r0 = Math.floor(row);
  const c1 = Math.min(c0 + 1, cols - 1);
  const r1 = Math.min(r0 + 1, rows - 1);
  const fc = col - c0;
  const fr = row - r0;
  const top = terrain.heights[r0][c0] * (1 - fc) + terrain.heights[r0][c1] * fc;
  const bottom = terrain.heights[r1][c0] * (1 - fc) + terrain.heights[r1][c1] * fc;
  return top * (1 - fr) + bottom * fr;
}

// ---------------------------------------------------------------------------
// Scenery
// ---------------------------------------------------------------------------

function terrainMesh(terrain: Terrain, texture: THREE.Texture | null): THREE.Group {
  const rows = terrain.heights.length;
  const cols = terrain.heights[0].length;
  const width = (cols - 1) * terrain.spacing;
  const depth = (rows - 1) * terrain.spacing;
  const geometry = new THREE.PlaneGeometry(width, depth, cols - 1, rows - 1);
  const position = geometry.attributes.position;

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      // PlaneGeometry runs top-left to bottom-right; the grid runs south to north.
      position.setZ(row * cols + col, terrain.heights[rows - 1 - row][col]);
    }
  }
  geometry.computeVertexNormals();

  // The ground cover is painted into `texture`; smooth shading lets the light
  // model the hills instead of breaking them into triangles.
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial(
      texture
        ? { map: texture, roughness: 1 }
        : { color: sceneColors().terrain, roughness: 0.95 },
    ),
  );
  mesh.rotation.x = -Math.PI / 2;
  // The grid's own origin is its south-west corner; centre it in the scene.
  mesh.position.set(terrain.origin_x + width / 2, 0, -(terrain.origin_y + depth / 2));
  mesh.userData.isTerrain = true;   // what a "pick a starting point" click lands on
  mesh.receiveShadow = true;

  const group = new THREE.Group();
  group.add(mesh, terrainSkirt(terrain, building_light));
  return group;
}

/** The ground mesh inside a built scene, for raycasting a dragged pad onto. */
function findTerrain(root: THREE.Object3D): THREE.Mesh | null {
  let found: THREE.Mesh | null = null;
  root.traverse((object) => {
    if (!found && object.userData.isTerrain) found = object as THREE.Mesh;
  });
  return found;
}

/**
 * The ground across a pad-sized patch, for live feedback while dragging.
 *
 * `world/landing.py` is the authority — it decides what is usable and its
 * answer is what the flight uses. This is the same measurement done locally so
 * the pad can turn red under your pointer instead of after a round trip.
 */
export const PAD_RADIUS = 3;
export const MAX_PAD_SLOPE_DEG = 8;
export const MAX_PAD_DROP = 0.9;

export function probeGround(terrain: Terrain | null | undefined, x: number, y: number) {
  const centre = elevationAt(terrain, x, y);
  let highest = centre;
  let lowest = centre;
  let slope = 0;
  for (let step = 0; step < 8; step++) {
    const angle = (step * Math.PI) / 4;
    const height = elevationAt(
      terrain,
      x + PAD_RADIUS * Math.cos(angle),
      y + PAD_RADIUS * Math.sin(angle),
    );
    highest = Math.max(highest, height);
    lowest = Math.min(lowest, height);
    slope = Math.max(slope, (Math.atan2(Math.abs(height - centre), PAD_RADIUS) * 180) / Math.PI);
  }
  const drop = highest - lowest;
  return { elevation: centre, slope, drop, usable: slope <= MAX_PAD_SLOPE_DEG && drop <= MAX_PAD_DROP };
}

/**
 * Every building of one kind, merged into a single mesh.
 *
 * A city is hundreds of buildings, and drawing each as its own mesh plus its
 * own edge lines was hundreds of draw calls a frame and hundreds of
 * `EdgesGeometry` builds on load. They share a material, so they can share a
 * buffer: two draw calls per kind instead of two per building.
 */
function mergedBuildings(buildings: Building[]): THREE.Object3D[] {
  const byKind = new Map<string, Building[]>();
  for (const building of buildings) {
    const kind = building.kind === "tree" ? "tree" : building.kind || "building";
    (byKind.get(kind) ?? byKind.set(kind, []).get(kind)!).push(building);
  }

  const objects: THREE.Object3D[] = [];
  for (const [kind, group] of byKind) {
    if (kind === "tree") {
      objects.push(...mergedTrees(group));
      continue;
    }
    const solids: THREE.BufferGeometry[] = [];
    const outlines: THREE.BufferGeometry[] = [];
    const finish = finishFor(kind, building_light);
    for (const building of group) {
      const geometry = buildingGeometry(building);
      paintBuilding(geometry, building, finish);
      solids.push(geometry);
      outlines.push(new THREE.EdgesGeometry(geometry, 25));
    }
    const surface = surfaceFor(kind, building_light);
    const walls = mergeGeometries(solids, false);
    for (const geometry of solids) geometry.dispose();
    if (walls) {
      const mesh = new THREE.Mesh(
        walls,
        new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.85,
          metalness: 0.02,
          flatShading: true,
        }),
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      objects.push(mesh);
    }
    const edges = mergeGeometries(outlines, false);
    for (const geometry of outlines) geometry.dispose();
    if (edges) {
      objects.push(
        new THREE.LineSegments(
          edges,
          new THREE.LineBasicMaterial({
            color: surface.edge,
            transparent: true,
            opacity: kind === "house" ? 0.22 : 0.38,
          }),
        ),
      );
    }
  }
  return objects;
}

/**
 * Colour one building's vertices: a wall and a roof drawn from its kind's
 * finishes by its name, so it keeps its colour between visits, and walls a
 * shade darker at the foot, which is where the light really is dimmer.
 */
function paintBuilding(
  geometry: THREE.BufferGeometry,
  building: Building,
  finish: { walls: number[]; roofs: number[] },
): void {
  const seed = hash(building.name + building.footprint[0].x.toFixed(1));
  const wall = new THREE.Color(finish.walls[seed % finish.walls.length]);
  const roof = new THREE.Color(finish.roofs[(seed >>> 8) % finish.roofs.length]);
  // A few percent of lightness either way: a terrace, not a row of clones.
  const jitter = 1 + (((seed >>> 16) % 100) / 100 - 0.5) * 0.08;
  wall.multiplyScalar(jitter);
  roof.multiplyScalar(jitter);

  const position = geometry.attributes.position;
  const normal = geometry.attributes.normal;
  const colors = new Float32Array(position.count * 3);
  const shade = new THREE.Color();
  for (let i = 0; i < position.count; i++) {
    if (normal.getY(i) > 0.5) {
      shade.copy(roof);
    } else {
      const up = Math.min(Math.max((position.getY(i) - building.base) / Math.max(building.height, 1), 0), 1);
      shade.copy(wall).multiplyScalar(0.8 + 0.2 * Math.min(1, up * 1.6));
    }
    colors[i * 3] = shade.r;
    colors[i * 3 + 1] = shade.g;
    colors[i * 3 + 2] = shade.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

/** Tall narrow trees are drawn as columns: ashoka, cypress, silver oak, gums. */
const COLUMNAR = /ashoka|cypress|silver oak|eucalyptus|maulsari/i;

function tint(geometry: THREE.BufferGeometry, color: THREE.Color, bottom: number, top: number): void {
  const position = geometry.attributes.position;
  const colors = new Float32Array(position.count * 3);
  const shade = new THREE.Color();
  for (let i = 0; i < position.count; i++) {
    const up = Math.min(Math.max((position.getY(i) - bottom) / Math.max(top - bottom, 0.1), 0), 1);
    shade.copy(color).multiplyScalar(0.72 + 0.34 * up);
    colors[i * 3] = shade.r;
    colors[i * 3 + 1] = shade.g;
    colors[i * 3 + 2] = shade.b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

/** Trunks in one mesh, canopies in another. */
function mergedTrees(trees: Building[]): THREE.Object3D[] {
  const trunks: THREE.BufferGeometry[] = [];
  const canopies: THREE.BufferGeometry[] = [];

  for (const tree of trees) {
    const xs = tree.footprint.map((p) => p.x);
    const ys = tree.footprint.map((p) => p.y);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const spread = Math.max((Math.max(...xs) - Math.min(...xs)) / 2, 1.5);
    const seed = hash(tree.name + cx.toFixed(1));
    const columnar = COLUMNAR.test(tree.name);
    const trunkHeight = tree.height * (columnar ? 0.25 : 0.4);
    const at = toScene(cx, cy, tree.base);

    const trunk = new THREE.CylinderGeometry(spread * 0.1, spread * 0.16, trunkHeight, 5);
    trunk.translate(at.x, at.y + trunkHeight / 2, at.z);
    trunks.push(trunk);

    const crown = tree.height - trunkHeight;
    const palette = building_light ? CANOPY_LIGHT : CANOPY_DARK;
    const color = new THREE.Color(palette[seed % palette.length]);
    let canopy: THREE.BufferGeometry;
    if (columnar) {
      canopy = new THREE.ConeGeometry(spread * 0.85, crown, 7, 1).toNonIndexed();
      canopy.deleteAttribute("uv");
      canopy.translate(at.x, at.y + trunkHeight + crown / 2, at.z);
    } else {
      // Detail 1 is round enough to read as a crown and still a handful of
      // triangles; a small random turn stops a row of them looking stamped.
      canopy = new THREE.IcosahedronGeometry(spread, 1);
      canopy.deleteAttribute("uv");
      canopy.rotateY(((seed >>> 8) % 360) * (Math.PI / 180));
      canopy.scale(1, crown / (2 * spread), 1);
      canopy.translate(at.x, at.y + trunkHeight + crown * 0.5, at.z);
    }
    tint(canopy, color, at.y + trunkHeight, at.y + tree.height);
    canopies.push(canopy);
  }

  const objects: THREE.Object3D[] = [];
  for (const geometry of trunks) geometry.deleteAttribute("uv");
  const trunkGeometry = mergeGeometries(trunks.map((g) => g.toNonIndexed()), false);
  for (const geometry of trunks) geometry.dispose();
  if (trunkGeometry) {
    const mesh = new THREE.Mesh(
      trunkGeometry,
      new THREE.MeshStandardMaterial({ color: sceneColors().trunk, roughness: 1 }),
    );
    mesh.castShadow = true;
    objects.push(mesh);
  }
  const canopyGeometry = mergeGeometries(canopies, false);
  for (const geometry of canopies) geometry.dispose();
  if (canopyGeometry) {
    const mesh = new THREE.Mesh(
      canopyGeometry,
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.95,
        flatShading: true,
      }),
    );
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    objects.push(mesh);
  }
  return objects;
}

function buildingGeometry(building: Building): THREE.BufferGeometry {
  // The footprint goes in as (x, y) in ENU. Extrude builds it in the XY plane
  // and rotateX(-90) maps (x, y, h) -> (x, h, -y), which is exactly toScene —
  // so do NOT flip y here, or the building lands on the mirror image of its own
  // plot and floats above whatever ground is on the other side.
  const shape = new THREE.Shape(building.footprint.map((p) => new THREE.Vector2(p.x, p.y)));
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: building.height,
    bevelEnabled: false,
  });
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, building.base, 0);
  return geometry;
}

export function labelSprite(text: string, color?: string): THREE.Sprite {
  const pad = 12;
  const font = 30;
  const measure = document.createElement("canvas").getContext("2d")!;
  measure.font = `${font}px ui-monospace, Menlo, monospace`;
  const width = Math.ceil(measure.measureText(text).width) + pad * 2;
  const height = font + pad * 2;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  // A name plate has to stay readable against whatever is behind it, which in
  // daylight is a pale hillside and at night is nearly black.
  const light = building_light;
  ctx.fillStyle = light ? "rgba(251, 250, 248, 0.82)" : "rgba(10, 13, 16, 0.76)";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = light ? "rgba(13, 110, 168, 0.45)" : "rgba(76, 194, 255, 0.4)";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  ctx.font = `${font}px ui-monospace, Menlo, monospace`;
  ctx.fillStyle = color ?? sceneColors().label;
  ctx.textBaseline = "middle";
  ctx.fillText(text, pad, height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }),
  );
  const scale = 0.55;
  sprite.scale.set((width / font) * scale * 8, (height / font) * scale * 8, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function landmarkMarkers(world: World): THREE.Group {
  const group = new THREE.Group();
  // A world with its own water draws it properly; a disc on top would double it.
  const drawnWater = hasWater(world);
  for (const landmark of world.landmarks ?? []) {
    const ground = elevationAt(world.terrain, landmark.x, landmark.y);

    if (landmark.kind === "water" && drawnWater) {
      // nothing: the lake is already there
    } else if (landmark.kind === "water") {
      const water = new THREE.Mesh(
        new THREE.CircleGeometry(landmark.radius, 48),
        new THREE.MeshStandardMaterial({
          color: COLORS.water,
          transparent: true,
          opacity: 0.6,
          roughness: 0.2,
          metalness: 0.35,
        }),
      );
      water.rotation.x = -Math.PI / 2;
      water.position.copy(toScene(landmark.x, landmark.y, ground + 0.4));
      group.add(water);
    } else {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(landmark.radius * 0.94, landmark.radius, 48),
        new THREE.MeshBasicMaterial({
          color: COLORS.flown,
          transparent: true,
          opacity: 0.22,
          side: THREE.DoubleSide,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.copy(toScene(landmark.x, landmark.y, ground + 0.3));
      group.add(ring);
    }

    const label = labelSprite(landmark.name);
    label.position.copy(toScene(landmark.x, landmark.y, ground + (landmark.height || 0) + 14));
    label.userData.isLabel = true;
    group.add(label);
  }
  return group;
}

function scenery(world: World): THREE.Group {
  const group = new THREE.Group();
  const frame = frameFor(world);
  const texture = paintGround(world, frame, building_light);
  if (world.terrain) group.add(terrainMesh(world.terrain, texture));
  else group.add(groundSlab(frame, texture, building_light));
  for (const water of waterSurfaces(world, (x, y) => elevationAt(world.terrain, x, y), building_light)) {
    group.add(water);
  }
  for (const object of mergedBuildings(world.buildings ?? [])) group.add(object);
  group.add(landmarkMarkers(world));
  return group;
}

// ---------------------------------------------------------------------------
// The mission drawn over the place
// ---------------------------------------------------------------------------

function polygonOutline(
  points: GeoPoint[],
  color: number,
  height: number,
  opacity: number,
): THREE.Group {
  const group = new THREE.Group();
  const loop = [...points, points[0]];
  const material = new THREE.LineBasicMaterial({ color });
  group.add(
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(loop.map((p) => toScene(p.x, p.y, 0))),
      material,
    ),
  );
  group.add(
    new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(loop.map((p) => toScene(p.x, p.y, height))),
      material,
    ),
  );

  // Translucent walls, so the boundary reads as a volume from any angle.
  const positions: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    const a0 = toScene(a.x, a.y, 0);
    const b0 = toScene(b.x, b.y, 0);
    const a1 = toScene(a.x, a.y, height);
    const b1 = toScene(b.x, b.y, height);
    positions.push(...a0.toArray(), ...b0.toArray(), ...b1.toArray());
    positions.push(...a0.toArray(), ...b1.toArray(), ...a1.toArray());
  }
  const wall = new THREE.BufferGeometry();
  wall.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  wall.computeVertexNormals();
  group.add(
    new THREE.Mesh(
      wall,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    ),
  );
  return group;
}

function plannedRoute(waypoints: PlannedWaypoint[]): THREE.Group {
  const group = new THREE.Group();
  if (!waypoints.length) return group;

  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      toScene(0, 0, 0),
      ...waypoints.map((w) => toScene(w.x, w.y, w.z)),
    ]),
    new THREE.LineDashedMaterial({
      color: COLORS.planned,
      dashSize: 3,
      gapSize: 2,
      transparent: true,
      opacity: 0.8,
    }),
  );
  line.computeLineDistances();
  group.add(line);

  for (const waypoint of waypoints) {
    const color = ACTION_COLOR[waypoint.action] ?? 0x888888;
    const marker = new THREE.Mesh(
      new THREE.OctahedronGeometry(1.8),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35 }),
    );
    marker.position.copy(toScene(waypoint.x, waypoint.y, waypoint.z));
    marker.userData = { waypointIndex: waypoint.index };
    group.add(marker);
    group.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          toScene(waypoint.x, waypoint.y, 0),
          toScene(waypoint.x, waypoint.y, waypoint.z),
        ]),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.25 }),
      ),
    );
  }
  return group;
}

function obstacleMesh(obstacle: Obstacle): THREE.Mesh {
  const height = obstacle.height ?? 45;
  const radius = Math.max(obstacle.radius ?? 1, 0.8);
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, height, 24, 1, true),
    new THREE.MeshBasicMaterial({
      color: COLORS.obstacle,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  mesh.position.copy(toScene(obstacle.x, obstacle.y, height / 2));
  mesh.userData = { appearedAt: obstacle.appeared_at ?? 0, name: obstacle.name };
  return mesh;
}

/**
 * The launch pad: where the aircraft stands before it flies.
 *
 * It belongs to the view rather than to the world, because you can move it --
 * picking a different starting point should not rebuild a city.
 */
/**
 * The launch and landing pads.
 *
 * These are handles, not decoration: you drag them, so they have to be easy to
 * see from any angle and easy to hit with a pointer. A disc on the ground says
 * where, a beam says where even when the ground is behind a building, and a
 * name plate says which one you are looking at. The whole thing is one group so
 * moving it is one assignment.
 */
export function buildPad(color: number = COLORS.pad, label?: string): THREE.Group {
  const group = new THREE.Group();

  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(5, 32),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, depthWrite: false }),
  );
  disc.rotation.x = -Math.PI / 2;
  disc.position.y = 0.25;

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(7.5, 9, 48),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.75,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.3;

  // Drawn through whatever is in front of it, because a pad you cannot find is
  // a pad you cannot drag.
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.9, 0.9, 46, 8, 1, true),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.32,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  beam.position.y = 23;
  beam.renderOrder = 6;

  const cap = new THREE.Mesh(
    new THREE.OctahedronGeometry(2.4),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, depthTest: false }),
  );
  cap.position.y = 46;
  cap.renderOrder = 7;

  // A generous invisible cylinder: the thing the pointer actually hits, so you
  // do not have to catch a 1 m mast to pick the pad up.
  const grab = new THREE.Mesh(
    new THREE.CylinderGeometry(9, 9, 50, 10, 1, true),
    new THREE.MeshBasicMaterial({ visible: false }),
  );
  grab.position.y = 25;
  grab.userData.isGrab = true;

  group.add(disc, ring, beam, cap, grab);
  if (label) {
    const plate = labelSprite(label);
    plate.position.y = 52;
    group.add(plate);
  }
  group.renderOrder = 5;
  return group;
}

/**
 * Recolour a pad in place — used to warn while it is being dragged.
 *
 * Only the solid parts: tinting the name plate would tint its texture, and the
 * invisible grab cylinder has nothing to show.
 */
export function tintPad(pad: THREE.Group, color: number): void {
  pad.traverse((object) => {
    if (!(object as THREE.Mesh).isMesh || object.userData.isGrab) return;
    const material = (object as THREE.Mesh).material as THREE.MeshBasicMaterial | undefined;
    material?.color?.setHex(color);
  });
}


// ---------------------------------------------------------------------------
// The whole thing
// ---------------------------------------------------------------------------

export interface BuiltScene {
  group: THREE.Group;
  /** The obstacles that appear mid-flight, so playback can reveal them on cue. */
  obstacles: THREE.Mesh[];
  /** The flown path, drawn only as far as the playhead. */
  flown: THREE.Line | null;
  /** Half-width of everything, including the fence: sets the fog and the far plane. */
  extent: number;
  /** The middle of what there is to look at, in scene coordinates. */
  center: THREE.Vector3;
  /** How far the content reaches from that middle: what the camera frames on. */
  focus: number;
  /** The ground mesh, if this world has terrain: what a dragged pad follows. */
  terrain: THREE.Mesh | null;
}

export function buildScene(run: Run, theme: "light" | "dark" = isLight() ? "light" : "dark"): BuiltScene {
  building_light = theme === "light";
  const group = new THREE.Group();
  const constraints: Partial<Constraints> = run.constraints ?? {};
  const fenceHeight = constraints.max_altitude ?? 80;
  const fence = constraints.geofence ?? [];
  const world = run.world;

  // Two different questions. The fence and the terrain say how far the scene
  // reaches, which is what the fog and the far plane need. What the camera
  // should frame is where the *content* is: a vista whose buildings sit at one
  // end of a 1.3 km box should not be looked at from the middle of the box.
  const reaches = [
    60,
    ...fence.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y))),
    ...(run.waypoints ?? []).map((w) => Math.max(Math.abs(w.x), Math.abs(w.y))),
  ];
  const content: { x: number; y: number }[] = [{ x: 0, y: 0 }];
  for (const building of world?.buildings ?? []) {
    for (const point of building.footprint) {
      reaches.push(Math.max(Math.abs(point.x), Math.abs(point.y)));
      content.push(point);
    }
  }
  for (const waypoint of run.waypoints ?? []) content.push(waypoint);
  for (const landmark of world?.landmarks ?? []) content.push(landmark);
  const extent = Math.max(...reaches);

  const xs = content.map((p) => p.x);
  const ys = content.map((p) => p.y);
  const midX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const midY = (Math.min(...ys) + Math.max(...ys)) / 2;
  const focus = Math.max(
    60,
    (Math.max(...xs) - Math.min(...xs)) / 2,
    (Math.max(...ys) - Math.min(...ys)) / 2,
  );
  const center = toScene(midX, midY, 0);

  // Real ground replaces the abstract grid; only a run with no world at all
  // still needs something to stand the flight on.
  if (!world) {
    group.add(
      new THREE.GridHelper(extent * 2, Math.max(8, Math.round(extent / 10)), 0x2a3947, 0x18242f),
    );
  }
  if (world) group.add(scenery(world));

  if (fence.length >= 3) group.add(polygonOutline(fence, COLORS.fence, fenceHeight, 0.05));
  for (const zone of constraints.no_fly_zones ?? []) {
    if (zone.length >= 3) group.add(polygonOutline(zone, COLORS.nofly, fenceHeight * 0.8, 0.13));
  }
  group.add(plannedRoute(run.waypoints ?? []));

  const obstacles: THREE.Mesh[] = [];
  for (const obstacle of run.obstacles ?? []) {
    const mesh = obstacleMesh(obstacle);
    obstacles.push(mesh);
    group.add(mesh);
  }

  // The flown path is drawn up to the playhead, so it draws itself as it flies.
  // A world preview has no path yet.
  const ticks = run.ticks ?? [];
  let flown: THREE.Line | null = null;
  if (ticks.length) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(ticks.length * 3);
    ticks.forEach((tick, i) => {
      const v = toScene(tick.x, tick.y, tick.z);
      positions[i * 3] = v.x;
      positions[i * 3 + 1] = v.y;
      positions[i * 3 + 2] = v.z;
    });
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    flown = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: COLORS.flown }));
    flown.geometry.setDrawRange(0, 1);
    group.add(flown);
  }

  return { group, obstacles, flown, extent, center, focus, terrain: findTerrain(group) };
}

export function buildDrone(): { drone: THREE.Group; rotors: THREE.Mesh[] } {
  const drone = new THREE.Group();
  drone.add(
    new THREE.Mesh(
      new THREE.BoxGeometry(2.6, 0.9, 2.6),
      new THREE.MeshStandardMaterial({ color: 0xdfe8f0, metalness: 0.4, roughness: 0.4 }),
    ),
  );
  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(0.55, 1.8, 12),
    new THREE.MeshStandardMaterial({ color: COLORS.flown, emissive: 0x1d5f80 }),
  );
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 2.0;
  drone.add(nose);

  const rotors: THREE.Mesh[] = [];
  for (const [dx, dz] of [
    [2, 2],
    [2, -2],
    [-2, 2],
    [-2, -2],
  ]) {
    const arm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.18, 2.9, 6),
      new THREE.MeshStandardMaterial({ color: 0x6b7c8d }),
    );
    arm.position.set(dx / 2, 0, dz / 2);
    arm.rotation.set(Math.PI / 2, 0, Math.atan2(dx, dz));
    drone.add(arm);

    const rotor = new THREE.Mesh(
      new THREE.CylinderGeometry(1.5, 1.5, 0.12, 20),
      new THREE.MeshStandardMaterial({ color: 0x9fd8ff, transparent: true, opacity: 0.55 }),
    );
    rotor.position.set(dx, 0.45, dz);
    drone.add(rotor);
    rotors.push(rotor);
  }
  // Its shadow on the ground is the best altitude cue a 3D view has.
  drone.traverse((part) => {
    part.castShadow = true;
  });
  return { drone, rotors };
}

export function buildGhost(): { ghost: THREE.Mesh; link: THREE.Line } {
  const ghost = new THREE.Mesh(
    new THREE.SphereGeometry(1.6, 16, 12),
    new THREE.MeshBasicMaterial({
      color: COLORS.ghost,
      transparent: true,
      opacity: 0.55,
      wireframe: true,
    }),
  );
  const link = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
    new THREE.LineDashedMaterial({ color: COLORS.ghost, dashSize: 1.5, gapSize: 1.2 }),
  );
  ghost.visible = false;
  link.visible = false;
  return { ghost, link };
}

/** Free the GPU memory a built scene holds; React will not do it for us. */
export function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    const mesh = object as THREE.Mesh & { material?: THREE.Material | THREE.Material[] };
    mesh.geometry?.dispose?.();
    const material = mesh.material;
    for (const one of Array.isArray(material) ? material : material ? [material] : []) {
      const textured = one as THREE.Material & { map?: THREE.Texture | null };
      textured.map?.dispose?.();
      one.dispose();
    }
  });
}
