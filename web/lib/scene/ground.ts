/**
 * The ground: what it is, painted onto it.
 *
 * A world's ground cover (lawns, roads, water, tea) is flat and never in the
 * way, so rather than hundreds of overlapping meshes fighting over the same
 * depth it is painted into one texture and draped over the terrain. Relief
 * colours come first (valley green up to summit grass, rock where it is
 * steep), then the cover in the order the world lists it, then a soft contact
 * shadow under everything that stands on it.
 */

import * as THREE from "three";
import type { Area, GeoPoint, Terrain, World } from "@/lib/types";
import { groundPalette, type GroundPalette } from "./palette";

/** The patch of ground a world is drawn on, in ENU metres. */
export interface Frame {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function frameFor(world: World, margin = 40): Frame {
  if (world.terrain) {
    const t = world.terrain;
    return {
      minX: t.origin_x,
      minY: t.origin_y,
      maxX: t.origin_x + (t.heights[0].length - 1) * t.spacing,
      maxY: t.origin_y + (t.heights.length - 1) * t.spacing,
    };
  }
  const points: GeoPoint[] = [{ x: 0, y: 0 }];
  for (const building of world.buildings ?? []) points.push(...building.footprint);
  for (const area of world.ground ?? []) points.push(...area.footprint);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return {
    minX: Math.min(...xs) - margin,
    minY: Math.min(...ys) - margin,
    maxX: Math.max(...xs) + margin,
    maxY: Math.max(...ys) + margin,
  };
}

/** A cheap deterministic hash, so every run of the same world paints the same. */
export function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mix(a: string, b: string, t: number): string {
  const ca = new THREE.Color(a);
  const cb = new THREE.Color(b);
  return `#${ca.lerp(cb, Math.min(Math.max(t, 0), 1)).getHexString()}`;
}

const smooth = (t: number) => {
  const c = Math.min(Math.max(t, 0), 1);
  return c * c * (3 - 2 * c);
};

/**
 * Terrain colour at every grid post: one pixel per post, smoothed when it is
 * scaled up to the texture. Flat worlds get their base colour with the faintest
 * variation, hills get a proper ramp.
 */
function paintRelief(ctx: CanvasRenderingContext2D, terrain: Terrain, palette: GroundPalette, size: [number, number]) {
  const rows = terrain.heights.length;
  const cols = terrain.heights[0].length;
  const flat = terrain.heights.flat();
  const low = Math.min(...flat);
  const high = Math.max(...flat);
  const relief = high - low;

  const posts = document.createElement("canvas");
  posts.width = cols;
  posts.height = rows;
  const pctx = posts.getContext("2d")!;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const h = terrain.heights[row][col];
      const t = relief > 0 ? (h - low) / relief : 0;
      let colour: string;
      if (relief < 40) {
        // A city on nearly flat ground: the base colour, a shade darker in the dips.
        colour = mix(palette.base, palette.valley, 0.12 + 0.12 * (1 - t));
      } else {
        const east = terrain.heights[row][Math.min(col + 1, cols - 1)] - terrain.heights[row][Math.max(col - 1, 0)];
        const north = terrain.heights[Math.min(row + 1, rows - 1)][col] - terrain.heights[Math.max(row - 1, 0)][col];
        const slope = Math.atan(Math.hypot(east, north) / (2 * terrain.spacing));
        colour = mix(palette.valley, palette.slope, smooth(t / 0.45));
        colour = mix(colour, palette.summit, smooth((t - 0.55) / 0.35));
        colour = mix(colour, palette.rock, smooth((slope - 0.45) / 0.25) * 0.8);
      }
      pctx.fillStyle = colour;
      // Row 0 is the south edge; the canvas's top is north.
      pctx.fillRect(col, rows - 1 - row, 1, 1);
    }
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  // Pixel centres sit on the posts, so the image spans half a pixel either side.
  const [w, h] = size;
  const px = w / (cols - 1);
  const py = h / (rows - 1);
  ctx.drawImage(posts, -px / 2, -py / 2, w + px, h + py);
}

/** Fine grain, so a lawn is not a flat swatch of colour. */
function paintGrain(ctx: CanvasRenderingContext2D, size: [number, number], seed: number, light: boolean) {
  const tile = document.createElement("canvas");
  tile.width = tile.height = 128;
  const tctx = tile.getContext("2d")!;
  const image = tctx.createImageData(128, 128);
  let s = seed || 1;
  for (let i = 0; i < image.data.length; i += 4) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const v = s >>> 24;
    image.data[i] = image.data[i + 1] = image.data[i + 2] = v;
    image.data[i + 3] = 255;
  }
  tctx.putImageData(image, 0, 0);
  ctx.save();
  ctx.globalAlpha = light ? 0.05 : 0.07;
  ctx.globalCompositeOperation = "overlay";
  ctx.fillStyle = ctx.createPattern(tile, "repeat")!;
  ctx.fillRect(0, 0, size[0], size[1]);
  ctx.restore();
}

export function paintGround(world: World, frame: Frame, light: boolean): THREE.CanvasTexture {
  const palette = groundPalette(light);
  const spanX = frame.maxX - frame.minX;
  const spanY = frame.maxY - frame.minY;
  const longest = 2048;
  const scale = longest / Math.max(spanX, spanY);
  const width = Math.max(2, Math.round(spanX * scale));
  const height = Math.max(2, Math.round(spanY * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = palette.base;
  ctx.fillRect(0, 0, width, height);
  if (world.terrain) paintRelief(ctx, world.terrain, palette, [width, height]);

  const toPx = (p: GeoPoint): [number, number] => [(p.x - frame.minX) * scale, (frame.maxY - p.y) * scale];
  const trace = (points: GeoPoint[]) => {
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = toPx(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
  };

  for (const area of world.ground ?? []) paintArea(ctx, area, palette, scale, trace, !!world.terrain);

  paintGrain(ctx, [width, height], hash(world.name), light);

  // Contact shadows: a soft dark bloom under every footprint, so a building
  // sits on the ground rather than hovering over it when the sun is overhead.
  // One path, one blurred fill: blurring each footprint on its own costs a
  // filter pass per building, which is seconds on a city.
  ctx.save();
  ctx.fillStyle = palette.shadow;
  ctx.filter = `blur(${Math.max(1.5, 2.5 * scale)}px)`;
  ctx.beginPath();
  for (const building of world.buildings ?? []) {
    const points = building.kind === "tree" ? building.footprint : grow(building.footprint, 1.5);
    points.forEach((p, i) => {
      const [x, y] = toPx(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
  }
  ctx.fill("nonzero");
  ctx.restore();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  return texture;
}

function grow(points: GeoPoint[], by: number): GeoPoint[] {
  const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
  return points.map((p) => {
    const dx = p.x - cx;
    const dy = p.y - cy;
    const d = Math.hypot(dx, dy) || 1;
    return { x: p.x + (dx / d) * by, y: p.y + (dy / d) * by };
  });
}

function paintArea(
  ctx: CanvasRenderingContext2D,
  area: Area,
  palette: GroundPalette,
  scale: number,
  trace: (points: GeoPoint[]) => void,
  onTerrain: boolean,
) {
  const kind = area.kind in palette.kinds ? area.kind : "lawn";
  let fill = palette.kinds[kind];
  // Water on a hillside is drawn by its own surface; what is painted under
  // it is the shore, so a bank that rises through the water reads as land.
  if (kind === "water" && onTerrain && area.level != null) fill = palette.kinds.lawn;
  trace(area.footprint);
  ctx.fillStyle = fill;
  ctx.fill();

  if (kind === "tea" || kind === "field" || kind === "rail") {
    // Rows: tea bushes along the contour, the sleepers of a track.
    ctx.save();
    ctx.clip();
    const angle = (hash(area.name + area.footprint[0].x) % 180) * (Math.PI / 180);
    const gap = kind === "tea" ? Math.max(2, 3.2 * scale) : kind === "rail" ? Math.max(2, 1.6 * scale) : Math.max(3, 9 * scale);
    ctx.strokeStyle = palette.stripe[kind];
    ctx.lineWidth = Math.max(0.8, gap * (kind === "field" ? 0.2 : 0.45));
    ctx.globalAlpha = kind === "field" ? 0.35 : 0.8;
    const [x0, y0] = [ctx.canvas.width / 2, ctx.canvas.height / 2];
    const reach = Math.hypot(ctx.canvas.width, ctx.canvas.height);
    ctx.translate(x0, y0);
    ctx.rotate(angle);
    ctx.beginPath();
    for (let d = -reach; d < reach; d += gap) {
      ctx.moveTo(-reach, d);
      ctx.lineTo(reach, d);
    }
    ctx.stroke();
    ctx.restore();
  } else if (kind === "road" && scale > 0.9) {
    // A faint kerb line, only when the texture is fine enough to show it.
    ctx.save();
    ctx.strokeStyle = palette.stripe.road;
    ctx.lineWidth = Math.max(0.6, 0.35 * scale);
    ctx.globalAlpha = 0.6;
    ctx.stroke();
    ctx.restore();
  } else if (kind === "forest") {
    ctx.save();
    ctx.globalAlpha = 0.25;
    ctx.strokeStyle = palette.kinds.forest;
    ctx.lineWidth = 4 * scale;
    ctx.stroke();
    ctx.restore();
  }
}

/** Where the surface of a body of water sits. */
export function waterLevel(area: Area, elevationAt: (x: number, y: number) => number, hasTerrain: boolean): number {
  if (area.level != null) return area.level;
  if (!hasTerrain) return 0.12;
  return Math.min(...area.footprint.map((p) => elevationAt(p.x, p.y))) + 0.1;
}

/** Flat, slightly glossy water surfaces for every body of water in the cover. */
export function waterSurfaces(
  world: World,
  elevationAt: (x: number, y: number) => number,
  light: boolean,
): THREE.Mesh[] {
  const palette = groundPalette(light);
  const meshes: THREE.Mesh[] = [];
  const material = new THREE.MeshStandardMaterial({
    color: palette.kinds.water,
    roughness: 0.18,
    metalness: 0.05,
    transparent: true,
    opacity: 0.92,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });
  for (const area of world.ground ?? []) {
    if (area.kind !== "water" || area.footprint.length < 3) continue;
    const shape = new THREE.Shape(area.footprint.map((p) => new THREE.Vector2(p.x, p.y)));
    const geometry = new THREE.ShapeGeometry(shape);
    geometry.rotateX(-Math.PI / 2);
    geometry.translate(0, waterLevel(area, elevationAt, !!world.terrain), 0);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.receiveShadow = true;
    meshes.push(mesh);
  }
  return meshes;
}

/**
 * A flat world's ground, as a slab with a little thickness: the place reads as
 * a model on a table rather than a sheet of paper floating in fog.
 */
export function groundSlab(frame: Frame, texture: THREE.Texture, light: boolean): THREE.Group {
  const palette = groundPalette(light);
  const width = frame.maxX - frame.minX;
  const depth = frame.maxY - frame.minY;
  const group = new THREE.Group();
  const top = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth),
    new THREE.MeshStandardMaterial({ map: texture, roughness: 1 }),
  );
  top.rotation.x = -Math.PI / 2;
  top.position.set(frame.minX + width / 2, 0, -(frame.minY + depth / 2));
  top.receiveShadow = true;
  const thickness = Math.max(3, Math.max(width, depth) * 0.006);
  // The body's top face is left undrawn: two coplanar surfaces a hair apart
  // z-fight into stripes at a distance, and the textured top covers it anyway.
  const sides = new THREE.MeshStandardMaterial({ color: palette.skirt, roughness: 1 });
  const hidden = new THREE.MeshBasicMaterial({ visible: false });
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(width, thickness, depth),
    [sides, sides, hidden, sides, sides, sides],
  );
  body.position.set(frame.minX + width / 2, -thickness / 2 - 0.02, -(frame.minY + depth / 2));
  group.add(top, body);
  return group;
}

/**
 * Walls down the four edges of a terrain grid, to a common floor, so hills are
 * cut like a relief model instead of ending in a ragged paper edge.
 */
export function terrainSkirt(terrain: Terrain, light: boolean): THREE.Mesh {
  const rows = terrain.heights.length;
  const cols = terrain.heights[0].length;
  const low = Math.min(...terrain.heights.flat());
  const floor = low - Math.max(4, (Math.max(...terrain.heights.flat()) - low) * 0.08);
  const at = (row: number, col: number) =>
    new THREE.Vector3(
      terrain.origin_x + col * terrain.spacing,
      terrain.heights[row][col],
      -(terrain.origin_y + row * terrain.spacing),
    );
  const edges: THREE.Vector3[][] = [
    Array.from({ length: cols }, (_, c) => at(0, c)),
    Array.from({ length: rows }, (_, r) => at(r, cols - 1)),
    Array.from({ length: cols }, (_, c) => at(rows - 1, cols - 1 - c)),
    Array.from({ length: rows }, (_, r) => at(rows - 1 - r, 0)),
  ];
  const positions: number[] = [];
  for (const edge of edges) {
    for (let i = 0; i < edge.length - 1; i++) {
      const a = edge[i];
      const b = edge[i + 1];
      positions.push(a.x, a.y, a.z, a.x, floor, a.z, b.x, floor, b.z);
      positions.push(a.x, a.y, a.z, b.x, floor, b.z, b.x, b.y, b.z);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ color: groundPalette(light).skirt, roughness: 1, side: THREE.DoubleSide }),
  );
}

/** True when the world draws its own water, so landmark discs would double it. */
export const hasWater = (world: World | null | undefined) =>
  (world?.ground ?? []).some((area) => area.kind === "water");
