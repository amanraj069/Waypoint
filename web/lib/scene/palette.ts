/**
 * One palette for the whole 3D view.
 *
 * The scene is dark so the flight data reads as instrument colour against it:
 * the ground and the buildings are desaturated, and only the things a mission
 * is actually about — the route, the fence, a hazard, a landmark — carry hue.
 */

/**
 * Read a colour the stylesheet owns.
 *
 * The scene and the chrome must agree about what "background" means, and the
 * only way to guarantee that is for both to read the same custom property
 * rather than keeping two copies of the same hex.
 */
function fromCss(name: string, fallback: number): number {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (!value) return fallback;
  const parsed = Number.parseInt(value.replace("#", ""), 16);
  return Number.isNaN(parsed) ? fallback : parsed;
}

/** The theme-dependent half of the palette, re-read whenever the theme changes. */
export function sceneColors() {
  return {
    background: fromCss("--scene-bg", 0x0a0d11),
    terrain: fromCss("--scene-terrain", 0x1b2530),
    building: fromCss("--scene-building", 0x2b3742),
    edge: fromCss("--scene-edge", 0x415162),
    canopy: fromCss("--scene-canopy", 0x2f5d3f),
    trunk: fromCss("--scene-trunk", 0x3a3026),
    label: getComputedStyle(document.documentElement).getPropertyValue("--scene-label").trim()
      || "#a8bdd0",
  };
}

export const COLORS = {
  background: 0x0a0d11,
  flown: 0x4cc2ff,
  planned: 0x8899aa,
  fence: 0x45d08a,
  nofly: 0xff5c5c,
  obstacle: 0xff7a5c,
  ghost: 0xffb020,
  terrain: 0x1b2530,
  terrainWire: 0x2b3742,
  water: 0x1d4a63,
  trunk: 0x3a3026,
  canopy: 0x2f5d3f,
  label: "#a8bdd0",
  /** The launch pad is the same green as the fence: safe, known ground. */
  pad: 0x45d08a,
  /** A landing pad somewhere else is amber, so the two are never confused. */
  landing: 0xffb020,
} as const;

/**
 * Buildings by what they are. Sandstone for the monuments, because in these
 * worlds the monument usually *is* the mission, and a red fort that renders
 * the same grey as an office block is a worse drawing of the same data.
 */
export const SURFACE: Record<string, { wall: number; edge: number }> = {
  building: { wall: 0x2b3742, edge: 0x415162 },
  house: { wall: 0x3b4046, edge: 0x51565c },
  monument: { wall: 0x6b4a34, edge: 0x9a7150 },
  pavilion: { wall: 0x3c4756, edge: 0x5b6b7e },
  mast: { wall: 0x4a4f58, edge: 0x6f7885 },
};

/** The same surfaces in daylight: sandstone stays sandstone, stone lightens. */
export const SURFACE_LIGHT: Record<string, { wall: number; edge: number }> = {
  building: { wall: 0xa9b3bb, edge: 0x9c968c },
  house: { wall: 0xe6dfd1, edge: 0xa39a8c },
  monument: { wall: 0xc4a184, edge: 0x8f7052 },
  pavilion: { wall: 0xb5bec6, edge: 0x828d97 },
  mast: { wall: 0x9ea6ad, edge: 0x767e86 },
};

export const ACTION_COLOR: Record<string, number> = {
  MOVE: 0x4c78a8,
  HOVER: 0xf58518,
  CAPTURE: 0x54a24b,
  LAND: 0xe45756,
};

export const surfaceFor = (kind: string, light = false) => {
  const table = light ? SURFACE_LIGHT : SURFACE;
  return table[kind] ?? table.building;
};

/** True when the document is showing the daylight palette. */
export const isLight = () =>
  typeof document !== "undefined" && document.documentElement.dataset.theme === "light";

/**
 * Ground cover, by what it is. Map colours rather than photographs: quiet
 * enough that the route and the fence still carry the picture, distinct enough
 * that a park reads as a park and a road as a road.
 */
export interface GroundPalette {
  /** Bare ground where nothing else is drawn. */
  base: string;
  /** The sides of the ground slab. */
  skirt: string;
  /** Terrain with real relief: valley floor, hillside, high grass, rock. */
  valley: string;
  slope: string;
  summit: string;
  rock: string;
  kinds: Record<string, string>;
  /** Tea rows, road markings: the darker stroke drawn over a fill. */
  stripe: Record<string, string>;
  shadow: string;
}

export const GROUND_LIGHT: GroundPalette = {
  base: "#dcd6c8",
  skirt: "#b9b2a3",
  valley: "#a9bf8c",
  slope: "#93ad78",
  summit: "#c3c393",
  rock: "#b3aa98",
  kinds: {
    built: "#d9d3c7",
    lawn: "#b3c796",
    garden: "#bccd9f",
    field: "#c7c998",
    tea: "#8fb070",
    forest: "#86a377",
    pitch: "#a4c287",
    flowerbed: "#cfaea3",
    sand: "#e4d7b6",
    plaza: "#e6ded0",
    parking: "#cdc8be",
    path: "#dfceac",
    road: "#b7b4ad",
    rail: "#aaa59c",
    stream: "#94b9c9",
    water: "#8fb3c4",
  },
  stripe: {
    tea: "#789c5b",
    road: "#cbc8c1",
    rail: "#8b857b",
    field: "#b9bc88",
  },
  shadow: "rgba(60, 55, 45, 0.28)",
};

export const GROUND_DARK: GroundPalette = {
  base: "#1b2127",
  skirt: "#12171c",
  valley: "#1f2d25",
  slope: "#1b2a22",
  summit: "#2a3226",
  rock: "#2b2d2f",
  kinds: {
    built: "#1f252c",
    lawn: "#20302a",
    garden: "#223329",
    field: "#28301f",
    tea: "#1f3a27",
    forest: "#1a2d22",
    pitch: "#213826",
    flowerbed: "#3a2b2e",
    sand: "#39352b",
    plaza: "#2a2f35",
    parking: "#262b31",
    path: "#34332c",
    road: "#2e343b",
    rail: "#2a2d31",
    stream: "#17344a",
    water: "#16324a",
  },
  stripe: {
    tea: "#193020",
    road: "#3a4149",
    rail: "#3d4046",
    field: "#232a1b",
  },
  shadow: "rgba(0, 0, 0, 0.35)",
};

export const groundPalette = (light: boolean) => (light ? GROUND_LIGHT : GROUND_DARK);

/**
 * Walls and roofs by kind, a few of each so a street is not one colour.
 * Each building draws one of these by its name, so it keeps its colour.
 */
export interface Finish {
  walls: number[];
  roofs: number[];
}

export const FINISH_LIGHT: Record<string, Finish> = {
  building: { walls: [0xd9d3c7, 0xcfc9bd, 0xe3ded4, 0xc8c3b9, 0xd4cdbf], roofs: [0xbdb7ac, 0xc6c1b7, 0xb0aba2] },
  house: {
    walls: [0xe6dfd1, 0xdccbb3, 0xebe5da, 0xd8cdbb, 0xd2d8d9, 0xe0d2bf],
    roofs: [0xb57555, 0x9d6d55, 0x7f8d92, 0xa75d4d, 0xc38e64, 0x8e9a7c, 0xb9a48a],
  },
  monument: { walls: [0xc9a07c, 0xc49a76, 0xd1aa85, 0xbf9470], roofs: [0xb58a66, 0xc19774] },
  pavilion: { walls: [0xdadee1, 0xd2d7da], roofs: [0xb4bbc1] },
  mast: { walls: [0xaab1b7], roofs: [0x9aa1a7] },
};

export const FINISH_DARK: Record<string, Finish> = {
  building: { walls: [0x35414d, 0x2f3a46, 0x39444f, 0x323d48], roofs: [0x404c59, 0x3b4754] },
  house: {
    walls: [0x3b4046, 0x3e3b38, 0x373e45, 0x40403c],
    roofs: [0x5c3d33, 0x4c3c35, 0x3d4952, 0x583732, 0x5b4a39, 0x44503f],
  },
  monument: { walls: [0x6b4a34, 0x654531, 0x734f37], roofs: [0x7c583e, 0x76533a] },
  pavilion: { walls: [0x3c4756, 0x404b59], roofs: [0x4a5663] },
  mast: { walls: [0x4a4f58], roofs: [0x555b64] },
};

export const finishFor = (kind: string, light: boolean): Finish => {
  const table = light ? FINISH_LIGHT : FINISH_DARK;
  return table[kind] ?? table.building;
};

export const CANOPY_LIGHT = [0x6f9460, 0x62895a, 0x7a9d62, 0x5a7f51, 0x86a468, 0x6d8f5a];
export const CANOPY_DARK = [0x2f5d3f, 0x2a5238, 0x35633f, 0x284a33, 0x3a6843];
