/**
 * The shapes the API hands back.
 *
 * These mirror the pydantic models on the Python side rather than restating
 * them: a run file is written by `orchestrator/runlog.py` and read here, so if
 * one of these drifts the other is the thing to fix.
 */

export type Action = "MOVE" | "HOVER" | "CAPTURE" | "LAND";

export type Vec3 = [number, number, number];

export interface GeoPoint {
  x: number;
  y: number;
}

export interface Obstacle {
  x: number;
  y: number;
  radius: number;
  height: number | null;
  name: string;
  appeared_at?: number;
}

export interface Constraints {
  geofence: GeoPoint[];
  max_altitude: number;
  min_altitude: number;
  max_velocity: number;
  battery_budget_pct: number;
  no_fly_zones: GeoPoint[][];
  obstacles: Obstacle[];
  min_separation: number;
}

export interface Terrain {
  origin_x: number;
  origin_y: number;
  spacing: number;
  heights: number[][];
}

export interface Building {
  footprint: GeoPoint[];
  height: number;
  base: number;
  name: string;
  /** building, tree, mast, pavilion, monument: how it is drawn and read. */
  kind: string;
}

export interface Landmark {
  name: string;
  x: number;
  y: number;
  radius: number;
  height: number;
  kind: string;
  note: string;
}

/**
 * Ground cover: what the ground is somewhere: a lawn, a road, a lake. Flat,
 * drawn onto the ground, and never an obstacle.
 */
export interface Area {
  footprint: GeoPoint[];
  kind: string;
  name: string;
  /** For water: the surface's elevation, when the world knows it. */
  level?: number | null;
}

export interface World {
  name: string;
  source: string;
  origin_lat: number | null;
  origin_lon: number | null;
  terrain: Terrain | null;
  buildings: Building[];
  landmarks: Landmark[];
  /** Older run files predate ground cover, so it may be missing. */
  ground?: Area[];
}

/** The card a picker shows: enough to choose a place without loading it. */
export interface WorldCard {
  id: string;
  name: string;
  source: string;
  blurb: string;
  buildings: number;
  trees: number;
  landmarks: string[];
  tallest_roof: number;
  elevation_min: number;
  elevation_max: number;
  bounds: [number, number, number, number];
  has_terrain: boolean;
  origin: { lat: number; lon: number } | null;
  built_in: boolean;
}

export interface PlannedWaypoint {
  subtask: string;
  index: number;
  x: number;
  y: number;
  z: number;
  speed: number;
  action: Action;
}

export interface Tick {
  timestamp: number;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  velocity: number;
  heading: number;
  battery_pct: number;
  target_index: number | null;
  subtask: string;
  ground?: number;
  agl?: number;
  gps_offset?: Vec3;
  reported?: Vec3;
  wind?: Vec3;
  faults?: string[];
}

export interface FlightEvent {
  t: number;
  type: string;
  description: string;
  payload?: Record<string, unknown>;
}

export interface Violation {
  type: string;
  severity: "WARNING" | "ERROR" | "CRITICAL";
  waypoint_index: number | null;
  description: string;
}

/**
 * A run file, or a world preview, which is the same thing with no ticks in it,
 * so looking around a place and replaying a flight are one code path.
 */
export interface Run {
  schema_version?: number;
  mission_text: string;
  created_at?: string;
  preview?: boolean;
  constraints: Constraints | null;
  constraint_notes?: string[];
  verification?: { passed: boolean; violations: Violation[] };
  replans?: number;
  completed?: boolean;
  sim?: {
    wind: { vector: Vec3; variation: Vec3; period: number; turbulence: number };
    distance_flown_m: number;
    crashed: boolean;
    duration_s: number;
    battery_remaining_pct: number;
  };
  world: World | null;
  faults?: unknown[];
  events: FlightEvent[];
  obstacles: Obstacle[];
  waypoints: PlannedWaypoint[];
  ticks: Tick[];
}

export interface RunCard {
  id: string;
  mission_text: string;
  created_at: string;
  world: string | null;
  duration_s: number | null;
  distance_m: number | null;
  battery_pct: number | null;
  completed: boolean | null;
  crashed: boolean;
  replans: number;
  verified: boolean | null;
  ticks: number;
  waypoints: number;
  captures: number;
  phases: string[];
  flyable: boolean;
}

export interface PlanCard {
  id: string;
  mission_text: string;
  world: string | null;
  subtasks: string[];
  waypoints: number;
}

export interface Health {
  ok: boolean;
  planner_ready: boolean;
  planner_model: string;
  stages: Stage[];
  worlds: number;
  runs: number;
  plans: number;
}

export type Stage =
  | "queued"
  | "planning"
  | "verifying"
  | "replanning"
  | "flying"
  | "done"
  | "failed";

/** One line of a mission's log. `kind` says which of the shapes below it is. */
export interface MissionEvent {
  seq: number;
  t: number;
  kind:
    | "stage"
    | "world"
    | "plan"
    | "verification"
    | "telemetry"
    | "flight"
    | "note"
    | "end"
    | "heartbeat"
    | "close";
  message: string;
  stage?: Stage;
  // world
  world?: { name: string; buildings: number; trees: number; tallest_roof: number };
  landmarks?: Landmark[];
  constraints?: Constraints;
  notes?: string[];
  // plan
  phases?: { name: string; waypoints: PlannedWaypoint[] }[];
  attempt?: number;
  // verification
  passed?: boolean;
  violations?: Violation[];
  // telemetry
  x?: number;
  y?: number;
  z?: number;
  agl?: number;
  speed?: number;
  battery?: number;
  heading?: number;
  subtask?: string;
  waypoint?: number | null;
  of?: number;
  faults?: string[];
  // flight + end
  type?: string;
  run?: string | null;
  outcome?: string;
  duration_s?: number;
  distance_m?: number;
  battery_pct?: number;
  completed?: boolean;
  crashed?: boolean;
}

export interface MissionSnapshot {
  id: string;
  status: Stage;
  mission: string;
  plan_id: string | null;
  world: string | null;
  run: string | null;
  error: string | null;
  created_at: number;
  events: MissionEvent[];
}

/** What the mission box submits. */
export interface MissionBody {
  mission?: string;
  plan?: string | null;
  world?: string | null;
  wind?: Vec3;
  turbulence?: number;
  faults?: string;
  retries?: number;
  max_altitude?: number | null;
  blind?: boolean;
  /** Where to launch from, in metres east/north. Omit for the origin. */
  home?: [number, number] | null;
  /** Where to put down. Omit to land back where it started. */
  landing?: [number, number] | null;
}

/** A surveyed patch of ground: can the aircraft stand here, and if not, where? */
export interface Site {
  x: number;
  y: number;
  elevation: number;
  slope_deg: number;
  drop: number;
  obstructed: string | null;
  moved: number;
  usable: boolean;
  reason: string;
  asked: { x: number; y: number; usable: boolean; why_not: string };
}
