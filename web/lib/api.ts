/**
 * Talking to the waypoint API.
 *
 * In development `next.config.ts` rewrites /api/* onto the FastAPI server, so
 * everything here is a same-origin request and there is no CORS story to tell.
 * Point NEXT_PUBLIC_API_BASE somewhere else to talk to a remote one.
 */

import type {
  Health,
  MissionBody,
  MissionEvent,
  MissionSnapshot,
  PlanCard,
  Run,
  RunCard,
  Site,
  World,
  WorldCard,
} from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as T;
}

/**
 * FastAPI puts the useful part of an error in `detail`; anything else is a
 * server that fell over, and the status line is all we have.
 */
async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg?: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `${response.status} ${response.statusText}`;
}

export const api = {
  health: () => get<Health>("/api/health"),

  worlds: () => get<WorldCard[]>("/api/worlds"),
  world: (id: string) => get<World>(`/api/worlds/${id}`),
  /** A world shaped like a run, with nothing flown in it: somewhere to look around. */
  worldPreview: (id: string) => get<Run>(`/api/worlds/${id}/preview`),

  /** Where a pad may go. The server decides, so the view cannot disagree with the flight. */
  site: (id: string, x: number, y: number) =>
    get<Site>(`/api/worlds/${id}/site?x=${x}&y=${y}`),

  runs: (limit = 60) => get<RunCard[]>(`/api/runs?limit=${limit}`),
  run: (id: string) => get<Run>(`/api/runs/${id}`),
  deleteRun: (id: string) =>
    get<{ deleted: string }>(`/api/runs/${id}`, { method: "DELETE" }),

  plans: () => get<PlanCard[]>("/api/plans"),

  missions: () => get<MissionSnapshot[]>("/api/missions"),
  mission: (id: string) => get<MissionSnapshot>(`/api/missions/${id}`),
  startMission: (body: MissionBody) =>
    get<{ id: string; status: string }>("/api/missions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  cancelMission: (id: string) =>
    get<{ id: string; cancelled: boolean }>(`/api/missions/${id}/cancel`, { method: "POST" }),
};

/**
 * Follow a mission's event stream.
 *
 * `since` replays from an offset so a reconnect does not re-render the flight
 * from the beginning. Returns the function that closes the stream.
 */
export function followMission(
  id: string,
  handlers: {
    onEvent: (event: MissionEvent) => void;
    onClose?: () => void;
    onError?: (error: Error) => void;
  },
  since = 0,
): () => void {
  const source = new EventSource(`${API_BASE}/api/missions/${id}/events?since=${since}`);
  let closed = false;

  const shut = () => {
    if (closed) return;
    closed = true;
    source.close();
    handlers.onClose?.();
  };

  source.onmessage = (message) => {
    try {
      handlers.onEvent(JSON.parse(message.data) as MissionEvent);
    } catch {
      /* a half-delivered frame; the next one will be whole */
    }
  };
  source.addEventListener("close", shut);
  source.onerror = () => {
    // EventSource retries by itself while the mission is live. Once the server
    // has sent its close frame the connection is finished, and reconnecting
    // would only replay a mission that has already ended.
    if (source.readyState === EventSource.CLOSED) shut();
  };

  return shut;
}
