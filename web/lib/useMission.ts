"use client";

/**
 * Starting a mission and watching it happen.
 *
 * The server streams one line at a time; this folds those lines into the shape
 * the page draws — which stage we are in, what the planner came back with, what
 * the verifier said about it, and where the aircraft is now.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, followMission } from "./api";
import type {
  MissionBody,
  MissionEvent,
  PlannedWaypoint,
  Stage,
  Violation,
} from "./types";

export interface Phase {
  name: string;
  waypoints: PlannedWaypoint[];
}

export interface LogLine {
  seq: number;
  t: number;
  kind: MissionEvent["kind"];
  type?: string;
  message: string;
}

export interface MissionState {
  id: string | null;
  status: Stage | "idle";
  message: string;
  phases: Phase[];
  attempt: number;
  verification: { passed: boolean; violations: Violation[]; attempt: number } | null;
  telemetry: MissionEvent | null;
  log: LogLine[];
  notes: string[];
  runId: string | null;
  error: string | null;
}

const IDLE: MissionState = {
  id: null,
  status: "idle",
  message: "",
  phases: [],
  attempt: 0,
  verification: null,
  telemetry: null,
  log: [],
  notes: [],
  runId: null,
  error: null,
};

/** A flight can be hundreds of lines; keep the tail rather than all of it. */
const LOG_LIMIT = 300;

function reduce(state: MissionState, event: MissionEvent): MissionState {
  const next = { ...state };
  switch (event.kind) {
    case "stage":
      next.status = event.stage ?? state.status;
      next.message = event.message;
      if (event.attempt !== undefined) next.attempt = event.attempt;
      break;
    case "world":
      next.notes = event.notes ?? [];
      next.message = event.message;
      break;
    case "plan":
      next.phases = event.phases ?? [];
      break;
    case "verification":
      next.verification = {
        passed: Boolean(event.passed),
        violations: event.violations ?? [],
        attempt: event.attempt ?? 0,
      };
      break;
    case "telemetry":
      next.telemetry = event;
      return next; // telemetry is not worth a log line; there are thousands
    case "end":
      next.status = event.stage ?? "done";
      next.message = event.message;
      next.runId = event.run ?? null;
      if (event.stage === "failed") next.error = event.message;
      break;
    default:
      break;
  }
  if (event.kind !== "heartbeat" && event.kind !== "close" && event.message) {
    next.log = [
      ...state.log,
      { seq: event.seq, t: event.t, kind: event.kind, type: event.type, message: event.message },
    ].slice(-LOG_LIMIT);
  }
  return next;
}

export function useMission(onFinished?: (runId: string) => void) {
  const [state, setState] = useState<MissionState>(IDLE);
  const close = useRef<(() => void) | null>(null);
  // Held in a ref rather than a dependency so a caller can pass an inline
  // callback without tearing down the stream on every render.
  const finished = useRef(onFinished);
  useEffect(() => {
    finished.current = onFinished;
  }, [onFinished]);

  const busy = state.status !== "idle" && state.status !== "done" && state.status !== "failed";

  const follow = useCallback((id: string, since = 0) => {
    close.current?.();
    close.current = followMission(
      id,
      {
        onEvent: (event) =>
          setState((current) => {
            const next = reduce(current, event);
            if (event.kind === "end" && event.run) {
              // Let the page load the finished run once React has the state.
              queueMicrotask(() => finished.current?.(event.run as string));
            }
            return next;
          }),
        onError: (error) =>
          setState((current) => ({ ...current, status: "failed", error: error.message })),
      },
      since,
    );
  }, []);

  const start = useCallback(
    async (body: MissionBody) => {
      setState({ ...IDLE, status: "queued", message: "queued" });
      try {
        const { id } = await api.startMission(body);
        setState((current) => ({ ...current, id }));
        follow(id);
        return id;
      } catch (error) {
        setState({
          ...IDLE,
          status: "failed",
          error: error instanceof Error ? error.message : String(error),
        });
        return null;
      }
    },
    [follow],
  );

  const cancel = useCallback(async () => {
    if (!state.id) return;
    try {
      await api.cancelMission(state.id);
    } catch {
      /* it had already finished */
    }
  }, [state.id]);

  const reset = useCallback(() => {
    close.current?.();
    close.current = null;
    setState(IDLE);
  }, []);

  useEffect(() => () => close.current?.(), []);

  return { mission: state, busy, start, cancel, reset, follow };
}
