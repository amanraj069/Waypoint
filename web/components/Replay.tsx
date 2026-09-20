"use client";

/**
 * Replaying one flight.
 *
 * Everything on this page is the same run seen from a different side: the plan
 * the model wrote, the telemetry the simulator produced, the events either of
 * them raised, and (the one thing that is neither) the photographs, which are
 * the scene re-rendered from the aircraft's own pose at the moment it reported
 * reaching a CAPTURE waypoint.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Viewport, type Capture } from "./Viewport";
import { Transport } from "./Transport";
import { Badge, Button, Empty, Panel, Row } from "./ui";
import { Playback, frameAt, type Sample } from "@/lib/playback";
import type { Run } from "@/lib/types";

interface Shot {
  index: number;
  t: number;
  subtask: string;
  tickIndex: number;
  position: [number, number, number];
}

function phasesOf(run: Run) {
  // The plan's hierarchy is recoverable from the waypoints: each one carries
  // the subtask that produced it, in flight order.
  const phases: { name: string; count: number; first: number }[] = [];
  for (const waypoint of run.waypoints ?? []) {
    const last = phases[phases.length - 1];
    if (last && last.name === waypoint.subtask) last.count += 1;
    else phases.push({ name: waypoint.subtask, count: 1, first: waypoint.index });
  }
  return phases;
}

export function Replay({ run, title }: { run: Run; title?: string }) {
  const playback = useMemo(() => new Playback(), []);
  const captureRef = useRef<Capture | null>(null);
  const [sample, setSample] = useState<Sample | null>(null);
  const [labels, setLabels] = useState(true);
  const [developed, setDeveloped] = useState<Record<number, string>>({});
  const [lightbox, setLightbox] = useState<number | null>(null);

  const flying = (run.ticks?.length ?? 0) > 0;
  const phases = useMemo(() => phasesOf(run), [run]);

  useEffect(() => {
    const stop = playback.onSample(setSample);
    return () => {
      stop();
    };
  }, [playback]);

  // The shot list, but not the shots: nothing is rendered until you ask for a
  // photo, so opening a run stays instant however many captures it has.
  const shots = useMemo<Shot[]>(() => {
    const list: Shot[] = [];
    for (const waypoint of run.waypoints ?? []) {
      if (waypoint.action !== "CAPTURE") continue;
      const arrival = (run.events ?? []).find(
        (event) =>
          event.type === "WAYPOINT" &&
          (event.payload as { waypoint_index?: number } | undefined)?.waypoint_index ===
            waypoint.index,
      );
      if (!arrival) continue; // never reached: nothing was photographed
      const frame = frameAt(run.ticks ?? [], arrival.t);
      if (!frame) continue;
      list.push({
        index: waypoint.index,
        t: arrival.t,
        subtask: waypoint.subtask,
        tickIndex: frame.index,
        position: [frame.a.x, frame.a.y, frame.a.z],
      });
    }
    return list;
  }, [run]);

  const develop = useCallback(
    (position: number) => {
      const shot = shots[position];
      if (!shot || !captureRef.current) return;
      setDeveloped((current) => {
        if (current[shot.index]) return current;
        const tick = (run.ticks ?? [])[shot.tickIndex];
        const image = tick ? captureRef.current!(tick, shot.tickIndex) : null;
        return image ? { ...current, [shot.index]: image } : current;
      });
    },
    [run, shots],
  );

  const openShot = useCallback(
    (position: number) => {
      if (!shots.length) return;
      const wrapped = (position + shots.length) % shots.length;
      develop(wrapped);
      setLightbox(wrapped);
    },
    [shots.length, develop],
  );

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const typing = (event.target as HTMLElement)?.tagName?.match(/INPUT|TEXTAREA|SELECT/);
      if (typing) return;
      if (event.code === "Space") {
        event.preventDefault();
        playback.toggle();
      }
      if (event.code === "ArrowRight") playback.nudge(lightbox === null ? 1 : 0);
      if (event.code === "ArrowLeft") playback.nudge(lightbox === null ? -1 : 0);
      if (event.key === "l") setLabels((v) => !v);
      if (lightbox !== null) {
        if (event.code === "Escape") setLightbox(null);
        if (event.code === "ArrowRight") openShot(lightbox + 1);
        if (event.code === "ArrowLeft") openShot(lightbox - 1);
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [playback, lightbox, openShot]);

  const tick = sample?.tick;
  const gpsError = tick?.gps_offset ? Math.hypot(...tick.gps_offset) : 0;
  const sim = run.sim;

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-2 p-2 lg:grid-cols-[250px_minmax(0,1fr)_280px]">
      {/* telemetry + plan */}
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
        <Panel title="telemetry">
          <p className="mb-2 text-dim">{title ?? run.mission_text}</p>
          <div className="space-y-0.5">
            <Row label="time">{(sample?.time ?? 0).toFixed(1)} s</Row>
            <Row label="pos">
              {tick ? `${tick.x.toFixed(1)}, ${tick.y.toFixed(1)}, ${tick.z.toFixed(1)}` : "-"}
            </Row>
            {tick?.agl !== undefined && <Row label="agl">{tick.agl.toFixed(1)} m</Row>}
            <Row label="speed">{sample ? `${sample.speed.toFixed(2)} m/s` : "-"}</Row>
            <Row label="hdg">{tick ? `${tick.heading.toFixed(0)}°` : "-"}</Row>
            <Row label="phase">{tick?.subtask || "-"}</Row>
            <Row label="wind">
              {tick?.wind
                ? `${Math.hypot(tick.wind[0], tick.wind[1]).toFixed(1)} m/s`
                : "calm"}
            </Row>
            {gpsError > 0.05 && (
              <Row label="gps err">
                <span className="text-warn">{gpsError.toFixed(1)} m</span>
              </Row>
            )}
            <Row label="batt">{tick ? `${tick.battery_pct.toFixed(1)} %` : "-"}</Row>
          </div>
          <div className="mt-1.5 h-0.5 bg-line">
            <div
              className={`h-full ${
                (tick?.battery_pct ?? 100) < 15
                  ? "bg-bad"
                  : (tick?.battery_pct ?? 100) < 35
                    ? "bg-warn"
                    : "bg-good"
              }`}
              style={{ width: `${Math.max(0, Math.min(100, tick?.battery_pct ?? 100))}%` }}
            />
          </div>
          {tick?.faults && tick.faults.length > 0 && (
            <p className="mt-1.5 text-warn">⚠ {tick.faults.join(", ")}</p>
          )}
        </Panel>

        <Panel title="plan" right={<span>{phases.length}</span>}>
          {phases.length === 0 ? (
            <Empty>no plan in this file</Empty>
          ) : (
            <ol className="space-y-0.5">
              {phases.map((phase, index) => {
                const live = tick?.subtask === phase.name;
                return (
                  <li key={`${phase.name}-${index}`}>
                    <button
                      type="button"
                      onClick={() => {
                        const arrival = (run.events ?? []).find(
                          (event) =>
                            event.type === "WAYPOINT" &&
                            (event.payload as { waypoint_index?: number } | undefined)
                              ?.waypoint_index === phase.first,
                        );
                        if (arrival) playback.seek(Math.max(0, arrival.t - 2));
                      }}
                      className={`w-full border-l py-0.5 pl-2 text-left ${
                        live
                          ? "border-accent text-text"
                          : "border-line text-dim hover:border-line-bright hover:text-text"
                      }`}
                    >
                      <span className="text-faint">{index + 1}. </span>
                      {phase.name} <span className="text-faint">({phase.count} wp)</span>
                    </button>
                  </li>
                );
              })}
            </ol>
          )}
        </Panel>

        {sim && (
          <Panel title="outcome">
            <div className="space-y-0.5">
              <Row label="duration">{sim.duration_s.toFixed(1)} s</Row>
              <Row label="flown">{sim.distance_flown_m.toFixed(0)} m</Row>
              <Row label="battery left">{sim.battery_remaining_pct.toFixed(1)} %</Row>
              <Row label="result">
                {sim.crashed ? (
                  <span className="text-bad">crashed</span>
                ) : run.completed ? (
                  <span className="text-good">landed</span>
                ) : (
                  <span className="text-warn">did not land cleanly</span>
                )}
              </Row>
              {run.replans ? <Row label="replans">{run.replans}</Row> : null}
            </div>
          </Panel>
        )}
      </div>

      {/* the flight */}
      <div className="flex min-h-0 min-w-0 flex-col gap-2">
        <div className="relative min-h-[340px] flex-1 overflow-hidden rounded border border-line">
          <Viewport
            run={run}
            playback={playback}
            labels={labels}
            captureRef={captureRef}
            className="absolute inset-0"
          />
          <div className="absolute right-3 top-3">
            <Button onClick={() => setLabels((v) => !v)} active={labels} className="!py-1 !px-2">
              labels
            </Button>
          </div>
        </div>
        <Transport playback={playback} events={run.events ?? []} enabled={flying} />
      </div>

      {/* what the camera saw, and what happened */}
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto">
        <Panel title="photos" right={<span>{shots.length}</span>}>
          {shots.length === 0 ? (
            <Empty>no CAPTURE waypoints were reached in this flight</Empty>
          ) : (
            <div className="grid grid-cols-2 gap-1.5">
              {shots.map((shot, position) => (
                <figure key={shot.index} className="m-0">
                  <button
                    type="button"
                    onClick={() => openShot(position)}
                    className="block w-full"
                    title={`${shot.subtask} · waypoint ${shot.index}`}
                  >
                    {developed[shot.index] ? (
                      // The frame is a canvas data URL, so next/image has
                      // nothing to optimise and would only get in the way.
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={developed[shot.index]}
                        alt={`waypoint ${shot.index}`}
                        className="block w-full rounded-sm border border-line hover:border-accent"
                      />
                    ) : (
                      <span className="grid aspect-[4/3] place-items-center rounded-sm border border-dashed border-line text-[10px] text-faint hover:border-accent hover:text-accent">
                        develop
                      </span>
                    )}
                  </button>
                  <figcaption className="pt-0.5 text-[10px] text-faint">
                    #{shot.index} · t={shot.t.toFixed(0)}s
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </Panel>

        <Panel title="log" className="min-h-48 flex-1">
          {(run.events ?? []).length === 0 ? (
            <Empty>nothing happened</Empty>
          ) : (
            <div className="space-y-0.5">
              {run.events.map((event, index) => {
                const past = (sample?.time ?? 0) >= event.t;
                return (
                  <button
                    key={index}
                    type="button"
                    onClick={() => playback.seek(event.t)}
                    className={`block w-full text-left ${past ? "text-text" : "text-dim"} hover:text-accent`}
                  >
                    <span
                      className={`mr-1.5 tabular-nums ${
                        event.type === "FAULT" ||
                        event.type === "SEPARATION" ||
                        event.type === "COLLISION"
                          ? "text-bad"
                          : past
                            ? "text-accent"
                            : "text-faint"
                      }`}
                    >
                      t={event.t.toFixed(1)}s
                    </span>
                    {event.description}
                  </button>
                );
              })}
            </div>
          )}
        </Panel>
      </div>

      {lightbox !== null && shots[lightbox] && (
        <div
          className="fixed inset-0 z-20 grid place-items-center bg-black/93 p-6"
          onClick={(event) => event.target === event.currentTarget && setLightbox(null)}
        >
          <figure className="m-0 w-full max-w-3xl">
            {developed[shots[lightbox].index] && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={developed[shots[lightbox].index]}
                alt={`waypoint ${shots[lightbox].index}`}
                className="block w-full rounded border border-line"
              />
            )}
            <figcaption className="flex flex-wrap items-center gap-3 pt-2 text-dim">
              <span>
                {shots[lightbox].subtask} · waypoint {shots[lightbox].index} · t=
                {shots[lightbox].t.toFixed(1)}s · (
                {shots[lightbox].position.map((c) => c.toFixed(0)).join(", ")}) [{lightbox + 1}/
                {shots.length}]
              </span>
              <span className="ml-auto flex gap-1">
                <Button
                  onClick={() => {
                    playback.seek(Math.max(0, shots[lightbox].t - 1));
                    playback.setPlaying(false);
                    setLightbox(null);
                  }}
                >
                  go to moment
                </Button>
                <Button onClick={() => openShot(lightbox - 1)}>prev</Button>
                <Button onClick={() => openShot(lightbox + 1)}>next</Button>
                <Button onClick={() => setLightbox(null)}>close</Button>
              </span>
            </figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}

export function ReplayBadges({ run }: { run: Run }) {
  const sim = run.sim;
  return (
    <>
      {run.world && <Badge tone="accent">{run.world.name}</Badge>}
      {run.verification && (
        <Badge tone={run.verification.passed ? "good" : "bad"}>
          {run.verification.passed ? "verified" : "unverified"}
        </Badge>
      )}
      {sim?.crashed && <Badge tone="bad">crashed</Badge>}
    </>
  );
}
