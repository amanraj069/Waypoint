"use client";

/**
 * The studio: pick a place, look around it, say what you want, watch it happen.
 *
 * One 3D view serves the whole page. Before a mission it holds the world you
 * picked, with nothing flying in it; during one it holds the same world while
 * the console fills up; when the flight is written it loads the finished run
 * and plays it. That continuity is the point: you never lose the place you
 * were looking at.
 */

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Viewport } from "@/components/Viewport";
import { MissionBox } from "@/components/MissionBox";
import { MissionConsole } from "@/components/MissionConsole";
import { WorldPicker } from "@/components/WorldPicker";
import { Transport } from "@/components/Transport";
import { Badge, Button, Panel, TopBar, useFolded } from "@/components/ui";
import { api } from "@/lib/api";
import { Playback } from "@/lib/playback";
import { useMission } from "@/lib/useMission";
import type { Health, MissionBody, PlanCard, Run, WorldCard } from "@/lib/types";

export default function Studio() {
  const [health, setHealth] = useState<Health | null>(null);
  const [worlds, setWorlds] = useState<WorldCard[]>([]);
  const [plans, setPlans] = useState<PlanCard[]>([]);
  const [worldId, setWorldId] = useState<string | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [runLabel, setRunLabel] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [labels, setLabels] = useState(true);
  // Where the aircraft stands, and where it puts down. Both pads are on screen
  // from the start; null means "where the world puts it" for the first and
  // "back where it started" for the second, which is what gets sent.
  const [home, setHome] = useState<[number, number] | null>(null);
  const [landing, setLanding] = useState<[number, number] | null>(null);
  const [picking, setPicking] = useState<"home" | "landing" | null>(null);
  const [siteNote, setSiteNote] = useState<string | null>(null);
  const [softwareRenderer, setSoftwareRenderer] = useState<string | null>(null);

  const playback = useMemo(() => new Playback(), []);
  // The whole sidebar folds, not just the panels in it: sometimes you want the
  // place and nothing else.
  const sidebar = useFolded("sidebar");

  const showRun = useCallback(async (runId: string) => {
    try {
      const loaded = await api.run(runId);
      setRun(loaded);
      setRunLabel(runId);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const { mission, busy, start, cancel, reset } = useMission(showRun);

  // -- what the server has ------------------------------------------------
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const [health, worlds, plans] = await Promise.all([
          api.health(),
          api.worlds(),
          api.plans(),
        ]);
        if (!live) return;
        setHealth(health);
        setWorlds(worlds);
        setPlans(plans);
        setWorldId((current) => current ?? worlds[0]?.id ?? null);
      } catch (error) {
        if (live) {
          setLoadError(
            error instanceof Error
              ? `${error.message}. Is the API running? \`.venv/bin/python -m uvicorn api.main:app --port 8000\``
              : String(error),
          );
        }
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  // -- the place, whenever it changes and nothing has been flown in it ----
  useEffect(() => {
    if (!worldId) return;
    let live = true;
    api
      .worldPreview(worldId)
      .then((preview) => {
        if (!live) return;
        setRun(preview);
        setRunLabel("");
        // Pads belong to a place, not to the session.
        setHome(null);
        setLanding(null);
        setPicking(null);
        setSiteNote(null);
        reset();
      })
      .catch((error) => live && setLoadError(String(error)));
    return () => {
      live = false;
    };
  }, [worldId, reset]);

  const world = worlds.find((candidate) => candidate.id === worldId) ?? null;
  const landmarks = run?.world?.landmarks?.map((landmark) => landmark.name) ?? [];
  const flying = (run?.ticks?.length ?? 0) > 0;

  const submit = (body: MissionBody) => {
    setLoadError(null);
    start(body);
  };

  /**
   * Put a pad where the ground will take it.
   *
   * The server owns this decision -- it surveys the patch the click landed on
   * and, if the aircraft could not stand there, hands back the nearest spot
   * where it could. Asking rather than guessing is what stops the pad you see
   * disagreeing with the flight you get.
   */
  const placePad = useCallback(
    async (role: "home" | "landing", x: number, y: number) => {
      setPicking(null);
      if (!worldId) {
        (role === "home" ? setHome : setLanding)([x, y]);
        return;
      }
      try {
        const site = await api.site(worldId, x, y);
        if (!site.usable) {
          setSiteNote(`can't put down there: ${site.asked.why_not}`);
          return;
        }
        (role === "home" ? setHome : setLanding)([site.x, site.y]);
        setSiteNote(
          site.moved
            ? `moved the ${role === "home" ? "start" : "landing"} ${site.moved.toFixed(0)} m to level ground: ${site.asked.why_not}`
            : `${role === "home" ? "start" : "landing"}: ground at ${site.elevation.toFixed(1)} m, ${site.slope_deg.toFixed(0)}° slope`,
        );
      } catch (error) {
        setSiteNote(error instanceof Error ? error.message : String(error));
      }
    },
    [worldId],
  );

  // While a mission is in the air the console's own telemetry is the live view;
  // the finished run replaces it the moment the flight is written.
  const activePhase = mission.telemetry?.subtask;

  const seekPhase = (index: number) => {
    const phase = mission.phases[index];
    const first = phase?.waypoints[0];
    if (!first || !run?.events) return;
    const arrival = run.events.find(
      (event) =>
        event.type === "WAYPOINT" &&
        (event.payload as { waypoint_index?: number } | undefined)?.waypoint_index === first.index,
    );
    if (arrival) playback.seek(Math.max(0, arrival.t - 2));
  };

  return (
    <div className="flex h-dvh flex-col">
      <TopBar>
        {health && (
          <Badge tone={health.planner_ready ? "good" : "warn"}>
            {health.planner_ready ? `planner ${health.planner_model}` : "no planner key"}
          </Badge>
        )}
        <Link href="/runs" className="text-[11px] text-dim hover:text-text">
          {health ? `${health.runs} runs` : ""}
        </Link>
      </TopBar>

      {loadError && (
        <div className="border-b border-bad/40 bg-bad/10 px-4 py-2 text-bad">{loadError}</div>
      )}
      {softwareRenderer && (
        <div className="border-b border-warn/40 bg-warn/10 px-4 py-2 text-warn">
          This browser is drawing the 3D view in software ({softwareRenderer}), not on the GPU,
          which costs many times the CPU for the same picture. Restart the browser to get hardware
          acceleration back, and check <span className="font-bold">chrome://gpu</span> if it keeps
          happening.
        </div>
      )}

      <main
        className={`grid min-h-0 flex-1 grid-cols-1 gap-2 p-2 ${
          sidebar.open ? "lg:grid-cols-[300px_minmax(0,1fr)]" : "lg:grid-cols-[28px_minmax(0,1fr)]"
        }`}
      >
        {/* everything about the mission, in one column */}
        {sidebar.open ? (
          <aside className="hidden min-h-0 flex-col gap-2 overflow-y-auto pr-0.5 lg:flex">
            <div className="flex items-center gap-2 px-1">
              <span className="text-[10px] uppercase tracking-[0.14em] text-faint">panels</span>
              <button
                type="button"
                onClick={sidebar.toggle}
                title="Collapse the sidebar"
                className="ml-auto text-[11px] text-faint hover:text-text"
              >
                ⟨⟨
              </button>
            </div>

            <Panel title="where" right={<span>{worlds.length}</span>} className="max-h-[38vh]">
              <WorldPicker worlds={worlds} selected={worldId} onSelect={setWorldId} />
            </Panel>

            {world && (
              <Panel title="this place">
                <dl className="space-y-0.5">
                  <div className="flex justify-between gap-2">
                    <dt className="text-faint">ground</dt>
                    <dd className="tabular-nums">
                      {world.elevation_min.toFixed(0)} … {world.elevation_max.toFixed(0)} m
                    </dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt className="text-faint">tallest roof</dt>
                    <dd className="tabular-nums">{world.tallest_roof.toFixed(0)} m</dd>
                  </div>
                  <div className="flex justify-between gap-2">
                    <dt className="text-faint">spans</dt>
                    <dd className="tabular-nums">
                      {world.bounds[2] - world.bounds[0]} × {world.bounds[3] - world.bounds[1]} m
                    </dd>
                  </div>
                </dl>
                {run?.constraint_notes && run.constraint_notes.length > 0 && (
                  <ul className="mt-2 space-y-0.5 border-t border-line pt-2 text-faint">
                    {run.constraint_notes.map((note) => (
                      <li key={note}>· {note}</li>
                    ))}
                  </ul>
                )}
              </Panel>
            )}

            <MissionConsole
              mission={mission}
              activePhase={activePhase}
              onSeekPhase={flying ? seekPhase : undefined}
            />
          </aside>
        ) : (
          <aside className="hidden min-h-0 lg:block">
            <button
              type="button"
              onClick={sidebar.toggle}
              title="Show the panels"
              className="flex h-full w-full flex-col items-center gap-3 rounded border border-line bg-panel/60 py-2 text-[11px] text-faint hover:text-text"
            >
              <span>⟩⟩</span>
              <span className="[writing-mode:vertical-rl] tracking-[0.2em] uppercase">panels</span>
            </button>
          </aside>
        )}

        {/* the place itself */}
        <div className="flex min-h-0 min-w-0 flex-col gap-2">
          <div className="relative min-h-[320px] flex-1 overflow-hidden rounded border border-line">
            <Viewport
              run={run}
              playback={playback}
              labels={labels}
              home={home}
              landing={landing}
              picking={picking}
              onPick={(x, y) => picking && placePad(picking, x, y)}
              onMovePad={placePad}
              onSoftwareRenderer={setSoftwareRenderer}
              className="absolute inset-0"
            />
            <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2">
              <span className="rounded-sm border border-line bg-panel/80 px-2 py-1 text-[11px] text-dim backdrop-blur-sm">
                {run?.mission_text ?? "loading…"}
              </span>
              {runLabel && <Badge tone="accent">replaying</Badge>}
            </div>
            {siteNote && (
              <button
                onClick={() => setSiteNote(null)}
                className="absolute bottom-3 left-3 rounded-sm border border-line bg-panel/90 px-2 py-1 text-[11px] text-dim backdrop-blur-sm hover:text-text"
                title="dismiss"
              >
                {siteNote}
              </button>
            )}
            <div className="absolute right-3 top-3 flex gap-1">
              <Button
                onClick={() => setPicking((v) => (v === "home" ? null : "home"))}
                active={picking === "home"}
                className="!py-1 !px-2"
                title="Drag the green pad, or click here and then click the ground"
              >
                {picking === "home"
                  ? "click the ground"
                  : home
                    ? `start ${home[0].toFixed(0)}, ${home[1].toFixed(0)}`
                    : "start"}
              </Button>
              {home && (
                <Button onClick={() => setHome(null)} className="!py-1 !px-2" title="Back to the origin">
                  ✕
                </Button>
              )}
              <Button
                onClick={() => setPicking((v) => (v === "landing" ? null : "landing"))}
                active={picking === "landing"}
                className="!py-1 !px-2"
                title="Drag the amber pad, or click here and then click the ground"
              >
                {picking === "landing"
                  ? "click the ground"
                  : landing
                    ? `land ${landing[0].toFixed(0)}, ${landing[1].toFixed(0)}`
                    : "landing"}
              </Button>
              {landing && (
                <Button
                  onClick={() => setLanding(null)}
                  className="!py-1 !px-2"
                  title="Land back where it started"
                >
                  ✕
                </Button>
              )}
              <Button onClick={() => setLabels((v) => !v)} active={labels} className="!py-1 !px-2">
                labels
              </Button>
              {runLabel && (
                <Link href={`/runs/${runLabel}`}>
                  <Button className="!py-1 !px-2">open replay</Button>
                </Link>
              )}
            </div>
          </div>

          <Transport playback={playback} events={run?.events ?? []} enabled={flying} />
          <MissionBox
            home={home}
            landing={landing}
            world={world}
            landmarks={landmarks}
            plans={plans}
            plannerReady={health?.planner_ready ?? false}
            busy={busy}
            onSubmit={submit}
            onCancel={cancel}
          />
        </div>

      </main>
    </div>
  );
}
