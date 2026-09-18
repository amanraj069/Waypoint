"use client";

/**
 * What the pipeline is doing, while it does it.
 *
 * The four boxes here are the four stages of the project made visible: the
 * planner's decomposition, the verifier's verdict on it, the telemetry coming
 * off the simulator, and the log everything writes to. A rejected plan and the
 * correction that follows it show up in the same place, which is the point —
 * the safety gate is not a detail hidden behind a spinner.
 */

import { useEffect, useRef } from "react";
import { Badge, Empty, Panel, Row } from "./ui";
import type { MissionState } from "@/lib/useMission";
import type { Stage, Violation } from "@/lib/types";

const RAIL: { key: Stage; label: string }[] = [
  { key: "planning", label: "plan" },
  { key: "verifying", label: "verify" },
  { key: "replanning", label: "correct" },
  { key: "flying", label: "fly" },
  { key: "done", label: "done" },
];

const ORDER: Record<string, number> = {
  idle: -1,
  queued: 0,
  planning: 1,
  verifying: 2,
  replanning: 3,
  flying: 4,
  done: 5,
  failed: 5,
};

function StageRail({ status, message }: { status: MissionState["status"]; message: string }) {
  const at = ORDER[status] ?? -1;
  const failed = status === "failed";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1">
        {RAIL.map((step, index) => {
          const position = ORDER[step.key];
          const done = at > position;
          const live = at === position && !failed;
          return (
            <div key={step.key} className="flex flex-1 items-center gap-1">
              <span
                className={`flex-1 border-t ${
                  live
                    ? "border-accent"
                    : done
                      ? "border-good/50"
                      : failed && at === position
                        ? "border-bad"
                        : "border-line"
                }`}
              />
              <span
                className={`text-[10px] uppercase tracking-wide ${
                  live
                    ? "text-accent pulse"
                    : done
                      ? "text-good"
                      : failed && at >= position
                        ? "text-bad"
                        : "text-faint"
                }`}
              >
                {index === RAIL.length - 1 && failed ? "failed" : step.label}
              </span>
            </div>
          );
        })}
      </div>
      {message && (
        <p className={`truncate ${failed ? "text-bad" : "text-dim"}`} title={message}>
          {message}
        </p>
      )}
    </div>
  );
}

function ViolationRow({ violation }: { violation: Violation }) {
  const tone =
    violation.severity === "WARNING" ? "warn" : violation.severity === "ERROR" ? "bad" : "bad";
  return (
    <li className="border-l border-line py-1 pl-2">
      <div className="mb-0.5 flex items-center gap-1.5">
        <Badge tone={tone}>{violation.severity}</Badge>
        <span className="text-faint">{violation.type}</span>
        {violation.waypoint_index !== null && (
          <span className="text-faint">wp {violation.waypoint_index}</span>
        )}
      </div>
      <p className="text-dim">{violation.description}</p>
    </li>
  );
}

export function MissionConsole({
  mission,
  activePhase,
  onSeekPhase,
}: {
  mission: MissionState;
  /** The phase the replay is currently in, so the plan lights up as it flies. */
  activePhase?: string;
  onSeekPhase?: (index: number) => void;
}) {
  const logEnd = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logEnd.current?.scrollIntoView({ block: "nearest" });
  }, [mission.log.length]);

  const telemetry = mission.telemetry;
  const verification = mission.verification;
  const blocking = (verification?.violations ?? []).filter((v) => v.severity !== "WARNING");
  const warnings = (verification?.violations ?? []).filter((v) => v.severity === "WARNING");

  return (
    <div className="flex min-h-0 shrink-0 flex-col gap-2">
      <Panel title="pipeline">
        <StageRail status={mission.status} message={mission.message} />
        {mission.notes.length > 0 && (
          <ul className="mt-2 space-y-0.5 border-t border-line pt-2 text-faint">
            {mission.notes.map((note) => (
              <li key={note}>· {note}</li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel
        title="plan"
        right={
          mission.phases.length > 0 ? (
            <span>
              {mission.phases.length} phase
              {mission.phases.length === 1 ? "" : "s"}
              {mission.attempt > 0 ? ` · correction ${mission.attempt}` : ""}
            </span>
          ) : null
        }
        bodyClassName="max-h-52"
      >
        {mission.phases.length === 0 ? (
          <Empty>nothing planned yet</Empty>
        ) : (
          <ol className="space-y-0.5">
            {mission.phases.map((phase, index) => {
              const live = activePhase === phase.name;
              return (
                <li key={`${phase.name}-${index}`}>
                  <button
                    type="button"
                    onClick={() => onSeekPhase?.(index)}
                    disabled={!onSeekPhase}
                    className={`w-full border-l py-0.5 pl-2 text-left ${
                      live
                        ? "border-accent text-text"
                        : "border-line text-dim enabled:hover:border-line-bright enabled:hover:text-text"
                    }`}
                  >
                    <span className="text-faint">{index + 1}. </span>
                    {phase.name}{" "}
                    <span className="text-faint">({phase.waypoints.length} wp)</span>
                  </button>
                </li>
              );
            })}
          </ol>
        )}
      </Panel>

      <Panel
        title="verifier"
        right={
          verification ? (
            <Badge tone={verification.passed ? "good" : "bad"}>
              {verification.passed ? "passed" : "rejected"}
            </Badge>
          ) : null
        }
        bodyClassName="max-h-64"
      >
        {!verification ? (
          <Empty>no verdict yet</Empty>
        ) : verification.violations.length === 0 ? (
          <p className="text-good">no violations — the plan is inside the envelope</p>
        ) : (
          <ul className="space-y-1">
            {[...blocking, ...warnings].map((violation, index) => (
              <ViolationRow key={index} violation={violation} />
            ))}
          </ul>
        )}
      </Panel>

      {telemetry && (
        <Panel
          title="live telemetry"
          right={telemetry.subtask ? <span className="text-accent">{telemetry.subtask}</span> : null}
        >
          <div className="space-y-0.5">
            <Row label="t">{telemetry.t?.toFixed(1)} s</Row>
            <Row label="pos">
              {telemetry.x?.toFixed(0)}, {telemetry.y?.toFixed(0)}, {telemetry.z?.toFixed(0)}
            </Row>
            {telemetry.agl !== undefined && <Row label="agl">{telemetry.agl.toFixed(0)} m</Row>}
            <Row label="speed">{telemetry.speed?.toFixed(1)} m/s</Row>
            <Row label="waypoint">
              {telemetry.waypoint ?? "—"} / {telemetry.of}
            </Row>
            <Row label="batt">{telemetry.battery?.toFixed(0)} %</Row>
          </div>
          <div className="mt-1.5 h-0.5 bg-line">
            <div
              className="h-full bg-good transition-[width] duration-200"
              style={{ width: `${Math.max(0, Math.min(100, telemetry.battery ?? 0))}%` }}
            />
          </div>
          {telemetry.faults && telemetry.faults.length > 0 && (
            <p className="mt-1.5 text-warn">⚠ {telemetry.faults.join(", ")}</p>
          )}
        </Panel>
      )}

      <Panel title="log" className="max-h-[32vh] shrink-0" bodyClassName="">
        {mission.log.length === 0 ? (
          <Empty>nothing yet</Empty>
        ) : (
          <div className="space-y-0.5">
            {mission.log.map((line) => (
              <div key={line.seq} className="flex gap-2">
                <span className="shrink-0 text-faint tabular-nums">{line.t.toFixed(1)}s</span>
                <span
                  className={
                    line.type === "FAULT" || line.type === "SEPARATION" || line.type === "COLLISION"
                      ? "text-bad"
                      : line.kind === "stage"
                        ? "text-accent"
                        : "text-dim"
                  }
                >
                  {line.message}
                </span>
              </div>
            ))}
            <div ref={logEnd} />
          </div>
        )}
      </Panel>
    </div>
  );
}
