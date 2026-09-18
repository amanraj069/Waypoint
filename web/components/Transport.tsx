"use client";

/**
 * Play, scrub, and the marks on the track.
 *
 * The marks are the flight's events — a waypoint reached, a fault firing — so
 * the timeline shows where the interesting seconds are before you go looking
 * for them.
 */

import { useSyncExternalStore } from "react";
import { Button, Select } from "./ui";
import type { Playback } from "@/lib/playback";
import type { FlightEvent } from "@/lib/types";

const RATES = [0.5, 1, 2, 4, 8];

export function Transport({
  playback,
  events,
  enabled,
  right,
}: {
  playback: Playback;
  events: FlightEvent[];
  enabled: boolean;
  right?: React.ReactNode;
}) {
  // The playhead moves every frame but the controls only change when you touch
  // them, so this subscribes to the control channel rather than the samples.
  useSyncExternalStore(playback.subscribe, playback.getSnapshot, () => 0);
  const duration = playback.duration || 1;

  return (
    <div className="rounded border border-line bg-panel/90 px-3 py-2 backdrop-blur-sm">
      <div className="relative pb-1.5">
        <input
          className="scrub"
          type="range"
          min={0}
          max={1000}
          step={1}
          disabled={!enabled}
          value={Math.round((playback.time / duration) * 1000)}
          onChange={(event) => playback.seek((Number(event.target.value) / 1000) * duration)}
        />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1.5">
          {events.map((event, index) => (
            <i
              key={index}
              title={`t=${event.t.toFixed(1)}s ${event.description}`}
              className={`absolute block h-1.5 w-px ${
                event.type === "WAYPOINT"
                  ? "bg-good"
                  : event.type === "FAULT" || event.type === "SEPARATION"
                    ? "bg-bad"
                    : "bg-faint"
              }`}
              style={{ left: `${(event.t / duration) * 100}%` }}
            />
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={!enabled} active={playback.playing} onClick={() => playback.toggle()}>
          {playback.playing ? "pause" : "play"}
        </Button>
        <Button disabled={!enabled} onClick={() => playback.restart()}>
          restart
        </Button>
        <Select
          value={String(playback.rate)}
          onChange={(value) => playback.setRate(Number(value))}
          title="Playback speed"
        >
          {RATES.map((rate) => (
            <option key={rate} value={rate}>
              {rate}×
            </option>
          ))}
        </Select>
        <Button
          disabled={!enabled}
          active={playback.follow}
          onClick={() => playback.setFollow(!playback.follow)}
          title="Lock the camera to the aircraft"
        >
          follow
        </Button>
        {right}
        <span className="ml-auto tabular-nums text-dim">
          {enabled
            ? `${playback.time.toFixed(1)} / ${playback.duration.toFixed(1)} s`
            : "nothing flown here yet"}
        </span>
      </div>
    </div>
  );
}
