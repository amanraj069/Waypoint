"use client";

/**
 * Choosing where to fly.
 *
 * Each card says the three things that decide whether a mission is possible
 * there: how much is standing up, how high the tallest roof is, and how far the
 * ground moves. In Munnar the ground moves 260 m, and that is the whole reason
 * a plan written for Lodhi Garden would fly into a hillside.
 */

import { Badge } from "./ui";
import type { WorldCard } from "@/lib/types";

export function WorldPicker({
  worlds,
  selected,
  onSelect,
}: {
  worlds: WorldCard[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      {worlds.map((world) => {
        const live = world.id === selected;
        const relief = world.elevation_max - world.elevation_min;
        return (
          <button
            key={world.id}
            type="button"
            onClick={() => onSelect(world.id)}
            className={`block w-full rounded-sm border px-2.5 py-2 text-left transition-colors ${
              live
                ? "border-accent/60 bg-accent/5"
                : "border-line hover:border-line-bright hover:bg-panel-soft"
            }`}
          >
            <div className="flex items-baseline gap-2">
              <span className={live ? "text-accent" : "text-text"}>{world.name}</span>
              {!world.built_in && <Badge>{world.source}</Badge>}
            </div>
            {world.blurb && <p className="mt-0.5 text-faint">{world.blurb}</p>}
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-faint tabular-nums">
              <span>{world.buildings} buildings</span>
              {world.trees > 0 && <span>{world.trees} trees</span>}
              <span>roof {world.tallest_roof.toFixed(0)} m</span>
              <span>
                {world.has_terrain ? `relief ${relief.toFixed(0)} m` : "flat ground"}
              </span>
              <span>{world.landmarks.length} named places</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
