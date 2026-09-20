"use client";

/**
 * Where you tell the aircraft what to do.
 *
 * This is the front door of the whole project: a sentence in, a flight out. The
 * controls beside it are the things that change what the sentence means: which
 * place it is flown in, what the air is doing, and how many corrections the
 * verifier may demand before the mission is abandoned.
 */

import { useEffect, useRef, useState } from "react";
import { Badge, Button, Select } from "./ui";
import { suggestionsFor } from "@/lib/suggestions";
import type { MissionBody, PlanCard, WorldCard } from "@/lib/types";

const WINDS: { label: string; value: [number, number, number] }[] = [
  { label: "calm", value: [0, 0, 0] },
  { label: "3 m/s easterly", value: [3, 0, 0] },
  { label: "6 m/s easterly", value: [6, 0, 0] },
  { label: "4 m/s northerly", value: [0, 4, 0] },
];

export function MissionBox({
  world,
  landmarks,
  plans,
  plannerReady,
  busy,
  onSubmit,
  home,
  landing,
  onCancel,
}: {
  world: WorldCard | null;
  landmarks: string[];
  plans: PlanCard[];
  plannerReady: boolean;
  busy: boolean;
  onSubmit: (body: MissionBody) => void;
  /** Where the aircraft will start, if it has been moved off the origin. */
  home?: [number, number] | null;
  /** Where it should put down, if that is not where it started. */
  landing?: [number, number] | null;
  onCancel: () => void;
}) {
  const [text, setText] = useState("");
  const [windIndex, setWindIndex] = useState(0);
  const [faults, setFaults] = useState("none");
  const [retries, setRetries] = useState(2);
  const [blind, setBlind] = useState(false);
  const [open, setOpen] = useState(false);
  const [folded, setFolded] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);

  const suggestions = suggestionsFor(world?.id ?? null, landmarks);
  const forThisWorld = plans.filter((plan) => !world || plan.world === world.id);

  // Cmd/Ctrl-Enter flies it, because the box is a textarea and Enter has to
  // stay a newline for a two-sentence mission.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        submit();
      }
    };
    const node = box.current;
    node?.addEventListener("keydown", onKey);
    return () => node?.removeEventListener("keydown", onKey);
  });

  function submit() {
    if (busy || !text.trim() || !plannerReady) return;
    onSubmit({
      mission: text.trim(),
      world: world?.id ?? null,
      wind: WINDS[windIndex].value,
      faults,
      retries,
      blind,
      home: home ?? null,
      landing: landing ?? null,
    });
  }

  function insert(name: string) {
    const node = box.current;
    const at = node?.selectionStart ?? text.length;
    const next = `${text.slice(0, at)}${text.slice(0, at).match(/\S$/) ? " " : ""}${name}${text.slice(at)}`;
    setText(next);
    node?.focus();
  }

  return (
    <div className="rounded border border-line bg-panel/90 backdrop-blur-sm">
      <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
        <button
          type="button"
          onClick={() => setFolded((v) => !v)}
          aria-expanded={!folded}
          className="cursor-pointer text-[10px] uppercase tracking-[0.14em] text-faint hover:text-dim"
        >
          {folded ? "+ " : "− "}mission
        </button>
        {world ? (
          <Badge tone="accent">{world.name}</Badge>
        ) : (
          <Badge>open space</Badge>
        )}
        {!plannerReady && <Badge tone="warn">no planner key</Badge>}
        {!folded && (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="ml-auto text-[10px] text-faint hover:text-dim"
          >
            {open ? "− conditions" : "+ conditions"}
          </button>
        )}
      </div>

      {!folded && (
      <div className="p-3">
        <textarea
          ref={box}
          value={text}
          disabled={!plannerReady}
          onChange={(event) => setText(event.target.value)}
          rows={2}
          spellCheck={false}
          placeholder={
            plannerReady
              ? suggestions[0]
              : "Set GEMINI_API_KEY in the API server's environment to plan a mission in English. Until then, fly one of the bundled plans below."
          }
          className="w-full resize-none rounded-sm border border-line bg-panel-soft px-2.5 py-2 text-text placeholder:text-faint disabled:opacity-60"
        />

        {landmarks.length > 0 && (
          <div className="mt-2">
            <div className="mb-1 text-[10px] uppercase tracking-wide text-faint">
              named places here; click to put one in the mission
            </div>
            <div className="flex flex-wrap gap-1">
              {landmarks.map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => insert(name)}
                  disabled={!plannerReady}
                  className="rounded-sm border border-line px-1.5 py-0.5 text-[11px] text-dim hover:border-accent/50 hover:text-accent disabled:opacity-40"
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        )}

        {open && (
          <div className="mt-3 grid gap-2 border-t border-line pt-3 sm:grid-cols-4">
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-faint">wind</span>
              <Select value={String(windIndex)} onChange={(v) => setWindIndex(Number(v))}>
                {WINDS.map((wind, index) => (
                  <option key={wind.label} value={index}>
                    {wind.label}
                  </option>
                ))}
              </Select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-faint">faults</span>
              <Select value={faults} onChange={setFaults}>
                <option value="none">none</option>
                <option value="sample">sample (gust, drain, GPS slip, crane)</option>
                <option value="storm">storm</option>
                <option value="gps_slip">GPS slip</option>
              </Select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wide text-faint">
                corrections after a rejection
              </span>
              <Select value={String(retries)} onChange={(v) => setRetries(Number(v))}>
                {[0, 1, 2, 3].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </Select>
            </label>
            <label className="flex cursor-pointer items-end gap-2 pb-1.5">
              <input
                type="checkbox"
                checked={blind}
                onChange={(event) => setBlind(event.target.checked)}
                className="accent-[var(--accent)]"
              />
              <span className="text-dim">fly blind (no wind compensation)</span>
            </label>
          </div>
        )}

        <div className="mt-3 flex flex-wrap items-center gap-2">
          {busy ? (
            <Button onClick={onCancel}>cancel</Button>
          ) : (
            <Button tone="accent" onClick={submit} disabled={!plannerReady || !text.trim()}>
              fly it
            </Button>
          )}
          <span className="text-[10px] text-faint">⌘↵</span>

          {forThisWorld.length > 0 && (
            <div className="ml-auto flex flex-wrap items-center gap-1">
              <span className="text-[10px] uppercase tracking-wide text-faint">
                or fly a saved plan
              </span>
              {forThisWorld.map((plan) => (
                <Button
                  key={plan.id}
                  disabled={busy}
                  title={plan.mission_text}
                  onClick={() =>
                    onSubmit({
                      plan: plan.id,
                      world: world?.id ?? plan.world,
                      wind: WINDS[windIndex].value,
                      faults,
                      blind,
                    })
                  }
                >
                  {plan.id.replace(/_/g, " ")}
                </Button>
              ))}
            </div>
          )}
        </div>

        {plannerReady && !text.trim() && (
          <ul className="mt-3 space-y-0.5 border-t border-line pt-2">
            {suggestions.slice(0, 3).map((line) => (
              <li key={line}>
                <button
                  type="button"
                  onClick={() => setText(line)}
                  className="text-left text-dim hover:text-accent"
                >
                  <span className="text-faint">→ </span>
                  {line}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      )}
    </div>
  );
}
