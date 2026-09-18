"use client";

/**
 * Light or dark, remembered.
 *
 * The choice is written to the document as `data-theme` — CSS variables do the
 * rest, and the 3D view reads the same variables, so the scene can never be
 * dark while its chrome is light. Nothing here renders; a small script in the
 * layout applies the stored choice before the first paint, so there is no flash
 * of the wrong theme on load.
 */

import { useCallback, useSyncExternalStore } from "react";

export type Theme = "light" | "dark";

const KEY = "waypoint.theme";

/** Applied before React runs, so it is duplicated in `layout.tsx` on purpose. */
export const THEME_BOOTSTRAP = `
try {
  var stored = localStorage.getItem("${KEY}");
  var theme = stored || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.dataset.theme = theme;
} catch (e) {
  document.documentElement.dataset.theme = "dark";
}`;

function current(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

/**
 * The theme lives on the document, not in React state.
 *
 * That is what makes it an *external* store: the bootstrap script sets it
 * before React exists, and the 3D scene reads the same attribute. Components
 * subscribe rather than owning a copy that could drift out of step.
 */
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useTheme() {
  const theme = useSyncExternalStore<Theme>(subscribe, current, () => "dark");

  const choose = useCallback((next: Theme) => {
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* private browsing: the choice lasts as long as the tab does */
    }
    for (const listener of listeners) listener();
  }, []);

  const toggle = useCallback(() => {
    choose(current() === "light" ? "dark" : "light");
  }, [choose]);

  return { theme, choose, toggle };
}
