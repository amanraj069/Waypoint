"use client";

/**
 * The small pieces every panel is made of.
 *
 * There is no component library here on purpose: the whole interface is boxes
 * with hairline borders and rows of aligned numbers, and that is about six
 * components' worth of styling.
 */

import Link from "next/link";
import { useCallback, useSyncExternalStore, type ReactNode } from "react";

import { useTheme } from "@/lib/theme";

const FOLDED_KEY = "waypoint.folded";

function readFolded(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(FOLDED_KEY) ?? "{}") as Record<string, boolean>;
  } catch {
    return {};
  }
}

/**
 * Which panels are folded is browser state, shared by every panel on the page,
 * so it is subscribed to rather than copied into each one.
 */
const foldListeners = new Set<() => void>();

function subscribeFolds(listener: () => void) {
  foldListeners.add(listener);
  return () => {
    foldListeners.delete(listener);
  };
}

function setFolded(title: string, folded: boolean) {
  try {
    localStorage.setItem(FOLDED_KEY, JSON.stringify({ ...readFolded(), [title]: folded }));
  } catch {
    /* private browsing: the fold lasts as long as the tab does */
  }
  for (const listener of foldListeners) listener();
}

/**
 * Fold state for one named thing, shared with every panel on the page.
 *
 * Exported because the sidebar folds too, and a sidebar that remembered itself
 * differently from the panels inside it would be two mechanisms doing one job.
 */
export function useFolded(key: string, defaultOpen = true) {
  const open = useSyncExternalStore(
    subscribeFolds,
    () => !(readFolded()[key] ?? !defaultOpen),
    () => defaultOpen,
  );
  const toggle = useCallback(() => setFolded(key, open), [key, open]);
  return { open, toggle };
}

/**
 * A titled box you can fold away.
 *
 * Every panel folds — a mission you are watching does not need the world list,
 * and a place you are studying does not need the log. What you fold stays
 * folded: the state is keyed on the title and kept in localStorage, so the
 * layout you arrive at is the layout you left.
 */
export function Panel({
  title,
  right,
  children,
  className = "",
  bodyClassName = "",
  defaultOpen = true,
}: {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  defaultOpen?: boolean;
}) {
  // The server has no localStorage, so it renders the default and the client
  // takes over with whatever was stored.
  const { open, toggle } = useFolded(title, defaultOpen);

  return (
    <section
      className={`flex min-h-0 flex-col rounded border border-line bg-panel/85 backdrop-blur-sm ${
        open ? className : "shrink-0"
      }`}
    >
      <header className="flex shrink-0 items-center gap-2 px-3 py-1.5">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={open}
          className="cursor-pointer text-[10px] uppercase tracking-[0.14em] text-faint hover:text-dim"
        >
          {open ? "− " : "+ "}
          {title}
        </button>
        <div className="ml-auto flex items-center gap-2 text-[10px] text-faint">{right}</div>
      </header>
      {open && (
        <div className={`min-h-0 flex-1 overflow-y-auto px-3 pb-2.5 ${bodyClassName}`}>
          {children}
        </div>
      )}
    </section>
  );
}

/** Light or dark, one click. */
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  return (
    <button
      type="button"
      onClick={toggle}
      title={theme === "light" ? "Switch to dark" : "Switch to light"}
      aria-label={theme === "light" ? "Switch to dark" : "Switch to light"}
      className="rounded-sm border border-line px-2 py-0.5 text-[11px] text-dim hover:border-line-bright hover:text-text"
    >
      {theme === "light" ? "☾ dark" : "☀ light"}
    </button>
  );
}

/** A label on the left, a value on the right, both on the same baseline. */
export function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 tabular-nums">
      <span className="text-faint">{label}</span>
      <span className="text-text">{children}</span>
    </div>
  );
}

const TONES = {
  neutral: "border-line-bright text-dim",
  accent: "border-accent/50 text-accent",
  good: "border-good/50 text-good",
  warn: "border-warn/50 text-warn",
  bad: "border-bad/50 text-bad",
} as const;

export function Badge({
  tone = "neutral",
  children,
  className = "",
}: {
  tone?: keyof typeof TONES;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${TONES[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  active,
  tone = "neutral",
  type = "button",
  title,
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  active?: boolean;
  tone?: "neutral" | "accent";
  type?: "button" | "submit";
  title?: string;
  className?: string;
}) {
  const live = active || tone === "accent";
  return (
    <button
      type={type}
      title={title}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-sm border px-3 py-1.5 transition-colors disabled:cursor-default disabled:opacity-35 ${
        live
          ? "border-accent/60 bg-accent/10 text-accent enabled:hover:bg-accent/20"
          : "border-line text-dim enabled:hover:border-line-bright enabled:hover:text-text"
      } ${className}`}
    >
      {children}
    </button>
  );
}

export function Select({
  value,
  onChange,
  children,
  title,
  className = "",
}: {
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  title?: string;
  className?: string;
}) {
  return (
    <select
      title={title}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className={`rounded-sm border border-line bg-panel px-2 py-1.5 text-dim hover:border-line-bright hover:text-text ${className}`}
    >
      {children}
    </select>
  );
}

export function TopBar({ children }: { children?: ReactNode }) {
  return (
    <header className="flex shrink-0 items-center gap-4 border-b border-line bg-panel/70 px-4 py-2">
      <Link href="/" className="flex items-baseline gap-2 hover:opacity-80">
        <span className="text-[13px] uppercase tracking-[0.22em] text-text">waypoint</span>
        <span className="hidden text-[10px] text-faint sm:inline">UAV mission studio</span>
      </Link>
      <nav className="flex items-center gap-3 text-[11px]">
        <Link href="/" className="text-dim hover:text-text">
          studio
        </Link>
        <Link href="/runs" className="text-dim hover:text-text">
          runs
        </Link>
      </nav>
      <div className="ml-auto flex items-center gap-2">
        {children}
        <ThemeToggle />
      </div>
    </header>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-3 text-faint">{children}</p>;
}
