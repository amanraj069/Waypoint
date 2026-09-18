/**
 * The playhead.
 *
 * Playback runs at frame rate and React does not, so the transport state lives
 * in a plain object the 3D view mutates directly. Components subscribe to two
 * different things: *control* changes (play, rate, seek, follow) which are rare
 * and re-render immediately, and *samples* — where the aircraft is right now —
 * which arrive many times a second and are throttled before they reach state.
 */

import type { Tick } from "./types";

export interface Sample {
  time: number;
  tick: Tick;
  index: number;
  speed: number;
}

/** The tick either side of `time`, and how far between them we are. */
export function frameAt(ticks: Tick[], time: number) {
  if (!ticks.length) return null;
  let lo = 0;
  let hi = ticks.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ticks[mid].timestamp < time) lo = mid + 1;
    else hi = mid;
  }
  const b = ticks[lo];
  const a = ticks[Math.max(0, lo - 1)];
  const span = b.timestamp - a.timestamp;
  const k = span > 0 ? Math.min(1, Math.max(0, (time - a.timestamp) / span)) : 0;
  return { a, b, k, index: lo };
}

export const lerp = (a: number, b: number, k: number) => a + (b - a) * k;

/** How often the HUD is allowed to re-render, in seconds of wall clock. */
const SAMPLE_INTERVAL = 1 / 12;

export class Playback {
  time = 0;
  playing = false;
  rate = 1;
  follow = false;
  duration = 0;
  /** Bumped on every seek, so the 3D view knows to redraw a paused frame. */
  seekNonce = 0;
  /** Bumped on every control change, for `useSyncExternalStore`. */
  version = 0;

  private controlListeners = new Set<() => void>();
  private sampleListeners = new Set<(sample: Sample) => void>();
  private lastSampleAt = 0;

  // -- control ------------------------------------------------------------

  subscribe = (listener: () => void) => {
    this.controlListeners.add(listener);
    return () => this.controlListeners.delete(listener);
  };

  getSnapshot = () => this.version;

  private changed() {
    this.version += 1;
    for (const listener of this.controlListeners) listener();
  }

  load(duration: number) {
    this.duration = duration;
    this.time = 0;
    this.playing = false;
    this.seekNonce += 1;
    this.changed();
  }

  setPlaying(on: boolean) {
    if (this.playing === on) return;
    // Playing on from the very end would sit there doing nothing; start again.
    if (on && this.duration > 0 && this.time >= this.duration - 1e-6) this.time = 0;
    this.playing = on;
    this.changed();
  }

  toggle() {
    this.setPlaying(!this.playing);
  }

  setRate(rate: number) {
    this.rate = rate;
    this.changed();
  }

  setFollow(on: boolean) {
    this.follow = on;
    this.changed();
  }

  seek(time: number) {
    this.time = Math.min(Math.max(time, 0), this.duration);
    this.seekNonce += 1;
    this.changed();
  }

  nudge(seconds: number) {
    this.seek(this.time + seconds);
  }

  restart() {
    this.seek(0);
    this.setPlaying(true);
  }

  // -- samples ------------------------------------------------------------

  onSample = (listener: (sample: Sample) => void) => {
    this.sampleListeners.add(listener);
    return () => this.sampleListeners.delete(listener);
  };

  /** Called from the render loop. Throttled, and always fires on a seek. */
  publish(sample: Sample, now: number, force = false) {
    if (!force && now - this.lastSampleAt < SAMPLE_INTERVAL) return;
    this.lastSampleAt = now;
    for (const listener of this.sampleListeners) listener(sample);
  }

  /** Advance the playhead; returns true when it has just reached the end. */
  advance(dt: number): boolean {
    if (!this.playing || this.duration <= 0) return false;
    this.time += dt * this.rate;
    if (this.time >= this.duration) {
      this.time = this.duration;
      this.playing = false;
      this.changed();
      return true;
    }
    return false;
  }
}
