"""Start Gazebo on a run and fly it there, in real time.

Three processes: the gz server (the world), the gz GUI (the window; on macOS
the two cannot share a process), and `gazebo/bridge.py`, which this module
feeds poses over a pipe. Playback interpolates between the simulator's 10 Hz
ticks so the aircraft moves smoothly, and prints the flight log as it happens.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from waypoint.gazebo.scene import UAV_NAME, WORLD_NAME, Scene, write_scene
from waypoint.paths import ROOT

BRIDGE = Path(__file__).resolve().parent / "bridge.py"
# The project may live under a path with spaces, and URIs inside an SDF do not
# survive that; the scene goes somewhere that is guaranteed not to have any.
CACHE = Path.home() / ".cache" / "waypoint" / "gazebo"

FRAME_RATE = 30.0
TRAIL_EVERY = 0.5  # seconds of flight between trail points


class GazeboError(RuntimeError):
    pass


def gz_python() -> str:
    """An interpreter that can import Gazebo's Python bindings.

    `.venv-gz` first (Homebrew's Python with protobuf added, see the README),
    then `$WAYPOINT_GZ_PYTHON`, then whatever Homebrew Pythons are installed.
    """
    candidates = [str(ROOT / ".venv-gz" / "bin" / "python"), os.environ.get("WAYPOINT_GZ_PYTHON")]
    candidates += [f"/opt/homebrew/bin/python3.{minor}" for minor in range(14, 9, -1)]
    candidates += ["python3"]
    probe = "import gz.transport, gz.msgs.pose_pb2"
    for candidate in filter(None, candidates):
        if not (shutil.which(candidate) or Path(candidate).exists()):
            continue
        ok = subprocess.run([candidate, "-c", probe], capture_output=True).returncode == 0
        if ok:
            return candidate
    raise GazeboError(
        "no Python can import Gazebo's bindings (gz.transport, gz.msgs). Make one with:\n"
        "  /opt/homebrew/bin/python3.14 -m venv --system-site-packages .venv-gz\n"
        "  .venv-gz/bin/python -m pip install protobuf"
    )


def play(
    run: dict,
    *,
    speed: float = 1.0,
    gui: bool = True,
    follow: bool = True,
    out_dir: Path | None = None,
) -> int:
    """Open Gazebo on `run` and fly its ticks. Returns once the window is closed."""
    # The flight log is printed as it happens, including into a pipe or a file.
    sys.stdout.reconfigure(line_buffering=True)
    gz = shutil.which("gz")
    if gz is None:
        raise GazeboError("`gz` is not on PATH; install Gazebo first (brew install gz-jetty).")
    python = gz_python()

    slug = _slug(run.get("mission_text") or (run.get("world") or {}).get("name") or "world")
    out_dir = out_dir or CACHE / slug
    print(f"  writing the scene to {out_dir}")
    scene = write_scene(run, out_dir)

    env = {**os.environ, "GZ_SIM_RESOURCE_PATH": str(out_dir)}
    # gz-transport discovers peers by multicast; pin it to loopback so a VPN or
    # a second interface cannot split the server from the GUI.
    env.setdefault("GZ_IP", "127.0.0.1")

    processes: list[subprocess.Popen] = []

    def start(args: list[str], **kwargs) -> subprocess.Popen:
        process = subprocess.Popen(args, env=env, cwd=out_dir, start_new_session=True, **kwargs)
        processes.append(process)
        return process

    try:
        print("  starting gz sim (server)")
        start([gz, "sim", "-s", "-r", str(scene.sdf)])
        window = None
        if gui:
            print("  starting gz sim (GUI)")
            window = start([gz, "sim", "-g"])
        bridge = start(
            [python, str(BRIDGE), "--world", WORLD_NAME, "--model", UAV_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        status = bridge.stdout.readline().strip()
        if status != "ready":
            raise GazeboError(f"the Gazebo bridge did not come up: {status or 'no output'}")

        _fly(run, scene, bridge, speed=speed, follow=follow and gui, window=window)

        if window is None:
            return 0
        print("\n  flight over. Look around; close the Gazebo window (or Ctrl-C) to finish.")
        window.wait()
        return 0
    except KeyboardInterrupt:
        print("\n  stopping Gazebo")
        return 130
    finally:
        for process in reversed(processes):
            _stop(process)


def _fly(
    run: dict,
    scene: Scene,
    bridge: subprocess.Popen,
    *,
    speed: float,
    follow: bool,
    window: subprocess.Popen | None,
) -> None:
    ticks = run.get("ticks") or []

    def send(command: dict) -> None:
        bridge.stdin.write(json.dumps(command) + "\n")
        bridge.stdin.flush()

    if not ticks:
        print("  nothing to fly: showing the world only")
        return

    if follow:
        send({"op": "follow"})

    events = sorted(run.get("events") or [], key=lambda e: e["t"])
    late = sorted(scene.late, key=lambda o: o.t)
    fresh: list[list[float]] = []  # trail points the bridge has not been sent yet
    next_trail = 0.0
    next_status = 0.0
    duration = ticks[-1]["timestamp"]
    print(f"\n  flying {duration:.1f}s of flight at {speed:g}x\n")

    started = time.monotonic()
    index = 0
    frame = 0
    while True:
        if window is not None and window.poll() is not None:
            return
        t = (time.monotonic() - started) * speed
        while index + 1 < len(ticks) and ticks[index + 1]["timestamp"] <= t:
            index += 1
        a = ticks[index]
        b = ticks[min(index + 1, len(ticks) - 1)]
        span = b["timestamp"] - a["timestamp"]
        f = 0.0 if span <= 0 else min(max((t - a["timestamp"]) / span, 0.0), 1.0)
        x, y, z = (a[k] + (b[k] - a[k]) * f for k in ("x", "y", "z"))
        send({"op": "pose", "x": x, "y": y, "z": z, "yaw": _yaw(a, b, f)})

        while events and events[0]["t"] <= t:
            event = events.pop(0)
            print(f"  t={event['t']:7.1f}s  {event['type']:14} {event['description']}")
        while late and late[0].t <= t:
            send({"op": "spawn", "sdf": late.pop(0).sdf})
        if t >= next_trail:
            fresh.append([round(x, 2), round(y, 2), round(z, 2)])
            next_trail = t + TRAIL_EVERY
        # Only what is new: the bridge keeps the line and decides when to redraw.
        done = t >= duration
        if fresh or done:
            send({"op": "trail", "points": fresh, "final": done})
            fresh = []

        if done:
            print(f"  {_status(ticks[-1])}")
            for event in events:
                print(f"  t={event['t']:7.1f}s  {event['type']:14} {event['description']}")
            return
        if t >= next_status:
            print(f"  {_status(a)}")
            next_status = t + 5.0
        # Sleep to the next frame rather than for a whole frame, so the work
        # each one costs does not push playback behind the flight's own clock.
        # The window redraws at FRAME_RATE whatever `speed` is; speed only
        # changes how much flight each frame covers.
        frame += 1
        time.sleep(max(0.0, started + frame / FRAME_RATE - time.monotonic()))


def _yaw(a: dict, b: dict, f: float) -> float:
    """Heading between two ticks, turning the short way round."""
    start, end = math.radians(a.get("heading", 0.0)), math.radians(b.get("heading", 0.0))
    turn = (end - start + math.pi) % (2 * math.pi) - math.pi
    return start + turn * f


def _status(tick: dict) -> str:
    line = (
        f"t={tick['timestamp']:6.1f}s  pos=({tick['x']:7.1f}, {tick['y']:7.1f}, {tick['z']:6.1f})  "
        f"{tick.get('velocity', 0.0):5.1f} m/s  batt={tick.get('battery_pct', 0.0):5.1f}%"
    )
    if tick.get("faults"):
        line += f"  [{', '.join(tick['faults'])}]"
    return line


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _slug(text: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")
    return "-".join(filter(None, slug.split("-")))[:60] or "world"


if __name__ == "__main__":  # pragma: no cover
    sys.exit("run scripts/gazebo.py instead")
