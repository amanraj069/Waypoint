#!/usr/bin/env python3
"""Serve the 3D viewer and open a run in it.

    python scripts/serve_viewer.py                      # newest run
    python scripts/serve_viewer.py runs/my-run.json     # a specific run
    python scripts/serve_viewer.py --no-open --port 9000

Uses Python's own http.server, so there is nothing to install: it serves the
project root, writes `viewer/runs.json` so the viewer's dropdown can list past
runs, and opens `viewer/index.html?run=/runs/<file>`.

Serving is what makes the viewer able to fetch a run file; opening index.html
straight off disk works too, but then you have to drop the JSON onto the page.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
import webbrowser
from collections.abc import Callable
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
WORLDS_DIR = ROOT / "worlds"
VIEWER_DIR = ROOT / "viewer"

# Worlds that ship with the project, written out on first serve so the viewer's
# picker has a real file to fetch for each of them.
from world.generate import BUILT_IN as BUILT_IN_WORLDS  # noqa: E402

# A page can ask the session to change something (`/command?world=...`).
# Only whatever is driving the session registers handlers here; with none, the
# endpoint politely does nothing.
COMMAND_HANDLERS: dict[str, Callable[[str], str]] = {}


def python_command() -> str:
    """How to invoke this interpreter from the project directory.

    Printing a bare "python" is a lie on a machine where the virtualenv is not
    activated, which is most of them.
    """
    exe = Path(sys.executable)
    try:
        return str(exe.relative_to(Path.cwd()))
    except ValueError:
        return str(exe)

def flyable_runs() -> list[Path]:
    """Run files with telemetry in them, newest first."""
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if json.loads(path.read_text()).get("ticks"):
                runs.append(path)
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def write_index(runs: list[Path]) -> None:
    """The dropdown's data: label and URL for each run the viewer can load."""
    entries = []
    for path in runs:
        try:
            run = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        duration = run.get("sim", {}).get("duration_s")
        suffix = f" — {duration:.0f}s" if isinstance(duration, (int, float)) else ""
        entries.append(
            {
                "url": f"/runs/{path.name}",
                "label": f"{run.get('mission_text', path.stem)} ({when}{suffix})",
            }
        )
    VIEWER_DIR.mkdir(exist_ok=True)
    (VIEWER_DIR / "runs.json").write_text(json.dumps(entries, indent=2) + "\n")


def ensure_built_in_worlds() -> None:
    """Materialise the shipped worlds so the picker can offer them."""
    from world.generate import build  # noqa: PLC0415 -- only needed when serving

    WORLDS_DIR.mkdir(parents=True, exist_ok=True)
    for name in BUILT_IN_WORLDS:
        path = WORLDS_DIR / f"{name}.json"
        if not path.exists():
            path.write_text(build(name).model_dump_json(indent=2) + "\n")


def world_index() -> list[dict]:
    """Every world in worlds/, with enough detail to choose between them."""
    ensure_built_in_worlds()
    entries = []
    for path in sorted(WORLDS_DIR.glob("*.json")):
        try:
            world = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        buildings = world.get("buildings") or []
        trees = sum(1 for b in buildings if b.get("kind") == "tree")
        landmarks = [lm["name"] for lm in (world.get("landmarks") or [])]
        parts = []
        if trees:
            parts.append(f"{trees} trees")
        if len(buildings) - trees:
            parts.append(f"{len(buildings) - trees} buildings")
        if landmarks:
            parts.append(f"{len(landmarks)} landmarks")
        entries.append(
            {
                "url": f"/worlds/{path.name}",
                "name": world.get("name", path.stem),
                "detail": ", ".join(parts) or "empty",
                "landmarks": landmarks,
            }
        )
    return entries


def write_world_index() -> None:
    VIEWER_DIR.mkdir(exist_ok=True)
    (VIEWER_DIR / "worlds.json").write_text(json.dumps(world_index(), indent=2) + "\n")


def start_server(port: int, tries: int = 12) -> tuple[ThreadingHTTPServer, int]:
    """Bind the first free port at or above `port`.

    A viewer left running in another terminal should not stop you opening
    another one, so this steps forward rather than failing.
    """
    handler = partial(QuietHandler, directory=str(ROOT))
    for candidate in range(port, port + tries):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            # Report the port actually bound, not the one asked for: port 0
            # means "any free port", and the caller needs to know which.
            return server, server.server_address[1]
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            if candidate == port:
                print(
                    f"port {port} is already in use -- probably a viewer you left running.\n"
                    f"  (to reclaim it:  lsof -ti:{port} | xargs kill)",
                    file=sys.stderr,
                )
    raise OSError(
        errno.EADDRINUSE,
        f"ports {port}-{port + tries - 1} are all in use. Free one, or pass --port.",
    )


class QuietHandler(SimpleHTTPRequestHandler):
    """Same server, without a log line for every asset the page pulls.

    It also answers `/command?key=value`, which is how the page asks the
    session driving it to change something -- switching worlds, for instance.
    """

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802 -- http.server's spelling
        parsed = urlparse(self.path)
        if parsed.path != "/command":
            return super().do_GET()

        messages = []
        for key, values in parse_qs(parsed.query).items():
            handler = COMMAND_HANDLERS.get(key)
            if handler is None:
                messages.append(f"nothing is listening for '{key}'")
                continue
            try:
                messages.append(handler(values[0]))
            except Exception as exc:  # a bad command must not kill the server
                messages.append(f"{key} failed: {exc}")

        body = json.dumps({"messages": messages}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(run_file: Path | str | None = None, port: int = 8765, open_browser: bool = True) -> int:
    runs = flyable_runs()
    write_index(runs)
    write_world_index()

    target = Path(run_file) if run_file else (runs[0] if runs else None)
    if target is None:
        print(f"No run files with telemetry in {RUNS_DIR}. Fly one first:", file=sys.stderr)
        print(f"  {python_command()} scripts/simulate.py --plan <plan.json>", file=sys.stderr)
        return 1

    try:
        server, port = start_server(port)
    except OSError as exc:
        print(f"Could not serve: {exc}", file=sys.stderr)
        return 1

    url = f"http://localhost:{port}/viewer/index.html?run=/runs/{Path(target).name}"
    print(f"Serving {ROOT} at http://localhost:{port}")
    print(f"Viewer:  {url}")
    print("Ctrl-C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", help="Run file to open; defaults to the newest.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Serve without opening a browser.")
    args = parser.parse_args(argv)
    return serve(args.run, port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
