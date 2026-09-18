#!/usr/bin/env python3
"""Start the whole platform: the API, the web app, and a browser on it.

    .venv/bin/python scripts/studio.py

The two halves are separate processes because they are separate languages --
the pipeline is Python and the interface is Next.js -- and this is the one
command that puts them together, waits until both actually answer, and shuts
them both down on Ctrl-C.

    --api-port / --web-port   move either one
    --no-open                 do not open a browser
    --api-only                just the API, for talking to it yourself
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status < 500
    except (urllib.error.HTTPError,):
        return True            # it answered, which is all we are asking
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def wait_for(url: str, process: subprocess.Popen, label: str, seconds: float = 60.0) -> bool:
    """Poll until it answers, or until the process it belongs to gives up."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            print(f"  {label} exited with code {process.returncode}", file=sys.stderr)
            return False
        if reachable(url):
            return True
        time.sleep(0.4)
    print(f"  {label} did not answer on {url} within {seconds:.0f}s", file=sys.stderr)
    return False


def check_api_dependencies() -> bool:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "The API needs fastapi and uvicorn, which are not installed:\n"
            '  uv pip install -e ".[dev,web]"',
            file=sys.stderr,
        )
        return False
    return True


def check_web_dependencies() -> bool:
    if not (WEB / "node_modules").exists():
        print(
            f"The web app's dependencies are not installed:\n  cd {WEB} && npm install",
            file=sys.stderr,
        )
        return False
    if shutil.which("npm") is None:
        print("npm is not on PATH; install Node 20 or newer.", file=sys.stderr)
        return False
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--web-port", type=int, default=3000)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--reload", action="store_true", help="Restart the API on a code change.")
    args = parser.parse_args(argv)

    if not check_api_dependencies():
        return 1
    if not args.api_only and not check_web_dependencies():
        return 1

    processes: list[tuple[str, subprocess.Popen]] = []
    api_url = f"http://127.0.0.1:{args.api_port}"

    api = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "127.0.0.1", "--port", str(args.api_port),
            *(["--reload"] if args.reload else []),
        ],
        cwd=ROOT,
    )
    processes.append(("api", api))
    print(f"api   {api_url}")

    try:
        if not wait_for(f"{api_url}/api/health", api, "the API"):
            raise SystemExit(1)

        health = "unknown"
        try:
            import json  # noqa: PLC0415

            with urllib.request.urlopen(f"{api_url}/api/health", timeout=3) as response:
                body = json.load(response)
            health = (
                f"{body['worlds']} worlds, {body['runs']} runs, "
                + (f"planner {body['planner_model']}" if body["planner_ready"]
                   else "no planner key -- bundled plans only")
            )
        except Exception:  # noqa: BLE001 -- a summary line is not worth failing over
            pass
        print(f"      {health}")

        url = api_url
        if not args.api_only:
            web = subprocess.Popen(
                ["npm", "run", "dev", "--", "--port", str(args.web_port)],
                cwd=WEB,
                # The rewrite in next.config.ts needs to know where the API went.
                env={**os.environ, "WAYPOINT_API": api_url},
            )
            processes.append(("web", web))
            url = f"http://localhost:{args.web_port}"
            print(f"web   {url}")
            if not wait_for(url, web, "the web app", seconds=90):
                raise SystemExit(1)

        print("\nready. Ctrl-C to stop both.\n")
        if not args.no_open:
            webbrowser.open(url)

        # Wait on whichever falls over first, then take the other down with it.
        while True:
            for label, process in processes:
                if process.poll() is not None:
                    print(f"\n{label} exited ({process.returncode}); stopping the rest.")
                    return process.returncode or 0
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for label, process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for label, process in reversed(processes):
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
