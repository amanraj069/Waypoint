#!/usr/bin/env python3
"""Look around a place, then tell the aircraft what to do, and watch it happen.

    .venv/bin/python scripts/explore.py --world lodhi          # Lodhi Garden, Delhi
    .venv/bin/python scripts/explore.py --world munnar         # the hills above Munnar
    .venv/bin/python scripts/explore.py --world delhi --wind 3,0,0

The browser opens on the world with nothing flying in it, so you can orbit
around and see what you are dealing with. Then type a mission at the prompt in
plain English:

    mission> photograph Bara Gumbad from two sides, then fly north up the avenue

The planner decomposes it, the verifier checks it against the buildings and the
terrain, the simulator flies it, and the page you are already looking at loads
the finished run by itself. Type another mission and it does it again.

Commands: /help /world <spec> /wind x,y,z /faults <spec> /retries <n> /again /quit
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

# Both the project root (for the packages) and scripts/ (for the sibling
# modules), so this works whether it is run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner import (  # noqa: E402
    DEFAULT_CONSTRAINTS,
    MissionConstraints,
    Plan,
    PlannerError,
    plan as make_plan,
    replan as make_replan,
)
from orchestrator.loop import MissionReport  # noqa: E402
from orchestrator.runlog import write_run  # noqa: E402
from serve_viewer import (  # noqa: E402
    COMMAND_HANDLERS,
    flyable_runs,
    start_server,
    world_index,
    write_index,
    write_world_index,
)
from sim import Executor, SimConfig, Wind  # noqa: E402
from simulate import fence_around, load_faults, load_world, parse_wind, show_plan  # noqa: E402
from verifier import verify  # noqa: E402
from world.models import World  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viewer"
RULE = "-" * 72
_preview_version = itertools.count(1)


class Live:
    """What the next mission will be flown with.

    The prompt and the browser both change this, so it lives in one object
    rather than in local variables the page cannot reach.
    """

    def __init__(self, world: World | None, constraints: MissionConstraints,
                 config: SimConfig, wind: Wind, faults, retries: int):
        self.world = world
        self.constraints = constraints
        self.config = config
        self.wind = wind
        self.faults = faults
        self.retries = retries

    def use_world(self, world: World | None) -> None:
        """Switch worlds, and fence the new one."""
        self.world = world
        if world:
            self.constraints = self.constraints.model_copy(update={"geofence": fence_around(world)})


class Session:
    """The bit of shared state the browser polls: what to show, and what we are doing."""

    def __init__(self) -> None:
        self.path = VIEWER / "session.json"
        self.run: str | None = None
        self.write("starting up")

    def write(self, message: str, run: str | None = None) -> None:
        if run:
            self.run = run
        self.path.write_text(
            json.dumps({"run": self.run, "message": message, "updated": time.time()}) + "\n"
        )

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def preview_file(world: World, constraints: MissionConstraints) -> str:
    """A run file with a world and no flight: somewhere to look around."""
    payload = {
        "schema_version": 2,
        "mission_text": f"{world.name} — nothing has flown yet",
        "constraints": constraints.model_dump(mode="json"),
        "world": world.model_dump(mode="json"),
        "waypoints": [],
        "ticks": [],
        "events": [],
        "obstacles": [],
    }
    out = VIEWER / "preview.json"
    out.write_text(json.dumps(payload) + "\n")
    # Versioned, so a page that is polling notices a *different* world even
    # though it is always the same file. A counter rather than a clock: two
    # switches in the same millisecond must still look different.
    return f"/viewer/preview.json?v={next(_preview_version)}"


def serve(port: int) -> tuple[ThreadingHTTPServer, int]:
    """Start serving in the background on the first free port at or above `port`."""
    server, bound = start_server(port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, bound


def fly(mission: str, live: Live, session: Session, runs_dir: Path) -> Path | None:
    """Plan, verify (with up to `live.retries` corrections), fly, and save."""
    world, constraints = live.world, live.constraints
    config, wind, faults, retries = live.config, live.wind, live.faults, live.retries
    session.write(f"planning: {mission}")
    print(f"\n{RULE}\nplanning...")
    try:
        plan: Plan = make_plan(mission, constraints, world=world)
    except PlannerError as exc:
        print(f"  planning failed: {exc}")
        session.write("planning failed")
        return None
    show_plan(plan, None)

    result = verify(plan, constraints, world=world)
    for attempt in range(retries):
        if result.passed:
            break
        print(f"\n{RULE}\nverifier rejected it:")
        for violation in result.blocking:
            print(f"  [{violation.type.value}] {violation.description}")
        print(f"\nasking for a correction ({attempt + 1}/{retries})...")
        session.write(f"replanning ({attempt + 1}/{retries})")
        try:
            plan = make_replan(mission, plan, result.messages(), constraints, world=world)
        except PlannerError as exc:
            print(f"  replanning failed: {exc}")
            break
        show_plan(plan, None)
        result = verify(plan, constraints, world=world)

    for violation in result.warnings:
        print(f"  [warning] [{violation.type.value}] {violation.description}")
    if not result.passed:
        print("\nstill unsafe -- not flying it:")
        for violation in result.blocking:
            print(f"  [{violation.type.value}] {violation.description}")
        session.write("plan rejected as unsafe")
        return None
    print("  verification passed")

    session.write("flying")
    print(f"\n{RULE}\nflying...")
    executor = Executor(
        plan, constraints=constraints, config=config, wind=wind, faults=faults, world=world
    )
    report = MissionReport(
        mission_text=mission,
        plan=plan,
        verification=result,
        constraints=constraints,
        sim_config=config,
        wind=wind,
        faults=faults,
        world=world,
    )
    report.states = list(executor.run())
    report.events = executor.event_records()
    report.obstacles = executor.obstacle_records()
    report.distance_flown = executor.distance_flown
    report.crashed = executor.crashed

    for event in report.events:
        print(f"  t={event['t']:7.1f}s  {event['type']:14} {event['description']}")

    outcome = "landed" if report.completed else "CRASHED" if report.crashed else "did not land cleanly"
    print(
        f"\n  {report.duration:.0f}s | {report.distance_flown:.0f} m flown | "
        f"{report.battery_remaining:.1f}% battery left | {outcome}"
    )

    report.run_file = write_run(report, runs_dir=runs_dir)
    write_index(flyable_runs())
    session.write(f"{outcome} — {report.duration:.0f}s", run=f"/runs/{report.run_file.name}")
    print(f"  {report.run_file}\n  ...loading in the browser")
    return report.run_file


HELP = """
  <anything else>      a mission, in plain English
  /world <spec>        switch worlds: delhi, lodhi, chowk, munnar, or a worlds/*.json
  /worlds              list the worlds the picker offers
  /wind x,y,z          set the steady wind in m/s (/wind 0,0,0 for calm)
  /faults <spec>       sample, none, or a schedule JSON
  /retries <n>         how many times to ask for a correction after a rejection
  /again               re-fly the last mission
  /help  /quit
"""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default="lodhi",
                        help="delhi, lodhi, chowk, munnar, or a worlds/*.json")
    parser.add_argument("--wind", help="Steady wind as 'x,y,z' in m/s.")
    parser.add_argument("--turbulence", type=float, default=0.0)
    parser.add_argument("--faults", default="none", help="'sample', 'none', or a schedule JSON.")
    parser.add_argument("--retries", type=int, default=2, help="Corrections to ask for.")
    parser.add_argument("--max-altitude", type=float)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    constraints = DEFAULT_CONSTRAINTS
    if args.max_altitude:
        constraints = constraints.model_copy(update={"max_altitude": args.max_altitude})
    live = Live(
        world=None,
        constraints=constraints,
        config=SimConfig(),
        wind=parse_wind(args.wind, args.turbulence),
        faults=load_faults(args.faults, Plan()),
        retries=args.retries,
    )
    live.use_world(load_world(args.world))

    session = Session()
    write_index(flyable_runs())
    write_world_index()

    def switch_world(spec: str) -> str:
        """The viewer's world picker, arriving over /command?world=..."""
        path = spec.lstrip("/") if spec.startswith("/worlds/") else spec
        live.use_world(load_world(path))
        name = live.world.name if live.world else "open space"
        session.write(f"looking around {name}",
                      run=preview_file(live.world, live.constraints) if live.world else None)
        print(f"\n[viewer] switched to {name} -- the next mission flies there")
        print("mission> ", end="", flush=True)
        return f"now flying in {name}"

    COMMAND_HANDLERS["world"] = switch_world
    try:
        server, port = serve(args.port)
    except OSError as exc:
        print(f"Could not serve: {exc}", file=sys.stderr)
        return 1

    url = f"http://localhost:{port}/viewer/index.html?session=1"
    if live.world:
        session.write(f"looking around {live.world.name}",
                      run=preview_file(live.world, live.constraints))
        summary = live.world.summary()
        print(f"{RULE}\n{live.world.name} ({live.world.source}): {summary['buildings']} "
              f"obstacles ({summary['trees']} trees), tallest roof {summary['tallest_roof']:g} m, "
              f"ground {summary['elevation_min']:g}..{summary['elevation_max']:g} m")
        if live.world.landmarks:
            print("you can name any of these in a mission:")
            for landmark in live.world.landmarks:
                print(f"  - {landmark.describe()}")
    else:
        print(f"{RULE}\nno world: open space above a flat plane")
    print(f"viewer: {url}")
    print("have a look around, then tell it what to do. /help for commands.")
    print("(the world picker in the page switches worlds too)\n")
    if not args.no_open:
        webbrowser.open(url)

    last = ""
    try:
        while True:
            try:
                line = input("mission> ").strip()
            except EOFError:
                break
            if not line:
                continue

            if line in ("/quit", "/exit", "/q"):
                break
            if line == "/help":
                print(HELP)
                continue
            if line.startswith("/world "):
                try:
                    live.use_world(load_world(line.split(maxsplit=1)[1]))
                except (OSError, ValueError) as exc:
                    print(f"  could not load that world: {exc}")
                    continue
                name = live.world.name if live.world else "open space"
                session.write(f"looking around {name}",
                              run=preview_file(live.world, live.constraints) if live.world else None)
                print(f"  now flying in {name}")
                continue
            if line == "/worlds":
                for entry in world_index():
                    print(f"  {entry['name']:18} {entry['url']}  ({entry['detail']})")
                continue
            if line.startswith("/wind "):
                live.wind = parse_wind(line.split(maxsplit=1)[1], args.turbulence)
                print(f"  wind is now {live.wind.vector} m/s")
                continue
            if line.startswith("/faults "):
                live.faults = load_faults(line.split(maxsplit=1)[1], Plan())
                print(f"  {len(live.faults)} fault(s) armed")
                continue
            if line.startswith("/retries "):
                live.retries = int(line.split()[1])
                print(f"  will ask for up to {live.retries} correction(s)")
                continue
            if line == "/again":
                if not last:
                    print("  nothing flown yet")
                    continue
                line = last
            elif line.startswith("/"):
                print(f"  unknown command. {HELP}")
                continue

            last = line
            fly(line, live, session, args.runs_dir)
    except KeyboardInterrupt:
        pass
    finally:
        session.clear()
        server.shutdown()
        print("\nsession ended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
