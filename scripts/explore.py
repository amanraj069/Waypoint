#!/usr/bin/env python3
"""Look at a place, choose where to take off and land, say what to do, watch it fly.

    .venv/bin/python scripts/explore.py --world lodhi

Then, at the prompt:

    /look                               1. see the place in Gazebo
    /start Munnar town                  2. where it takes off   (a landmark, or x,y)
    /land 120,-40                          and where it finishes (default: back at the start)
    photograph Bara Gumbad, then land   3. the mission, in plain English

Each mission goes through the closed loop: Gemini plans it from your start to
your landing point, the verifier checks it, the simulator flies it, the monitor
replans in the air if something gets in the way, and Gazebo opens on the
finished flight. Close the Gazebo window to get the prompt back.

Pads are surveyed, not assumed: a point on a slope or inside a building is moved
to the nearest flat, clear ground within 120 m, and you are told how far.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waypoint.backends import make_backend  # noqa: E402
from waypoint.gazebo.launch import GazeboError, play  # noqa: E402
from waypoint.orchestrator import run_mission  # noqa: E402
from waypoint.orchestrator.options import system_from  # noqa: E402
from waypoint.orchestrator.runlog import read_run  # noqa: E402
from waypoint.paths import RUNS_DIR, WORLDS_DIR  # noqa: E402
from waypoint.planner import Plan  # noqa: E402
from waypoint.scenario import (  # noqa: E402
    constraints_for,
    load_faults,
    load_world,
    pad_at,
    parse_wind,
    resolve_point,
)
from shapely.geometry import Point, Polygon  # noqa: E402

from waypoint.world.landing import Site, survey  # noqa: E402
from waypoint.world.models import World  # noqa: E402

RULE = "-" * 72


class Session:
    """Everything the next mission is flown with; the prompt's commands change it."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.wind = parse_wind(args.wind, args.turbulence)
        self.faults = args.faults
        self.world: World | None = None
        self.start: Site = survey(None, 0.0, 0.0)
        self.landing: Site | None = None  # None: back where it started

    # -- where -----------------------------------------------------------------

    def use_world(self, world: World | None) -> None:
        self.world = world
        self.constraints, notes = constraints_for(world, self.args.max_altitude)
        # Every world keeps its origin clear, so that is the default pad.
        self.start = survey(world, 0.0, 0.0)
        self.landing = None
        describe_world(world)
        for note in notes:
            print(f"  ({note})")

    def _point(self, spec: str) -> tuple[float, float, str]:
        """Resolve a spec, and refuse anything outside the geofence."""
        x, y, label = resolve_point(self.world, spec)
        fence = Polygon([(p.x, p.y) for p in self.constraints.geofence])
        if not fence.covers(Point(x, y)):
            minx, miny, maxx, maxy = fence.bounds
            raise ValueError(
                f"({x:g}, {y:g}) is outside the geofence: x {minx:.0f}..{maxx:.0f}, "
                f"y {miny:.0f}..{maxy:.0f} m"
            )
        return x, y, label

    def set_start(self, spec: str) -> None:
        x, y, label = self._point(spec)
        self.start = pad_at(self.world, x, y, "take off")
        print(f"  start: {describe_site(self.start, label)}")

    def set_landing(self, spec: str) -> None:
        if spec.strip().lower() in ("start", "home", "same", "back"):
            self.landing = None
            print("  landing: back at the start")
            return
        x, y, label = self._point(spec)
        self.landing = pad_at(self.world, x, y, "land")
        print(f"  landing: {describe_site(self.landing, label)}")

    def show_pads(self) -> None:
        print(f"  start:   {describe_site(self.start)}")
        print(f"  landing: {describe_site(self.landing) if self.landing else 'back at the start'}")

    @property
    def landing_site(self) -> Site:
        return self.landing or self.start

    # -- looking and flying ----------------------------------------------------

    def look(self) -> None:
        """Gazebo on the world with the pads marked, and nothing flying."""
        if not self.world:
            print("  no world to look at: open space above a flat plane")
            return
        run = {
            "world": self.world.model_dump(mode="json"),
            "constraints": self.constraints.model_dump(mode="json"),
            "pads": {"start": list(self.start.position), "landing": list(self.landing_site.position)},
        }
        print("  green pad: start; amber pad: landing (shown when it is somewhere else)")
        watch(run, self.args.speed, self.world.name)

    def fly(self, mission: str) -> None:
        backend = make_backend("custom", world=self.world, wind=self.wind)
        faults = load_faults(self.faults, Plan())
        log = run_mission(
            mission,
            self.constraints,
            faults,
            backend,
            world=self.world,
            home=self.start.position,
            landing=self.landing_site.position,
            max_replan_attempts=self.args.max_attempts,
            max_total_replans=self.args.max_replans,
            runs_dir=self.args.runs_dir,
            system=system_from(self.args.system),
        )
        report = log.report
        if self.args.no_watch or not (report and report.run_file and report.states):
            return
        watch(read_run(report.run_file), self.args.speed, report.run_file.name)


def describe_world(world: World | None) -> None:
    if not world:
        print(f"{RULE}\nno world: open space above a flat plane")
        return
    summary = world.summary()
    minx, miny, maxx, maxy = world.bounds
    print(f"{RULE}\n{world.name} ({world.source}): {summary['buildings']} "
          f"obstacles ({summary['trees']} trees), tallest roof {summary['tallest_roof']:g} m, "
          f"ground {summary['elevation_min']:g}..{summary['elevation_max']:g} m")
    print(f"x runs {minx:.0f}..{maxx:.0f} m (east), y {miny:.0f}..{maxy:.0f} m (north), "
          "from the origin at (0, 0)")
    if world.landmarks:
        print("landmarks (use a name for /start and /land, or in a mission):")
        for landmark in world.landmarks:
            print(f"  - {landmark.describe()}")


def describe_site(site: Site, label: str = "") -> str:
    where = f"({site.x:.1f}, {site.y:.1f}), ground at {site.elevation:.1f} m"
    named = f"{label} " if label and not label.startswith("(") else ""
    moved = f"; moved {site.moved:.0f} m to the nearest flat, clear ground" if site.moved else ""
    return f"{named}{where}{moved}"


def world_names() -> list[str]:
    """Every world you can switch to: the built-in ones, then saved files."""
    from waypoint.world.generate import BUILT_IN  # noqa: PLC0415 -- keeps startup cheap

    saved = sorted(path.stem for path in WORLDS_DIR.glob("*.json"))
    return list(BUILT_IN) + [name for name in saved if name not in BUILT_IN]


def watch(run: dict, speed: float, label: str) -> None:
    """Open Gazebo on a run (or on a world with nothing flying) and wait for it."""
    print(f"  opening Gazebo on {label}; close the window to come back here")
    try:
        play(run, speed=speed)
    except GazeboError as exc:
        print(f"  Gazebo: {exc}")


HELP = """
  1. see it       /look               open Gazebo on the world, pads marked
                  /landmarks          list the named places
  2. pads         /start <where>      where it takes off: a landmark name, or x,y
                  /land <where>       where it finishes: a landmark, x,y, or 'start'
                  /pads               show both
  3. fly          <anything else>     a mission, in plain English
                  /again              fly the last mission again

  /world <name>   switch places (resets the pads): /worlds lists them
  /wind x,y,z     steady wind in m/s (/wind 0,0,0 for calm)
  /faults <spec>  none, crane_ahead, storm, gps_slip, or a schedule JSON
  /help  /quit
"""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", default="lodhi",
                        help="a name in data/worlds (lodhi, munnar, ...) or a world JSON")
    parser.add_argument("--start", help="Where it takes off: a landmark name, or x,y.")
    parser.add_argument("--land", help="Where it finishes: a landmark name, x,y, or 'start'.")
    parser.add_argument("--wind", help="Steady wind as 'x,y,z' in m/s.")
    parser.add_argument("--turbulence", type=float, default=0.0)
    parser.add_argument("--faults", default="none",
                        help="none, crane_ahead, storm, gps_slip, or a schedule JSON.")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Corrections to ask for after each verifier rejection.")
    parser.add_argument("--max-replans", type=int, default=3,
                        help="In-flight replans allowed before landing where it is.")
    parser.add_argument("--max-altitude", type=float)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--system", default="full",
                        help="full (default), classic, or a preset with switches, e.g. full,no-shield.")
    parser.add_argument("--speed", type=float, default=1.0, help="Gazebo playback speed.")
    parser.add_argument("--no-watch", action="store_true",
                        help="Write the run file but do not open Gazebo on it.")
    parser.add_argument("--look", action="store_true",
                        help="Open Gazebo on the world before the first mission.")
    args = parser.parse_args(argv)

    session = Session(args)
    session.use_world(load_world(args.world))
    try:
        if args.start:
            session.set_start(args.start)
        if args.land:
            session.set_landing(args.land)
    except ValueError as exc:
        print(f"  {exc}", file=sys.stderr)
        return 2
    print(RULE)
    session.show_pads()
    print("\n/look to see it, /start and /land to move the pads, then type a mission. /help for more.\n")
    if args.look:
        session.look()

    last = ""
    try:
        while True:
            try:
                line = input("mission> ").strip()
            except EOFError:
                break
            if not line:
                continue
            command, _, rest = line.partition(" ")
            rest = rest.strip()
            try:
                if command in ("/quit", "/exit", "/q"):
                    break
                elif command == "/help":
                    print(HELP)
                elif command == "/look":
                    session.look()
                elif command == "/landmarks":
                    describe_world(session.world)
                elif command == "/start" and rest:
                    session.set_start(rest)
                elif command in ("/land", "/landing") and rest:
                    session.set_landing(rest)
                elif command == "/pads":
                    session.show_pads()
                elif command == "/world" and rest:
                    session.use_world(load_world(rest))
                    session.show_pads()
                elif command == "/worlds":
                    for name in world_names():
                        print(f"  {name}")
                elif command == "/wind" and rest:
                    session.wind = parse_wind(rest, args.turbulence)
                    print(f"  wind is now {session.wind.vector} m/s")
                elif command == "/faults" and rest:
                    load_faults(rest, Plan())  # fail now, not after planning
                    session.faults = rest
                    print(f"  faults: {rest}")
                elif command == "/again":
                    if not last:
                        print("  nothing flown yet")
                    else:
                        session.fly(last)
                elif line.startswith("/"):
                    print(f"  not a command, or it needs an argument. {HELP}")
                else:
                    last = line
                    session.fly(line)
            except (ValueError, OSError) as exc:
                print(f"  {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nsession ended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
