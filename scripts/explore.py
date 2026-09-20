#!/usr/bin/env python3
"""Look around a place, then tell the aircraft what to do, and watch it in Gazebo.

    .venv/bin/python scripts/explore.py --world lodhi          # Lodhi Garden, Delhi
    .venv/bin/python scripts/explore.py --world munnar         # the hills above Munnar
    .venv/bin/python scripts/explore.py --world delhi --wind 3,0,0

Type a mission at the prompt in plain English:

    mission> photograph Bara Gumbad from two sides, then fly north up the avenue

The planner decomposes it, the verifier checks it against the buildings and the
terrain, the simulator flies it, and Gazebo opens on the finished flight. Close
the Gazebo window to get the prompt back, then type another mission.

Commands: /help /look /world <spec> /wind x,y,z /faults <spec> /retries <n> /again /quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Both the project root (for the packages) and scripts/ (for the sibling
# modules), so this works whether it is run as a script or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazebo.launch import GazeboError, play  # noqa: E402
from planner import (  # noqa: E402
    DEFAULT_CONSTRAINTS,
    MissionConstraints,
    Plan,
    PlannerError,
    plan as make_plan,
    replan as make_replan,
)
from orchestrator.loop import MissionReport  # noqa: E402
from orchestrator.runlog import read_run, write_run  # noqa: E402
from sim import Executor, SimConfig, Wind  # noqa: E402
from simulate import fence_around, load_faults, load_world, parse_wind, show_plan  # noqa: E402
from verifier import verify  # noqa: E402
from world.models import World  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORLDS = ROOT / "worlds"
RULE = "-" * 72


class Live:
    """What the next mission will be flown with.

    One object rather than a pile of local variables, because the prompt's
    commands all reach in and change it between flights.
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


def describe(world: World | None) -> None:
    """What you are about to fly in, and the names a mission can use."""
    if not world:
        print(f"{RULE}\nno world: open space above a flat plane")
        return
    summary = world.summary()
    print(f"{RULE}\n{world.name} ({world.source}): {summary['buildings']} "
          f"obstacles ({summary['trees']} trees), tallest roof {summary['tallest_roof']:g} m, "
          f"ground {summary['elevation_min']:g}..{summary['elevation_max']:g} m")
    if world.landmarks:
        print("you can name any of these in a mission:")
        for landmark in world.landmarks:
            print(f"  - {landmark.describe()}")


def world_names() -> list[str]:
    """Every world you can switch to: the built-in ones, then saved files."""
    from world.generate import BUILT_IN  # noqa: PLC0415 -- keeps startup cheap

    saved = sorted(path.stem for path in WORLDS.glob("*.json"))
    return list(BUILT_IN) + [name for name in saved if name not in BUILT_IN]


def watch(run: dict, speed: float, label: str) -> None:
    """Open Gazebo on a run (or on a world with nothing flying) and wait for it."""
    print(f"  opening Gazebo on {label}")
    try:
        play(run, speed=speed)
    except GazeboError as exc:
        print(f"  Gazebo: {exc}")


def fly(mission: str, live: Live, runs_dir: Path) -> Path | None:
    """Plan, verify (with up to `live.retries` corrections), fly, and save."""
    world, constraints = live.world, live.constraints
    config, wind, faults, retries = live.config, live.wind, live.faults, live.retries
    print(f"\n{RULE}\nplanning...")
    try:
        plan: Plan = make_plan(mission, constraints, world=world)
    except PlannerError as exc:
        print(f"  planning failed: {exc}")
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
        return None
    print("  verification passed")

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
    print(f"  {report.run_file}")
    return report.run_file


HELP = """
  <anything else>      a mission, in plain English
  /look                open Gazebo on the world with nothing flying
  /world <spec>        switch worlds: delhi, lodhi, chowk, munnar, or a worlds/*.json
  /worlds              list the worlds you can switch to
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
    parser.add_argument("--speed", type=float, default=1.0, help="Gazebo playback speed.")
    parser.add_argument("--no-watch", action="store_true",
                        help="Write the run file but do not open Gazebo on it.")
    parser.add_argument("--look", action="store_true",
                        help="Open Gazebo on the world before the first mission.")
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

    def look() -> None:
        """Gazebo on the current world, nothing flying: somewhere to look around."""
        if not live.world:
            print("  no world to look at: open space above a flat plane")
            return
        watch({"world": live.world.model_dump(mode="json")}, args.speed, live.world.name)

    describe(live.world)
    print("tell it what to do, and it flies in Gazebo. /help for commands.\n")
    if args.look:
        look()

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
            if line == "/look":
                look()
                continue
            if line.startswith("/world "):
                try:
                    live.use_world(load_world(line.split(maxsplit=1)[1]))
                except (OSError, ValueError) as exc:
                    print(f"  could not load that world: {exc}")
                    continue
                describe(live.world)
                continue
            if line == "/worlds":
                for name in world_names():
                    print(f"  {name}")
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
            run_file = fly(line, live, args.runs_dir)
            if run_file and not args.no_watch:
                watch(read_run(run_file), args.speed, run_file.name)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nsession ended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
