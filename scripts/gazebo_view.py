#!/usr/bin/env python3
"""Watch a mission in Gazebo: replay a run file, or just walk around a world.

    # replay a run the simulator already flew
    python scripts/gazebo_view.py runs/<mission>_<stamp>.json

    # the newest run, at double speed
    python scripts/gazebo_view.py --latest --speed 2

    # a world on its own, nothing flying
    python scripts/gazebo_view.py --world lodhi

To plan, verify, fly *and* watch in one go, use `scripts/simulate.py ... --gazebo`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazebo.launch import GazeboError, play  # noqa: E402
from orchestrator.runlog import read_run  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", nargs="?", type=Path, help="A run file from runs/.")
    parser.add_argument("--latest", action="store_true", help="Replay the newest run in runs/.")
    parser.add_argument("--world", help="Show a world with nothing flying: 'lodhi', a worlds/*.json...")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--headless", action="store_true", help="Server only, no Gazebo window.")
    parser.add_argument("--no-follow", action="store_true", help="Leave the camera where it is.")
    parser.add_argument("--out", type=Path, help="Where to write the generated scene.")
    args = parser.parse_args(argv)

    if args.latest:
        runs = sorted(Path("runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not runs:
            parser.error("no runs in runs/ yet; fly one with scripts/simulate.py first")
        args.run = runs[-1]

    if args.run:
        run = read_run(args.run)
        print(f"  replaying {args.run}")
    elif args.world:
        from simulate import load_world  # noqa: PLC0415 -- sibling script

        world = load_world(args.world)
        run = {"world": world.model_dump(mode="json") if world else None}
    else:
        parser.error("give a run file, --latest, or --world")

    try:
        return play(
            run,
            speed=args.speed,
            gui=not args.headless,
            follow=not args.no_follow,
            out_dir=args.out,
        )
    except GazeboError as exc:
        print(f"Gazebo: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main(sys.argv[1:]))
