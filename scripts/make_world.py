#!/usr/bin/env python3
"""Build a world to fly in and save it to worlds/<name>.json.

    # the Central Vista, New Delhi: India Gate up Raisina Hill
    python scripts/make_world.py delhi

    # Chandni Chowk: a 30 m street between solid shopfronts
    python scripts/make_world.py chowk

    # the hills above Munnar, with more relief than the default
    python scripts/make_world.py munnar --relief-m 300

    # a real place, from OpenStreetMap (needs the internet)
    python scripts/make_world.py osm --lat 28.6129 --lon 77.2295 --radius 400 --name india_gate
    python scripts/make_world.py osm --lat 18.9220 --lon 72.8347 --radius 400 --name colaba --relief

The point of origin -- the pad of a built-in world, or the lat/lon you give --
is where the aircraft takes off, and the ground there is the zero datum every
altitude is measured from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world.generate import (  # noqa: E402
    BUILDERS,
    BUILT_IN,
    chandni_chowk,
    lodhi_garden,
    lutyens_delhi,
    munnar,
    rolling_terrain,
)
from world.models import World  # noqa: E402

WORLDS = Path("worlds")


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

def save(world: World, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(world.model_dump_json(indent=2) + "\n")
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "kind",
        choices=[*BUILDERS, "osm"],
        help="A built-in place (" + ", ".join(BUILT_IN) + "), or 'osm' for a real one.",
    )
    parser.add_argument("--name", help="World name; also the file name.")
    parser.add_argument("--out", type=Path, help="Where to write it (default worlds/<name>.json).")
    parser.add_argument("--seed", type=int, default=None, help="Varies the planting and the bazaar.")
    parser.add_argument("--street", type=float, default=30.0, help="chowk: metres between shopfronts.")
    parser.add_argument("--relief-m", type=float, default=235.0,
                        help="munnar: metres from the pad to the top of Anamudi.")
    parser.add_argument("--masts", type=int, default=2, help="munnar: relay towers on the ridges.")
    parser.add_argument("--lat", type=float, help="osm: latitude of the launch point.")
    parser.add_argument("--lon", type=float, help="osm: longitude of the launch point.")
    parser.add_argument("--radius", type=float, default=350.0, help="osm: metres around it.")
    parser.add_argument("--limit", type=int, default=400, help="osm: most buildings to keep.")
    parser.add_argument(
        "--relief",
        action="store_true",
        help="osm: add synthetic rolling ground (OSM has footprints, not elevation).",
    )
    args = parser.parse_args(argv)

    if args.kind in BUILDERS:
        builder = BUILDERS[args.kind][0]
        if builder == "lutyens_delhi":
            world = lutyens_delhi(name=args.name or "central vista", seed=args.seed or 7)
        elif builder == "lodhi_garden":
            world = lodhi_garden(name=args.name or "lodhi garden", seed=args.seed or 5)
        elif builder == "chandni_chowk":
            world = chandni_chowk(name=args.name or "chandni chowk",
                                  street=args.street, seed=args.seed or 3)
        else:
            world = munnar(name=args.name or "munnar", relief=args.relief_m,
                           seed=args.seed or 12, masts=args.masts)
    else:
        if args.lat is None or args.lon is None:
            parser.error("osm needs --lat and --lon")
        from world.osm import OSMError, fetch  # noqa: PLC0415 -- only this path needs the network

        terrain = (
            rolling_terrain(half_extent=args.radius * 1.5, amplitude=14.0,
                            seed=args.seed or 7).levelled_at()
            if args.relief
            else None
        )
        print(f"asking Overpass for buildings within {args.radius:g} m of "
              f"{args.lat}, {args.lon}...", file=sys.stderr)
        try:
            world = fetch(
                args.lat,
                args.lon,
                radius=args.radius,
                name=args.name or f"osm_{args.lat:.4f}_{args.lon:.4f}",
                limit=args.limit,
                terrain=terrain,
            )
        except OSMError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1

    out = args.out or WORLDS / f"{world.name}.json"
    save(world, out)

    summary = world.summary()
    if summary["landmarks"]:
        print("\nnamed places you can put in a mission:")
        for landmark in world.landmarks:
            print(f"  - {landmark.describe()}")
    print(f"\n{world.name}: {summary['buildings']} buildings ({summary['trees']} trees), tallest roof "
          f"{summary['tallest_roof']:g} m, ground {summary['elevation_min']:g}..."
          f"{summary['elevation_max']:g} m")
    minx, miny, maxx, maxy = world.bounds
    print(f"spans x {minx:.0f}..{maxx:.0f}, y {miny:.0f}..{maxy:.0f} m around the pad")
    print(f"written to {out}")
    print(f"\nlook around it:  {python_command()} scripts/explore.py --world {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
