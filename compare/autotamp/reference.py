"""A correct trajectory for each HouseWorld instruction family, and HouseWorld2's time limits.

"HouseWorld2 (T, single-agent): This scenario is identical to HouseWorld1, but
each planned trajectory is subjected to a hard time constraint. This time limit
is pre-determined by completing the correct trajectory with 0.8 maximum
velocity." (paper, section IV)

The release has neither the correct trajectories nor the limits. Here each
family's meaning is written directly as a specification tree for the STL
planner (not through the LLM or the token parser), the planner finds its
time-optimal trajectory at full speed, and the limit is that time divided by
0.8: the time the correct trajectory takes at 0.8 of the maximum velocity.

The limits are computed once for the nominal start and end of each test-case
variant and stored in `data/reference_times.json`:

    python -m compare.autotamp.reference            # recompute and store
    python -m compare.autotamp.reference --only-missing --time-limit 900

A case whose reference trajectory the solver does not find in the time allowed
has no limit, and is scored without the time constraint; the experiment's
summary says how many cases that is.
"""

from __future__ import annotations

import json
from pathlib import Path

from compare.autotamp.envs import regions
from compare.autotamp.stl.pwl_plan import Node, PlanStats, plan

STORE = Path(__file__).parent / "data" / "reference_times.json"
TMAX = 50.0
SPEED_SHARE = 0.8

#: Nominal start and end of each test-case variant: the centre of the release's random ranges.
#: Family 9 (index 8 in the release's loop) has its own pair, used for both of its variants.
VARIANTS: dict[str, tuple[list[float], list[float]]] = {
    "a": ([1.305, -1.305], [0.0, 1.305]),
    "b": ([-1.0, -0.35], [0.0, 1.305]),
    "nine": ([-1.35, -1.1], [1.15, 1.15]),
}


def variant_of(family: int, variant: str) -> str:
    return "nine" if family == 9 else variant


def reference_spec(family: int) -> Node:
    """What instruction family 1 to 10 asks for, as a specification tree."""
    env = regions("HouseWorld")
    blocks = {entry[2]: entry[3] for entry in env.prop2block.values() if entry[0] != "black"}

    def inside(room):
        return Node("mu", info={"A": blocks[room][0], "b": blocks[room][1]})

    def outside(room):
        return Node("negmu", info={"A": blocks[room][0], "b": blocks[room][1]})

    def F(node, a=0, b=TMAX):
        return Node("F", deps=[node], info={"int": [a, b]})

    def G(node, a=0, b=TMAX):
        return Node("A", deps=[node], info={"int": [a, b]})

    def U(first, second):
        return Node("U", deps=[first, second], info={"int": [0, TMAX]})

    def AND(*nodes):
        node = nodes[0]
        for other in nodes[1:]:
            node = Node("and", deps=[node, other])
        return node

    def then(*steps):
        """Each step holds, and the next one holds some time after it."""
        node = steps[-1]
        for step in reversed(steps[:-1]):
            node = AND(step, F(node))
        return F(node)

    rooms = list(blocks)
    no_blue = AND(outside("LivingRoom"), outside("ExerciseRoom"))
    if family == 1:
        return then(inside("MasterBedroom"), inside("RestRoom"))
    if family == 2:
        return AND(
            then(inside("RestRoom"), inside("Bedroom"), inside("RestRoom2")), F(inside("Kitchen")), F(inside("DiningRoom"))
        )
    if family == 3:
        return AND(then(inside("MasterBedroom"), inside("RestRoom")), G(outside("LivingRoom")))
    if family == 4:
        return AND(then(inside("Kitchen"), G(inside("Bedroom"), 0, 5), inside("RestRoom")), G(no_blue))
    if family == 5:
        return AND(F(inside("Bedroom")), F(inside("DiningRoom")))
    if family == 6:
        return AND(*[F(inside(r)) for r in rooms if r not in ("Bedroom", "Kitchen")])
    if family == 7:
        return AND(
            U(AND(outside("RestRoom"), outside("RestRoom2")), inside("Kitchen")),
            F(inside("RestRoom")), F(inside("RestRoom2")), G(no_blue),
        )
    if family == 8:
        return AND(
            then(inside("Kitchen"), G(inside("Bedroom"), 0, 10), inside("LivingRoom"), inside("Kitchen")),
            U(no_blue, inside("Bedroom")),
        )
    if family == 9:
        return AND(*[F(inside(r)) for r in rooms], U(outside("Bedroom"), inside("LivingRoom")))
    if family == 10:
        others = [r for r in rooms if r != "LivingRoom"]
        rest = [r for r in rooms if r != "ExerciseRoom"]
        return AND(
            *[F(inside(r)) for r in rooms],
            U(AND(*[outside(r) for r in others]), inside("LivingRoom")),
            F(AND(inside("ExerciseRoom"), G(AND(*[outside(r) for r in rest])))),
        )
    raise ValueError(f"no reference for family {family}")


def reference_trajectory(family: int, start, end, *, time_limit: float = 120.0, vmax: float = 5.0, stats: PlanStats | None = None):
    """The planner's time-optimal trajectory for the family's reference spec: [[x, y, t], ...] or None."""
    PWLs = plan(
        [list(start)], [reference_spec(family)], bloat=0.05, size=0.11 / 2, num_segs=20, tmax=TMAX, vmax=vmax,
        hard_goals=[list(end)], time_limit=time_limit, stats=stats,
    )
    if PWLs == [None]:
        return None
    return [[P[0][0], P[0][1], P[1]] for P in PWLs[0]]


def time_limits() -> dict[str, float]:
    """`<family><variant>` -> HouseWorld2's time limit, e.g. "4a" -> 9.7."""
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} is missing: run python -m compare.autotamp.reference")
    stored = json.loads(STORE.read_text(encoding="utf-8"))
    return {key: entry["limit"] for key, entry in stored["cases"].items() if entry.get("limit") is not None}


def main(argv: list[str] | None = None) -> None:
    import argparse  # noqa: PLC0415

    from compare.autotamp.checkers import check_houseworld  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Compute HouseWorld2's time limits from reference trajectories")
    parser.add_argument("--time-limit", type=float, default=120.0, help="seconds the planner may spend on each case")
    parser.add_argument("--only-missing", action="store_true", help="keep the stored cases that already have a limit")
    args = parser.parse_args(argv)

    cases: dict[str, dict] = {}
    if args.only_missing and STORE.exists():
        cases = json.loads(STORE.read_text(encoding="utf-8"))["cases"]
    for family in range(1, 11):
        for variant in ("a", "b"):
            if cases.get(f"{family}{variant}", {}).get("limit") is not None:
                continue
            start, end = VARIANTS[variant_of(family, variant)]
            stats = PlanStats()
            trajectory = reference_trajectory(family, start, end, time_limit=args.time_limit, stats=stats)
            duration = trajectory[-1][2] if trajectory else None
            cases[f"{family}{variant}"] = {
                "start": start, "end": end, "status": stats.status, "solve_seconds": stats.seconds,
                "duration": None if duration is None else round(duration, 4),
                "limit": None if duration is None else round(duration / SPEED_SHARE, 4),
                "passes_checker": bool(trajectory) and check_houseworld(family, trajectory),
            }
            print(f"family {family}{variant}: {cases[f'{family}{variant}']}", flush=True)
            STORE.write_text(json.dumps({"speed_share": SPEED_SHARE, "vmax": 5.0, "cases": cases}, indent=1), encoding="utf-8")
    STORE.write_text(json.dumps({"speed_share": SPEED_SHARE, "vmax": 5.0, "cases": cases}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
