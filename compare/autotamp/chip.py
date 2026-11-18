"""Is Chip's Challenge solvable as released? A probe, with no LLM in it.

The paper reports 87.7% task success on Chip's Challenge with GPT-4. The
release's script runs layout `env2`, and the trajectories stored in the
release for that domain are all the "no solution" placeholder. This probe
asks the planner directly, with the release's settings for the domain (30
segments, bloat 0.18, first trajectory found), for a trajectory that only
reaches every goal and never touches a wall: no keys, no doors. Any formula a
translation could produce for the task asks for at least that, so if this
relaxed problem has no trajectory, no translation can succeed on that layout.

    python -m compare.autotamp.chip                  # all three layouts
    python -m compare.autotamp.chip --env env2 --time-limit 900

`infeasible` is a proof by the solver. `limit` means the solver ran out of
time without finding a trajectory or proving there is none. HiGHS reaches
`limit` on env2 and env3; SCIP proves both infeasible in about 20 s
(see README.md).
"""

from __future__ import annotations

import argparse

from compare.autotamp.envs import regions
from compare.autotamp.stl.pwl_plan import Node, PlanStats, plan

TMAX = 50.0
START = [4.5, 2.0]


def relaxed_spec(environment: str) -> Node:
    """Eventually every goal, and always outside every wall."""
    env = regions("chip", environment)
    goals, walls = [], []
    for colour, _, function, block in env.prop2block.values():
        if function == "goal":
            goals.append(Node("F", deps=[Node("mu", info={"A": block[0], "b": block[1]})], info={"int": [0, TMAX]}))
        elif colour == "black":
            walls.append(Node("negmu", info={"A": block[0], "b": block[1]}))

    def conjunction(nodes: list[Node]) -> Node:
        node = nodes[0]
        for other in nodes[1:]:
            node = Node("and", deps=[node, other])
        return node

    return Node("and", deps=[conjunction(goals), Node("A", deps=[conjunction(walls)], info={"int": [0, TMAX]})])


def probe(environment: str, time_limit: float = 600.0) -> PlanStats:
    stats = PlanStats()
    plan([START], [relaxed_spec(environment)], bloat=0.18, MIPGap=0.99, num_segs=30, tmax=TMAX, vmax=5.0, time_limit=time_limit, stats=stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probe whether each Chip's Challenge layout has any goal-reaching, wall-avoiding trajectory")
    parser.add_argument("--env", default="env1,env2,env3")
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args(argv)
    for environment in [e.strip() for e in args.env.split(",") if e.strip()]:
        stats = probe(environment, args.time_limit)
        found = "a trajectory exists" if stats.objective is not None else "no trajectory"
        print(
            f"{environment}: {stats.status} ({found}) after {stats.seconds:.0f} s; "
            f"{stats.binaries} binaries, {stats.constraints} constraints",
            flush=True,
        )


if __name__ == "__main__":
    main()
