"""AutoTAMP's STL trajectory planner (PWLPlan.py), on HiGHS instead of Gurobi.

The paper, section III-C: "We use a state-of-the-art multi-agent STL planner
[23] that uses piece-wise linear reference paths defined by timed waypoints to
recursively encode the constraints expressed in the provided STL expression."
[23] is Sun, Chen, Mitra and Fan, "Multi-agent motion planning from signal
temporal logic specifications" (RA-L 2022); AutoTAMP ships its code as
PWLPlan.py, and this is that file.

A trajectory is `num_segs` straight segments between timed waypoints
(x_i, t_i), all decision variables. Each STL node gets, for every segment i, a
tree of conjunctions and disjunctions over linear inequalities (`zs[i]`) that
holds when the formula holds on that segment; the root's tree at segment 0 is
turned into big-M constraints with one binary per disjunct. The objective is
the final time.

Kept exactly: every encoding (mu, negmu, always, eventually, until, release,
bounded eventually), the constants (M = 1e3, EPS = 1e-2, T_MIN_SEP = 1e-2),
the mu predicate's fixed 0.1 shrink, the L1 velocity limit, the constraint
caching on shared tree nodes, and the objective.

Changed: the solver. Gurobi's `abs_` general constraint becomes two
inequalities where the absolute value is only bounded from above (the
velocity limit), which is exact, and a binary split where it is used with both
signs (mutual clearance between agents).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from compare.autotamp.stl.milp import INF, LinExpr, Model

M = 1e3
# a large M causes numerical issues and makes the model infeasible to the solver
T_MIN_SEP = 1e-2
EPS = 1e-2


def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]


def _add(x1, x2):
    return [x1[i] + x2[i] for i in range(len(x1))]


def L1Norm(model: Model, x, exact: bool = False):
    """sum |x_i|. With `exact` false the result is only an upper bound's worth: abs_i >= |x_i|."""
    total = LinExpr()
    for component in x:
        if exact:
            positive, negative = model.add_var(), model.add_var()
            side = model.add_var(binary=True)
            model.add_eq(component - positive + negative)
            model.add_ge(M * side - positive)
            model.add_ge(M * (1 - side) - negative)
            total = total + positive + negative
        else:
            magnitude = model.add_var()
            model.add_ge(magnitude - component)
            model.add_ge(magnitude + component)
            total = total + magnitude
    return total


class Conjunction:
    def __init__(self, deps=None):
        self.deps = deps if deps is not None else []
        self.constraints = []


class Disjunction:
    def __init__(self, deps=None):
        self.deps = deps if deps is not None else []
        self.constraints = []


def noIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has no intersection
    # b < c or d < a
    return Disjunction([c - b - EPS, a - d - EPS])


def hasIntersection(a, b, c, d):
    # z = 1 iff. [a, b] and [c, d] has intersection
    # b >= c and d >= a
    return Conjunction([b - c, d - a])


def always(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    conjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        conjunctions.append(Disjunction([noIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Conjunction(conjunctions)


def eventually(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    z_intervalWidth = b - a - (t_i_1 - t_i) - EPS
    disjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        disjunctions.append(Conjunction([hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphis[j]]))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])


def bounded_eventually(i, a, b, zphis, PWL, tmax):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    z_intervalWidth = b - a - (t_i_1 - t_i) - EPS
    disjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        disjunctions.append(Conjunction([hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphis[j]]))
    return Disjunction([Conjunction([z_intervalWidth, Disjunction(disjunctions)]), t_i + b - tmax - EPS])


def until(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    z_intervalWidth = b - a - (t_i_1 - t_i) - EPS
    disjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        conjunctions = [hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j + 1):  # noqa: E741
            t_l = PWL[l][1]
            t_l_1 = PWL[l + 1][1]
            conjunctions.append(Disjunction([noIntersection(t_l, t_l_1, t_i, t_i_1 + b), zphi1s[l]]))
        disjunctions.append(Conjunction(conjunctions))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])


def release(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i + 1][1]
    conjunctions = []
    for j in range(len(PWL) - 1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j + 1][1]
        disjunctions = [noIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j):  # noqa: E741
            t_l = PWL[l][1]
            t_l_1 = PWL[l + 1][1]
            disjunctions.append(Conjunction([hasIntersection(t_l, t_l_1, t_i_1, t_i_1 + b), zphi1s[l]]))
        conjunctions.append(Disjunction(disjunctions))
    return Conjunction(conjunctions)


def mu(i, PWL, bloat_factor, A, b):
    bloat_factor = max(0.0, bloat_factor)
    # this segment is fully contained in Ax<=b (shrunk)
    b = np.asarray(b, dtype=float).reshape(-1)
    conjunctions = []
    for e in range(len(b)):
        a = A[e, :]
        norm = float(np.linalg.norm(a))
        for j in [i, i + 1]:
            x = PWL[j][0]
            inner = LinExpr()
            for k in range(len(x)):
                inner = inner + float(a[k]) * x[k]
            conjunctions.append(float(b[e]) - norm * bloat_factor - inner - EPS)
    return Conjunction(conjunctions)


def negmu(i, PWL, bloat_factor, A, b):
    # this segment is outside Ax<=b (bloated)
    b = np.asarray(b, dtype=float).reshape(-1)
    disjunctions = []
    for e in range(len(b)):
        a = A[e, :]
        norm = float(np.linalg.norm(a))
        conjunctions = []
        for j in [i, i + 1]:
            x = PWL[j][0]
            inner = LinExpr()
            for k in range(len(x)):
                inner = inner + float(a[k]) * x[k]
            conjunctions.append(inner - (float(b[e]) + norm * bloat_factor) - EPS)
        disjunctions.append(Conjunction(conjunctions))
    return Disjunction(disjunctions)


def add_space_constraints(model: Model, xlist, limits, bloat=0.0):
    xlim, ylim = limits
    for x in xlist:
        model.add_ge(x[0] - (xlim[0] + bloat))
        model.add_ge(x[1] - (ylim[0] + bloat))
        model.add_ge((xlim[1] - bloat) - x[0])
        model.add_ge((ylim[1] - bloat) - x[1])


def add_time_constraints(model: Model, PWL, tmax=None):
    if tmax is not None:
        model.add_ge((tmax - T_MIN_SEP) - PWL[-1][1])
    for i in range(len(PWL) - 1):
        _, t1 = PWL[i]
        _, t2 = PWL[i + 1]
        model.add_ge(t2 - t1 - T_MIN_SEP)


def add_velocity_constraints(model: Model, PWL, vmax=3):
    for i in range(len(PWL) - 1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i + 1]
        L1_dist = L1Norm(model, _sub(x1, x2))
        model.add_ge(vmax * (t2 - t1) - L1_dist)


def disjoint_segments(model: Model, seg1, seg2, bloat):
    # assuming that bloat is the error bound in two norm for one agent
    return (
        0.5 * L1Norm(model, _sub(_add(seg1[0], seg1[1]), _add(seg2[0], seg2[1])), exact=True)
        - 0.5 * (L1Norm(model, _sub(seg1[0], seg1[1]), exact=True) + L1Norm(model, _sub(seg2[0], seg2[1]), exact=True))
        - 2 * bloat * math.sqrt(len(seg1[0]))
        - EPS
    )


def add_mutual_clearance_constraints(model: Model, PWLs, bloat):
    for i in range(len(PWLs)):
        for j in range(i + 1, len(PWLs)):
            PWL1 = PWLs[i]
            PWL2 = PWLs[j]
            for k in range(len(PWL1) - 1):
                for l in range(len(PWL2) - 1):  # noqa: E741
                    x11, t11 = PWL1[k]
                    x12, t12 = PWL1[k + 1]
                    x21, t21 = PWL2[l]
                    x22, t22 = PWL2[l + 1]
                    z_noIntersection = noIntersection(t11, t12, t21, t22)
                    z_disjoint_segments = disjoint_segments(model, [x11, x12], [x21, x22], bloat)
                    add_CDTree_Constraints(model, Disjunction([z_noIntersection, z_disjoint_segments]))


class Node:
    """One STL operator: op is mu, negmu, and, or, A (always), F (eventually), U (until), BF."""

    def __init__(self, op, deps=None, zs=None, info=None):
        self.op = op
        self.deps = deps if deps is not None else []
        self.zs = zs if zs is not None else []
        self.info = info if info is not None else []


def clearSpecTree(spec):
    for dep in spec.deps:
        clearSpecTree(dep)
    spec.zs = []


def handleSpecTree(spec, PWL, bloat_factor, size):
    for dep in spec.deps:
        handleSpecTree(dep, PWL, bloat_factor, size)
    if len(spec.zs) == len(PWL) - 1:
        return
    elif len(spec.zs) > 0:
        raise ValueError("incomplete zs")
    segments = range(len(PWL) - 1)
    if spec.op == "mu":
        spec.zs = [mu(i, PWL, 0.1, spec.info["A"], spec.info["b"]) for i in segments]
    elif spec.op == "negmu":
        spec.zs = [negmu(i, PWL, bloat_factor + size, spec.info["A"], spec.info["b"]) for i in segments]
    elif spec.op == "and":
        spec.zs = [Conjunction([dep.zs[i] for dep in spec.deps]) for i in segments]
    elif spec.op == "or":
        spec.zs = [Disjunction([dep.zs[i] for dep in spec.deps]) for i in segments]
    elif spec.op == "U":
        spec.zs = [until(i, spec.info["int"][0], spec.info["int"][1], spec.deps[0].zs, spec.deps[1].zs, PWL) for i in segments]
    elif spec.op == "F":
        spec.zs = [eventually(i, spec.info["int"][0], spec.info["int"][1], spec.deps[0].zs, PWL) for i in segments]
    elif spec.op == "BF":
        spec.zs = [
            bounded_eventually(i, spec.info["int"][0], spec.info["int"][1], spec.deps[0].zs, PWL, spec.info["tmax"])
            for i in segments
        ]
    elif spec.op == "A":
        spec.zs = [always(i, spec.info["int"][0], spec.info["int"][1], spec.deps[0].zs, PWL) for i in segments]
    else:
        raise ValueError("wrong op code")


def gen_CDTree_constraints(model: Model, root):
    if not hasattr(root, "deps"):
        return [root]
    if len(root.constraints) > 0:
        return root.constraints
    dep_constraints = []
    for dep in root.deps:
        dep_constraints.append(gen_CDTree_constraints(model, dep))
    zs = []
    for dep_con in dep_constraints:
        if isinstance(root, Disjunction):
            z = model.add_var(binary=True)
            zs.append(z)
            dep_con = [con + M * (1 - z) for con in dep_con]
        root.constraints += dep_con
    if len(zs) > 0:
        total = LinExpr()
        for z in zs:
            total = total + z
        root.constraints.append(total - 1)
    return root.constraints


def add_CDTree_Constraints(model: Model, root):
    for con in gen_CDTree_constraints(model, root):
        model.add_ge(con)


@dataclass
class PlanStats:
    """What the solve came to, for the experiment's records."""

    status: str = "not_run"
    seconds: float = 0.0
    binaries: int = 0
    variables: int = 0
    constraints: int = 0
    objective: float | None = None
    num_segs: int = 0


def plan(
    x0s,
    specs,
    bloat,
    limits=None,
    num_segs=None,
    tasks=None,
    vmax=3.0,
    MIPGap=1e-4,
    max_segs=None,
    tmax=None,
    hard_goals=None,
    size=0.11 * 4 / 2,
    time_limit: float = 2000.0,
    stats: PlanStats | None = None,
):
    """Plan one piece-wise linear trajectory per agent satisfying its spec, minimising the total final time.

    Returns one trajectory per agent as [[position, time], ...], or `[None]`
    when no trajectory was found (infeasible, or no solution within
    `time_limit`), exactly as the released planner does.
    """
    if num_segs is None:
        min_segs = 1
        assert max_segs is not None
    else:
        min_segs = num_segs
        max_segs = num_segs

    for num_segs in range(min_segs, max_segs + 1):
        for spec in specs:
            clearSpecTree(spec)
        if tasks:
            for task in tasks:
                for t in task:
                    clearSpecTree(t)

        PWLs = []
        m = Model()

        for idx_a in range(len(x0s)):
            x0 = np.array(x0s[idx_a], dtype=float).reshape(-1).tolist()
            spec = specs[idx_a]
            dims = len(x0)

            PWL = []
            for _ in range(num_segs + 1):
                PWL.append([[m.add_var(lb=-INF) for _ in range(dims)], m.add_var()])
            PWLs.append(PWL)

            # the initial constraint
            for i in range(dims):
                m.add_eq(PWL[0][0][i] - x0[i])
            m.add_eq(PWL[0][1])

            if hard_goals is not None:
                goal = hard_goals[idx_a]
                for i in range(dims):
                    m.add_eq(PWL[-1][0][i] - float(goal[i]))

            if limits is not None:
                add_space_constraints(m, [P[0] for P in PWL], limits)

            add_velocity_constraints(m, PWL, vmax=vmax)
            add_time_constraints(m, PWL, tmax)

            handleSpecTree(spec, PWL, bloat, size)
            add_CDTree_Constraints(m, spec.zs[0])

        if tasks is not None:
            for idx_agent in range(len(tasks)):
                for idx_task in range(len(tasks[idx_agent])):
                    handleSpecTree(tasks[idx_agent][idx_task], PWLs[idx_agent], bloat, size)
            conjunctions = []
            for idx_task in range(len(tasks[0])):
                disjunctions = [tasks[idx_agent][idx_task].zs[0] for idx_agent in range(len(tasks))]
                conjunctions.append(Disjunction(disjunctions))
            add_CDTree_Constraints(m, Conjunction(conjunctions))

        add_mutual_clearance_constraints(m, PWLs, bloat)

        obj = LinExpr()
        for PWL in PWLs:
            obj = obj + PWL[-1][1]

        solution = m.minimize(obj, mip_gap=MIPGap, time_limit=time_limit)
        if stats is not None:
            stats.status = solution.status
            stats.seconds = round(solution.seconds, 3)
            stats.binaries = m.num_binaries
            stats.variables = m.num_vars
            stats.constraints = m.num_constraints
            stats.objective = solution.objective
            stats.num_segs = num_segs
        if not solution.found:
            continue
        PWLs_output = []
        for PWL in PWLs:
            PWLs_output.append([[[solution.value(P[0][i]) for i in range(len(P[0]))], solution.value(P[1])] for P in PWL])
        return PWLs_output
    return [None]
