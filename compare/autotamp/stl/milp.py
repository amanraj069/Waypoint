"""A small mixed-integer linear model, solved by HiGHS.

AutoTAMP's STL planner is written against Gurobi, which needs a licence. The
planner only ever builds linear expressions over continuous and binary
variables and asks for `expr >= 0` or `expr == 0`, so this module offers just
that: enough of Gurobi's modelling surface for the planner's code to read the
same, and HiGHS (through SciPy) underneath.

    model = Model()
    x = model.add_var(lb=-INF)          # continuous; like Gurobi, lb defaults to 0
    z = model.add_var(binary=True)
    model.add_ge(x + 5 * z - 2)         # x + 5 z - 2 >= 0
    model.add_eq(x - 1)
    solution = model.minimize(x, mip_gap=1e-4, time_limit=60)
    solution.value(x)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

INF = math.inf


class LinExpr:
    """A linear expression: sum of coefficient * variable, plus a constant."""

    __slots__ = ("terms", "const")
    # NumPy scalars must defer to the reflected operators here rather than broadcast.
    __array_ufunc__ = None

    def __init__(self, terms: dict[int, float] | None = None, const: float = 0.0):
        self.terms = terms or {}
        self.const = float(const)

    def __add__(self, other):
        if isinstance(other, LinExpr):
            terms = dict(self.terms)
            for index, coef in other.terms.items():
                terms[index] = terms.get(index, 0.0) + coef
            return LinExpr(terms, self.const + other.const)
        return LinExpr(dict(self.terms), self.const + float(other))

    __radd__ = __add__

    def __neg__(self):
        return LinExpr({index: -coef for index, coef in self.terms.items()}, -self.const)

    def __sub__(self, other):
        return self + (-other if isinstance(other, LinExpr) else -float(other))

    def __rsub__(self, other):
        return (-self) + float(other)

    def __mul__(self, factor):
        factor = float(factor)
        return LinExpr({index: coef * factor for index, coef in self.terms.items()}, self.const * factor)

    __rmul__ = __mul__


@dataclass
class Solution:
    status: str
    values: np.ndarray | None
    objective: float | None
    seconds: float
    gap: float | None = None

    @property
    def found(self) -> bool:
        return self.values is not None

    def value(self, expr: LinExpr | float) -> float:
        if not isinstance(expr, LinExpr):
            return float(expr)
        return float(sum(coef * self.values[index] for index, coef in expr.terms.items()) + expr.const)


class Model:
    def __init__(self) -> None:
        self._lb: list[float] = []
        self._ub: list[float] = []
        self._integer: list[int] = []
        self._rows: list[tuple[dict[int, float], float, float]] = []

    @property
    def num_vars(self) -> int:
        return len(self._lb)

    @property
    def num_binaries(self) -> int:
        return sum(self._integer)

    @property
    def num_constraints(self) -> int:
        return len(self._rows)

    def add_var(self, lb: float = 0.0, ub: float = INF, binary: bool = False) -> LinExpr:
        if binary:
            lb, ub = 0.0, 1.0
        self._lb.append(lb)
        self._ub.append(ub)
        self._integer.append(1 if binary else 0)
        return LinExpr({len(self._lb) - 1: 1.0})

    def add_ge(self, expr) -> None:
        """expr >= 0"""
        self._add(expr, 0.0, INF)

    def add_eq(self, expr) -> None:
        """expr == 0"""
        self._add(expr, 0.0, 0.0)

    def _add(self, expr, low: float, high: float) -> None:
        if not isinstance(expr, LinExpr):
            expr = LinExpr(const=float(expr))
        terms = {index: coef for index, coef in expr.terms.items() if coef != 0.0}
        if not terms:
            if not (low - 1e-9 <= expr.const <= high + 1e-9):
                # A constant that cannot hold: keep it as an infeasible row so the solver says so.
                self._rows.append(({}, 1.0, 1.0))
            return
        self._rows.append((terms, low - expr.const, high - expr.const))

    def minimize(self, objective, *, mip_gap: float = 1e-4, time_limit: float | None = None) -> Solution:
        from scipy.optimize import Bounds, LinearConstraint, milp  # noqa: PLC0415
        from scipy.sparse import csr_matrix  # noqa: PLC0415

        began = time.perf_counter()
        n = self.num_vars
        cost = np.zeros(n)
        if isinstance(objective, LinExpr):
            for index, coef in objective.terms.items():
                cost[index] = coef
        data, rows, cols, lows, highs = [], [], [], [], []
        for row, (terms, low, high) in enumerate(self._rows):
            lows.append(low)
            highs.append(high)
            for index, coef in terms.items():
                rows.append(row)
                cols.append(index)
                data.append(coef)
        matrix = csr_matrix((data, (rows, cols)), shape=(len(self._rows), n))
        options: dict = {"disp": False, "mip_rel_gap": mip_gap}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        result = milp(
            cost,
            constraints=LinearConstraint(matrix, np.array(lows), np.array(highs)) if self._rows else None,
            integrality=np.array(self._integer),
            bounds=Bounds(np.array(self._lb), np.array(self._ub)),
            options=options,
        )
        seconds = time.perf_counter() - began
        status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(result.status, "other")
        if result.x is None:
            return Solution(status, None, None, seconds)
        return Solution(status, np.asarray(result.x), float(result.fun), seconds, getattr(result, "mip_gap", None))
