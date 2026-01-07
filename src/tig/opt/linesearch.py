"""
Line search methods (minimal, math-first).

Implements Armijo backtracking for a descent direction:
φ(x + t p) <= φ(x) + c1 t <∇φ(x), p>.

This is used by gradient-based methods and trust-region outer loops.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.norms import inner
from tig.core.types import TensorLike, as_float64
from tig.opt.objectives import Objective

__all__ = ["LineSearchResult", "ArmijoBacktracking"]


@dataclass(frozen=True)
class LineSearchResult:
    """
    Result of a line search.
    """

    step: float
    value_new: float
    accepted: bool
    num_backtracks: int


@dataclass(frozen=True)
class ArmijoBacktracking:
    """
    Armijo backtracking line search.
    """

    c1: float = 1e-4
    shrink: float = 0.5
    step0: float = 1.0
    max_backtracks: int = 30

    def __call__(
        self,
        obj: Objective,
        x: TensorLike,
        p: TensorLike,
        g: TensorLike | None = None,
    ) -> LineSearchResult:
        xx = as_float64(x)
        pp = as_float64(p)
        gg = as_float64(g) if g is not None else as_float64(obj.gradient(xx))

        f0 = as_float64(obj.value(xx))
        slope = as_float64(inner(gg, pp))

        step = float(self.step0)
        accepted = False
        f_new = float(f0.numpy())
        k = 0

        for k in range(int(self.max_backtracks) + 1):
            x_try = xx + tf.cast(step, tf.float64) * pp
            f_try = as_float64(obj.value(x_try))

            rhs = f0 + tf.cast(self.c1 * step, tf.float64) * slope
            if bool((f_try <= rhs).numpy()):
                accepted = True
                f_new = float(f_try.numpy())
                break

            step *= float(self.shrink)

        return LineSearchResult(
            step=float(step),
            value_new=float(f_new),
            accepted=bool(accepted),
            num_backtracks=int(k),
        )
