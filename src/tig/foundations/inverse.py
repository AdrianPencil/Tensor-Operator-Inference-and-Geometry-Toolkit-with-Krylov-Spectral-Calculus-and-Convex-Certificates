"""
Inverse problem core framing (foundations).

This module is intentionally light and mathematical:
- forward map F
- data y
- misfit functional Phi(x) = 1/2 ||F(x) - y||^2
- optional regularizer R(x)

Concrete solvers live later in tig/inverse/.
"""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["LeastSquaresInverseProblem"]


@dataclass(frozen=True)
class LeastSquaresInverseProblem:
    """
    Least squares inverse problem: minimize 1/2 ||F(x) - y||^2 + lam * R(x).
    """

    forward: Callable[[tf.Tensor], tf.Tensor]
    data: tf.Tensor
    lam: float = 0.0
    regularizer: Callable[[tf.Tensor], tf.Tensor] | None = None

    def misfit(self, x: TensorLike) -> tf.Tensor:
        xx = as_float64(x)
        yhat = as_float64(self.forward(xx))
        rr = yhat - as_float64(self.data)
        return 0.5 * tf.reduce_sum(rr * rr)

    def objective(self, x: TensorLike) -> tf.Tensor:
        val = self.misfit(x)
        if self.regularizer is None or self.lam == 0.0:
            return val
        return val + float(self.lam) * as_float64(self.regularizer(as_float64(x)))
