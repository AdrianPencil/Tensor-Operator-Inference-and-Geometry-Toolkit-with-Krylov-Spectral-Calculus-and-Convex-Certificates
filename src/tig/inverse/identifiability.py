"""
Identifiability tools (inverse-facing, math-first).

This module exposes small, reusable quantities:
- Jacobian Gram operator J^T J as a matrix-free operator
- Fisher-like local information operator (for least squares with i.i.d. noise)

The goal is to reason about identifiability and conditioning without materializing J.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.operators import ExplicitLinearOperator
from tig.core.types import TensorLike, as_float64
from tig.inverse.forward_models import ForwardModel

__all__ = ["jacobian_gram_operator", "FisherOperator"]


def jacobian_gram_operator(model: ForwardModel, x: TensorLike) -> ExplicitLinearOperator:
    """
    Return a matrix-free operator G = J(x)^T J(x).
    """
    xx = as_float64(x)

    def matvec(v: tf.Tensor) -> tf.Tensor:
        jv = as_float64(model.jvp(xx, v))
        return as_float64(model.vjp(xx, jv))

    return ExplicitLinearOperator(matvec_fn=matvec, rmatvec_fn=matvec)


@dataclass(frozen=True)
class FisherOperator:
    """
    Fisher-like operator for least squares with Gaussian noise variance sigma^2:
    I(x) = (1/sigma^2) J(x)^T J(x).
    """

    model: ForwardModel
    x: tf.Tensor
    sigma: float = 1.0

    def operator(self) -> ExplicitLinearOperator:
        gram = jacobian_gram_operator(self.model, self.x)
        scale = 1.0 / (float(self.sigma) ** 2)

        def mv(v: tf.Tensor) -> tf.Tensor:
            return tf.cast(scale, tf.float64) * gram.matvec(v)

        return ExplicitLinearOperator(matvec_fn=mv, rmatvec_fn=mv)
