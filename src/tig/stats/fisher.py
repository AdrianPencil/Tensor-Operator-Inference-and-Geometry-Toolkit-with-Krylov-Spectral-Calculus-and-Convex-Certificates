"""
Fisher information objects (minimal, math-first).

This module provides:
- dense Fisher matrix construction for small problems (explicit J)
- matrix-free Fisher operator access for large problems (via vjp/jvp)

Least squares (Gaussian noise):
I(x) = (1/sigma^2) J(x)^T J(x).
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64
from tig.inverse.forward_models import ForwardModel
from tig.inverse.identifiability import jacobian_gram_operator

__all__ = ["fisher_operator", "fisher_dense_from_jacobian"]


def fisher_operator(model: ForwardModel, x: TensorLike, sigma: float = 1.0):
    """
    Return a matrix-free Fisher operator.
    """
    gram = jacobian_gram_operator(model, x)
    scale = 1.0 / (float(sigma) ** 2)

    def mv(v: tf.Tensor) -> tf.Tensor:
        return tf.cast(scale, tf.float64) * gram.matvec(v)

    return mv


def fisher_dense_from_jacobian(j: TensorLike, sigma: float = 1.0) -> tf.Tensor:
    """
    Dense Fisher matrix from an explicit Jacobian J (m x n): I = (1/sigma^2) J^T J.
    """
    jj = as_float64(j)
    scale = tf.cast(1.0 / (float(sigma) ** 2), tf.float64)
    return scale * (tf.transpose(jj) @ jj)
