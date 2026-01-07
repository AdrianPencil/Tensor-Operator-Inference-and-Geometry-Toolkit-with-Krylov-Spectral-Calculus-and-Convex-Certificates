"""
Green's functions / resolvent-as-kernel (minimal).

For an operator L, a Green's operator G satisfies:
L G = I  (with boundary conditions).

In discrete settings, G is a matrix inverse (or a linear solve operator).
This module provides a dense Green operator builder for small problems.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["GreensOperator", "greens_from_matrix"]


@dataclass(frozen=True)
class GreensOperator:
    """
    Dense Green operator for a linear system matrix L: G = L^{-1}.
    """

    g: tf.Tensor

    def apply(self, f: TensorLike) -> tf.Tensor:
        gg = as_float64(self.g)
        ff = tf.reshape(as_float64(f), (-1, 1))
        return (gg @ ff)[:, 0]


def greens_from_matrix(l: TensorLike, ridge: float = 1e-12) -> GreensOperator:
    """
    Construct a dense Green operator G ≈ (L + ridge I)^{-1}.
    """
    ll = as_float64(l)
    n = int(ll.shape[0])
    reg = tf.cast(float(ridge), tf.float64) * tf.eye(n, dtype=tf.float64)
    g = tf.linalg.inv(ll + reg)
    return GreensOperator(g=g)
