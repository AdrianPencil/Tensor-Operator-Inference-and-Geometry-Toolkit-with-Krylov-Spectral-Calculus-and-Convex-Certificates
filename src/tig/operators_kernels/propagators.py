"""
Propagators as semigroups (minimal).

For a linear ODE d/dt u = A u, the propagator is:
U(t) = exp(t A),  u(t) = U(t) u(0)

This module provides a dense propagator action for small/medium matrices.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["Propagator", "linear_propagator"]


@dataclass(frozen=True)
class Propagator:
    """
    Dense propagator U(t) = exp(tA).
    """

    a: tf.Tensor

    def apply(self, u0: TensorLike, t: float) -> tf.Tensor:
        aa = as_float64(self.a)
        tt = tf.cast(float(t), tf.float64)
        u0v = tf.reshape(as_float64(u0), (-1, 1))
        return (tf.linalg.expm(tt * aa) @ u0v)[:, 0]


def linear_propagator(a: TensorLike) -> Propagator:
    return Propagator(a=as_float64(a))
