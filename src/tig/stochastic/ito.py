"""
Itô calculus primitives (minimal, math-first).

This module provides small, explicit building blocks:
- Brownian increments
- Itô integral approximation via left-point sums

The goal is to keep the stochastic meaning explicit and composable.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import ShapeLike, TensorLike, as_float64

__all__ = ["BrownianPath", "brownian_increments", "ito_integral_left"]


@dataclass(frozen=True)
class BrownianPath:
    """
    Discrete Brownian path represented by increments dW on a uniform grid.

    dW has shape (n_steps, *state_shape).
    """

    dt: float
    dW: tf.Tensor

    @property
    def n_steps(self) -> int:
        return int(self.dW.shape[0])

    def path(self) -> tf.Tensor:
        """
        Cumulative Brownian path W_t with W_0 = 0.
        """
        return tf.cumsum(as_float64(self.dW), axis=0)


def brownian_increments(
    dt: float,
    n_steps: int,
    shape: ShapeLike,
    rng: Rng,
) -> BrownianPath:
    """
    Generate Brownian increments dW ~ Normal(0, dt) on a uniform grid.
    """
    dd = float(dt)
    ns = int(n_steps)
    dW = tf.sqrt(tf.cast(dd, tf.float64)) * rng.normal((ns, *tuple(shape)), dtype=tf.float64)
    return BrownianPath(dt=dd, dW=dW)


def ito_integral_left(
    g: Callable[[tf.Tensor, float], tf.Tensor],
    x: TensorLike,
    t0: float,
    brownian: BrownianPath,
) -> tf.Tensor:
    """
    Approximate Itô integral ∫ g(X_t, t) dW_t by left-point sums.

    Parameters
    ----------
    g:
        Coefficient function g(x, t) returning same shape as x.
    x:
        State at initial time t0.
    t0:
        Initial time.
    brownian:
        BrownianPath with dt and dW increments.

    Returns
    -------
    tf.Tensor
        Approximation to the Itô integral as a tensor with the same shape as x.
    """
    xx = as_float64(x)
    dt = float(brownian.dt)
    acc = tf.zeros_like(xx)
    t = float(t0)

    for k in range(brownian.n_steps):
        gk = as_float64(g(xx, t))
        dWk = as_float64(brownian.dW[k])
        acc = acc + gk * dWk
        t += dt

    return acc
