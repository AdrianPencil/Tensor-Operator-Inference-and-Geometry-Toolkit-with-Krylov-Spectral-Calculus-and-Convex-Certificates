"""
SDE solvers (minimal, math-first).

Implements Euler–Maruyama for:
dX_t = f(X_t, t) dt + g(X_t, t) dW_t

The solver is TF-first and vectorized over batch dimensions when present.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import ShapeLike, TensorLike, as_float64

__all__ = ["SdeSpec", "euler_maruyama"]


@dataclass(frozen=True)
class SdeSpec:
    """
    SDE specification.
    """

    drift: Callable[[tf.Tensor, float], tf.Tensor]
    diffusion: Callable[[tf.Tensor, float], tf.Tensor]


def euler_maruyama(
    spec: SdeSpec,
    x0: TensorLike,
    t0: float,
    dt: float,
    n_steps: int,
    rng: Rng,
) -> tf.Tensor:
    """
    Simulate an SDE by Euler–Maruyama.

    Returns a tensor of shape (n_steps+1, *x0.shape).
    """
    xx = as_float64(x0)
    dd = float(dt)
    ns = int(n_steps)

    traj = [xx]
    t = float(t0)

    for _ in range(ns):
        dW = tf.sqrt(tf.cast(dd, tf.float64)) * rng.normal(xx.shape, dtype=tf.float64)
        f = as_float64(spec.drift(xx, t))
        g = as_float64(spec.diffusion(xx, t))
        xx = xx + tf.cast(dd, tf.float64) * f + g * dW
        traj.append(xx)
        t += dd

    return tf.stack(traj, axis=0)
