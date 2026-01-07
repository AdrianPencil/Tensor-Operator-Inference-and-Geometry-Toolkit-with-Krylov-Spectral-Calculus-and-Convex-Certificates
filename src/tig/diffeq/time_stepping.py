"""
Time stepping (minimal).

Provides:
- RK4 explicit stepper for ODEs x' = f(x,t)
- implicit Euler for linear systems x' = A x + b(t) (dense)

This is kept compact to support stability-focused notebooks and experiments.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["rk4_step", "implicit_euler_linear_step"]


def rk4_step(f: Callable[[tf.Tensor, float], tf.Tensor], x: TensorLike, t: float, dt: float) -> tf.Tensor:
    """
    One RK4 step.
    """
    xx = as_float64(x)
    tt = float(t)
    dd = float(dt)

    k1 = as_float64(f(xx, tt))
    k2 = as_float64(f(xx + 0.5 * dd * k1, tt + 0.5 * dd))
    k3 = as_float64(f(xx + 0.5 * dd * k2, tt + 0.5 * dd))
    k4 = as_float64(f(xx + dd * k3, tt + dd))

    return xx + (dd / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def implicit_euler_linear_step(
    a: TensorLike,
    x: TensorLike,
    dt: float,
    b: Optional[TensorLike] = None,
) -> tf.Tensor:
    """
    Implicit Euler for linear system x' = A x + b (b treated constant over step):
    (I - dt A) x_{n+1} = x_n + dt b
    """
    aa = as_float64(a)
    xx = tf.reshape(as_float64(x), (-1, 1))
    n = int(xx.shape[0])
    dd = float(dt)

    rhs = xx
    if b is not None:
        bb = tf.reshape(as_float64(b), (-1, 1))
        rhs = rhs + tf.cast(dd, tf.float64) * bb

    mat = tf.eye(n, dtype=tf.float64) - tf.cast(dd, tf.float64) * aa
    xn1 = tf.linalg.solve(mat, rhs)
    return tf.reshape(xn1[:, 0], as_float64(x).shape)
