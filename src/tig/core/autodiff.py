"""
Autodiff primitives (TF-first).

This module exposes math-facing JVP/VJP operations:
- JVP: Jacobian-vector product
- VJP: vector-Jacobian product (adjoint mode)

The interface is intentionally small and composable.
"""

from typing import Callable, Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["jvp", "vjp", "grad"]


def jvp(f: Callable[[tf.Tensor], tf.Tensor], x: TensorLike, v: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute (f(x), J(x) v) using TF forward-mode accumulation.
    """
    xx = as_float64(x)
    vv = as_float64(v)

    with tf.autodiff.ForwardAccumulator(primals=xx, tangents=vv) as acc:
        y = f(xx)
    jv = acc.jvp(y)
    return y, jv


def vjp(f: Callable[[tf.Tensor], tf.Tensor], x: TensorLike, u: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute (f(x), J(x)^T u) using TF reverse-mode (GradientTape).
    """
    xx = as_float64(x)
    uu = as_float64(u)

    with tf.GradientTape() as tape:
        tape.watch(xx)
        y = f(xx)

    dot = tf.reduce_sum(as_float64(y) * uu)
    jt_u = tape.gradient(dot, xx)
    return y, jt_u


def grad(f: Callable[[tf.Tensor], tf.Tensor], x: TensorLike) -> tf.Tensor:
    """
    Gradient of a scalar-valued function f at x.
    """
    xx = as_float64(x)
    with tf.GradientTape() as tape:
        tape.watch(xx)
        y = f(xx)
    return tape.gradient(y, xx)
