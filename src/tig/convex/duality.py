"""
Convex duality (minimal, math-first).

Provides small Fenchel conjugates and dual-gap style quantities for canonical functions.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["fenchel_conjugate_quadratic", "duality_gap_quadratic"]


def fenchel_conjugate_quadratic(y: TensorLike) -> tf.Tensor:
    """
    f(x) = 0.5 ||x||^2  =>  f*(y) = 0.5 ||y||^2.
    """
    yy = as_float64(y)
    return 0.5 * tf.reduce_sum(yy * yy)


def duality_gap_quadratic(x: TensorLike, y: TensorLike) -> tf.Tensor:
    """
    For f(x)=0.5||x||^2 and its conjugate, the Fenchel-Young gap:
    f(x) + f*(y) - <x,y> >= 0.
    """
    xx = as_float64(x)
    yy = as_float64(y)
    inner = tf.reduce_sum(xx * yy)
    return 0.5 * tf.reduce_sum(xx * xx) + 0.5 * tf.reduce_sum(yy * yy) - inner
