"""
Proximal operators (minimal, math-first).

The proximal operator of g is:
prox_{λ g}(v) = argmin_x 0.5||x - v||^2 + λ g(x).

This module keeps a small set of commonly used proximal maps.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["prox_l1", "proj_l2_ball", "proj_nonneg"]


def prox_l1(v: TensorLike, lam: float) -> tf.Tensor:
    """
    Soft-thresholding: prox_{λ ||·||_1}(v).
    """
    vv = as_float64(v)
    t = tf.cast(float(lam), tf.float64)
    return tf.sign(vv) * tf.maximum(tf.abs(vv) - t, 0.0)


def proj_l2_ball(v: TensorLike, radius: float) -> tf.Tensor:
    """
    Projection onto the Euclidean ball {x : ||x||_2 <= radius}.
    """
    vv = as_float64(v)
    r = float(radius)
    n = tf.linalg.norm(vv)
    n_val = float(n.numpy())
    if n_val <= r or n_val == 0.0:
        return vv
    return vv * tf.cast(r / n_val, tf.float64)


def proj_nonneg(v: TensorLike) -> tf.Tensor:
    """
    Projection onto the nonnegative orthant.
    """
    vv = as_float64(v)
    return tf.maximum(vv, 0.0)
