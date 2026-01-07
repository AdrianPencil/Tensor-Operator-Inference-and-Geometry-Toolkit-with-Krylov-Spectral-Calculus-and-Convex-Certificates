"""
Stiffness metrics and stability proxies (minimal).

For linear systems x' = A x, stiffness relates to the eigenvalue spread.
This module provides:
- spectral radius proxy via ||A||_2
- stiffness ratio proxy via max|λ| / min|λ| (dense eigvals)
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64
from tig.core.norms import op_norm_2

__all__ = ["spectral_radius_proxy", "stiffness_ratio_proxy"]


def spectral_radius_proxy(a: TensorLike) -> tf.Tensor:
    """
    Proxy: rho(A) <= ||A||_2.
    """
    return op_norm_2(a)


def stiffness_ratio_proxy(a: TensorLike, ridge: float = 1e-12) -> tf.Tensor:
    """
    Dense eigenvalue-based proxy max|λ| / (min|λ| + ridge).
    """
    aa = as_float64(a)
    w = tf.linalg.eigvals(tf.cast(aa, tf.complex128))
    mag = tf.abs(w)
    mx = tf.reduce_max(mag)
    mn = tf.reduce_min(mag)
    return tf.cast(mx, tf.float64) / (tf.cast(mn, tf.float64) + tf.cast(float(ridge), tf.float64))
