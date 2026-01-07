"""
Nuclear norm tools (minimal).

Nuclear norm ||A||_* = sum_i sigma_i(A).
Proximal operator (singular value thresholding):
prox_{λ||·||_*}(A) = U diag((s-λ)_+) V^T
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["nuclear_norm", "prox_nuclear_norm"]


def nuclear_norm(a: TensorLike) -> tf.Tensor:
    aa = as_float64(a)
    s = tf.linalg.svd(aa, compute_uv=False)
    return tf.reduce_sum(s)


def prox_nuclear_norm(a: TensorLike, lam: float) -> tf.Tensor:
    aa = as_float64(a)
    u, s, vt = tf.linalg.svd(aa, full_matrices=False)
    t = tf.cast(float(lam), tf.float64)
    s_shrink = tf.maximum(s - t, 0.0)
    return u @ tf.linalg.diag(s_shrink) @ vt
