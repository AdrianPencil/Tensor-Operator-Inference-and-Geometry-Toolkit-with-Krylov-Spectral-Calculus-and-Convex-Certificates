"""
Resolvent computations (minimal, math-first).

Resolvent of A at z:
R(z; A) = (z I - A)^{-1}.

This file provides:
- dense resolvent-vector solve
- a simple bound proxy via ||R(z;A)||_2 using SVD for small matrices
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128, as_float64

__all__ = ["resolvent_solve", "resolvent_norm_2_proxy"]


def resolvent_solve(a: TensorLike, z: complex, v: TensorLike) -> tf.Tensor:
    """
    Solve (z I - A) x = v for x (dense).
    """
    aa = as_complex128(a)
    vv = as_complex128(v)
    n = int(aa.shape[0])
    zz = tf.cast(z, tf.complex128)
    mat = zz * tf.eye(n, dtype=tf.complex128) - aa
    return tf.linalg.solve(mat, tf.reshape(vv, (-1, 1)))[:, 0]


def resolvent_norm_2_proxy(a: TensorLike, z: complex) -> tf.Tensor:
    """
    Proxy for ||(zI - A)^{-1}||_2 using sigma_min of (zI - A).
    """
    aa = as_complex128(a)
    n = int(aa.shape[0])
    zz = tf.cast(z, tf.complex128)
    mat = zz * tf.eye(n, dtype=tf.complex128) - aa
    s = tf.linalg.svd(mat, compute_uv=False)
    smin = tf.reduce_min(tf.abs(s))
    return 1.0 / tf.cast(smin, tf.float64)
