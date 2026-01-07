"""
Rank notions (math-first).

This module starts from definitions and exposes only a few canonical quantities:
- matrix rank (numerical)
- stable rank

Tensor CP/Tucker/TT ranks are handled in linalg/tensor_decomp/.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["matrix_rank", "stable_rank"]


def matrix_rank(a: TensorLike, tol: float | None = None) -> tf.Tensor:
    """
    Numerical rank of a matrix via singular values.

    If tol is None, a TF-default tolerance is used.
    """
    aa = as_float64(a)
    s = tf.linalg.svd(aa, compute_uv=False)
    if tol is None:
        tol = float(tf.reduce_max(s).numpy()) * max(aa.shape) * 1e-12
    return tf.reduce_sum(tf.cast(s > tol, tf.int32))


def stable_rank(a: TensorLike) -> tf.Tensor:
    """
    Stable rank: ||A||_F^2 / ||A||_2^2.
    """
    aa = as_float64(a)
    s = tf.linalg.svd(aa, compute_uv=False)
    fro2 = tf.reduce_sum(s * s)
    op2 = tf.reduce_max(s) ** 2
    return fro2 / op2
