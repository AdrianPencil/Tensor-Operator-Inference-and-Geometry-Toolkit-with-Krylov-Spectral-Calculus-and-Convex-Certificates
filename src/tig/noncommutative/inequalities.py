"""
Noncommutative inequalities (minimal).

This module provides small building blocks used for sanity checks:
- positivity proxy via eigenvalues of Hermitian part
- Golden–Thompson proxy check for small dense matrices:
  tr(exp(A+B)) <= tr(exp(A) exp(B)) for Hermitian A,B

These are intentionally small - deeper theory belongs in docs/theory.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128, as_float64

__all__ = ["min_eig_hermitian_part", "golden_thompson_gap"]


def min_eig_hermitian_part(a: TensorLike) -> tf.Tensor:
    aa = as_complex128(a)
    h = 0.5 * (aa + tf.linalg.adjoint(aa))
    w = tf.linalg.eigvalsh(h)
    return tf.reduce_min(tf.cast(w, tf.float64))


def golden_thompson_gap(a: TensorLike, b: TensorLike) -> tf.Tensor:
    aa = as_complex128(a)
    bb = as_complex128(b)
    lhs = tf.linalg.trace(tf.linalg.expm(aa + bb))
    rhs = tf.linalg.trace(tf.linalg.expm(aa) @ tf.linalg.expm(bb))
    return tf.cast(rhs - lhs, tf.float64)
