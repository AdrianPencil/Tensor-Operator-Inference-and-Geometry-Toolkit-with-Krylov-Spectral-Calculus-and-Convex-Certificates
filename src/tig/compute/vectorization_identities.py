"""
Vectorization identities enabling tensor/matrix-free computations (math-first).

Core identities (column-stacking convention for vec):
- vec(A X B^T) = (B ⊗ A) vec(X)
- inner( vec(X), vec(Y) ) = inner(X, Y) under Frobenius

These primitives support noncommutative/superoperator tests and structured linalg.
"""

from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["vec", "unvec", "vec_axbt", "kron_apply_to_vec"]


def vec(x: TensorLike) -> tf.Tensor:
    """
    Column-stacking vec for matrices/tensors: returns a 1D view.
    """
    xx = as_float64(x)
    return tf.reshape(xx, (-1,))


def unvec(v: TensorLike, shape: Tuple[int, ...]) -> tf.Tensor:
    """
    Inverse of vec: reshape a 1D vector to the given shape.
    """
    vv = as_float64(v)
    return tf.reshape(vv, tuple(int(s) for s in shape))


def vec_axbt(a: TensorLike, x: TensorLike, b: TensorLike) -> tf.Tensor:
    """
    Compute vec(A X B^T) without forming Kronecker explicitly.
    """
    aa = as_float64(a)
    xx = as_float64(x)
    bb = as_float64(b)
    y = aa @ xx @ tf.transpose(bb)
    return tf.reshape(y, (-1,))


def kron_apply_to_vec(a: TensorLike, b: TensorLike, v: TensorLike, x_shape: Tuple[int, int]) -> tf.Tensor:
    """
    Apply (A ⊗ B) to vec(X) using vec(B X A^T).
    """
    aa = as_float64(a)
    bb = as_float64(b)
    vv = as_float64(v)
    m, n = int(x_shape[0]), int(x_shape[1])
    x = tf.reshape(vv, (m, n))
    y = bb @ x @ tf.transpose(aa)
    return tf.reshape(y, (-1,))
