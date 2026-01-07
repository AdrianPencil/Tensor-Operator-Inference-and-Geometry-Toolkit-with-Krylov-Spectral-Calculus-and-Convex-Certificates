"""
CP (CANDECOMP/PARAFAC) decomposition (minimal).

Represents an order-3 tensor X ∈ R^{I×J×K} as:
X ≈ sum_{r=1}^R a_r ⊗ b_r ⊗ c_r

This file provides:
- CPFactors container
- reconstruction
- a minimal ALS solver for order-3 tensors

This is intentionally compact and math-driven, not a fully-featured CP package.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import TensorLike, as_float64

__all__ = ["CPFactors", "cp_reconstruct", "cp_als"]


@dataclass(frozen=True)
class CPFactors:
    """
    CP factors for a 3-way tensor: (A, B, C) with rank R.
    """

    a: tf.Tensor
    b: tf.Tensor
    c: tf.Tensor


def cp_reconstruct(f: CPFactors) -> tf.Tensor:
    """
    Reconstruct X_hat[i,j,k] = sum_r A[i,r] B[j,r] C[k,r].
    """
    a = as_float64(f.a)
    b = as_float64(f.b)
    c = as_float64(f.c)
    return tf.einsum("ir,jr,kr->ijk", a, b, c)


def _khatri_rao(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = as_float64(x)
    y = as_float64(y)
    return tf.reshape(tf.einsum("ir,jr->ijr", x, y), (int(x.shape[0]) * int(y.shape[0]), int(x.shape[1])))


def _solve_normal(gram: tf.Tensor, rhs: tf.Tensor, ridge: float = 1e-10) -> tf.Tensor:
    gram = as_float64(gram)
    rhs = as_float64(rhs)
    r = int(gram.shape[0])
    reg = ridge * tf.eye(r, dtype=tf.float64)
    return tf.linalg.solve(gram + reg, rhs)


def cp_als(
    x: TensorLike,
    rank: int,
    n_iter: int = 50,
    rng: Optional[Rng] = None,
) -> CPFactors:
    """
    Minimal CP-ALS for 3-way tensors.

    Parameters
    ----------
    x:
        Tensor of shape (I, J, K).
    rank:
        Target CP rank R.
    n_iter:
        ALS iterations.
    rng:
        Optional reproducible RNG.

    Returns
    -------
    CPFactors
        Factor matrices (A, B, C).
    """
    xx = as_float64(x)
    i, j, k = int(xx.shape[0]), int(xx.shape[1]), int(xx.shape[2])
    r = int(rank)

    if rng is None:
        rng = Rng(seed=0)

    a = rng.normal((i, r), dtype=tf.float64)
    b = rng.normal((j, r), dtype=tf.float64)
    c = rng.normal((k, r), dtype=tf.float64)

    x1 = tf.reshape(xx, (i, j * k))
    x2 = tf.reshape(tf.transpose(xx, perm=[1, 0, 2]), (j, i * k))
    x3 = tf.reshape(tf.transpose(xx, perm=[2, 0, 1]), (k, i * j))

    for _ in range(int(n_iter)):
        kr = _khatri_rao(c, b)
        gram = (tf.transpose(b) @ b) * (tf.transpose(c) @ c)
        a = tf.transpose(_solve_normal(gram, tf.transpose(x1 @ kr)))

        kr = _khatri_rao(c, a)
        gram = (tf.transpose(a) @ a) * (tf.transpose(c) @ c)
        b = tf.transpose(_solve_normal(gram, tf.transpose(x2 @ kr)))

        kr = _khatri_rao(b, a)
        gram = (tf.transpose(a) @ a) * (tf.transpose(b) @ b)
        c = tf.transpose(_solve_normal(gram, tf.transpose(x3 @ kr)))

        a = a / tf.maximum(tf.linalg.norm(a, axis=0, keepdims=True), 1e-12)
        b = b / tf.maximum(tf.linalg.norm(b, axis=0, keepdims=True), 1e-12)
        c = c / tf.maximum(tf.linalg.norm(c, axis=0, keepdims=True), 1e-12)

    return CPFactors(a=a, b=b, c=c)
