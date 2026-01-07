"""
Tucker decomposition (minimal, math-first).

For an order-3 tensor X ∈ R^{I×J×K}, Tucker-HOSVD computes:
X ≈ G ×1 U ×2 V ×3 W

This file implements a compact HOSVD-based Tucker decomposition for 3-way tensors.
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["TuckerFactors", "tucker_hosvd", "tucker_reconstruct"]


@dataclass(frozen=True)
class TuckerFactors:
    """
    Tucker factors for a 3-way tensor.
    """

    core: tf.Tensor
    u: tf.Tensor
    v: tf.Tensor
    w: tf.Tensor


def _unfold(x: tf.Tensor, mode: int) -> tf.Tensor:
    x = as_float64(x)
    if mode == 0:
        return tf.reshape(x, (int(x.shape[0]), int(x.shape[1]) * int(x.shape[2])))
    if mode == 1:
        return tf.reshape(tf.transpose(x, perm=[1, 0, 2]), (int(x.shape[1]), int(x.shape[0]) * int(x.shape[2])))
    if mode == 2:
        return tf.reshape(tf.transpose(x, perm=[2, 0, 1]), (int(x.shape[2]), int(x.shape[0]) * int(x.shape[1])))
    raise ValueError("mode must be 0, 1, or 2.")


def tucker_hosvd(x: TensorLike, ranks: Tuple[int, int, int]) -> TuckerFactors:
    """
    Tucker-HOSVD for 3-way tensors using mode-wise SVD on unfoldings.
    """
    xx = as_float64(x)
    r1, r2, r3 = int(ranks[0]), int(ranks[1]), int(ranks[2])

    u1, _, _ = tf.linalg.svd(_unfold(xx, 0), full_matrices=False)
    u2, _, _ = tf.linalg.svd(_unfold(xx, 1), full_matrices=False)
    u3, _, _ = tf.linalg.svd(_unfold(xx, 2), full_matrices=False)

    u = u1[:, :r1]
    v = u2[:, :r2]
    w = u3[:, :r3]

    g = tf.einsum("ia,jb,kc,ijk->abc", tf.transpose(u), tf.transpose(v), tf.transpose(w), xx)
    return TuckerFactors(core=g, u=u, v=v, w=w)


def tucker_reconstruct(f: TuckerFactors) -> tf.Tensor:
    """
    Reconstruct X_hat = G ×1 U ×2 V ×3 W.
    """
    g = as_float64(f.core)
    u = as_float64(f.u)
    v = as_float64(f.v)
    w = as_float64(f.w)
    return tf.einsum("ia,jb,kc,abc->ijk", u, v, w, g)
