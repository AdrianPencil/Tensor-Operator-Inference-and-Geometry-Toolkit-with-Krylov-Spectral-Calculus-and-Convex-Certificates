"""
Matrix Product States (MPS) - minimal.

An MPS for a length-L chain with physical dimensions d_k consists of cores:
G_k of shape (r_{k-1}, d_k, r_k), with r_0 = r_L = 1.

This file provides:
- MPS container
- dense reconstruction (for small L)
- inner product by contracting cores (kept simple)
"""

from dataclasses import dataclass
from typing import List, Sequence

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["MPS", "mps_to_dense", "mps_inner"]


@dataclass(frozen=True)
class MPS:
    """
    Matrix product state cores.
    """

    cores: List[tf.Tensor]


def mps_to_dense(mps: MPS) -> tf.Tensor:
    """
    Reconstruct a dense tensor from an MPS (intended for small systems).
    """
    cores = [as_float64(g) for g in mps.cores]
    x = cores[0]
    x = tf.reshape(x, (int(x.shape[1]), int(x.shape[2])))

    for k in range(1, len(cores)):
        g = cores[k]
        g_mat = tf.reshape(g, (int(g.shape[0]), int(g.shape[1]) * int(g.shape[2])))
        x = x @ g_mat
        x = tf.reshape(x, (-1, int(g.shape[2])))

    out_shape = [int(g.shape[1]) for g in cores]
    return tf.reshape(x, tuple(out_shape))


def mps_inner(a: MPS, b: MPS) -> tf.Tensor:
    """
    Inner product <a, b> by sequential core contraction (real case).
    """
    ac = [as_float64(g) for g in a.cores]
    bc = [as_float64(g) for g in b.cores]
    if len(ac) != len(bc):
        raise ValueError("MPS lengths must match.")

    env = tf.constant([[1.0]], dtype=tf.float64)
    for ga, gb in zip(ac, bc):
        env = tf.einsum("ab, aic, bjc -> ij", env, ga, gb)
    return tf.reshape(env, ())
