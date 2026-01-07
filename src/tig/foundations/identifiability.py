"""
Identifiability primitives (foundations).

Identifiability is framed in terms of local sensitivity and rank:
- Jacobian rank (local identifiability)
- linearized information measures (Fisher-style objects)

This module provides minimal quantities used later in stats/ and inverse/.
"""

import tensorflow as tf

from tig.core.autodiff import jvp
from tig.core.types import TensorLike, as_float64

__all__ = ["finite_difference_jacobian_rank_proxy", "jvp_energy"]


def jvp_energy(f, x: TensorLike, v: TensorLike) -> tf.Tensor:
    """
    Energy of a direction under the linearization: ||J(x) v||_2^2.
    """
    _, jv = jvp(f, as_float64(x), as_float64(v))
    jv = as_float64(jv)
    return tf.reduce_sum(jv * jv)


def finite_difference_jacobian_rank_proxy(
    f,
    x: TensorLike,
    directions: tf.Tensor,
    eps: float = 1e-6,
) -> tf.Tensor:
    """
    Rank proxy via finite differences on a set of directions.

    directions: shape (k, n) interpreted as k probe directions in parameter space.
    Returns the numerical rank of the k x m matrix of directional differences.
    """
    xx = as_float64(x)
    dirs = as_float64(directions)

    fx = as_float64(f(xx))
    probes = []
    for i in range(dirs.shape[0]):
        v = tf.reshape(dirs[i], xx.shape)
        fp = as_float64(f(xx + eps * v))
        probes.append(tf.reshape((fp - fx) / eps, (-1,)))

    mat = tf.stack(probes, axis=0)
    s = tf.linalg.svd(mat, compute_uv=False)
    tol = float(tf.reduce_max(s).numpy()) * max(mat.shape) * 1e-12
    return tf.reduce_sum(tf.cast(s > tol, tf.int32))
