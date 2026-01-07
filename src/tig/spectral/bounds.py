"""
Spectral bounds (minimal).

This module provides a few canonical inequalities used throughout:
- Gershgorin disks bound for eigenvalue locations (dense)
- spectral radius upper bound via matrix norms
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64
from tig.core.norms import op_norm_2

__all__ = ["gershgorin_bound_radius", "spectral_radius_upper_bound"]


def gershgorin_bound_radius(a: TensorLike) -> tf.Tensor:
    """
    Return the maximum Gershgorin radius max_i sum_{j!=i} |a_ij|.
    """
    aa = as_float64(a)
    abs_a = tf.abs(aa)
    row_sum = tf.reduce_sum(abs_a, axis=1)
    diag = tf.abs(tf.linalg.diag_part(aa))
    radii = row_sum - diag
    return tf.reduce_max(radii)


def spectral_radius_upper_bound(a: TensorLike) -> tf.Tensor:
    """
    Upper bound on spectral radius: rho(A) <= ||A||_2.
    """
    return op_norm_2(a)
