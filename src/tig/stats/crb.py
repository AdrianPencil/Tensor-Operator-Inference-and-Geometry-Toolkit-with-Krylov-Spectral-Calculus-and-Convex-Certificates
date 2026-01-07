"""
Cramér-Rao bounds (minimal).

For an unbiased estimator under regularity conditions, covariance satisfies:
Cov(theta_hat) >= I(theta)^{-1}.

This module provides:
- dense CRB for small problems
- diagonal bound extraction for reporting
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["crb_dense", "crb_diag"]


def crb_dense(fisher: TensorLike, ridge: float = 1e-12) -> tf.Tensor:
    """
    Compute (I + ridge*I)^{-1} for a dense Fisher matrix.
    """
    ff = as_float64(fisher)
    n = int(ff.shape[0])
    reg = tf.cast(float(ridge), tf.float64) * tf.eye(n, dtype=tf.float64)
    return tf.linalg.inv(ff + reg)


def crb_diag(fisher: TensorLike, ridge: float = 1e-12) -> tf.Tensor:
    """
    Return diagonal of the dense CRB matrix.
    """
    inv = crb_dense(fisher, ridge=ridge)
    return tf.linalg.diag_part(inv)
