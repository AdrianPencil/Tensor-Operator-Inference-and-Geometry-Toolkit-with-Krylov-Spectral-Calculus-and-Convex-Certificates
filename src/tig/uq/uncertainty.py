"""
Uncertainty objects and interval summaries (minimal).

Provides:
- Gaussian confidence intervals for scalar quantities
- covariance -> standard deviations extraction
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["ScalarInterval", "gaussian_confidence_interval", "std_from_cov"]


@dataclass(frozen=True)
class ScalarInterval:
    """
    Scalar interval [lo, hi].
    """

    lo: float
    hi: float


def gaussian_confidence_interval(mean: float, var: float, z: float = 1.96) -> ScalarInterval:
    """
    Symmetric Gaussian confidence interval mean ± z * sqrt(var).

    z=1.96 corresponds to ~95% (two-sided) under a standard normal approximation.
    """
    m = float(mean)
    v = float(var)
    s = (v ** 0.5) if v > 0.0 else 0.0
    return ScalarInterval(lo=m - float(z) * s, hi=m + float(z) * s)


def std_from_cov(cov: TensorLike) -> tf.Tensor:
    """
    Standard deviations from a dense covariance matrix.
    """
    cc = as_float64(cov)
    d = tf.linalg.diag_part(cc)
    return tf.sqrt(tf.maximum(d, tf.constant(0.0, dtype=tf.float64)))
