"""
Conditioning diagnostics (minimal).

This module returns conditioning-related arrays suitable for plotting:
- condition number proxy via singular values (dense)
- residual histories formatting helpers

Plotting is optional; notebooks can render these with matplotlib.
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["SVDConditioning", "svd_conditioning"]


@dataclass(frozen=True)
class SVDConditioning:
    """
    Dense SVD-based conditioning diagnostics.
    """

    s_max: float
    s_min: float
    cond: float


def svd_conditioning(a: TensorLike, ridge: float = 1e-12) -> SVDConditioning:
    aa = as_float64(a)
    s = tf.linalg.svd(aa, compute_uv=False)
    smax = float(tf.reduce_max(s).numpy())
    smin = float((tf.reduce_min(s) + tf.cast(float(ridge), tf.float64)).numpy())
    cond = float(smax / smin) if smin != 0.0 else float("inf")
    return SVDConditioning(s_max=float(smax), s_min=float(smin), cond=float(cond))
