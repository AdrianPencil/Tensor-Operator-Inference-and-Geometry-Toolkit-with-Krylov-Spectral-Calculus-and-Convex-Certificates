"""
Convex certificates (minimal).

Provides:
- PSD certificate check for symmetric matrices (eigenvalue-based, dense)
- primal-dual gap proxy for scalar objective values
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["is_psd", "primal_dual_gap"]


def is_psd(a: TensorLike, tol: float = 1e-10) -> bool:
    aa = as_float64(a)
    sym = 0.5 * (aa + tf.transpose(aa))
    w = tf.linalg.eigvalsh(sym)
    return bool((tf.reduce_min(w) >= tf.cast(-float(tol), tf.float64)).numpy())


def primal_dual_gap(primal_value: float, dual_value: float) -> float:
    return float(primal_value) - float(dual_value)
