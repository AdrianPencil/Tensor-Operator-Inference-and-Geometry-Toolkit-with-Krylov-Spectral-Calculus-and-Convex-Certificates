"""
Verification and validation constraints (minimal).

This module provides small checks that can be used as constraints or post-hoc
verification for simulations and optimizations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["VerificationResult", "check_bounds", "check_monotone_nonincreasing"]


@dataclass(frozen=True)
class VerificationResult:
    """
    Result of a verification check.
    """

    ok: bool
    message: str


def check_bounds(x: TensorLike, lo: float, hi: float, name: str = "x") -> VerificationResult:
    xx = as_float64(x)
    lo_t = tf.cast(float(lo), tf.float64)
    hi_t = tf.cast(float(hi), tf.float64)
    ok = tf.reduce_all(tf.logical_and(xx >= lo_t, xx <= hi_t))
    if bool(ok.numpy()):
        return VerificationResult(ok=True, message=f"{name} within bounds.")
    return VerificationResult(ok=False, message=f"{name} violates bounds [{lo}, {hi}].")


def check_monotone_nonincreasing(x: TensorLike, name: str = "x") -> VerificationResult:
    xx = tf.reshape(as_float64(x), (-1,))
    if int(xx.shape[0]) <= 1:
        return VerificationResult(ok=True, message=f"{name} trivially monotone.")
    dif = xx[1:] - xx[:-1]
    ok = tf.reduce_all(dif <= tf.cast(0.0, tf.float64))
    if bool(ok.numpy()):
        return VerificationResult(ok=True, message=f"{name} is nonincreasing.")
    return VerificationResult(ok=False, message=f"{name} is not nonincreasing.")
