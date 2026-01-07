"""
Regularizers (math-first).

A regularizer is a functional R: X -> R, optionally equipped with a proximal map.
This module keeps a small set of canonical regularizers used in inverse problems:
- L2 (Tikhonov)
- L1 (sparsity)

Proximal maps are TF-first and intended to compose with ADMM/prox solvers.
"""

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["Regularizer", "L2Regularizer", "L1Regularizer"]


class Regularizer(Protocol):
    """
    Regularizer interface with optional proximal map.
    """

    def value(self, x: TensorLike) -> tf.Tensor: ...
    def prox(self, v: TensorLike, lam: float) -> tf.Tensor: ...


@dataclass(frozen=True)
class L2Regularizer:
    """
    R(x) = 0.5 ||x||_2^2.
    """

    def value(self, x: TensorLike) -> tf.Tensor:
        xx = as_float64(x)
        return 0.5 * tf.reduce_sum(xx * xx)

    def prox(self, v: TensorLike, lam: float) -> tf.Tensor:
        """
        prox_{λ R}(v) = v / (1 + λ).
        """
        vv = as_float64(v)
        den = tf.cast(1.0 + float(lam), tf.float64)
        return vv / den


@dataclass(frozen=True)
class L1Regularizer:
    """
    R(x) = ||x||_1.
    """

    def value(self, x: TensorLike) -> tf.Tensor:
        xx = as_float64(x)
        return tf.reduce_sum(tf.abs(xx))

    def prox(self, v: TensorLike, lam: float) -> tf.Tensor:
        """
        Soft-thresholding prox for λ||·||_1.
        """
        vv = as_float64(v)
        t = tf.cast(float(lam), tf.float64)
        return tf.sign(vv) * tf.maximum(tf.abs(vv) - t, 0.0)
