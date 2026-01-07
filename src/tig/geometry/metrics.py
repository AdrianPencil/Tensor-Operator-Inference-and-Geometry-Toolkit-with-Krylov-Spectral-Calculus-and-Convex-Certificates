"""
Riemannian metrics (minimal).

This module defines inner products and norms used by manifold implementations.
For matrix manifolds, the default is the Frobenius metric.
"""

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["RiemannianMetric", "FrobeniusMetric", "metric_norm"]


class RiemannianMetric(Protocol):
    """
    Metric g_x(u, v) returning a scalar tensor.
    """

    def __call__(self, x: TensorLike, u: TensorLike, v: TensorLike) -> tf.Tensor: ...


@dataclass(frozen=True)
class FrobeniusMetric:
    """
    Frobenius metric g_x(u, v) = trace(u^T v) = sum_ij u_ij v_ij.
    """

    def __call__(self, x: TensorLike, u: TensorLike, v: TensorLike) -> tf.Tensor:
        uu = as_float64(u)
        vv = as_float64(v)
        return tf.reduce_sum(uu * vv)


def metric_norm(metric: RiemannianMetric, x: TensorLike, u: TensorLike) -> tf.Tensor:
    """
    Norm induced by a metric: ||u||_x = sqrt(g_x(u, u)).
    """
    val = as_float64(metric(x, u, u))
    return tf.sqrt(tf.maximum(val, tf.constant(0.0, dtype=tf.float64)))
