"""
Error propagation (math-first).

Linearized (delta-method) propagation through y = F(x):
Cov_y ≈ J Cov_x J^T

This module supports:
- dense pushforward for small problems (explicit J via JVP probes)
- scalar-output variance propagation
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from tig.core.types import TensorLike, as_float64
from tig.inverse.forward_models import ForwardModel
from tig.uq.sensitivity import explicit_jacobian_small

__all__ = ["PropagationResult", "pushforward_cov_dense", "scalar_variance_pushforward"]


@dataclass(frozen=True)
class PropagationResult:
    """
    Linearized propagation result.
    """

    mean_y: tf.Tensor
    cov_y: tf.Tensor


def pushforward_cov_dense(
    model: ForwardModel,
    x_mean: TensorLike,
    cov_x: TensorLike,
) -> PropagationResult:
    """
    Dense linearized pushforward covariance.

    Parameters
    ----------
    model:
        Forward model F.
    x_mean:
        Mean point x̄.
    cov_x:
        Dense covariance Cov_x in flattened coordinates.

    Returns
    -------
    PropagationResult
        mean_y = F(x̄), cov_y ≈ J Cov_x J^T.
    """
    xx = as_float64(x_mean)
    cov = as_float64(cov_x)

    fx = as_float64(model(xx))
    j = as_float64(explicit_jacobian_small(model, xx))

    cov_y = j @ cov @ tf.transpose(j)
    return PropagationResult(mean_y=fx, cov_y=cov_y)


def scalar_variance_pushforward(
    model: ForwardModel,
    x_mean: TensorLike,
    cov_x: TensorLike,
) -> tf.Tensor:
    """
    For scalar y = F(x) (y shape () or (1,)), approximate Var(y) ≈ g^T Cov_x g,
    where g = ∇F(x̄) in flattened coordinates.

    Returns
    -------
    tf.Tensor
        Scalar variance.
    """
    xx = as_float64(x_mean)
    cov = as_float64(cov_x)

    with tf.GradientTape() as tape:
        tape.watch(xx)
        y = tf.reshape(as_float64(model(xx)), ())
    g = as_float64(tape.gradient(y, xx))
    g_flat = tf.reshape(g, (-1, 1))
    var = tf.transpose(g_flat) @ cov @ g_flat
    return tf.reshape(as_float64(var), ())
