"""
Filtering (minimal, math-first).

Provides a compact Kalman filter for a linear Gaussian state-space model:
x_{k+1} = A x_k + w_k,   w_k ~ N(0, Q)
y_k     = H x_k + v_k,   v_k ~ N(0, R)

This is used as a principled baseline and a bridge to inverse/UQ sections.
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["KalmanSpec", "KalmanState", "kalman_filter"]


@dataclass(frozen=True)
class KalmanSpec:
    """
    Linear Gaussian model parameters.
    """

    a: tf.Tensor
    h: tf.Tensor
    q: tf.Tensor
    r: tf.Tensor


@dataclass(frozen=True)
class KalmanState:
    """
    Gaussian belief state N(m, P).
    """

    mean: tf.Tensor
    cov: tf.Tensor


def kalman_filter(spec: KalmanSpec, init: KalmanState, ys: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Run the Kalman filter over observations.

    Parameters
    ----------
    spec:
        Model matrices (A,H,Q,R).
    init:
        Initial Gaussian belief.
    ys:
        Observations of shape (T, m).

    Returns
    -------
    (means, covs)
        Filtered means and covariances.
    """
    a = as_float64(spec.a)
    h = as_float64(spec.h)
    q = as_float64(spec.q)
    r = as_float64(spec.r)

    y = as_float64(ys)
    t_steps = int(y.shape[0])

    m = as_float64(init.mean)
    p = as_float64(init.cov)

    means = []
    covs = []

    for k in range(t_steps):
        m_pred = a @ m
        p_pred = a @ p @ tf.transpose(a) + q

        s = h @ p_pred @ tf.transpose(h) + r
        k_gain = p_pred @ tf.transpose(h) @ tf.linalg.inv(s)

        innov = tf.reshape(y[k], (-1, 1)) - (h @ m_pred)
        m = m_pred + k_gain @ innov
        p = (tf.eye(int(p.shape[0]), dtype=tf.float64) - k_gain @ h) @ p_pred

        means.append(tf.reshape(m, (-1,)))
        covs.append(p)

    return tf.stack(means, axis=0), tf.stack(covs, axis=0)
