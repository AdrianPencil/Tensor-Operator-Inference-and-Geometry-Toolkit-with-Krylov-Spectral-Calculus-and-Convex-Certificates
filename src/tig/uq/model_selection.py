"""
Model selection criteria (minimal).

This module provides small, explicit criteria used in inverse/UQ:
- Gaussian negative log-likelihood (NLL)
- AIC / BIC for Gaussian residual models
- ELBO-style proxy for variational approximations (lightweight)
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["SelectionScores", "gaussian_nll_iid", "aic", "bic", "elbo_proxy"]


@dataclass(frozen=True)
class SelectionScores:
    """
    Common scalar model selection scores.
    """

    nll: float
    aic: float
    bic: float


def gaussian_nll_iid(residuals: TensorLike, sigma: float) -> tf.Tensor:
    """
    IID Gaussian negative log-likelihood up to constant:
    0.5 * sum (r^2/sigma^2 + log(2π sigma^2)).
    """
    rr = as_float64(residuals)
    ss2 = tf.cast(float(sigma) ** 2, tf.float64)
    const = tf.math.log(tf.cast(2.0 * 3.141592653589793, tf.float64) * ss2)
    return 0.5 * tf.reduce_sum(rr * rr / ss2 + const)


def aic(nll: float, k_params: int) -> float:
    """
    Akaike Information Criterion: AIC = 2k + 2*NLL.
    """
    return 2.0 * float(k_params) + 2.0 * float(nll)


def bic(nll: float, k_params: int, n_obs: int) -> float:
    """
    Bayesian Information Criterion: BIC = k log(n) + 2*NLL.
    """
    return float(k_params) * float(tf.math.log(tf.cast(int(n_obs), tf.float64)).numpy()) + 2.0 * float(nll)


def elbo_proxy(expected_loglik: float, kl: float) -> float:
    """
    ELBO proxy: E_q[log p(y|x)] - KL(q||p).
    """
    return float(expected_loglik) - float(kl)
