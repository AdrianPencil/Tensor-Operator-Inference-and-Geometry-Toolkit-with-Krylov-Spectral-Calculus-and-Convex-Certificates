"""
Likelihoods for stochastic models (minimal).

Provides Gaussian log-likelihood building blocks used in filtering and inverse problems.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["gaussian_logpdf", "gaussian_loglik_iid"]


def gaussian_logpdf(x: TensorLike, mean: TensorLike, cov: TensorLike) -> tf.Tensor:
    """
    Log density of N(mean, cov) at x (dense, small/medium).

    Returns a scalar tensor.
    """
    xx = tf.reshape(as_float64(x), (-1, 1))
    mm = tf.reshape(as_float64(mean), (-1, 1))
    cc = as_float64(cov)

    d = int(xx.shape[0])
    diff = xx - mm
    sol = tf.linalg.solve(cc, diff)
    quad = tf.reduce_sum(diff * sol)
    logdet = tf.linalg.logdet(cc)
    const = tf.cast(d, tf.float64) * tf.math.log(tf.cast(2.0 * 3.141592653589793, tf.float64))
    return -0.5 * (quad + logdet + const)


def gaussian_loglik_iid(residuals: TensorLike, sigma: float) -> tf.Tensor:
    """
    IID Gaussian log-likelihood for residuals with variance sigma^2:
    sum_k -0.5*(r_k^2/sigma^2 + log(2π sigma^2)).
    """
    rr = as_float64(residuals)
    ss2 = tf.cast(float(sigma) ** 2, tf.float64)
    const = tf.math.log(tf.cast(2.0 * 3.141592653589793, tf.float64) * ss2)
    return tf.reduce_sum(-0.5 * (rr * rr / ss2 + const))
