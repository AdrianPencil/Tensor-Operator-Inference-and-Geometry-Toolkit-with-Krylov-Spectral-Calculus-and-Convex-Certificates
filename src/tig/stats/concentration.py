"""
Concentration inequalities (minimal, math-first).

This module is used for reasoning about randomized estimators and noise:
- sub-Gaussian tail bound
- Hoeffding-style bound for bounded variables

These return bounds on probabilities, keeping the interface explicit.
"""

import tensorflow as tf

__all__ = ["subgaussian_tail", "hoeffding_tail"]


def subgaussian_tail(t: float, sigma: float) -> tf.Tensor:
    """
    For a centered sub-Gaussian X with parameter sigma:
    P(|X| >= t) <= 2 exp(-t^2 / (2 sigma^2)).
    """
    tt = tf.cast(float(t), tf.float64)
    ss = tf.cast(float(sigma), tf.float64)
    return 2.0 * tf.exp(-(tt * tt) / (2.0 * ss * ss))


def hoeffding_tail(t: float, n: int, a: float, b: float) -> tf.Tensor:
    """
    For average of n i.i.d. bounded variables in [a,b]:
    P(|mean - E| >= t) <= 2 exp(-2 n t^2 / (b-a)^2).
    """
    tt = tf.cast(float(t), tf.float64)
    nn = tf.cast(int(n), tf.float64)
    width = tf.cast(float(b - a), tf.float64)
    return 2.0 * tf.exp(-2.0 * nn * (tt * tt) / (width * width))
