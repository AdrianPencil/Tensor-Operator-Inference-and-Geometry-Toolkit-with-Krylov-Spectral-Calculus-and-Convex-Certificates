"""
k-space / ω-space conventions (minimal).

This module provides compact helpers for:
- angular frequency ω = 2π f
- wavenumber k = 2π / λ
"""

import tensorflow as tf

__all__ = ["omega_from_f", "k_from_lambda"]


def omega_from_f(f: float) -> tf.Tensor:
    return tf.cast(2.0 * 3.141592653589793 * float(f), tf.float64)


def k_from_lambda(lam: float) -> tf.Tensor:
    return tf.cast(2.0 * 3.141592653589793 / float(lam), tf.float64)
