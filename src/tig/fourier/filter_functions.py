"""
Filter functions as ω-weights (minimal).

A filter function weights noise PSD in sensitivity calculations:
variance ~ ∫ S(ω) F(ω) dω.

This module provides a few canonical filters used in practice as math objects.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["lowpass_ideal", "bandpass_ideal"]


def lowpass_ideal(omega: TensorLike, omega_c: float) -> tf.Tensor:
    """
    Ideal low-pass filter: F(ω) = 1 for |ω|<=ωc else 0.
    """
    ww = as_float64(omega)
    oc = tf.cast(float(omega_c), tf.float64)
    return tf.where(tf.abs(ww) <= oc, tf.ones_like(ww), tf.zeros_like(ww))


def bandpass_ideal(omega: TensorLike, omega_lo: float, omega_hi: float) -> tf.Tensor:
    """
    Ideal band-pass filter: 1 for ω_lo <= |ω| <= ω_hi else 0.
    """
    ww = as_float64(omega)
    lo = tf.cast(float(omega_lo), tf.float64)
    hi = tf.cast(float(omega_hi), tf.float64)
    inside = tf.logical_and(tf.abs(ww) >= lo, tf.abs(ww) <= hi)
    return tf.where(inside, tf.ones_like(ww), tf.zeros_like(ww))
