"""
Fourier/Laplace operator wrappers (minimal, TF-first).

This module provides:
- FFT / IFFT wrappers for 1D signals
- frequency grid helper

SciPy can be used as an optional oracle elsewhere; TF is the default backend.
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128, as_float64

__all__ = ["fft1", "ifft1", "freq_grid"]


def fft1(x: TensorLike) -> tf.Tensor:
    xx = tf.reshape(as_complex128(x), (-1,))
    return tf.signal.fft(xx)


def ifft1(x: TensorLike) -> tf.Tensor:
    xx = tf.reshape(as_complex128(x), (-1,))
    return tf.signal.ifft(xx)


def freq_grid(n: int, dt: float) -> tf.Tensor:
    """
    Angular frequency grid ω for FFT ordering matching tf.signal.fft.
    """
    nn = int(n)
    dd = float(dt)
    fs = 1.0 / dd
    f = tf.signal.fftshift(tf.linspace(-0.5 * fs, 0.5 * fs, nn))
    omega = 2.0 * 3.141592653589793 * f
    return tf.cast(omega, tf.float64)
