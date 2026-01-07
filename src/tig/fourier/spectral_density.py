"""
Spectral densities (minimal).

Provides a compact power spectral density (PSD) estimator using FFT:
PSD(ω) ≈ |FFT(x)|^2 (with simple normalization conventions).

This is math-facing and intended for filter-function and identification demos.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128

__all__ = ["periodogram_psd"]


def periodogram_psd(x: TensorLike) -> tf.Tensor:
    """
    Periodogram PSD proxy: |X(ω)|^2 / n where X is FFT(x).
    """
    xx = tf.reshape(as_complex128(x), (-1,))
    n = int(xx.shape[0])
    xw = tf.signal.fft(xx)
    psd = tf.abs(xw) ** 2 / tf.cast(n, tf.float64)
    return tf.cast(psd, tf.float64)
