"""
Convolution operators (minimal).

Discrete convolution is treated as a kernel operator with Toeplitz structure.
This module provides:
- FFT-based circular convolution
- linear convolution via zero-padding
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128, as_float64

__all__ = ["circular_convolution", "linear_convolution"]


def circular_convolution(x: TensorLike, h: TensorLike) -> tf.Tensor:
    """
    Circular convolution via FFT on equal-length vectors.
    """
    xx = tf.reshape(as_complex128(x), (-1,))
    hh = tf.reshape(as_complex128(h), (-1,))
    if int(xx.shape[0]) != int(hh.shape[0]):
        raise ValueError("circular_convolution requires equal-length inputs.")
    y = tf.signal.ifft(tf.signal.fft(xx) * tf.signal.fft(hh))
    return tf.math.real(tf.cast(y, tf.complex128))


def linear_convolution(x: TensorLike, h: TensorLike) -> tf.Tensor:
    """
    Linear convolution via zero-padding and FFT.
    """
    xx = tf.reshape(as_complex128(x), (-1,))
    hh = tf.reshape(as_complex128(h), (-1,))
    n = int(xx.shape[0])
    m = int(hh.shape[0])
    size = n + m - 1

    xxp = tf.pad(xx, [[0, size - n]])
    hhp = tf.pad(hh, [[0, size - m]])

    y = tf.signal.ifft(tf.signal.fft(xxp) * tf.signal.fft(hhp))
    return tf.math.real(tf.cast(y, tf.complex128))
