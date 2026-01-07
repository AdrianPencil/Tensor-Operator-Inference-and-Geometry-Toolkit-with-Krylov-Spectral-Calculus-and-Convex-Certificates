"""
Convolution adjoint pairing.

For circular convolution A(x) = ifft(fft(x) * H),
the adjoint is A*(y) = ifft(fft(y) * conj(H)) for the standard complex inner product.
We test the real-valued specialization (still using complex FFT internally).
"""

import tensorflow as tf

from tig.core.random import Rng


def _circ_conv_real(x: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
    xx = tf.cast(tf.reshape(x, (-1,)), tf.complex128)
    hh = tf.cast(tf.reshape(h, (-1,)), tf.complex128)
    y = tf.signal.ifft(tf.signal.fft(xx) * tf.signal.fft(hh))
    return tf.cast(tf.math.real(y), tf.float64)


def _circ_conv_adjoint_real(y: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
    yy = tf.cast(tf.reshape(y, (-1,)), tf.complex128)
    hh = tf.cast(tf.reshape(h, (-1,)), tf.complex128)
    hhat = tf.signal.fft(hh)
    x = tf.signal.ifft(tf.signal.fft(yy) * tf.math.conj(hhat))
    return tf.cast(tf.math.real(x), tf.float64)


def test_circular_convolution_adjoint_pairing() -> None:
    rng = Rng(seed=0)
    n = 256
    x = rng.normal((n,), dtype=tf.float64)
    y = rng.normal((n,), dtype=tf.float64)
    h = rng.normal((n,), dtype=tf.float64)

    ax = _circ_conv_real(x, h)
    aty = _circ_conv_adjoint_real(y, h)

    left = tf.reduce_sum(ax * y)
    right = tf.reduce_sum(x * aty)

    gap = tf.abs(left - right)
    denom = tf.abs(left) + tf.abs(right) + tf.cast(1e-15, tf.float64)
    assert float((gap / denom).numpy()) < 1e-8
