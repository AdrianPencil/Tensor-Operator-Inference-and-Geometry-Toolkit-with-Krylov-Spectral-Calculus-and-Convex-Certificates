"""
Fourier Parseval sanity and filter weights behavior.

TensorFlow FFT is unnormalized, so Parseval holds as:
sum |x|^2 = (1/n) sum |X|^2, where X = FFT(x).
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.fourier.filter_functions import bandpass_ideal, lowpass_ideal


def test_parseval_tf_fft_convention() -> None:
    rng = Rng(seed=0)
    n = 1024
    x = rng.normal((n,), dtype=tf.float64)
    xc = tf.cast(x, tf.complex128)

    X = tf.signal.fft(xc)
    lhs = tf.reduce_sum(tf.cast(x * x, tf.float64))
    rhs = (1.0 / tf.cast(n, tf.float64)) * tf.reduce_sum(tf.cast(tf.abs(X) ** 2, tf.float64))

    rel = tf.abs(lhs - rhs) / (tf.abs(rhs) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-10


def test_filter_functions_are_binary_weights() -> None:
    omega = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float64)
    lp = lowpass_ideal(omega, omega_c=1.5)
    bp = bandpass_ideal(omega, omega_lo=0.5, omega_hi=2.0)

    assert set(lp.numpy().tolist()) <= {0.0, 1.0}
    assert set(bp.numpy().tolist()) <= {0.0, 1.0}
