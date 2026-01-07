"""
Spectral expmv sanity.

Compares expmv(A, v, t) against dense expm(tA) v.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.spectral.expmv import expmv


def test_expmv_matches_dense_expm() -> None:
    rng = Rng(seed=0)
    a = rng.normal((32, 32), dtype=tf.float64)
    v = rng.normal((32,), dtype=tf.float64)
    t = 0.3

    out = expmv(a, v, t=t)
    dense = tf.linalg.expm(tf.cast(t, tf.float64) * a) @ tf.reshape(v, (-1, 1))
    dense = dense[:, 0]

    rel = tf.linalg.norm(out - dense) / (tf.linalg.norm(dense) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-10
