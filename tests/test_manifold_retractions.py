"""
Manifold retractions sanity.

This test assumes geometry.stiefel provides a retraction producing orthonormal columns.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.geometry.stiefel import retract_qr  # implemented earlier in your sequence


def test_stiefel_qr_retraction_orthonormal() -> None:
    rng = Rng(seed=0)
    x = rng.normal((32, 5), dtype=tf.float64)
    u = rng.normal((32, 5), dtype=tf.float64)
    y = retract_qr(x, u)
    gram = tf.transpose(y) @ y
    err = tf.linalg.norm(gram - tf.eye(5, dtype=tf.float64))
    assert float(err.numpy()) < 1e-8
