"""
Tensor identity checks.

These tests validate vec/unvec and Kronecker-style identities used for
vectorization and operator algebra.
"""

import tensorflow as tf

from tig.compute.vectorization_identities import kron_apply_to_vec, unvec, vec, vec_axbt
from tig.core.random import Rng


def test_vec_unvec_roundtrip() -> None:
    rng = Rng(seed=0)
    x = rng.normal((4, 5), dtype=tf.float64)
    v = vec(x)
    xr = unvec(v, (4, 5))
    err = tf.linalg.norm(tf.reshape(x - xr, (-1,)))
    assert float(err.numpy()) == 0.0


def test_vec_axbt_matches_kron_apply() -> None:
    rng = Rng(seed=1)
    a = rng.normal((3, 3), dtype=tf.float64)
    b = rng.normal((4, 4), dtype=tf.float64)
    x = rng.normal((4, 3), dtype=tf.float64)

    v = vec(x)
    left = vec_axbt(a, x, b)
    right = kron_apply_to_vec(a, b, v, x_shape=(4, 3))

    diff = tf.linalg.norm(left - right)
    denom = tf.linalg.norm(right) + tf.cast(1e-15, tf.float64)
    assert float((diff / denom).numpy()) < 1e-10
