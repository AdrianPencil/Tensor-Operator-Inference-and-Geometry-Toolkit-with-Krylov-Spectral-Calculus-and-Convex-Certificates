"""
Superoperator and vec identity sanity.

Validates vec(A X B^T) identity and a commutator consistency check.
"""

import tensorflow as tf

from tig.compute.vectorization_identities import vec, vec_axbt
from tig.core.random import Rng
from tig.noncommutative.commutators import commutator


def test_vec_axbt_identity_consistency() -> None:
    rng = Rng(seed=0)
    a = rng.normal((5, 5), dtype=tf.float64)
    b = rng.normal((4, 4), dtype=tf.float64)
    x = rng.normal((4, 5), dtype=tf.float64)
    y1 = vec(a @ tf.transpose(tf.transpose(x) @ b))
    y2 = vec_axbt(a, x, b)
    rel = tf.linalg.norm(y1 - y2) / (tf.linalg.norm(y2) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-10


def test_commutator_zero_when_commuting() -> None:
    rng = Rng(seed=1)
    d = tf.linalg.diag(rng.normal((6,), dtype=tf.float64))
    c = tf.linalg.diag(rng.normal((6,), dtype=tf.float64))
    k = commutator(d, c)
    assert float(tf.linalg.norm(tf.reshape(k, (-1,))).numpy()) == 0.0
