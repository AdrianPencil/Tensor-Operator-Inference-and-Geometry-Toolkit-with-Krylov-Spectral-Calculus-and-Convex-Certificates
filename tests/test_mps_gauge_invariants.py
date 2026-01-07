"""
MPS consistency sanity.

Compares MPS inner product computed by contraction vs dense reconstruction inner product.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.tensor_networks.mps import MPS, mps_inner, mps_to_dense


def _random_mps(rng: Rng, length: int, phys_dim: int, bond_dim: int) -> MPS:
    cores = []
    r_prev = 1
    for k in range(length):
        r_next = 1 if k == length - 1 else bond_dim
        cores.append(rng.normal((r_prev, phys_dim, r_next), dtype=tf.float64))
        r_prev = r_next
    return MPS(cores=cores)


def test_mps_inner_matches_dense_inner() -> None:
    rng = Rng(seed=0)
    a = _random_mps(rng, length=5, phys_dim=2, bond_dim=3)
    b = _random_mps(rng, length=5, phys_dim=2, bond_dim=3)

    inner_mps = mps_inner(a, b)

    da = tf.reshape(mps_to_dense(a), (-1,))
    db = tf.reshape(mps_to_dense(b), (-1,))
    inner_dense = tf.reduce_sum(da * db)

    rel = tf.abs(inner_mps - inner_dense) / (tf.abs(inner_dense) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-8
