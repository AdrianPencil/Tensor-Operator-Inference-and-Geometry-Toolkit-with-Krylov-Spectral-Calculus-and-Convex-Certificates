"""
Proximal operator sanity.

Checks L1 soft-thresholding and nuclear norm shrinkage behavior.
"""

import tensorflow as tf

from tig.inverse.regularizers import L1Regularizer
from tig.convex.nuclear_norm import prox_nuclear_norm
from tig.core.random import Rng


def test_l1_prox_soft_threshold() -> None:
    reg = L1Regularizer()
    v = tf.constant([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=tf.float64)
    out = reg.prox(v, lam=1.0)
    expected = tf.constant([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=tf.float64)
    err = tf.linalg.norm(out - expected)
    assert float(err.numpy()) == 0.0


def test_nuclear_prox_shrinks_singular_values() -> None:
    rng = Rng(seed=0)
    a = rng.normal((10, 8), dtype=tf.float64)
    lam = 0.2
    out = prox_nuclear_norm(a, lam=lam)

    s_in = tf.linalg.svd(a, compute_uv=False)
    s_out = tf.linalg.svd(out, compute_uv=False)

    diff = s_out - tf.maximum(s_in - tf.cast(lam, tf.float64), 0.0)
    assert float(tf.linalg.norm(diff).numpy()) < 1e-8
