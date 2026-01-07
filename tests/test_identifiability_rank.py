"""
Identifiability and rank-facing sanity.

Checks Jacobian Gram operator matches A^T A for a linear forward model.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.inverse.forward_models import CallableForwardModel
from tig.inverse.identifiability import jacobian_gram_operator


def test_jacobian_gram_matches_linear_case() -> None:
    rng = Rng(seed=0)
    a = rng.normal((40, 20), dtype=tf.float64)

    def f(x: tf.Tensor) -> tf.Tensor:
        return (a @ tf.reshape(x, (-1, 1)))[:, 0]

    model = CallableForwardModel(f=f)
    x0 = rng.normal((20,), dtype=tf.float64)
    v = rng.normal((20,), dtype=tf.float64)

    gop = jacobian_gram_operator(model, x0)
    left = gop.matvec(v)

    right = tf.transpose(a) @ tf.reshape((a @ tf.reshape(v, (-1, 1)))[:, 0], (-1, 1))
    right = right[:, 0]

    rel = tf.linalg.norm(left - right) / (tf.linalg.norm(right) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-10
