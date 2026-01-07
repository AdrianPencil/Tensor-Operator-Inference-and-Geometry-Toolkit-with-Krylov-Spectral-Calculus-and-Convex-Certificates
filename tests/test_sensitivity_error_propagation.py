"""
Sensitivity and error propagation sanity for a linear forward model.

For y = A x:
J = A
Cov_y = A Cov_x A^T
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.inverse.forward_models import CallableForwardModel
from tig.uq.error_propagation import pushforward_cov_dense
from tig.uq.sensitivity import explicit_jacobian_small


def test_linear_jacobian_and_cov_pushforward() -> None:
    rng = Rng(seed=0)
    a = rng.normal((12, 7), dtype=tf.float64)

    def f(x: tf.Tensor) -> tf.Tensor:
        return (a @ tf.reshape(x, (-1, 1)))[:, 0]

    model = CallableForwardModel(f=f)
    x = rng.normal((7,), dtype=tf.float64)

    j = explicit_jacobian_small(model, x)
    rel_j = tf.linalg.norm(j - a) / (tf.linalg.norm(a) + tf.cast(1e-15, tf.float64))
    assert float(rel_j.numpy()) < 1e-12

    m = rng.normal((7,), dtype=tf.float64)
    cov_x = tf.eye(7, dtype=tf.float64) * tf.cast(0.3, tf.float64)
    res = pushforward_cov_dense(model=model, x_mean=m, cov_x=cov_x)
    cov_ref = a @ cov_x @ tf.transpose(a)

    rel = tf.linalg.norm(res.cov_y - cov_ref) / (tf.linalg.norm(cov_ref) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-10
