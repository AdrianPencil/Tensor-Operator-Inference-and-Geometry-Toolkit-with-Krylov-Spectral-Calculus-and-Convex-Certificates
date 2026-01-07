"""
Autodiff JVP/VJP sanity.

Checks consistency between JVP and VJP using the adjoint identity:
<u, Jv> = <J^T u, v>.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.inverse.forward_models import CallableForwardModel


def test_jvp_vjp_adjoint_identity() -> None:
    rng = Rng(seed=0)

    def f(x: tf.Tensor) -> tf.Tensor:
        return tf.math.sin(x) + 0.1 * (x * x)

    model = CallableForwardModel(f=f)

    x = rng.normal((32,), dtype=tf.float64)
    v = rng.normal((32,), dtype=tf.float64)
    u = rng.normal((32,), dtype=tf.float64)

    jv = model.jvp(x, v)
    jtu = model.vjp(x, u)

    left = tf.reduce_sum(u * jv)
    right = tf.reduce_sum(jtu * v)
    gap = tf.abs(left - right)
    denom = tf.abs(left) + tf.abs(right) + tf.cast(1e-15, tf.float64)
    assert float((gap / denom).numpy()) < 1e-10
