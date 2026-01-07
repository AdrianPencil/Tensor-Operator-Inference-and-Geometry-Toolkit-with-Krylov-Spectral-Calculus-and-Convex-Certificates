"""
Krylov convergence sanity.

Solves an SPD system using CG and checks residual reduction.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.linalg.krylov import cg


class _Op:
    def __init__(self, a: tf.Tensor) -> None:
        self._a = a

    def matvec(self, x: tf.Tensor) -> tf.Tensor:
        return self._a @ tf.reshape(x, (-1, 1))[:, 0]

    def rmatvec(self, y: tf.Tensor) -> tf.Tensor:
        return self.matvec(y)


def test_cg_solves_spd() -> None:
    rng = Rng(seed=0)
    m = rng.normal((64, 64), dtype=tf.float64)
    a = tf.transpose(m) @ m + 1e-3 * tf.eye(64, dtype=tf.float64)

    x_true = rng.normal((64,), dtype=tf.float64)
    b = a @ tf.reshape(x_true, (-1, 1))[:, 0]

    res = cg(op=_Op(a), b=b, x0=tf.zeros_like(b), tol=1e-10, max_iter=300)
    r = b - (_Op(a).matvec(res.x))
    rel = tf.linalg.norm(r) / (tf.linalg.norm(b) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-8
