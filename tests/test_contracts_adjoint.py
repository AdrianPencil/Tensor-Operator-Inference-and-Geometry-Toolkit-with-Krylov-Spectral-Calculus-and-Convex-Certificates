"""
Adjoint pairing contracts.

Verifies <Ax, y> == <x, A* y> for a simple explicit operator using randomized probes.
"""

import tensorflow as tf

from tig.compute.benchmarking_contracts import assert_adjoint_pairing
from tig.core.operators import ExplicitLinearOperator
from tig.core.random import Rng


def test_explicit_operator_adjoint_pairing() -> None:
    rng = Rng(seed=0)
    a = rng.normal((16, 16), dtype=tf.float64)

    def mv(x: tf.Tensor) -> tf.Tensor:
        return a @ tf.reshape(x, (-1, 1))[:, 0]

    def rmv(y: tf.Tensor) -> tf.Tensor:
        return tf.transpose(a) @ tf.reshape(y, (-1, 1))[:, 0]

    op = ExplicitLinearOperator(matvec_fn=mv, rmatvec_fn=rmv)
    assert_adjoint_pairing(op=op, shape_x=(16,), shape_y=(16,), n_probe=5, tol=1e-10, rng=rng)
