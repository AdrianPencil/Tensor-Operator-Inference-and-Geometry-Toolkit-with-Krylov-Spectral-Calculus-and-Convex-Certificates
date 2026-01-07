"""
Boundary constraint consistency.

Checks residual is zero when u satisfies C u = d.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.operators_kernels.boundary_constraints import BoundaryConstraint


def test_boundary_constraint_residual_zero() -> None:
    rng = Rng(seed=0)
    c = rng.normal((3, 5), dtype=tf.float64)
    u = rng.normal((5,), dtype=tf.float64)
    d = (c @ tf.reshape(u, (-1, 1)))[:, 0]

    bc = BoundaryConstraint(c=c, d=d)
    r = bc.residual(u)
    assert float(tf.linalg.norm(tf.reshape(r, (-1,))).numpy()) < 1e-12
