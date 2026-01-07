"""
PDE operators (minimal).

This module provides discrete operator constructors for:
- 1D Laplacian (Dirichlet-like stencil)
- 1D advection (upwind)

These are building blocks for operator-theoretic PDE framing elsewhere.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import as_float64

__all__ = ["laplacian_1d", "advection_upwind_1d"]


def laplacian_1d(n: int, dx: float) -> tf.Tensor:
    """
    1D Laplacian with second-order stencil on an n-grid.
    """
    nn = int(n)
    dd = float(dx)
    main = -2.0 * tf.ones((nn,), dtype=tf.float64)
    off = tf.ones((nn - 1,), dtype=tf.float64)
    a = tf.linalg.diag(main) + tf.linalg.diag(off, k=1) + tf.linalg.diag(off, k=-1)
    return a / tf.cast(dd * dd, tf.float64)


def advection_upwind_1d(n: int, dx: float, vel: float) -> tf.Tensor:
    """
    1D upwind advection operator for velocity vel.
    """
    nn = int(n)
    dd = float(dx)
    v = float(vel)

    main = tf.zeros((nn,), dtype=tf.float64)
    sub = tf.zeros((nn - 1,), dtype=tf.float64)
    sup = tf.zeros((nn - 1,), dtype=tf.float64)

    if v >= 0.0:
        main = tf.ones((nn,), dtype=tf.float64)
        sub = -tf.ones((nn - 1,), dtype=tf.float64)
        a = tf.linalg.diag(main) + tf.linalg.diag(sub, k=-1)
    else:
        main = -tf.ones((nn,), dtype=tf.float64)
        sup = tf.ones((nn - 1,), dtype=tf.float64)
        a = tf.linalg.diag(main) + tf.linalg.diag(sup, k=1)

    return (tf.cast(v, tf.float64) / tf.cast(dd, tf.float64)) * a
