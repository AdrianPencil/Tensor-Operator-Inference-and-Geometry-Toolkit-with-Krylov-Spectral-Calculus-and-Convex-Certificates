"""
Boundary constraints as operator constraints (minimal).

A boundary constraint is represented as:
C u = 0  (homogeneous) or C u = d (inhomogeneous)

This module provides a compact representation and a residual function.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["BoundaryConstraint", "boundary_residual"]


@dataclass(frozen=True)
class BoundaryConstraint:
    """
    Linear boundary constraint C u = d.
    """

    c: tf.Tensor
    d: tf.Tensor | None = None

    def residual(self, u: TensorLike) -> tf.Tensor:
        uu = tf.reshape(as_float64(u), (-1, 1))
        cc = as_float64(self.c)
        lhs = (cc @ uu)[:, 0]
        if self.d is None:
            return lhs
        return lhs - tf.reshape(as_float64(self.d), (-1,))


def boundary_residual(constraint: BoundaryConstraint, u: TensorLike) -> tf.Tensor:
    return constraint.residual(u)
