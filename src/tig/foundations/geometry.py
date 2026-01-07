"""
Geometric structure on tensor spaces.

This is a foundations-level module: it defines the minimal objects used later
for manifold optimization (Stiefel/Grassmann/fixed-rank modules).

Here we provide:
- a Euclidean metric induced by the standard inner product
- a projection operator template (used by constrained geometries)
"""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from tig.core.norms import inner
from tig.core.types import TensorLike, as_float64

__all__ = ["EuclideanMetric", "Projector"]


@dataclass(frozen=True)
class EuclideanMetric:
    """
    Euclidean metric g_x(u, v) = <u, v>.
    """

    def __call__(self, u: TensorLike, v: TensorLike) -> tf.Tensor:
        return inner(as_float64(u), as_float64(v))


@dataclass(frozen=True)
class Projector:
    """
    Projection operator P: ambient -> tangent (or feasible) subspace.
    """

    apply: Callable[[tf.Tensor], tf.Tensor]

    def __call__(self, u: TensorLike) -> tf.Tensor:
        return self.apply(as_float64(u))
