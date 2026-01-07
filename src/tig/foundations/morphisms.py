"""
Morphisms between tensor spaces.

A morphism is represented as a callable with explicit domain/codomain.
We keep the interface small and mathematical.
"""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from tig.foundations.spaces import TensorSpace
from tig.core.types import TensorLike, as_float64

__all__ = ["LinearMap"]


@dataclass(frozen=True)
class LinearMap:
    """
    Linear map T : X -> Y between tensor spaces, with an optional adjoint.

    If the adjoint is provided, it should satisfy:
    <T x, y>_Y = <x, T* y>_X.
    """

    domain: TensorSpace
    codomain: TensorSpace
    apply: Callable[[tf.Tensor], tf.Tensor]
    adjoint: Callable[[tf.Tensor], tf.Tensor] | None = None

    def __call__(self, x: TensorLike) -> tf.Tensor:
        xx = self.domain.element(x)
        yy = self.apply(xx)
        return self.codomain.element(yy)

    def star(self, y: TensorLike) -> tf.Tensor:
        if self.adjoint is None:
            raise ValueError("Adjoint not available for this LinearMap.")
        yy = self.codomain.element(y)
        xx = self.adjoint(yy)
        return self.domain.element(xx)
