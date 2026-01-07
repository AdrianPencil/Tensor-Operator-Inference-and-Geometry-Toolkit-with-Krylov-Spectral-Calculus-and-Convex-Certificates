"""
Tensor spaces and basic structure.

This module is intentionally minimal:
- a TensorSpace is a finite-dimensional real inner-product space represented by
  a shape and dtype, with an explicit inner product.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import tensorflow as tf

from tig.core.norms import inner
from tig.core.types import ShapeLike, TensorLike, as_float64, tf_float

__all__ = ["TensorSpace"]


@dataclass(frozen=True)
class TensorSpace:
    """
    Finite-dimensional tensor space with a fixed shape and an inner product.
    """

    shape: ShapeLike
    dtype: tf.DType = tf_float

    def element(self, x: TensorLike) -> tf.Tensor:
        t = tf.cast(tf.convert_to_tensor(x), self.dtype)
        return tf.reshape(t, tuple(self.shape))

    def ip(self, x: TensorLike, y: TensorLike) -> tf.Tensor:
        """
        Euclidean inner product on this space.
        """
        xx = self.element(x)
        yy = self.element(y)
        return inner(xx, yy)

    def zeros(self) -> tf.Tensor:
        return tf.zeros(tuple(self.shape), dtype=self.dtype)

    def ones(self) -> tf.Tensor:
        return tf.ones(tuple(self.shape), dtype=self.dtype)
