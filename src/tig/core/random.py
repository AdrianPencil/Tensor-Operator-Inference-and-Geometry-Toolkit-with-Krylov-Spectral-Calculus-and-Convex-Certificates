"""
Randomness utilities with controlled seeding.

TensorFlow is the default RNG source. The generator object is used so that
experiments can be reproducible without relying on global state.
"""

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf

from tig.core.types import ShapeLike, tf_float

__all__ = ["Rng"]


@dataclass
class Rng:
    """
    Reproducible random number generator wrapper.
    """

    seed: int

    def __post_init__(self) -> None:
        self._gen = tf.random.Generator.from_seed(int(self.seed))

    def normal(self, shape: ShapeLike, stddev: float = 1.0, dtype: tf.DType = tf_float) -> tf.Tensor:
        return self._gen.normal(shape=tuple(shape), stddev=stddev, dtype=dtype)

    def uniform(
        self,
        shape: ShapeLike,
        minval: float = 0.0,
        maxval: float = 1.0,
        dtype: tf.DType = tf_float,
    ) -> tf.Tensor:
        return self._gen.uniform(shape=tuple(shape), minval=minval, maxval=maxval, dtype=dtype)
