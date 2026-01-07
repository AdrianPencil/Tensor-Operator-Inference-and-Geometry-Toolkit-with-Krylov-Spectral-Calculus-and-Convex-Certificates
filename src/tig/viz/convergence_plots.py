"""
Convergence diagnostics (minimal).

Provides a small container for iteration traces and helpers to normalize them.
"""

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf

__all__ = ["Trace", "normalize_trace"]


@dataclass(frozen=True)
class Trace:
    """
    Iteration trace.
    """

    values: tf.Tensor

    def as_float_list(self) -> list[float]:
        return [float(x) for x in self.values.numpy().tolist()]


def normalize_trace(values: Sequence[float], eps: float = 1e-15) -> Trace:
    v = tf.convert_to_tensor(list(values), dtype=tf.float64)
    v0 = tf.maximum(tf.abs(v[0]), tf.cast(float(eps), tf.float64))
    return Trace(values=v / v0)
