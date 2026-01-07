"""
Integro-differential equation skeletons (minimal).

A generic form:
d/dt x(t) = f(x(t), t) + ∫_0^t K(t, s) x(s) ds

This module provides a simple discrete memory term accumulator for kernels
K(t_k, t_j) on a uniform grid.
"""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["MemoryKernel", "memory_term"]


@dataclass(frozen=True)
class MemoryKernel:
    """
    Discrete memory kernel K(k, j) evaluated on a time grid.
    """

    k: Callable[[int, int], tf.Tensor]


def memory_term(kernel: MemoryKernel, history: TensorLike, dt: float, k: int) -> tf.Tensor:
    """
    Compute ∑_{j=0}^{k-1} K(k, j) x_j dt for history x_j.

    history has shape (k, *state_shape), returns tensor with shape state_shape.
    """
    hh = as_float64(history)
    dd = float(dt)
    if int(hh.shape[0]) != int(k):
        raise ValueError("history length must equal k.")

    acc = tf.zeros_like(hh[0])
    for j in range(int(k)):
        acc = acc + as_float64(kernel.k(int(k), int(j))) * hh[j]
    return tf.cast(dd, tf.float64) * acc
