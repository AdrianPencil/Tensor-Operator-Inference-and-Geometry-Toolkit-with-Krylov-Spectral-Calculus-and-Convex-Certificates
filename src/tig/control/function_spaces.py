"""
Function-space views for controls (minimal, math-first).

Controls are represented on a time grid as tensors u_k = u(t_k).
This module provides:
- TimeGrid
- ControlSignal (piecewise-constant interpretation)
- inner products on the grid (L2 with dt weight)
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["TimeGrid", "ControlSignal", "grid_inner_l2"]


@dataclass(frozen=True)
class TimeGrid:
    """
    Uniform time grid.
    """

    t0: float
    dt: float
    n_steps: int

    def times(self) -> tf.Tensor:
        t0 = tf.cast(float(self.t0), tf.float64)
        dt = tf.cast(float(self.dt), tf.float64)
        k = tf.range(0, int(self.n_steps) + 1, dtype=tf.float64)
        return t0 + k * dt


@dataclass(frozen=True)
class ControlSignal:
    """
    Control values on a grid, interpreted as piecewise-constant over intervals.

    values has shape (n_steps, *u_shape) for intervals [t_k, t_{k+1}).
    """

    grid: TimeGrid
    values: tf.Tensor

    def u(self, k: int) -> tf.Tensor:
        return as_float64(self.values[int(k)])


def grid_inner_l2(u: TensorLike, v: TensorLike, dt: float) -> tf.Tensor:
    """
    Discrete L2 inner product: <u, v> = sum_k u_k · v_k * dt.
    """
    uu = as_float64(u)
    vv = as_float64(v)
    dd = tf.cast(float(dt), tf.float64)
    return dd * tf.reduce_sum(uu * vv)
