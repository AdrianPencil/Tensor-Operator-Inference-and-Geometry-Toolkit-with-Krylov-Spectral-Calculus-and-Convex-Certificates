"""
Phase-diagram style parameter sweeps (minimal).

This module provides a generic 2D sweep helper that evaluates a scalar statistic
on a grid (a,b) -> s(a,b). The result is a dense matrix for plotting.
"""

from typing import Callable, Tuple

import tensorflow as tf

__all__ = ["sweep_2d"]


def sweep_2d(
    a_vals: tf.Tensor,
    b_vals: tf.Tensor,
    statistic: Callable[[float, float], float],
) -> tf.Tensor:
    aa = tf.cast(a_vals, tf.float64)
    bb = tf.cast(b_vals, tf.float64)

    out = []
    for i in range(int(aa.shape[0])):
        row = []
        for j in range(int(bb.shape[0])):
            row.append(float(statistic(float(aa[i].numpy()), float(bb[j].numpy()))))
        out.append(row)

    return tf.convert_to_tensor(out, dtype=tf.float64)
