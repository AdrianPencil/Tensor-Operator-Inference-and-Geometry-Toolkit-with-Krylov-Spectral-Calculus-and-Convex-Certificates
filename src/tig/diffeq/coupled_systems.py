"""
Coupled systems and stability diagnostics (minimal).

Provides a compact block-operator constructor for coupled linear systems:
[ x' ] = [A  B][x] + [f]
[ y' ]   [C  D][y]   [g]

This supports multiscale/stability demonstrations without adding solver bloat.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["block_operator_2x2"]


def block_operator_2x2(a: TensorLike, b: TensorLike, c: TensorLike, d: TensorLike) -> tf.Tensor:
    """
    Construct the dense block matrix [[A,B],[C,D]].
    """
    aa = as_float64(a)
    bb = as_float64(b)
    cc = as_float64(c)
    dd = as_float64(d)

    top = tf.concat([aa, bb], axis=1)
    bot = tf.concat([cc, dd], axis=1)
    return tf.concat([top, bot], axis=0)
