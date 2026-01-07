"""
Tensor slicing helpers (minimal).

Provides canonical slices for visualization/inspection:
- fixed index slicing
- central slice extraction
"""

from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, to_tensor

__all__ = ["central_slice", "slice_along_axis"]


def central_slice(x: TensorLike, axis: int = 0) -> tf.Tensor:
    """
    Return the central slice along a given axis.
    """
    xx = to_tensor(x)
    ax = int(axis)
    n = int(xx.shape[ax])
    idx = n // 2
    return slice_along_axis(xx, axis=ax, index=idx)


def slice_along_axis(x: TensorLike, axis: int, index: int) -> tf.Tensor:
    """
    Slice tensor x at x[..., index, ...] along a given axis.
    """
    xx = to_tensor(x)
    ax = int(axis)
    idx = int(index)
    if idx < 0 or idx >= int(xx.shape[ax]):
        raise ValueError("index out of range for the chosen axis.")
    begin = [0] * len(xx.shape)
    size = [int(s) for s in xx.shape]
    begin[ax] = idx
    size[ax] = 1
    sl = tf.slice(xx, begin=begin, size=size)
    return tf.squeeze(sl, axis=ax)
