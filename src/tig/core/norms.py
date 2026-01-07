"""
Norms and inner products.

This file keeps the mathematical interface explicit:
- inner products
- vector/matrix/tensor norms (TF-native)

Where a norm has multiple conventions, the convention is spelled out in the docstring.
"""

from typing import Literal

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["inner", "fro_norm", "l2_norm", "op_norm_2"]


def inner(x: TensorLike, y: TensorLike) -> tf.Tensor:
    """
    Euclidean inner product <x, y> = sum_i x_i y_i for real tensors.
    """
    xx = as_float64(x)
    yy = as_float64(y)
    return tf.reduce_sum(xx * yy)


def fro_norm(x: TensorLike) -> tf.Tensor:
    """
    Frobenius norm for any tensor: ||x||_F = sqrt(sum_i x_i^2).
    """
    xx = as_float64(x)
    return tf.linalg.norm(xx)


def l2_norm(x: TensorLike) -> tf.Tensor:
    """
    Vector 2-norm. For non-1D tensors, this is the Frobenius norm.
    """
    return fro_norm(x)


def op_norm_2(a: TensorLike) -> tf.Tensor:
    """
    Matrix operator 2-norm (spectral norm): ||A||_2 = sigma_max(A).
    """
    aa = as_float64(a)
    s = tf.linalg.svd(aa, compute_uv=False)
    return tf.reduce_max(s)
