"""
Core typing and tensor conventions.

TensorFlow is the default tensor backend. NumPy exists as glue for SciPy/SymPy
and small reference calculations, but library-facing APIs should primarily use
TensorFlow tensors.
"""

from typing import Any, Protocol, Sequence, TypeAlias, Union

import numpy as np
import tensorflow as tf

__all__ = [
    "DTypeLike",
    "ShapeLike",
    "TensorLike",
    "to_tensor",
    "as_float64",
    "as_complex128",
    "tf_float",
    "tf_complex",
]


ShapeLike: TypeAlias = Sequence[int]
DTypeLike: TypeAlias = Union[tf.DType, np.dtype, str]

TensorLike: TypeAlias = Union[tf.Tensor, tf.Variable]


tf_float: tf.DType = tf.float64
tf_complex: tf.DType = tf.complex128


def to_tensor(x: Any, dtype: DTypeLike | None = None) -> tf.Tensor:
    """
    Convert input to a TensorFlow tensor (default dtype float64 unless already typed).
    """
    if dtype is None:
        if isinstance(x, (tf.Tensor, tf.Variable)):
            return tf.convert_to_tensor(x)
        return tf.convert_to_tensor(x, dtype=tf_float)
    return tf.convert_to_tensor(x, dtype=dtype)


def as_float64(x: Any) -> tf.Tensor:
    """
    Cast/convert to a float64 TF tensor.
    """
    return tf.cast(to_tensor(x), tf_float)


def as_complex128(x: Any) -> tf.Tensor:
    """
    Cast/convert to a complex128 TF tensor.
    """
    return tf.cast(to_tensor(x), tf_complex)


class CallableTensorMap(Protocol):
    """
    Protocol for a map that acts on a tensor and returns a tensor.
    """

    def __call__(self, x: tf.Tensor) -> tf.Tensor: ...
