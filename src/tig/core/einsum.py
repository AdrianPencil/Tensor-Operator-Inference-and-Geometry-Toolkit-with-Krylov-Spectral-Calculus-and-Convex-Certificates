"""
Einstein summation utilities.

TensorFlow is the default backend:
- tf.einsum is the primary contraction primitive
- expressions are explicit and math-readable

Higher-level contraction planning lives elsewhere (tensor networks module).
"""

from typing import Any

import tensorflow as tf

from tig.core.types import TensorLike, to_tensor

__all__ = ["einsum"]


def einsum(expr: str, *tensors: TensorLike) -> tf.Tensor:
    """
    Tensor contraction using Einstein summation with a TF backend.

    Parameters
    ----------
    expr:
        Einstein summation expression, e.g. "ij,jk->ik".
    tensors:
        Input tensors (converted to TF tensors if needed).

    Returns
    -------
    tf.Tensor
        Result of the contraction.
    """
    ts = [to_tensor(t) for t in tensors]
    return tf.einsum(expr, *ts)
