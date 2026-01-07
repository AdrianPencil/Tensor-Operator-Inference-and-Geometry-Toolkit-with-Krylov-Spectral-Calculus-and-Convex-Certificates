"""
Contractions and tensor-product structure.

This module provides a small set of contraction primitives framed as maps.
Higher-level graph-based contraction planning is handled in tensor_networks/.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import tensorflow as tf

from tig.core.einsum import einsum
from tig.core.types import TensorLike, to_tensor

__all__ = ["Contraction"]


@dataclass(frozen=True)
class Contraction:
    """
    A contraction specified by an einsum expression.

    Example
    -------
    expr = "ij,jk->ik" defines matrix multiplication.
    """

    expr: str

    def __call__(self, *tensors: TensorLike) -> tf.Tensor:
        ts = [to_tensor(t) for t in tensors]
        return einsum(self.expr, *ts)
