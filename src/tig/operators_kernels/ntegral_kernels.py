"""
Integral kernels as operators (minimal, math-first).

A kernel K defines an operator (K f)(x) = ∫ K(x, y) f(y) dy.
In discrete settings, this becomes matrix multiplication.

This module provides a small discrete kernel operator abstraction:
- apply to a discretized vector f via K @ f
- composition via matrix product
"""

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["KernelOperator", "DenseKernelOperator"]


class KernelOperator(Protocol):
    """
    Kernel operator interface.
    """

    def apply(self, f: TensorLike) -> tf.Tensor: ...
    def compose(self, other: "KernelOperator") -> "KernelOperator": ...


@dataclass(frozen=True)
class DenseKernelOperator:
    """
    Dense discrete kernel operator with matrix K.
    """

    k: tf.Tensor

    def apply(self, f: TensorLike) -> tf.Tensor:
        kk = as_float64(self.k)
        ff = tf.reshape(as_float64(f), (-1, 1))
        return (kk @ ff)[:, 0]

    def compose(self, other: "DenseKernelOperator") -> "DenseKernelOperator":
        kk = as_float64(self.k)
        oo = as_float64(other.k)
        return DenseKernelOperator(k=kk @ oo)
