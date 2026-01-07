"""
Superoperators (minimal).

A superoperator acts on operators. A canonical example is:
L_A(X) = A X,   R_A(X) = X A

Vectorization identity (column-stacking):
vec(A X B^T) = (B ⊗ A) vec(X)

This module provides:
- left/right multiplication maps
- Liouvillian commutator superoperator: 𝓛_H(X) = -i [H, X] (complex form)
"""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128, as_float64

__all__ = ["left_mul", "right_mul", "Liouvillian"]


def left_mul(a: TensorLike) -> Callable[[tf.Tensor], tf.Tensor]:
    aa = as_float64(a)

    def op(x: tf.Tensor) -> tf.Tensor:
        return aa @ as_float64(x)

    return op


def right_mul(a: TensorLike) -> Callable[[tf.Tensor], tf.Tensor]:
    aa = as_float64(a)

    def op(x: tf.Tensor) -> tf.Tensor:
        return as_float64(x) @ aa

    return op


@dataclass(frozen=True)
class Liouvillian:
    """
    𝓛_H(X) = -i [H, X].
    """

    h: tf.Tensor

    def __call__(self, x: TensorLike) -> tf.Tensor:
        hh = as_complex128(self.h)
        xx = as_complex128(x)
        comm = hh @ xx - xx @ hh
        return -1j * comm
