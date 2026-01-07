"""
Preconditioners (minimal, math-first).

A preconditioner represents an approximation to A^{-1} applied as a map z = M^{-1} r.
This module keeps a small interface that composes naturally with Krylov methods.
"""

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["Preconditioner", "IdentityPreconditioner", "DiagonalPreconditioner"]


class Preconditioner(Protocol):
    """
    Preconditioner interface: apply returns M^{-1} r.
    """

    def apply(self, r: tf.Tensor) -> tf.Tensor: ...


@dataclass(frozen=True)
class IdentityPreconditioner:
    """
    M^{-1} = I.
    """

    def apply(self, r: TensorLike) -> tf.Tensor:
        return as_float64(r)


@dataclass(frozen=True)
class DiagonalPreconditioner:
    """
    Diagonal preconditioner: M^{-1} r = r / diag.

    diag should be broadcastable to r.
    """

    diag: tf.Tensor

    def apply(self, r: TensorLike) -> tf.Tensor:
        rr = as_float64(r)
        dd = as_float64(self.diag)
        return rr / dd
