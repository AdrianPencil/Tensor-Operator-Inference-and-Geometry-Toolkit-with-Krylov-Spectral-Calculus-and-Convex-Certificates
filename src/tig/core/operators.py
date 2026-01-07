"""
Operator abstractions (math-first).

The project treats many objects as operators between finite-dimensional tensor spaces.
We keep the interface minimal: matvec + adjoint-matvec (rmatvec).
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = [
    "LinearOperator",
    "ExplicitLinearOperator",
    "AdjointOperator",
    "ComposeOperator",
]


class LinearOperator(Protocol):
    """
    Minimal linear operator interface.
    """

    def matvec(self, x: tf.Tensor) -> tf.Tensor: ...
    def rmatvec(self, y: tf.Tensor) -> tf.Tensor: ...


@dataclass(frozen=True)
class ExplicitLinearOperator:
    """
    Explicit operator defined by callables.

    This supports matrix-free implementations where A is not materialized,
    but the action x -> A x and y -> A* y are available.
    """

    matvec_fn: Callable[[tf.Tensor], tf.Tensor]
    rmatvec_fn: Callable[[tf.Tensor], tf.Tensor]

    def matvec(self, x: TensorLike) -> tf.Tensor:
        return self.matvec_fn(as_float64(x))

    def rmatvec(self, y: TensorLike) -> tf.Tensor:
        return self.rmatvec_fn(as_float64(y))


@dataclass(frozen=True)
class AdjointOperator:
    """
    Operator wrapper that swaps matvec and rmatvec (A -> A*).
    """

    op: LinearOperator

    def matvec(self, x: TensorLike) -> tf.Tensor:
        return self.op.rmatvec(as_float64(x))

    def rmatvec(self, y: TensorLike) -> tf.Tensor:
        return self.op.matvec(as_float64(y))


@dataclass(frozen=True)
class ComposeOperator:
    """
    Composition B ∘ A (apply A then B).
    """

    b: LinearOperator
    a: LinearOperator

    def matvec(self, x: TensorLike) -> tf.Tensor:
        return self.b.matvec(self.a.matvec(as_float64(x)))

    def rmatvec(self, y: TensorLike) -> tf.Tensor:
        return self.a.rmatvec(self.b.rmatvec(as_float64(y)))
