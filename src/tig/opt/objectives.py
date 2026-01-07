"""
Objective function abstractions (math-first).

An objective is a scalar functional φ(x). This module provides a small TF-first
interface for evaluation and gradients, keeping the mathematical meaning explicit
while allowing different backends to serve as optional oracles elsewhere.
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import tensorflow as tf

from tig.core.autodiff import grad as tf_grad
from tig.core.types import TensorLike, as_float64

__all__ = ["Objective", "ScalarObjective", "SumObjective"]


class Objective(Protocol):
    """
    Scalar functional φ: X -> R with optional gradient access.
    """

    def value(self, x: TensorLike) -> tf.Tensor: ...
    def gradient(self, x: TensorLike) -> tf.Tensor: ...


@dataclass(frozen=True)
class ScalarObjective:
    """
    Scalar objective defined by a value function and optional gradient function.

    If grad_fn is not provided, it is computed using TensorFlow autodiff.
    """

    value_fn: Callable[[tf.Tensor], tf.Tensor]
    grad_fn: Callable[[tf.Tensor], tf.Tensor] | None = None

    def value(self, x: TensorLike) -> tf.Tensor:
        xx = as_float64(x)
        y = self.value_fn(xx)
        return tf.reshape(as_float64(y), ())

    def gradient(self, x: TensorLike) -> tf.Tensor:
        xx = as_float64(x)
        if self.grad_fn is not None:
            return as_float64(self.grad_fn(xx))
        return as_float64(tf_grad(self.value_fn, xx))


@dataclass(frozen=True)
class SumObjective:
    """
    Sum objective: φ(x) = φ1(x) + φ2(x) (+ ...).

    Gradients add linearly.
    """

    terms: tuple[Objective, ...]

    def value(self, x: TensorLike) -> tf.Tensor:
        vals = [t.value(x) for t in self.terms]
        return tf.add_n([tf.reshape(as_float64(v), ()) for v in vals])

    def gradient(self, x: TensorLike) -> tf.Tensor:
        grads = [t.gradient(x) for t in self.terms]
        return tf.add_n([as_float64(g) for g in grads])
