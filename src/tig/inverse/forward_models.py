"""
Forward models for inverse problems (math-first).

A forward model is a map F: X -> Y. This module provides a minimal TF-first
abstraction plus default JVP/VJP constructions via autodiff, so downstream
inverse routines can treat the Jacobian as an operator without materializing it.
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import tensorflow as tf

from tig.core.autodiff import jvp as tf_jvp
from tig.core.autodiff import vjp as tf_vjp
from tig.core.types import TensorLike, as_float64

__all__ = ["ForwardModel", "CallableForwardModel"]


class ForwardModel(Protocol):
    """
    Minimal forward model interface.

    The Jacobian is accessed through JVP/VJP, which supports matrix-free solvers.
    """

    def __call__(self, x: TensorLike) -> tf.Tensor: ...
    def jvp(self, x: TensorLike, v: TensorLike) -> tf.Tensor: ...
    def vjp(self, x: TensorLike, u: TensorLike) -> tf.Tensor: ...


@dataclass(frozen=True)
class CallableForwardModel:
    """
    Forward model defined by a callable F.

    JVP/VJP are computed by TensorFlow autodiff by default.
    """

    f: Callable[[tf.Tensor], tf.Tensor]

    def __call__(self, x: TensorLike) -> tf.Tensor:
        return as_float64(self.f(as_float64(x)))

    def jvp(self, x: TensorLike, v: TensorLike) -> tf.Tensor:
        _, jv = tf_jvp(self.f, as_float64(x), as_float64(v))
        return as_float64(jv)

    def vjp(self, x: TensorLike, u: TensorLike) -> tf.Tensor:
        _, jt_u = tf_vjp(self.f, as_float64(x), as_float64(u))
        return as_float64(jt_u)
