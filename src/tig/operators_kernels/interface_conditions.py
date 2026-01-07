"""
Interface conditions as abstract maps (minimal).

This module frames interface conditions (e.g., continuity/jumps) as linear maps
acting on traces of fields at interfaces.
"""

from dataclasses import dataclass
from typing import Callable, Protocol

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["InterfaceMap", "CallableInterfaceMap"]


class InterfaceMap(Protocol):
    """
    Interface map acting on a stacked interface state vector.
    """

    def __call__(self, u: TensorLike) -> tf.Tensor: ...


@dataclass(frozen=True)
class CallableInterfaceMap:
    """
    Interface map defined by a callable.
    """

    f: Callable[[tf.Tensor], tf.Tensor]

    def __call__(self, u: TensorLike) -> tf.Tensor:
        return as_float64(self.f(as_float64(u)))
