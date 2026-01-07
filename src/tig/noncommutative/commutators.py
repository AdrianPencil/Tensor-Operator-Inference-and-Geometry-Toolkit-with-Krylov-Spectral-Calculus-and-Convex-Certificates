"""
Commutators and basic noncommutative identities (minimal).

For operators A, B:
[A, B] = AB - BA
{A, B} = AB + BA (anticommutator)

These are used across noncommutative inequalities and superoperator models.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["commutator", "anticommutator"]


def commutator(a: TensorLike, b: TensorLike) -> tf.Tensor:
    aa = as_float64(a)
    bb = as_float64(b)
    return aa @ bb - bb @ aa


def anticommutator(a: TensorLike, b: TensorLike) -> tf.Tensor:
    aa = as_float64(a)
    bb = as_float64(b)
    return aa @ bb + bb @ aa
