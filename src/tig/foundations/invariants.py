"""
Invariants and sanity identities.

This module collects small, mathematically-motivated invariants used as
contracts in tests and examples: traces, similarity invariance, and basic
unitary/orthogonal invariances.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["trace", "similarity_invariant_trace"]


def trace(a: TensorLike) -> tf.Tensor:
    """
    Trace of a square matrix.
    """
    aa = as_float64(a)
    return tf.linalg.trace(aa)


def similarity_invariant_trace(a: TensorLike, s: TensorLike) -> tf.Tensor:
    """
    Check trace(S^{-1} A S) equals trace(A) for invertible S.

    Returns the difference trace(S^{-1} A S) - trace(A).
    """
    aa = as_float64(a)
    ss = as_float64(s)
    s_inv = tf.linalg.inv(ss)
    conj = s_inv @ aa @ ss
    return trace(conj) - trace(aa)
