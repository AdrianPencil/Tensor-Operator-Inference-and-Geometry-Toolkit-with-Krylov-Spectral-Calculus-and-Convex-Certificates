"""
Completely positive (CP) maps (minimal Kraus form).

A CP map Φ acts on matrices X by:
Φ(X) = sum_k K_k X K_k^*

This module provides:
- apply_kraus for a list of Kraus operators
- trace-preserving check proxy: sum_k K_k^* K_k ≈ I
"""

from typing import Sequence

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128

__all__ = ["apply_kraus", "trace_preserving_residual"]


def apply_kraus(kraus_ops: Sequence[TensorLike], x: TensorLike) -> tf.Tensor:
    xx = as_complex128(x)
    acc = tf.zeros_like(xx)
    for k in kraus_ops:
        kk = as_complex128(k)
        acc = acc + kk @ xx @ tf.linalg.adjoint(kk)
    return acc


def trace_preserving_residual(kraus_ops: Sequence[TensorLike]) -> tf.Tensor:
    if len(kraus_ops) == 0:
        raise ValueError("At least one Kraus operator is required.")
    d = int(as_complex128(kraus_ops[0]).shape[0])
    acc = tf.zeros((d, d), dtype=tf.complex128)
    for k in kraus_ops:
        kk = as_complex128(k)
        acc = acc + tf.linalg.adjoint(kk) @ kk
    return acc - tf.eye(d, dtype=tf.complex128)
