"""
Toeplitz-structured operators.

A Toeplitz matrix T is determined by its first column c and first row r, with c[0]=r[0].
Matrix-vector products can be performed via FFT using embedding into a circulant.

This module provides a TF-first ToeplitzOperator with matvec and rmatvec.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from tig.core.operators import LinearOperator
from tig.core.types import TensorLike, as_float64

__all__ = ["ToeplitzOperator"]


def _toeplitz_embed_fft_kernel(c: tf.Tensor, r: tf.Tensor) -> tf.Tensor:
    n = int(c.shape[0])
    tail = tf.reverse(r[1:], axis=[0])
    g = tf.concat([c, tf.zeros((1,), dtype=c.dtype), tail], axis=0)
    return tf.signal.fft(tf.cast(g, tf.complex128))


@dataclass(frozen=True)
class ToeplitzOperator:
    """
    Toeplitz operator T defined by first column c and first row r.

    Uses FFT-based matvec with O(n log n) complexity.
    """

    c: tf.Tensor
    r: tf.Tensor
    _kernel_fft: Optional[tf.Tensor] = None

    def __post_init__(self) -> None:
        cc = as_float64(self.c)
        rr = as_float64(self.r)
        if int(cc.shape[0]) != int(rr.shape[0]):
            raise ValueError("ToeplitzOperator requires c and r of equal length.")
        if float(cc[0].numpy()) != float(rr[0].numpy()):
            raise ValueError("ToeplitzOperator requires c[0] == r[0].")

        if self._kernel_fft is None:
            object.__setattr__(self, "_kernel_fft", _toeplitz_embed_fft_kernel(cc, rr))

        object.__setattr__(self, "c", cc)
        object.__setattr__(self, "r", rr)

    def matvec(self, x: TensorLike) -> tf.Tensor:
        xx = tf.reshape(as_float64(x), (-1,))
        n = int(self.c.shape[0])
        if int(xx.shape[0]) != n:
            raise ValueError("x has incompatible shape for Toeplitz matvec.")

        x_pad = tf.concat([xx, tf.zeros((n,), dtype=tf.float64)], axis=0)
        xf = tf.signal.fft(tf.cast(x_pad, tf.complex128))
        yf = xf * tf.cast(self._kernel_fft, tf.complex128)
        y_pad = tf.signal.ifft(yf)
        y = tf.math.real(y_pad[:n])
        return y

    def rmatvec(self, y: TensorLike) -> tf.Tensor:
        """
        Adjoint for real Toeplitz: T* corresponds to Toeplitz with swapped (c,r) and reversed tails.
        """
        cc = tf.concat([self.r[:1], tf.reverse(self.c[1:], axis=[0])], axis=0)
        rr = tf.concat([self.c[:1], tf.reverse(self.r[1:], axis=[0])], axis=0)
        return ToeplitzOperator(c=cc, r=rr).matvec(y)
