"""
Transfer functions and identification hooks (minimal).

A transfer function H(ω) maps input spectrum to output spectrum.
This module provides:
- frequency response for a linear time-invariant (LTI) state-space model:
  x' = A x + B u, y = C x
  H(ω) = C (iωI - A)^{-1} B
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128

__all__ = ["StateSpaceLTI"]


@dataclass(frozen=True)
class StateSpaceLTI:
    """
    Continuous-time LTI system in state-space form.
    """

    a: tf.Tensor
    b: tf.Tensor
    c: tf.Tensor

    def h(self, omega: float) -> tf.Tensor:
        aa = as_complex128(self.a)
        bb = as_complex128(self.b)
        cc = as_complex128(self.c)

        n = int(aa.shape[0])
        iw = 1j * tf.cast(float(omega), tf.complex128)
        mat = iw * tf.eye(n, dtype=tf.complex128) - aa
        x = tf.linalg.solve(mat, bb)
        return cc @ x
