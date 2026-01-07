"""
Operator-valued response functions (minimal).

Frequency-domain linear response is represented as:
χ(ω) = (i ω I - A)^{-1} B

This module provides a compact dense construction for small problems.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import TensorLike, as_complex128

__all__ = ["FrequencyResponse"]


@dataclass(frozen=True)
class FrequencyResponse:
    """
    Dense frequency response χ(ω) = (iωI - A)^{-1} B.
    """

    a: tf.Tensor
    b: tf.Tensor

    def eval(self, omega: float) -> tf.Tensor:
        aa = as_complex128(self.a)
        bb = as_complex128(self.b)
        n = int(aa.shape[0])
        iw = 1j * tf.cast(float(omega), tf.complex128)
        mat = iw * tf.eye(n, dtype=tf.complex128) - aa
        return tf.linalg.solve(mat, bb)
