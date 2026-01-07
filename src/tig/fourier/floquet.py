"""
Floquet structure for periodic drives (minimal).

A periodic drive with base frequency Ω induces sidebands at ω + nΩ.
This module provides a helper to generate sideband frequencies.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["floquet_sidebands"]


def floquet_sidebands(omega0: float, omega_drive: float, n_side: int) -> tf.Tensor:
    """
    Return frequencies omega0 + n * omega_drive for n in [-n_side, ..., n_side].
    """
    o0 = tf.cast(float(omega0), tf.float64)
    od = tf.cast(float(omega_drive), tf.float64)
    ns = int(n_side)
    n = tf.range(-ns, ns + 1, dtype=tf.float64)
    return o0 + n * od
