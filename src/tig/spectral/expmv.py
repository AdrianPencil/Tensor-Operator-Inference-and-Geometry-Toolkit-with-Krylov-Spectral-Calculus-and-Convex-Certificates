"""
Matrix exponential action exp(tA) v (minimal).

TF-first path:
- dense expm via tf.linalg.expm for small/medium matrices

Optional SciPy path (reference / large sparse):
- scipy.sparse.linalg.expm_multiply if SciPy is available

This module keeps one primary function: expmv.
"""

from typing import Optional

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["expmv", "expmv_scipy"]


def expmv(a: TensorLike, v: TensorLike, t: float = 1.0) -> tf.Tensor:
    """
    Compute exp(tA) v using dense tf.linalg.expm.
    """
    aa = as_float64(a)
    vv = as_float64(v)
    tt = tf.cast(float(t), tf.float64)
    e = tf.linalg.expm(tt * aa)
    return e @ tf.reshape(vv, (-1, 1))[:, 0]


def expmv_scipy(a: TensorLike, v: TensorLike, t: float = 1.0) -> tf.Tensor:
    """
    Optional SciPy expm_multiply reference.
    """
    try:
        import numpy as np
        import scipy.sparse.linalg as spla
    except Exception as exc:
        raise ImportError("SciPy is required for expmv_scipy.") from exc

    aa = as_float64(a).numpy()
    vv = as_float64(v).numpy()
    out = spla.expm_multiply(float(t) * aa, vv)
    return tf.convert_to_tensor(out, dtype=tf.float64)
