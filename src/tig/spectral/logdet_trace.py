"""
Log-determinants and trace estimators (minimal, math-first).

Provides:
- logdet for dense SPD matrices with ridge
- Hutchinson trace estimator for functionals that accept probe vectors

The trace estimator is used for A-optimal design and other UQ tasks.
"""

from typing import Callable, Optional

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import ShapeLike, tf_float

__all__ = ["logdet_spd", "hutchinson_trace"]


def logdet_spd(a: tf.Tensor, ridge: float = 1e-12) -> tf.Tensor:
    """
    Compute log det(A + ridge I) for a dense SPD-like matrix.
    """
    aa = tf.cast(a, tf.float64)
    n = int(aa.shape[0])
    reg = tf.cast(float(ridge), tf.float64) * tf.eye(n, dtype=tf.float64)
    return tf.linalg.logdet(aa + reg)


def hutchinson_trace(
    quad_form: Callable[[tf.Tensor], tf.Tensor],
    shape: ShapeLike,
    n_probe: int = 16,
    rng: Optional[Rng] = None,
) -> tf.Tensor:
    """
    Hutchinson estimator for tr(K) where quad_form(v) approximates v^T K v.

    Uses Rademacher probes by default.
    """
    if rng is None:
        rng = Rng(seed=0)

    n = int(tf.reduce_prod(tf.constant(tuple(shape))).numpy())
    acc = tf.constant(0.0, dtype=tf.float64)

    for _ in range(int(n_probe)):
        u = rng.uniform((n,), minval=0.0, maxval=1.0, dtype=tf.float64)
        v = tf.where(u < 0.5, -tf.ones_like(u), tf.ones_like(u))
        acc = acc + tf.reshape(tf.cast(quad_form(v), tf.float64), ())

    return acc / tf.cast(int(n_probe), tf.float64)
