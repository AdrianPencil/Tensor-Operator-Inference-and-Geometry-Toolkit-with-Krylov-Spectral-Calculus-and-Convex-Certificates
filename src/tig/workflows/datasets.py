"""
Datasets for workflows (minimal).

This module provides small synthetic dataset generators for experiments and tests:
- linear regression style inverse problems
- noisy forward-model observations

The emphasis is reproducible tensors and clear mathematical structure.
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import as_float64

__all__ = ["LinearDataset", "make_linear_inverse_dataset"]


@dataclass(frozen=True)
class LinearDataset:
    """
    Dataset for y = A x + noise.
    """

    a: tf.Tensor
    x_true: tf.Tensor
    y: tf.Tensor
    sigma: float


def make_linear_inverse_dataset(
    m: int,
    n: int,
    sigma: float = 0.01,
    rng: Rng | None = None,
) -> LinearDataset:
    if rng is None:
        rng = Rng(seed=0)

    mm = int(m)
    nn = int(n)
    ss = float(sigma)

    a = rng.normal((mm, nn), dtype=tf.float64)
    x_true = rng.normal((nn,), dtype=tf.float64)
    noise = tf.cast(ss, tf.float64) * rng.normal((mm,), dtype=tf.float64)
    y = (a @ tf.reshape(x_true, (-1, 1)))[:, 0] + noise

    return LinearDataset(a=as_float64(a), x_true=as_float64(x_true), y=as_float64(y), sigma=float(ss))
