"""
Robust optimization under uncertainty (minimal).

Provides Monte Carlo risk estimators:
- expected value (risk-neutral)
- CVaR_alpha (risk-averse tail expectation)

These are thin wrappers intended for control/inverse demonstrations.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import TensorLike, as_float64

__all__ = ["RiskEstimates", "mc_expected_risk", "mc_cvar_risk"]


@dataclass(frozen=True)
class RiskEstimates:
    """
    Risk estimates from Monte Carlo samples.
    """

    mean: float
    cvar: float


def mc_expected_risk(
    loss_fn: Callable[[tf.Tensor], tf.Tensor],
    param_sampler: Callable[[Rng], tf.Tensor],
    n_samples: int,
    rng: Optional[Rng] = None,
) -> tf.Tensor:
    """
    Estimate E[ loss_fn(theta) ] by Monte Carlo.
    """
    if rng is None:
        rng = Rng(seed=0)

    vals = []
    for _ in range(int(n_samples)):
        theta = param_sampler(rng)
        vals.append(tf.reshape(as_float64(loss_fn(as_float64(theta))), ()))
    return tf.reduce_mean(tf.stack(vals, axis=0))


def mc_cvar_risk(
    loss_fn: Callable[[tf.Tensor], tf.Tensor],
    param_sampler: Callable[[Rng], tf.Tensor],
    n_samples: int,
    alpha: float = 0.9,
    rng: Optional[Rng] = None,
) -> tf.Tensor:
    """
    Estimate CVaR_alpha of loss by Monte Carlo.

    CVaR_alpha = E[ loss | loss >= VaR_alpha ].
    """
    if rng is None:
        rng = Rng(seed=0)

    vals = []
    for _ in range(int(n_samples)):
        theta = param_sampler(rng)
        vals.append(tf.reshape(as_float64(loss_fn(as_float64(theta))), ()))
    v = tf.sort(tf.stack(vals, axis=0))
    n = int(v.shape[0])
    idx = int(tf.floor(tf.cast(alpha, tf.float64) * tf.cast(n - 1, tf.float64)).numpy())
    tail = v[idx:]
    return tf.reduce_mean(tail)
