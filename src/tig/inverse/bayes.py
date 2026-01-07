"""
Bayesian inverse problems (scoped, optional).

This module is intentionally small to avoid bloat. It provides a PyMC-based
wrapper when PyMC is installed. If PyMC is not installed, importing still works,
but the PyMC construction routine raises a clear error.

The forward model remains TF-first; PyMC is used to build probabilistic inference
around the forward map and observation model.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from tig.core.types import TensorLike, as_float64
from tig.inverse.forward_models import ForwardModel

__all__ = ["BayesModelSpec", "build_pymc_model"]


@dataclass(frozen=True)
class BayesModelSpec:
    """
    Minimal specification for a Bayesian inverse problem.

    y ~ Normal(F(x), sigma)
    prior on x is provided by a callable returning a PyMC random variable.
    """

    model: ForwardModel
    y: tf.Tensor
    sigma: float
    prior: Callable  # PyMC prior factory, kept untyped to avoid hard dependency


def build_pymc_model(spec: BayesModelSpec):
    """
    Build a PyMC model if PyMC is installed.
    """
    try:
        import pymc as pm
    except Exception as exc:
        raise ImportError("PyMC is required for build_pymc_model (install extra 'bayes').") from exc

    y_obs = as_float64(spec.y).numpy()
    sigma = float(spec.sigma)

    with pm.Model() as m:
        x_rv = spec.prior()
        def _forward_np(x_np):
            x_tf = tf.convert_to_tensor(x_np, dtype=tf.float64)
            y_tf = as_float64(spec.model(x_tf))
            return y_tf.numpy()

        mu = pm.Deterministic("mu", _forward_np(x_rv))
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

    return m
