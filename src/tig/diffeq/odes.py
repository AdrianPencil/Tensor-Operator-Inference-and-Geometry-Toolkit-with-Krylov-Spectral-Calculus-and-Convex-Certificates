"""
ODEs as operator flows (minimal).

We represent an ODE as:
d/dt x = f(x, t)

This module provides a compact explicit Euler integrator.
More advanced time-stepping is separated into time_stepping.py.
"""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["OdeSpec", "euler"]


@dataclass(frozen=True)
class OdeSpec:
    """
    ODE specification.
    """

    f: Callable[[tf.Tensor, float], tf.Tensor]


def euler(spec: OdeSpec, x0: TensorLike, t0: float, dt: float, n_steps: int) -> tf.Tensor:
    """
    Explicit Euler trajectory of shape (n_steps+1, *x0.shape).
    """
    xx = as_float64(x0)
    dd = float(dt)
    ns = int(n_steps)
    t = float(t0)

    traj = [xx]
    for _ in range(ns):
        xx = xx + tf.cast(dd, tf.float64) * as_float64(spec.f(xx, t))
        traj.append(xx)
        t += dd

    return tf.stack(traj, axis=0)
