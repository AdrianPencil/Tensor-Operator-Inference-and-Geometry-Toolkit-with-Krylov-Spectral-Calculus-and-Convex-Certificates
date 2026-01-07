"""
Trust-region (minimal Cauchy-point variant).

This is a deliberately compact trust-region method that only needs:
- objective value φ(x)
- gradient ∇φ(x)

Model: m(p) = φ(x) + <g, p>
Trust region constraint: ||p|| <= Δ

Cauchy step: p = -α g where α = min(Δ/||g||, 1).

This preserves the theoretical trust-region structure without requiring Hessians.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.norms import inner, l2_norm
from tig.core.types import TensorLike, as_float64
from tig.opt.objectives import Objective

__all__ = ["TrustRegionState", "TrustRegionResult", "cauchy_trust_region_step", "trust_region_solve"]


@dataclass(frozen=True)
class TrustRegionState:
    """
    Trust-region radius state.
    """

    delta: float


@dataclass(frozen=True)
class TrustRegionResult:
    """
    Result of trust-region optimization.
    """

    x: tf.Tensor
    value: float
    converged: bool
    num_iter: int
    delta: float


def cauchy_trust_region_step(g: TensorLike, delta: float) -> tf.Tensor:
    """
    Cauchy step within ||p|| <= delta using Euclidean norm.
    """
    gg = as_float64(g)
    gn = tf.reshape(as_float64(l2_norm(gg)), ())
    gn_val = float(gn.numpy())
    if gn_val == 0.0:
        return tf.zeros_like(gg)
    alpha = min(float(delta) / gn_val, 1.0)
    return -tf.cast(alpha, tf.float64) * gg


def trust_region_solve(
    obj: Objective,
    x0: TensorLike,
    state: TrustRegionState | None = None,
    tol_grad: float = 1e-8,
    max_iter: int = 200,
    eta: float = 0.1,
    delta_min: float = 1e-12,
    delta_max: float = 1e3,
    grow: float = 2.0,
    shrink: float = 0.5,
) -> TrustRegionResult:
    """
    Trust-region loop using the Cauchy point and actual/predicted reduction ratio.

    Ratio:
    ρ = (φ(x) - φ(x+p)) / (m(0) - m(p))
      = (φ(x) - φ(x+p)) / (-<g, p>)
    """
    xx = as_float64(x0)
    delta = float(state.delta) if state is not None else 1.0

    f = float(as_float64(obj.value(xx)).numpy())
    converged = False
    k = 0

    for k in range(int(max_iter)):
        g = as_float64(obj.gradient(xx))
        gn = float(as_float64(l2_norm(g)).numpy())
        if gn <= tol_grad:
            converged = True
            break

        p = cauchy_trust_region_step(g, delta)
        pred = -float(as_float64(inner(g, p)).numpy())
        if pred <= 0.0:
            delta = max(delta * shrink, delta_min)
            continue

        x_try = xx + p
        f_try = float(as_float64(obj.value(x_try)).numpy())
        ared = f - f_try
        rho = ared / pred

        if rho >= eta:
            xx = x_try
            f = f_try
            if rho > 0.75:
                delta = min(delta * grow, delta_max)
        else:
            delta = max(delta * shrink, delta_min)

        if delta <= delta_min:
            break

    return TrustRegionResult(
        x=xx,
        value=float(f),
        converged=bool(converged),
        num_iter=int(k),
        delta=float(delta),
    )
