"""
ADMM (minimal consensus form).

Consensus splitting:
min_x f(x) + g(z)  s.t. x = z

Scaled-form iterations:
x^{k+1} = prox_{f/ρ}(z^k - u^k)
z^{k+1} = prox_{g/ρ}(x^{k+1} + u^k)
u^{k+1} = u^k + x^{k+1} - z^{k+1}

This file keeps the implementation compact and math-faithful.
"""

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["AdmmResult", "consensus_admm"]


@dataclass(frozen=True)
class AdmmResult:
    """
    Result of ADMM consensus iterations.
    """

    x: tf.Tensor
    z: tf.Tensor
    u: tf.Tensor
    converged: bool
    num_iter: int
    prim_res: float
    dual_res: float


def consensus_admm(
    prox_f: Callable[[tf.Tensor, float], tf.Tensor],
    prox_g: Callable[[tf.Tensor, float], tf.Tensor],
    x0: TensorLike,
    rho: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> AdmmResult:
    """
    Run ADMM for min f(x) + g(z) subject to x=z given proximal maps.

    prox_f(v, rho) returns prox_{f/ρ}(v)
    prox_g(v, rho) returns prox_{g/ρ}(v)
    """
    rr = float(rho)
    x = as_float64(x0)
    z = tf.identity(x)
    u = tf.zeros_like(x)

    converged = False
    k = 0
    prim = 0.0
    dual = 0.0

    for k in range(int(max_iter)):
        x = as_float64(prox_f(z - u, rr))
        z_prev = z
        z = as_float64(prox_g(x + u, rr))
        u = u + (x - z)

        r = x - z
        s = rr * (z - z_prev)

        prim = float(tf.linalg.norm(r).numpy())
        dual = float(tf.linalg.norm(s).numpy())

        if prim <= tol and dual <= tol:
            converged = True
            break

    return AdmmResult(
        x=x,
        z=z,
        u=u,
        converged=bool(converged),
        num_iter=int(k),
        prim_res=float(prim),
        dual_res=float(dual),
    )
