"""
Generic optimal control problems (minimal, math-first).

We solve:
min_u  J(u) = loss(simulate(u), u)

using gradient descent with optional Armijo line search on the control space.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from tig.core.types import TensorLike, as_float64
from tig.control.adjoint_methods import DiscreteDynamics, simulate_with_grad_u
from tig.opt.linesearch import ArmijoBacktracking
from tig.opt.objectives import ScalarObjective

__all__ = ["OptimalControlProblem", "ControlSolveResult", "solve_control_gd"]


@dataclass(frozen=True)
class OptimalControlProblem:
    """
    Optimal control problem definition.
    """

    dynamics: DiscreteDynamics
    loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    t0: float
    dt: float


@dataclass(frozen=True)
class ControlSolveResult:
    """
    Result of control optimization.
    """

    u: tf.Tensor
    value: float
    converged: bool
    num_iter: int
    grad_norm: float


def solve_control_gd(
    prob: OptimalControlProblem,
    x0: TensorLike,
    u0: TensorLike,
    tol_grad: float = 1e-8,
    max_iter: int = 200,
    step0: float = 1.0,
) -> ControlSolveResult:
    """
    Gradient descent in control space with Armijo line search.
    """
    x_init = as_float64(x0)
    u = as_float64(u0)

    ls = ArmijoBacktracking(step0=float(step0))
    converged = False
    k = 0

    def value_fn(u_var: tf.Tensor) -> tf.Tensor:
        L, _ = simulate_with_grad_u(
            dyn=prob.dynamics,
            x0=x_init,
            u0=u_var,
            t0=float(prob.t0),
            dt=float(prob.dt),
            loss=prob.loss,
        )
        return tf.reshape(as_float64(L), ())

    obj = ScalarObjective(value_fn=value_fn)

    for k in range(int(max_iter)):
        L, g = simulate_with_grad_u(
            dyn=prob.dynamics,
            x0=x_init,
            u0=u,
            t0=float(prob.t0),
            dt=float(prob.dt),
            loss=prob.loss,
        )
        gn = float(tf.linalg.norm(tf.reshape(as_float64(g), (-1,))).numpy())
        if gn <= tol_grad:
            converged = True
            return ControlSolveResult(u=u, value=float(L.numpy()), converged=True, num_iter=int(k), grad_norm=gn)

        p = -as_float64(g)
        step_res = ls(obj=obj, x=u, p=p, g=g)
        if not step_res.accepted:
            break
        u = u + tf.cast(step_res.step, tf.float64) * p

    L_final = float(as_float64(value_fn(u)).numpy())
    g_final = simulate_with_grad_u(
        dyn=prob.dynamics,
        x0=x_init,
        u0=u,
        t0=float(prob.t0),
        dt=float(prob.dt),
        loss=prob.loss,
    )[1]
    gn_final = float(tf.linalg.norm(tf.reshape(as_float64(g_final), (-1,))).numpy())

    return ControlSolveResult(
        u=u,
        value=float(L_final),
        converged=bool(converged),
        num_iter=int(k),
        grad_norm=float(gn_final),
    )
