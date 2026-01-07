"""
Inverse problem solvers (minimal, math-first).

This module focuses on MAP-style optimization for least squares objectives using
matrix-free Jacobian access (JVP/VJP).

Primary routine:
- solve_map: minimize 0.5||F(x)-y||^2 + lam * R(x)

The implementation is TF-first and uses Armijo backtracking for robust descent.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from tig.core.norms import inner
from tig.core.types import TensorLike, as_float64
from tig.inverse.forward_models import ForwardModel
from tig.inverse.regularizers import Regularizer
from tig.opt.linesearch import ArmijoBacktracking

__all__ = ["MapSolveResult", "solve_map"]


@dataclass(frozen=True)
class MapSolveResult:
    """
    Result for MAP solve.
    """

    x: tf.Tensor
    value: float
    converged: bool
    num_iter: int
    grad_norm: float


def _misfit_value(residual: tf.Tensor) -> tf.Tensor:
    rr = as_float64(residual)
    return 0.5 * tf.reduce_sum(rr * rr)


def solve_map(
    model: ForwardModel,
    y: TensorLike,
    x0: TensorLike,
    lam: float = 0.0,
    reg: Optional[Regularizer] = None,
    tol_grad: float = 1e-8,
    max_iter: int = 200,
    linesearch: Optional[ArmijoBacktracking] = None,
) -> MapSolveResult:
    """
    Minimize: Φ(x) = 0.5||F(x)-y||^2 + lam * R(x).

    Gradient:
    ∇Φ(x) = J(x)^T (F(x)-y) + lam * ∇R(x) (when available).
    For L1-style regularizers, use proximal solvers instead of gradient descent.
    """
    yy = as_float64(y)
    x = as_float64(x0)
    ls = linesearch if linesearch is not None else ArmijoBacktracking()

    converged = False
    k = 0

    for k in range(int(max_iter)):
        fx = as_float64(model(x))
        r = fx - yy
        f_val = _misfit_value(r)

        g = as_float64(model.vjp(x, r))

        if reg is not None and float(lam) != 0.0:
            if hasattr(reg, "gradient"):
                g = g + tf.cast(float(lam), tf.float64) * as_float64(reg.gradient(x))  # type: ignore[attr-defined]
            else:
                if reg.__class__.__name__.lower().startswith("l2"):
                    g = g + tf.cast(float(lam), tf.float64) * x

        gn = float(tf.linalg.norm(g).numpy())
        if gn <= tol_grad:
            converged = True
            val_total = float(f_val.numpy())
            if reg is not None and float(lam) != 0.0:
                val_total += float(lam) * float(as_float64(reg.value(x)).numpy())
            return MapSolveResult(x=x, value=float(val_total), converged=True, num_iter=int(k), grad_norm=gn)

        p = -g
        step_res = ls(
            obj=_MapObjective(model=model, y=yy, lam=float(lam), reg=reg),
            x=x,
            p=p,
            g=g,
        )
        if not step_res.accepted:
            break

        x = x + tf.cast(step_res.step, tf.float64) * p

    fx = as_float64(model(x))
    r = fx - yy
    val_total = float(_misfit_value(r).numpy())
    if reg is not None and float(lam) != 0.0:
        val_total += float(lam) * float(as_float64(reg.value(x)).numpy())

    g = as_float64(model.vjp(x, r))
    gn = float(tf.linalg.norm(g).numpy())

    return MapSolveResult(x=x, value=float(val_total), converged=bool(converged), num_iter=int(k), grad_norm=gn)


class _MapObjective:
    """
    Internal objective wrapper for line search.
    """

    def __init__(self, model: ForwardModel, y: tf.Tensor, lam: float, reg: Optional[Regularizer]) -> None:
        self._model = model
        self._y = y
        self._lam = float(lam)
        self._reg = reg

    def value(self, x: TensorLike) -> tf.Tensor:
        xx = as_float64(x)
        r = as_float64(self._model(xx)) - self._y
        val = _misfit_value(r)
        if self._reg is not None and self._lam != 0.0:
            val = val + tf.cast(self._lam, tf.float64) * as_float64(self._reg.value(xx))
        return tf.reshape(val, ())

    def gradient(self, x: TensorLike) -> tf.Tensor:
        xx = as_float64(x)
        r = as_float64(self._model(xx)) - self._y
        g = as_float64(self._model.vjp(xx, r))
        if self._reg is not None and self._lam != 0.0:
            if self._reg.__class__.__name__.lower().startswith("l2"):
                g = g + tf.cast(self._lam, tf.float64) * xx
        return g
