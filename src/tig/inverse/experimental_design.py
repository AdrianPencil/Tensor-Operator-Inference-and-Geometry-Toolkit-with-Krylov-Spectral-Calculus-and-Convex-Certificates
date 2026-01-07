"""
Experimental design criteria (minimal, math-first).

This module provides scalar criteria computed from a Fisher/information matrix I:
- A-optimal: tr(I^{-1})
- D-optimal: log det(I)

In matrix-free settings, these are approximated using:
- CG solves for trace estimators (Hutchinson)
- logdet for small dense matrices

This file stays compact: advanced design loops live in workflows/experiments.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import TensorLike, as_float64
from tig.linalg.krylov import cg
from tig.spectral.logdet_trace import hutchinson_trace

__all__ = ["DesignCriteria", "a_optimal_trace_inv", "d_optimal_logdet"]


@dataclass(frozen=True)
class DesignCriteria:
    """
    Container for experimental design scalar criteria.
    """

    a_opt: float
    d_opt: float


def a_optimal_trace_inv(
    info_matvec,
    shape: tuple[int, ...],
    n_probe: int = 16,
    tol: float = 1e-8,
    max_iter: int = 200,
    rng: Optional[Rng] = None,
) -> float:
    """
    Estimate tr(I^{-1}) using Hutchinson probes and CG solves.
    """
    if rng is None:
        rng = Rng(seed=0)

    def solve_for(v: tf.Tensor) -> tf.Tensor:
        op = _MatvecOp(matvec=info_matvec)
        res = cg(op=op, b=v, x0=tf.zeros_like(v), tol=tol, max_iter=max_iter)
        return res.x

    def quad(v: tf.Tensor) -> tf.Tensor:
        x = solve_for(v)
        return tf.reduce_sum(as_float64(v) * as_float64(x))

    return float(hutchinson_trace(quad_form=quad, shape=shape, n_probe=int(n_probe), rng=rng).numpy())


def d_optimal_logdet(i_dense: TensorLike, ridge: float = 1e-10) -> float:
    """
    Compute log det(I + ridge * I) for a dense information matrix.
    """
    ii = as_float64(i_dense)
    n = int(ii.shape[0])
    reg = tf.cast(float(ridge), tf.float64) * tf.eye(n, dtype=tf.float64)
    return float(tf.linalg.logdet(ii + reg).numpy())


class _MatvecOp:
    """
    Minimal operator wrapper for CG using a callable matvec.
    """

    def __init__(self, matvec) -> None:
        self._mv = matvec

    def matvec(self, x: tf.Tensor) -> tf.Tensor:
        return as_float64(self._mv(as_float64(x)))

    def rmatvec(self, y: tf.Tensor) -> tf.Tensor:
        return self.matvec(y)
