"""
Krylov subspace methods (math-first, matrix-free).

This module provides a minimal TF-first implementation of Conjugate Gradient (CG)
for symmetric positive definite operators, plus an optional SciPy reference
GMRES/CG adapter when SciPy is available.

The operator interface follows tig.core.operators.LinearOperator:
- matvec(x) returns A x
- rmatvec(y) returns A* y (not required for CG)

The emphasis is on mathematical clarity: iterative methods are expressed in
inner-product form with explicit residual updates.
"""

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from tig.core.operators import LinearOperator
from tig.core.types import TensorLike, as_float64

__all__ = ["KrylovResult", "cg", "gmres_scipy"]


@dataclass(frozen=True)
class KrylovResult:
    """
    Result of an iterative linear solve.
    """

    x: tf.Tensor
    converged: bool
    num_iter: int
    rel_residual: float


def cg(
    op: LinearOperator,
    b: TensorLike,
    x0: Optional[TensorLike] = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> KrylovResult:
    """
    Conjugate Gradient for SPD operators.

    Solves A x = b with A represented by op.matvec.

    Parameters
    ----------
    op:
        Linear operator with a matvec method.
    b:
        Right-hand side tensor.
    x0:
        Optional initial guess. If None, uses zeros.
    tol:
        Relative residual tolerance.
    max_iter:
        Maximum iterations.

    Returns
    -------
    KrylovResult
        Solution and convergence diagnostics.
    """
    bb = as_float64(b)
    x = as_float64(x0) if x0 is not None else tf.zeros_like(bb)

    r = bb - as_float64(op.matvec(x))
    p = tf.identity(r)

    rtr = tf.reduce_sum(r * r)
    btb = tf.reduce_sum(bb * bb)
    b_norm = tf.sqrt(tf.maximum(btb, tf.constant(0.0, dtype=tf.float64)))

    if float(b_norm.numpy()) == 0.0:
        return KrylovResult(x=x, converged=True, num_iter=0, rel_residual=0.0)

    rel = float((tf.sqrt(rtr) / b_norm).numpy())
    if rel <= tol:
        return KrylovResult(x=x, converged=True, num_iter=0, rel_residual=rel)

    converged = False
    k = 0

    for k in range(1, int(max_iter) + 1):
        ap = as_float64(op.matvec(p))
        denom = tf.reduce_sum(p * ap)

        denom_val = float(denom.numpy())
        if denom_val == 0.0:
            break

        alpha = rtr / denom
        x = x + alpha * p
        r = r - alpha * ap

        new_rtr = tf.reduce_sum(r * r)
        rel = float((tf.sqrt(new_rtr) / b_norm).numpy())
        if rel <= tol:
            converged = True
            rtr = new_rtr
            break

        beta = new_rtr / rtr
        p = r + beta * p
        rtr = new_rtr

    return KrylovResult(x=x, converged=converged, num_iter=int(k), rel_residual=float(rel))


def gmres_scipy(
    op: LinearOperator,
    b: TensorLike,
    x0: Optional[TensorLike] = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> KrylovResult:
    """
    Optional SciPy GMRES reference solve for correctness comparisons.

    This is intended for tests, benchmarks, and parity checking, not as the main path.
    """
    try:
        import numpy as np
        import scipy.sparse.linalg as spla
    except Exception as exc:
        raise ImportError("SciPy is required for gmres_scipy.") from exc

    bb = as_float64(b)
    b_flat = tf.reshape(bb, (-1,))
    n = int(b_flat.shape[0])

    def mv(v: "np.ndarray") -> "np.ndarray":
        tv = tf.reshape(tf.convert_to_tensor(v, dtype=tf.float64), bb.shape)
        out = as_float64(op.matvec(tv))
        return tf.reshape(out, (-1,)).numpy()

    a_linop = spla.LinearOperator((n, n), matvec=mv, dtype=np.float64)
    x0_flat = None
    if x0 is not None:
        x0_flat = tf.reshape(as_float64(x0), (-1,)).numpy()

    x_hat, info = spla.gmres(a_linop, b_flat.numpy(), x0=x0_flat, tol=tol, maxiter=max_iter)
    x_tf = tf.reshape(tf.convert_to_tensor(x_hat, dtype=tf.float64), bb.shape)

    r = b_flat - tf.reshape(as_float64(op.matvec(x_tf)), (-1,))
    rel = float((tf.linalg.norm(r) / tf.linalg.norm(b_flat)).numpy())
    converged = bool(info == 0)

    iters = int(max_iter) if info > 0 else 0
    return KrylovResult(x=x_tf, converged=converged, num_iter=iters, rel_residual=rel)
