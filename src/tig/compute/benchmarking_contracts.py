"""
Benchmarking contracts and invariants (math-first).

This module provides small checks used by tests and microbenchmarks:
- finite checks
- approximate equality checks
- adjoint pairing checks for operators: <Ax, y> == <x, A* y>
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from tig.core.norms import inner
from tig.core.random import Rng
from tig.core.types import ShapeLike, TensorLike, as_float64
from tig.core.operators import LinearOperator

__all__ = [
    "ContractViolation",
    "assert_finite",
    "assert_allclose",
    "adjoint_pairing_gap",
    "assert_adjoint_pairing",
]


@dataclass(frozen=True)
class ContractViolation(Exception):
    """
    Raised when a mathematical contract is violated.
    """

    message: str


def assert_finite(x: TensorLike, name: str = "x") -> None:
    xx = as_float64(x)
    ok = tf.reduce_all(tf.math.is_finite(xx))
    if not bool(ok.numpy()):
        raise ContractViolation(f"{name} contains non-finite values.")


def assert_allclose(a: TensorLike, b: TensorLike, rtol: float = 1e-7, atol: float = 1e-9) -> None:
    aa = as_float64(a)
    bb = as_float64(b)
    diff = tf.linalg.norm(tf.reshape(aa - bb, (-1,)))
    denom = tf.linalg.norm(tf.reshape(bb, (-1,))) + tf.cast(float(atol), tf.float64)
    rel = float((diff / denom).numpy())
    if rel > float(rtol) and float(diff.numpy()) > float(atol):
        raise ContractViolation(f"allclose failed: rel={rel:g}, atol={float(diff.numpy()):g}.")


def adjoint_pairing_gap(op: LinearOperator, x: TensorLike, y: TensorLike) -> tf.Tensor:
    """
    Return |<A x, y> - <x, A* y>|.
    """
    xx = as_float64(x)
    yy = as_float64(y)
    ax = as_float64(op.matvec(xx))
    aty = as_float64(op.rmatvec(yy))
    left = as_float64(inner(ax, yy))
    right = as_float64(inner(xx, aty))
    return tf.abs(left - right)


def assert_adjoint_pairing(
    op: LinearOperator,
    shape_x: ShapeLike,
    shape_y: ShapeLike,
    n_probe: int = 5,
    tol: float = 1e-7,
    rng: Optional[Rng] = None,
) -> None:
    """
    Randomized adjoint pairing contract check.
    """
    if rng is None:
        rng = Rng(seed=0)

    for _ in range(int(n_probe)):
        x = rng.normal(tuple(shape_x), dtype=tf.float64)
        y = rng.normal(tuple(shape_y), dtype=tf.float64)
        gap = float(adjoint_pairing_gap(op, x, y).numpy())
        if gap > float(tol):
            raise ContractViolation(f"Adjoint pairing gap too large: {gap:g}.")
