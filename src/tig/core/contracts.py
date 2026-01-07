"""
Mathematical contracts and invariants.

Contracts provide executable checks of the theory:
- adjoint pairing consistency
- autodiff consistency (JVP/VJP versus finite differences on small problems)
- basic linearity checks

These are meant for tests and debugging, not production hot paths.
"""

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import tensorflow as tf

from tig.core.types import TensorLike, as_float64, to_tensor

__all__ = [
    "ContractViolation",
    "ContractResult",
    "assert_allclose",
    "check_linearity",
    "check_adjoint_pairing",
]


class ContractViolation(AssertionError):
    """
    Raised when a mathematical contract fails.
    """


@dataclass(frozen=True)
class ContractResult:
    """
    Result object for a contract check.
    """

    name: str
    passed: bool
    max_abs_error: float
    details: Mapping[str, float]


def assert_allclose(a: TensorLike, b: TensorLike, rtol: float = 1e-7, atol: float = 1e-9) -> None:
    """
    Assert elementwise closeness using float64 evaluation.
    """
    aa = as_float64(a)
    bb = as_float64(b)
    diff = tf.abs(aa - bb)
    denom = atol + rtol * tf.abs(bb)
    ok = tf.reduce_all(diff <= denom)

    if not bool(ok.numpy()):
        max_abs = float(tf.reduce_max(diff).numpy())
        raise ContractViolation(f"assert_allclose failed: max_abs_error={max_abs}")


def check_linearity(
    f: Callable[[tf.Tensor], tf.Tensor],
    x: TensorLike,
    y: TensorLike,
    alpha: float = 0.37,
    beta: float = -1.11,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> ContractResult:
    """
    Check f(alpha x + beta y) == alpha f(x) + beta f(y) for a map f.
    """
    xx = as_float64(x)
    yy = as_float64(y)
    lhs = f(alpha * xx + beta * yy)
    rhs = alpha * f(xx) + beta * f(yy)

    diff = tf.reduce_max(tf.abs(as_float64(lhs) - as_float64(rhs)))
    max_abs = float(diff.numpy())
    passed = max_abs <= float((atol + rtol * tf.reduce_max(tf.abs(as_float64(rhs)))).numpy())

    return ContractResult(
        name="linearity",
        passed=passed,
        max_abs_error=max_abs,
        details={"alpha": float(alpha), "beta": float(beta)},
    )


def check_adjoint_pairing(
    matvec: Callable[[tf.Tensor], tf.Tensor],
    rmatvec: Callable[[tf.Tensor], tf.Tensor],
    x: TensorLike,
    y: TensorLike,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> ContractResult:
    """
    Check the adjoint pairing identity: <A x, y> = <x, A* y> in the Euclidean inner product.
    """
    xx = as_float64(x)
    yy = as_float64(y)

    ax = as_float64(matvec(xx))
    aty = as_float64(rmatvec(yy))

    left = tf.reduce_sum(ax * yy)
    right = tf.reduce_sum(xx * aty)

    diff = tf.abs(left - right)
    max_abs = float(diff.numpy())
    scale = float((atol + rtol * tf.abs(right)).numpy())
    passed = max_abs <= scale

    return ContractResult(
        name="adjoint_pairing",
        passed=passed,
        max_abs_error=max_abs,
        details={"left": float(left.numpy()), "right": float(right.numpy())},
    )
