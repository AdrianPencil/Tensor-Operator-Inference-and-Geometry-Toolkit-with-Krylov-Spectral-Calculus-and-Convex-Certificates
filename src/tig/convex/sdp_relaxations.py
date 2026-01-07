"""
SDP relaxations (scoped, minimal).

This module defines an SDP problem data structure and an optional CVXPY solve path
(if cvxpy is installed). We keep this minimal to avoid bloat: the main goal is
mathematical framing and small demonstrators.

Standard primal form:
min <C, X>
s.t. <A_i, X> = b_i
     X ⪰ 0
"""

from dataclasses import dataclass
from typing import List, Optional

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["SdpProblem", "solve_sdp_cvxpy_optional"]


@dataclass(frozen=True)
class SdpProblem:
    """
    SDP data in dense matrix form (small/medium intended).
    """

    c: tf.Tensor
    a_list: List[tf.Tensor]
    b: tf.Tensor

    def dim(self) -> int:
        return int(self.c.shape[0])


def solve_sdp_cvxpy_optional(prob: SdpProblem):
    """
    Solve an SDP using CVXPY if available, otherwise raise ImportError.
    """
    try:
        import cvxpy as cp
        import numpy as np
    except Exception as exc:
        raise ImportError("cvxpy is required to solve SDPs (optional dependency).") from exc

    n = prob.dim()
    c = as_float64(prob.c).numpy()
    a_list = [as_float64(a).numpy() for a in prob.a_list]
    b = as_float64(prob.b).numpy()

    x = cp.Variable((n, n), PSD=True)
    constraints = []
    for a_i, b_i in zip(a_list, b):
        constraints.append(cp.trace(a_i @ x) == float(b_i))

    obj = cp.Minimize(cp.trace(c @ x))
    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    return tf.convert_to_tensor(x.value, dtype=tf.float64)
