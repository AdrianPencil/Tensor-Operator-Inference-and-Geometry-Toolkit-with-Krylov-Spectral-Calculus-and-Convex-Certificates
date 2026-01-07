"""
Fixed-rank matrix manifold (minimal).

Set:
M_r = {X ∈ R^{m×n} : rank(X) = r}

This module provides:
- best rank-r approximation via truncated SVD
- factorization X ≈ U diag(s) V^T with rank r

More advanced tangent-space geometry can build on these primitives.
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["FixedRank", "truncate_svd"]


def truncate_svd(a: TensorLike, rank: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Truncated SVD: returns (U_r, s_r, Vt_r).
    """
    aa = as_float64(a)
    u, s, vt = tf.linalg.svd(aa, full_matrices=False)
    r = int(rank)
    return u[:, :r], s[:r], vt[:r, :]


@dataclass(frozen=True)
class FixedRank:
    """
    Fixed-rank utilities for matrices of shape (m, n).
    """

    m: int
    n: int
    r: int

    def project(self, x: TensorLike) -> tf.Tensor:
        """
        Best rank-r approximation in Frobenius norm via truncated SVD.
        """
        u, s, vt = truncate_svd(x, self.r)
        return u @ tf.linalg.diag(s) @ vt

    def factors(self, x: TensorLike) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Return truncated factors (U_r, s_r, Vt_r).
        """
        return truncate_svd(x, self.r)

    def is_rank_at_most_r(self, x: TensorLike, tol: float = 1e-10) -> bool:
        """
        Numerical check that singular values beyond r are <= tol.
        """
        xx = as_float64(x)
        s = tf.linalg.svd(xx, compute_uv=False)
        if int(s.shape[0]) <= self.r:
            return True
        tail = s[self.r :]
        return bool((tf.reduce_max(tail) <= tf.cast(tol, tf.float64)).numpy())
