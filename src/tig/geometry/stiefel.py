"""
Stiefel manifold St(n, p) = {X ∈ R^{n×p} : X^T X = 𝕀_p}.

This module provides:
- tangent projection
- QR-based retraction
- feasibility projection (QR)

The implementation is TF-first and math-driven.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["Stiefel"]


def _sym(a: tf.Tensor) -> tf.Tensor:
    return 0.5 * (a + tf.transpose(a))


@dataclass(frozen=True)
class Stiefel:
    """
    Stiefel manifold operations for matrices of shape (n, p).
    """

    n: int
    p: int

    def project(self, x: TensorLike) -> tf.Tensor:
        """
        Project to Stiefel via QR: X = QR -> Q.
        """
        xx = as_float64(x)
        q, r = tf.linalg.qr(xx, full_matrices=False)
        diag = tf.linalg.diag_part(r)
        s = tf.sign(diag)
        s = tf.where(tf.equal(s, 0.0), tf.ones_like(s), s)
        q = q * tf.reshape(s, (1, -1))
        return q

    def proj_tangent(self, x: TensorLike, u: TensorLike) -> tf.Tensor:
        """
        Tangent projection: P_X(U) = U - X sym(X^T U).
        """
        xx = as_float64(x)
        uu = as_float64(u)
        xtu = tf.transpose(xx) @ uu
        return uu - xx @ _sym(xtu)

    def retract(self, x: TensorLike, u: TensorLike) -> tf.Tensor:
        """
        QR retraction: R_X(U) = qf(X + U).
        """
        return self.project(as_float64(x) + as_float64(u))

    def is_feasible(self, x: TensorLike, tol: float = 1e-8) -> bool:
        """
        Check ||X^T X - 𝕀||_F <= tol.
        """
        xx = as_float64(x)
        xtx = tf.transpose(xx) @ xx
        eye = tf.eye(self.p, dtype=tf.float64)
        err = tf.linalg.norm(xtx - eye)
        return bool((err <= tf.cast(tol, tf.float64)).numpy())
