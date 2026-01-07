"""
Grassmann manifold Gr(n, p): p-dimensional subspaces of R^n.

A representative is an orthonormal matrix X ∈ St(n, p). The Grassmann tangent
space uses the horizontal projection:
P_X(U) = U - X (X^T U).

We use QR-based retraction on representatives.
"""

from dataclasses import dataclass

import tensorflow as tf

from tig.core.types import TensorLike, as_float64
from tig.geometry.stiefel import Stiefel

__all__ = ["Grassmann"]


@dataclass(frozen=True)
class Grassmann:
    """
    Grassmann manifold operations represented by Stiefel matrices.
    """

    n: int
    p: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "_stiefel", Stiefel(self.n, self.p))

    def project(self, x: TensorLike) -> tf.Tensor:
        return self._stiefel.project(x)

    def proj_tangent(self, x: TensorLike, u: TensorLike) -> tf.Tensor:
        """
        Horizontal projection: U - X (X^T U).
        """
        xx = as_float64(x)
        uu = as_float64(u)
        return uu - xx @ (tf.transpose(xx) @ uu)

    def retract(self, x: TensorLike, u: TensorLike) -> tf.Tensor:
        """
        Retraction by projecting X+U back to an orthonormal representative.
        """
        return self.project(as_float64(x) + as_float64(u))

    def principal_angles_cos(self, x: TensorLike, y: TensorLike) -> tf.Tensor:
        """
        Cosines of principal angles via singular values of X^T Y.
        """
        xx = self.project(x)
        yy = self.project(y)
        s = tf.linalg.svd(tf.transpose(xx) @ yy, compute_uv=False)
        return tf.clip_by_value(as_float64(s), 0.0, 1.0)
