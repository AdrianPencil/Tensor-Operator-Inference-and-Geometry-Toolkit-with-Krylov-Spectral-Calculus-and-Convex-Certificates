"""
Kronecker-structured operators.

Provides a matrix-free representation of (A ⊗ B) acting on vec(X), where
vec stacks columns by default under reshape conventions.

Main identity used:
(A ⊗ B) vec(X) = vec(B X A^T).
"""

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tig.core.operators import LinearOperator
from tig.core.types import TensorLike, as_float64

__all__ = ["kron_matvec", "KronOperator"]


def kron_matvec(a: TensorLike, b: TensorLike, x: TensorLike, x_shape: Tuple[int, int]) -> tf.Tensor:
    """
    Compute y = (A ⊗ B) vec(X) using the identity vec(B X A^T).

    Parameters
    ----------
    a:
        Matrix A with shape (n, n).
    b:
        Matrix B with shape (m, m).
    x:
        Vectorized X of shape (m*n,).
    x_shape:
        (m, n) shape of X.

    Returns
    -------
    tf.Tensor
        Vectorized result vec(B X A^T).
    """
    aa = as_float64(a)
    bb = as_float64(b)
    xx = as_float64(x)

    m, n = int(x_shape[0]), int(x_shape[1])
    x_mat = tf.reshape(xx, (m, n))
    y_mat = bb @ x_mat @ tf.transpose(aa)
    return tf.reshape(y_mat, (-1,))


@dataclass(frozen=True)
class KronOperator:
    """
    Matrix-free linear operator for A ⊗ B acting on vec(X).

    The adjoint is (A ⊗ B)* = (A^T ⊗ B^T) in the real Euclidean setting.
    """

    a: tf.Tensor
    b: tf.Tensor
    x_shape: Tuple[int, int]

    def matvec(self, x: TensorLike) -> tf.Tensor:
        return kron_matvec(self.a, self.b, x, self.x_shape)

    def rmatvec(self, y: TensorLike) -> tf.Tensor:
        return kron_matvec(tf.transpose(self.a), tf.transpose(self.b), y, self.x_shape)
