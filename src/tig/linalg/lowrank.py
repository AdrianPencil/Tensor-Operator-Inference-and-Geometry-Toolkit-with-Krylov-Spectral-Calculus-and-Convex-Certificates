"""
Low-rank linear algebra (minimal, TF-first).

This module provides:
- randomized SVD (rank-k) for dense matrices
- reconstruction helpers

SciPy/JAX/Torch can be used elsewhere as parity oracles, but the main path here
is TF-native and math-centered.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import TensorLike, as_float64

__all__ = ["LowRankSVD", "randomized_svd", "reconstruct_from_svd"]


@dataclass(frozen=True)
class LowRankSVD:
    """
    Low-rank SVD representation A ≈ U diag(s) V^T.
    """

    u: tf.Tensor
    s: tf.Tensor
    vt: tf.Tensor


def randomized_svd(
    a: TensorLike,
    rank: int,
    n_oversamples: int = 5,
    n_iter: int = 1,
    rng: Optional[Rng] = None,
) -> LowRankSVD:
    """
    Randomized SVD using a Gaussian range finder.

    Parameters
    ----------
    a:
        Matrix tensor of shape (m, n).
    rank:
        Target rank k.
    n_oversamples:
        Oversampling parameter p, using k+p random probes.
    n_iter:
        Power iterations to improve spectral separation.
    rng:
        Optional Rng for reproducibility.

    Returns
    -------
    LowRankSVD
        Low-rank factors.
    """
    aa = as_float64(a)
    m = int(aa.shape[0])
    n = int(aa.shape[1])
    k = int(rank)
    p = int(n_oversamples)
    ell = int(k + p)

    if rng is None:
        rng = Rng(seed=0)

    omega = rng.normal((n, ell), dtype=tf.float64)
    y = aa @ omega

    for _ in range(int(n_iter)):
        y = aa @ (tf.transpose(aa) @ y)

    q, _ = tf.linalg.qr(y, full_matrices=False)
    b = tf.transpose(q) @ aa

    ub, s, vt = tf.linalg.svd(b, full_matrices=False)
    u = q @ ub

    u_k = u[:, :k]
    s_k = s[:k]
    vt_k = vt[:k, :]

    return LowRankSVD(u=u_k, s=s_k, vt=vt_k)


def reconstruct_from_svd(lr: LowRankSVD) -> tf.Tensor:
    """
    Reconstruct A_hat = U diag(s) V^T.
    """
    s_mat = tf.linalg.diag(lr.s)
    return lr.u @ s_mat @ lr.vt
