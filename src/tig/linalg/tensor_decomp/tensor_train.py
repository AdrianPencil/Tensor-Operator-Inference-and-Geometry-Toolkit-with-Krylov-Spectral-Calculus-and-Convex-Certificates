"""
Tensor Train (TT) decomposition via TT-SVD (minimal).

TT represents an order-d tensor X as cores G1,...,Gd:
X[i1,...,id] = G1[i1] G2[i2] ... Gd[id]
where each Gk has shape (r_{k-1}, n_k, r_k).

This file implements TT-SVD for dense TF tensors.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["TTCores", "tt_svd", "tt_reconstruct"]


@dataclass(frozen=True)
class TTCores:
    """
    Tensor Train cores.
    """

    cores: List[tf.Tensor]


def tt_svd(x: TensorLike, ranks: Optional[Sequence[int]] = None, tol: Optional[float] = None) -> TTCores:
    """
    TT-SVD for a dense tensor.

    Parameters
    ----------
    x:
        Input tensor of shape (n1, ..., nd).
    ranks:
        Optional TT ranks (r1,...,r_{d-1}). If provided, used as hard caps.
    tol:
        Optional truncation tolerance on singular values.

    Returns
    -------
    TTCores
        TT core list.
    """
    xx = as_float64(x)
    shape = [int(s) for s in xx.shape]
    d = len(shape)
    if d < 2:
        raise ValueError("TT decomposition requires tensor order >= 2.")

    if ranks is None:
        ranks_list = [min(shape[k], shape[k + 1]) for k in range(d - 1)]
    else:
        ranks_list = [int(r) for r in ranks]
        if len(ranks_list) != d - 1:
            raise ValueError("ranks must have length d-1.")

    cores: List[tf.Tensor] = []
    r_prev = 1
    tensor = xx

    for k in range(d - 1):
        n_k = shape[k]
        tensor = tf.reshape(tensor, (r_prev * n_k, -1))
        u, s, vt = tf.linalg.svd(tensor, full_matrices=False)

        r_cap = ranks_list[k]
        r_keep = min(int(s.shape[0]), int(r_cap))

        if tol is not None:
            mask = s > tf.cast(tol, tf.float64)
            r_keep = min(r_keep, int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()))
            r_keep = max(r_keep, 1)

        u = u[:, :r_keep]
        s = s[:r_keep]
        vt = vt[:r_keep, :]

        core = tf.reshape(u, (r_prev, n_k, r_keep))
        cores.append(core)

        tensor = tf.linalg.diag(s) @ vt
        r_prev = r_keep

    n_last = shape[-1]
    core_last = tf.reshape(tensor, (r_prev, n_last, 1))
    cores.append(core_last)

    return TTCores(cores=cores)


def tt_reconstruct(tt: TTCores) -> tf.Tensor:
    """
    Reconstruct a dense tensor from TT cores.
    """
    cores = [as_float64(g) for g in tt.cores]
    d = len(cores)

    x = cores[0]
    x = tf.reshape(x, (int(x.shape[1]), int(x.shape[2])))

    for k in range(1, d):
        g = cores[k]
        g_mat = tf.reshape(g, (int(g.shape[0]), int(g.shape[1]) * int(g.shape[2])))
        x = x @ g_mat
        x = tf.reshape(x, (-1, int(g.shape[2])))

    out_shape = [int(g.shape[1]) for g in cores]
    return tf.reshape(x, tuple(out_shape))
