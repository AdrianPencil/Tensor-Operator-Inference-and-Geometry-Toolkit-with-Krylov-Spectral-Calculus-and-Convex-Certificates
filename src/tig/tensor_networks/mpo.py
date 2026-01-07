"""
Matrix Product Operators (MPO) - minimal.

An MPO represents an operator on a chain via cores:
W_k of shape (r_{k-1}, d_in, d_out, r_k).

This file provides:
- MPO container
- apply to an MPS by local contraction (kept compact, real case)
"""

from dataclasses import dataclass
from typing import List

import tensorflow as tf

from tig.core.types import as_float64
from tig.tensor_networks.mps import MPS

__all__ = ["MPO", "mpo_apply_mps"]


@dataclass(frozen=True)
class MPO:
    """
    Matrix product operator cores.
    """

    cores: List[tf.Tensor]


def mpo_apply_mps(mpo: MPO, mps: MPS) -> MPS:
    """
    Apply an MPO to an MPS, producing a new MPS with enlarged bond dimensions.
    """
    wc = [as_float64(w) for w in mpo.cores]
    gc = [as_float64(g) for g in mps.cores]
    if len(wc) != len(gc):
        raise ValueError("MPO and MPS lengths must match.")

    out_cores: List[tf.Tensor] = []
    for w, g in zip(wc, gc):
        out = tf.einsum("aijb, cjd -> ac i d b", w, g)
        out = tf.reshape(out, (int(out.shape[0]), int(out.shape[2]), int(out.shape[3])))
        out_cores.append(out)

    return MPS(cores=out_cores)
