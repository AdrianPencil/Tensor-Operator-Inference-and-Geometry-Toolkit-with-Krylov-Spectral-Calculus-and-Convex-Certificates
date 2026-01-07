"""
Sensitivity kernels and Jacobian-operator views (math-first).

This module exposes matrix-free sensitivity primitives for forward models:
- directional sensitivity via JVP: J(x) v
- adjoint sensitivity via VJP: J(x)^T u
- small-dimension explicit Jacobian construction (optional, for diagnostics)

These are used by UQ, experimental design, and identifiability tooling.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import tensorflow as tf

from tig.core.random import Rng
from tig.core.types import TensorLike, as_float64
from tig.inverse.forward_models import ForwardModel

__all__ = [
    "SensitivityBatch",
    "jvp_batch",
    "vjp_batch",
    "explicit_jacobian_small",
    "local_sensitivity_norms",
]


@dataclass(frozen=True)
class SensitivityBatch:
    """
    Batch of sensitivities for multiple directions.
    """

    directions: tf.Tensor
    responses: tf.Tensor


def jvp_batch(model: ForwardModel, x: TensorLike, vs: TensorLike) -> SensitivityBatch:
    """
    Compute J(x) v_i for a batch of directions v_i stacked in the first axis.

    Parameters
    ----------
    model:
        ForwardModel with jvp.
    x:
        Point at which to linearize.
    vs:
        Batch of directions with shape (k, *x.shape).

    Returns
    -------
    SensitivityBatch
        directions=vs, responses stacked with shape (k, *F(x).shape).
    """
    xx = as_float64(x)
    vv = as_float64(vs)
    k = int(vv.shape[0])
    outs = []
    for i in range(k):
        outs.append(as_float64(model.jvp(xx, vv[i])))
    return SensitivityBatch(directions=vv, responses=tf.stack(outs, axis=0))


def vjp_batch(model: ForwardModel, x: TensorLike, us: TensorLike) -> SensitivityBatch:
    """
    Compute J(x)^T u_i for a batch of adjoint directions u_i.

    Parameters
    ----------
    us:
        Batch of adjoint directions with shape (k, *F(x).shape).

    Returns
    -------
    SensitivityBatch
        directions=us, responses stacked with shape (k, *x.shape).
    """
    xx = as_float64(x)
    uu = as_float64(us)
    k = int(uu.shape[0])
    outs = []
    for i in range(k):
        outs.append(as_float64(model.vjp(xx, uu[i])))
    return SensitivityBatch(directions=uu, responses=tf.stack(outs, axis=0))


def explicit_jacobian_small(model: ForwardModel, x: TensorLike) -> tf.Tensor:
    """
    Build an explicit Jacobian J for small problems by probing basis vectors with JVP.

    Notes
    -----
    This is diagnostic only. For large problems, use jvp/vjp operators.
    """
    xx = as_float64(x)
    x_flat = tf.reshape(xx, (-1,))
    n = int(x_flat.shape[0])

    fx = as_float64(model(xx))
    y_flat = tf.reshape(fx, (-1,))
    m = int(y_flat.shape[0])

    cols = []
    for j in range(n):
        e = tf.one_hot(j, n, dtype=tf.float64)
        v = tf.reshape(e, xx.shape)
        jv = as_float64(model.jvp(xx, v))
        cols.append(tf.reshape(jv, (-1,)))
    j_mat = tf.stack(cols, axis=1)
    if int(j_mat.shape[0]) != m:
        raise ValueError("Explicit Jacobian shape mismatch.")
    return j_mat


def local_sensitivity_norms(
    model: ForwardModel,
    x: TensorLike,
    n_probe: int = 16,
    rng: Optional[Rng] = None,
) -> tf.Tensor:
    """
    Probe local sensitivity magnitudes by sampling random directions v and returning ||Jv||_2.

    Returns
    -------
    tf.Tensor
        Vector of length n_probe containing norms.
    """
    if rng is None:
        rng = Rng(seed=0)

    xx = as_float64(x)
    norms = []
    for _ in range(int(n_probe)):
        v = rng.normal(xx.shape, dtype=tf.float64)
        jv = as_float64(model.jvp(xx, v))
        norms.append(tf.linalg.norm(tf.reshape(jv, (-1,))))
    return tf.stack(norms, axis=0)
