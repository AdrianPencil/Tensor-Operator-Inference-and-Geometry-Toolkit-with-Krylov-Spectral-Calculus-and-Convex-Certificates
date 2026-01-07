"""
Perturbation viewpoints (minimal).

Perturbation theory enters throughout:
- conditioning and sensitivity
- identifiability
- operator calculus

This file provides small, reusable sensitivity quantities as TF computations.
"""

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["relative_perturbation", "condition_number_2"]


def relative_perturbation(a: TensorLike, da: TensorLike) -> tf.Tensor:
    """
    Relative perturbation magnitude ||dA||_F / ||A||_F.
    """
    aa = as_float64(a)
    dd = as_float64(da)
    num = tf.linalg.norm(dd)
    den = tf.linalg.norm(aa)
    return num / den


def condition_number_2(a: TensorLike) -> tf.Tensor:
    """
    Spectral condition number kappa_2(A) = sigma_max / sigma_min for full-rank A.
    """
    aa = as_float64(a)
    s = tf.linalg.svd(aa, compute_uv=False)
    smax = tf.reduce_max(s)
    smin = tf.reduce_min(s)
    return smax / smin
