"""
Stiffness / stability proxies sanity.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.diffeq.stiff import spectral_radius_proxy, stiffness_ratio_proxy


def test_stiffness_ratio_ge_one_for_nonzero_spectrum() -> None:
    rng = Rng(seed=0)
    a = rng.normal((16, 16), dtype=tf.float64)
    val = stiffness_ratio_proxy(a, ridge=1e-12)
    assert float(val.numpy()) >= 0.0


def test_spectral_radius_proxy_nonnegative() -> None:
    rng = Rng(seed=1)
    a = rng.normal((16, 16), dtype=tf.float64)
    rho = spectral_radius_proxy(a)
    assert float(rho.numpy()) >= 0.0
