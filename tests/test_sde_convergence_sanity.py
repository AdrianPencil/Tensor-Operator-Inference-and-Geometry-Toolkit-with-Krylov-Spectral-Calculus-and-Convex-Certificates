"""
SDE sanity.

Checks Brownian increment statistics and a simple Euler–Maruyama diffusion scale.
"""

import tensorflow as tf

from tig.core.random import Rng
from tig.stochastic.ito import brownian_increments
from tig.stochastic.sde_solvers import SdeSpec, euler_maruyama


def test_brownian_increment_variance() -> None:
    rng = Rng(seed=0)
    dt = 1e-3
    n_steps = 2000
    shape = (128,)
    path = brownian_increments(dt=dt, n_steps=n_steps, shape=shape, rng=rng)

    dW = path.dW
    var_emp = tf.reduce_mean(dW * dW)
    assert abs(float(var_emp.numpy()) - dt) < 2e-4


def test_euler_maruyama_pure_brownian_scale() -> None:
    rng = Rng(seed=1)
    dt = 1e-3
    n_steps = 1000
    x0 = tf.zeros((256,), dtype=tf.float64)

    def drift(x: tf.Tensor, t: float) -> tf.Tensor:
        return tf.zeros_like(x)

    def diffusion(x: tf.Tensor, t: float) -> tf.Tensor:
        return tf.ones_like(x)

    traj = euler_maruyama(SdeSpec(drift=drift, diffusion=diffusion), x0=x0, t0=0.0, dt=dt, n_steps=n_steps, rng=rng)
    xT = traj[-1]
    var_emp = tf.reduce_mean(xT * xT)
    assert abs(float(var_emp.numpy()) - (n_steps * dt)) < 0.05
