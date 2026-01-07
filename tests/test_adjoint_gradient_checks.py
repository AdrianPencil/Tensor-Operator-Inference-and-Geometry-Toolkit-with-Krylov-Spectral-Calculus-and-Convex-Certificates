"""
Adjoint gradient checks for control.

Compares TF reverse-mode gradient against finite differences for a small
discrete-time control problem.
"""

import tensorflow as tf

from tig.control.adjoint_methods import DiscreteDynamics, simulate_with_grad_u
from tig.core.random import Rng


def test_control_gradient_matches_finite_difference() -> None:
    rng = Rng(seed=0)
    dt = 0.1
    n_steps = 12

    a = tf.eye(3, dtype=tf.float64) * tf.cast(0.9, tf.float64)
    b = rng.normal((3, 2), dtype=tf.float64) * tf.cast(0.1, tf.float64)
    x_target = rng.normal((3,), dtype=tf.float64)

    def f(x: tf.Tensor, u: tf.Tensor, t: float) -> tf.Tensor:
        return (a @ tf.reshape(x, (-1, 1)))[:, 0] + (b @ tf.reshape(u, (-1, 1)))[:, 0]

    dyn = DiscreteDynamics(f=f)
    x0 = rng.normal((3,), dtype=tf.float64)
    u0 = rng.normal((n_steps, 2), dtype=tf.float64)

    def loss(traj: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
        xT = traj[-1]
        e = xT - x_target
        return tf.reduce_sum(e * e) + 1e-3 * tf.reduce_sum(u * u)

    L, g = simulate_with_grad_u(dyn=dyn, x0=x0, u0=u0, t0=0.0, dt=dt, loss=loss)

    eps = 1e-6
    idx = (3, 1)
    e_ij = tf.scatter_nd(indices=[[idx[0], idx[1]]], updates=[1.0], shape=u0.shape)
    u_plus = u0 + tf.cast(eps, tf.float64) * e_ij
    u_minus = u0 - tf.cast(eps, tf.float64) * e_ij

    Lp = simulate_with_grad_u(dyn=dyn, x0=x0, u0=u_plus, t0=0.0, dt=dt, loss=loss)[0]
    Lm = simulate_with_grad_u(dyn=dyn, x0=x0, u0=u_minus, t0=0.0, dt=dt, loss=loss)[0]

    fd = (tf.reshape(Lp, ()) - tf.reshape(Lm, ())) / tf.cast(2.0 * eps, tf.float64)
    ad = g[idx[0], idx[1]]

    rel = tf.abs(fd - ad) / (tf.abs(fd) + tf.abs(ad) + tf.cast(1e-15, tf.float64))
    assert float(rel.numpy()) < 1e-5
