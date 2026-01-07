"""
Adjoint/JVP/VJP view for control gradients (minimal).

We compute gradients of a loss functional depending on a simulated trajectory
by unrolling the dynamics and using TensorFlow reverse-mode autodiff.

This keeps the theory explicit while avoiding a large bespoke adjoint framework.
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import tensorflow as tf

from tig.core.types import TensorLike, as_float64

__all__ = ["DiscreteDynamics", "simulate", "simulate_with_grad_u"]


@dataclass(frozen=True)
class DiscreteDynamics:
    """
    Discrete-time dynamics: x_{k+1} = f(x_k, u_k, t_k).
    """

    f: Callable[[tf.Tensor, tf.Tensor, float], tf.Tensor]


def simulate(
    dyn: DiscreteDynamics,
    x0: TensorLike,
    u: TensorLike,
    t0: float,
    dt: float,
) -> tf.Tensor:
    """
    Simulate trajectory for controls u of shape (n_steps, *u_shape).

    Returns
    -------
    tf.Tensor
        Trajectory of shape (n_steps+1, *x_shape).
    """
    xx = as_float64(x0)
    uu = as_float64(u)
    n_steps = int(uu.shape[0])

    traj = [xx]
    t = float(t0)
    for k in range(n_steps):
        xx = as_float64(dyn.f(xx, uu[k], t))
        traj.append(xx)
        t += float(dt)

    return tf.stack(traj, axis=0)


def simulate_with_grad_u(
    dyn: DiscreteDynamics,
    x0: TensorLike,
    u0: TensorLike,
    t0: float,
    dt: float,
    loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute loss(traj, u) and gradient w.r.t. control array u.

    loss(traj, u) must return a scalar tensor.
    """
    x_init = as_float64(x0)
    u_init = as_float64(u0)

    with tf.GradientTape() as tape:
        tape.watch(u_init)
        traj = simulate(dyn=dyn, x0=x_init, u=u_init, t0=float(t0), dt=float(dt))
        L = tf.reshape(as_float64(loss(traj, u_init)), ())

    g_u = tape.gradient(L, u_init)
    if g_u is None:
        g_u = tf.zeros_like(u_init)

    return L, as_float64(g_u)
