# 24 - Optimal control as function-space optimization

Control is a function u(t) over a time interval.
Discrete representation: u_k ≈ u(t_k) piecewise-constant.

Objective:
min_u  J(u) = loss(traj(u), u)

Gradients computed via adjoint / reverse-mode:
∇_u J from unrolled dynamics + autodiff.

Robust variants use risk measures:
- E[loss]
- CVaR tail risk

In code: `tig.control.function_spaces`, `tig.control.adjoint_methods`, `tig.control.optimal_control`, `tig.control.robust_optim`.
