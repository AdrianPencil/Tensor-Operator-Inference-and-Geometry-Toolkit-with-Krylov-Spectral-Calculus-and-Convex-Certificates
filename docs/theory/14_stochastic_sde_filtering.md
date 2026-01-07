# 14 - Stochastic processes, SDEs, filtering

## Itô SDE

dX_t = f(X_t,t) dt + g(X_t,t) dW_t

Euler–Maruyama:
X_{k+1} = X_k + f(X_k,t_k) dt + g(X_k,t_k) ΔW_k

## Filtering

Linear Gaussian model has Kalman filter with closed-form recursions.

In code: `tig.stochastic.ito`, `tig.stochastic.sde_solvers`, `tig.stochastic.filtering`.
