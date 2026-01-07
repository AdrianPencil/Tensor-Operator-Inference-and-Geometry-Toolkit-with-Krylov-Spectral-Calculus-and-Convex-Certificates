# 18 - ODE/PDE/integro-diff, stiffness, multiscale

## ODE

x' = f(x,t)

## PDE operator framing

u_t = L u + … where L is a differential operator (Laplacian, advection).

Discrete view:
u' = A u + …

## Integro-diff

x'(t) = f(x,t) + ∫ K(t,s) x(s) ds

## Stiffness

Large eigenvalue spreads cause explicit steps to require tiny dt.
Diagnostics use spectral radius and eigenvalue ratio proxies.

In code: `tig.diffeq.odes`, `tig.diffeq.pdes`, `tig.diffeq.integro_diff`, `tig.diffeq.stiff`.
