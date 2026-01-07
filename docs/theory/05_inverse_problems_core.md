# 05 - Inverse problems (core)

We observe y and want x with model:
y = F(x) + ε

## MAP / variational view

MAP estimate solves:
x̂ = argmin_x  (1/2σ²)||F(x)-y||² + R(x)

R is a regularizer (prior).

## Linear case

If F(x)=A x:
min_x  0.5||A x - y||² + λ R(x)

Matrix-free methods use A matvec and A* rmatvec.

In code: `tig.inverse.solvers`, `tig.inverse.regularizers`, `tig.linalg.krylov`.
