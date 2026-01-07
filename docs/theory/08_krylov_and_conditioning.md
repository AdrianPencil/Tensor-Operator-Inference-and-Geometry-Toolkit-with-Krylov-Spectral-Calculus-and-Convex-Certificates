# 08 - Krylov methods and conditioning

Solve A x = b using Krylov subspaces:
K_k(A,b) = span{b, A b, …, A^{k-1} b}

For SPD:
- CG converges with rate controlled by κ(A)

Preconditioning:
Solve (M^{-1}A) x = M^{-1}b to improve conditioning.

In code: `tig.linalg.krylov`, `tig.linalg.precond`, `tig.viz.conditioning_plots`.
