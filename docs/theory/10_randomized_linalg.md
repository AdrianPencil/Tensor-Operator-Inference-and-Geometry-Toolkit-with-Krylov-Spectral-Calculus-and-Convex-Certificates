# 10 - Randomized linear algebra

Random projections approximate subspaces.

Example: randomized range finder
Y = A Ω,  Q = orth(Y)

Then A ≈ Q(QᵀA).

Uses:
- low-rank approximations
- faster trace/logdet estimators
- sketching for conditioning diagnostics

In code: `tig.linalg.lowrank`, `tig.spectral.logdet_trace`, `tig.linalg.structured`.
