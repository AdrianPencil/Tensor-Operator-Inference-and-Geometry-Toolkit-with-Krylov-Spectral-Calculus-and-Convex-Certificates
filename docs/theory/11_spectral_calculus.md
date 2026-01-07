# 11 - Spectral calculus

For diagonalizable A with eigenpairs (λ_i, v_i):
f(A) = V f(Λ) V^{-1}

Key operations:
- exp(tA) v (expmv) without dense expm
- resolvent (zI - A)^{-1}
- logdet(A), trace(f(A))

In code: `tig.spectral.expmv`, `tig.spectral.resolvent`, `tig.spectral.logdet_trace`.
