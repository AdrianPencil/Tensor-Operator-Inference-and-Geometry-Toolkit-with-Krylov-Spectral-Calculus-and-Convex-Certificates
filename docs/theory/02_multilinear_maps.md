# 02 - Multilinear maps

## k-linear maps

A map f: V₁×…×V_k → W is multilinear if linear in each argument.

By tensor universality, multilinear maps ↔ linear maps on tensor products:
f(v₁,…,v_k) = F(v₁ ⊗ … ⊗ v_k)

## Contractions

Given T_{a₁…a_k} and U_{b₁…b_m}, contractions sum over repeated indices:
(T ⋅ U)_{…} = Σ_{i} T_{…i…} U_{…i…}

In code this is `einsum`-driven and aligns with `tig.core.einsum`.
