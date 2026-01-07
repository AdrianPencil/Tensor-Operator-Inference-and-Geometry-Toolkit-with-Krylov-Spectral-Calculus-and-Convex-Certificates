# 15 - SDP relaxations and certificates

## SDP primal

min ⟨C,X⟩
s.t. ⟨A_i,X⟩ = b_i
     X ⪰ 0

Relaxations appear when lifting nonconvex problems to convex constraints.

## Certificates

- PSD check: eigenvalues ≥ 0 (up to tolerance)
- primal-dual gap as optimality proxy

In code: `tig.convex.sdp_relaxations`, `tig.convex.certificates`.
