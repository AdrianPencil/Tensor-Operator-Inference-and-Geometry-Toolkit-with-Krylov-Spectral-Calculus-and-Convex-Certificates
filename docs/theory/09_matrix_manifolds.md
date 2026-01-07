# 09 - Matrix manifolds

## Stiefel

St(n,p) = {X ∈ ℝ^{n×p} : XᵀX = 𝕀_p}

Tangent vectors U satisfy:
XᵀU + UᵀX = 0

Retractions map X + U back onto the manifold, e.g. QR-based.

## Grassmann

Gr(n,p) identifies subspaces: X ~ XQ for Q ∈ O(p).

In code: `tig.geometry.stiefel`, `tig.geometry.grassmann`, `tig.geometry.metrics`.
