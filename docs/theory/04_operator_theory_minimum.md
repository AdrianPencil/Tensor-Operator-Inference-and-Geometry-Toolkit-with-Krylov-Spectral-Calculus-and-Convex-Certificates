# 04 - Operator theory (minimum)

## Linear operators

A linear operator A: X → Y is implemented via:
- matvec: x ↦ A x
- rmatvec: y ↦ A* y

Adjoint pairing contract:
⟨A x, y⟩_Y = ⟨x, A* y⟩_X

## Norms (proxies)

- operator norm ||A||₂ (spectral norm) via SVD on dense cases
- conditioning κ(A) ≈ σ_max / σ_min

In code: `tig.core.operators`, `tig.core.norms`, `tig.viz.conditioning_plots`.
