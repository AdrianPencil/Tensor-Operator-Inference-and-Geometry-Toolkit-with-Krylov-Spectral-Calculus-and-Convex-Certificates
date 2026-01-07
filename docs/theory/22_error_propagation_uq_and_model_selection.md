# 22 - Error propagation, UQ, model selection

## Delta method

Cov_y ≈ J Cov_x Jᵀ

Scalar variance:
Var(y) ≈ gᵀ Cov_x g

## Sensitivity

Measure ||J v|| for random v to detect ill-conditioned directions.

## Model selection

- NLL, AIC, BIC (residual models)
- ELBO proxy (variational)

In code: `tig.uq.sensitivity`, `tig.uq.error_propagation`, `tig.uq.model_selection`.
