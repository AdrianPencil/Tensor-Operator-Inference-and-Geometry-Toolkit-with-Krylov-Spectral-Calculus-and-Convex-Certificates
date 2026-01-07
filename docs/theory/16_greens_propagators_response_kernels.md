# 16 - Green operators, propagators, response kernels

## Green operator

Solve L u = f with boundary constraints.
Discrete Green operator G ≈ (L + ϵI)^{-1} so u = G f.

## Propagators

For x' = A x:
x(t) = exp(tA) x(0)

## Response

Frequency response for LTI:
H(ω) = C (iωI - A)^{-1} B

In code: `tig.operators_kernels.greens`, `tig.operators_kernels.propagators`, `tig.operators_kernels.response`.
