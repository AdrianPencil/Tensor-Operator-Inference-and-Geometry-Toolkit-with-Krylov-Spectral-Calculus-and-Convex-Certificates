# 03 - Fréchet derivatives

## Definition

F: X → Y between normed spaces is Fréchet differentiable at x if:
F(x+h) = F(x) + DF(x)[h] + o(||h||)

DF(x) is a bounded linear operator.

## JVP / VJP

- JVP: DF(x)[v]
- VJP: DF(x)^*[u] under chosen inner products

Adjoint identity:
⟨DF(x)[v], u⟩_Y = ⟨v, DF(x)^*[u]⟩_X

In TF, `GradientTape` provides reverse-mode (VJP) naturally; JVP can be built via forward-over-reverse patterns or explicit implementations.

In code: see `tig.inverse.forward_models` and `tig.uq.sensitivity`.
