# 06 - Regularization and Bayes

Regularization corresponds to priors:

- L2: R(x)=0.5||x||² ↔ Gaussian prior
- L1: R(x)=||x||₁ ↔ Laplace prior
- nuclear norm: ||X||_* ↔ low-rank bias

Posterior:
p(x|y) ∝ p(y|x) p(x)

Approximate inference may use:
- Laplace approximation (Gaussian around MAP)
- variational proxies (ELBO-style)

In code: `tig.inverse.bayes`, `tig.inverse.regularizers`, `tig.convex.nuclear_norm`.
