# 07 - Convex duality and proximal operators

## Fenchel conjugate

f*(y) = sup_x (⟨y,x⟩ - f(x))

Fenchel-Young:
f(x) + f*(y) ≥ ⟨y,x⟩

## Proximal operator

prox_{λf}(v) = argmin_x 0.5||x-v||² + λ f(x)

Key example:
f(x)=||x||₁ → soft thresholding

In code: `tig.opt.proximal`, `tig.convex.duality`, `tig.inverse.regularizers`.
