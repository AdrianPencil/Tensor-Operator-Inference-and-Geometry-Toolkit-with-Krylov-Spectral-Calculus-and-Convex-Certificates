# Glossary

**Adjoint (A\*)**  
The unique operator satisfying ⟨A x, y⟩ = ⟨x, A\* y⟩ under a chosen inner product.

**JVP / VJP**  
- JVP: Jacobian-vector product J(x) v  
- VJP: vector-Jacobian product J(x)^T u  
Used for matrix-free sensitivity and gradients.

**Fréchet derivative**  
The linear map DF(x) approximating F(x+h) ≈ F(x) + DF(x)[h].

**Conditioning**  
Sensitivity of solutions to perturbations; often proxied by singular value ratios or operator norms.

**Identifiability**  
Whether parameters are uniquely recoverable from data under a model; often studied via rank and sensitivity.

**Prox operator**  
prox_{λR}(v) = argmin_x 0.5||x-v||^2 + λR(x). Central to convex optimization.

**Stiefel manifold**  
Set of matrices with orthonormal columns: St(n,p) = {X ∈ ℝ^{n×p} : XᵀX = 𝕀_p}.

**Grassmann manifold**  
Subspaces of dimension p in ℝ^n (equivalence classes of Stiefel points).

**MPS / MPO**  
Tensor network formats for vectors (MPS) and operators (MPO) enabling structured contraction.

**Green operator**  
Inverse-like operator for linear systems with boundary/interface constraints.

**Transfer function H(ω)**  
Frequency response mapping input spectrum to output spectrum for linear systems.

**PSD**  
Power spectral density, describing variance distribution across frequencies.
