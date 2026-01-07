# 01 - Tensor spaces

## Multilinear tensor product

Given vector spaces V₁,…,V_k over 𝔽, the tensor product space V₁ ⊗ … ⊗ V_k is defined by:

- a multilinear map ι: V₁×…×V_k → V₁⊗…⊗V_k
- universal property: for any multilinear f: V₁×…×V_k → W there exists a unique linear F such that f = F ∘ ι

## Coordinates

If dim(V_i)=n_i, a tensor T ∈ ⊗_{i=1}^k V_i has coordinates:

T_{a₁…a_k},  a_i ∈ {1,…,n_i}

In code: tensors are represented as multi-index arrays with shapes (n₁,…,n_k).

## Inner products

If each V_i has inner product, then ⊗ inherits Frobenius-type inner products:
⟨A,B⟩ = Σ_{indices} A_{…} B_{…}

Used for adjoints and VJP identities.
