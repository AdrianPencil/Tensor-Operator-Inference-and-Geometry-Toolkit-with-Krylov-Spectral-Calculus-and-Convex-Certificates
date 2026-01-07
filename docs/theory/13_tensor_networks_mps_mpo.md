# 13 - Tensor networks: MPS/MPO

## MPS

A vector in (ℝ^d)^{⊗L} is represented by cores:
G^{(k)} ∈ ℝ^{r_{k-1}×d×r_k}

Contraction reconstructs the dense tensor, but algorithms avoid full reconstruction.

## MPO

Linear operators in tensor product spaces are represented by MPO cores.

In code: `tig.tensor_networks.mps`, `tig.tensor_networks.mpo`, `tig.tensor_networks.contraction`.
