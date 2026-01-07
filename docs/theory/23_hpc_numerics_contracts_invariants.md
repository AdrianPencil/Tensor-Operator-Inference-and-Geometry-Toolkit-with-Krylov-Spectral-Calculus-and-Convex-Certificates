# 23 - HPC numerics: contracts and invariants

This project treats performance and correctness as mathematical objects:

- vectorization identities (vec, Kronecker laws)
- layout descriptors (shape, strides, contiguity)
- complexity proxies (flops, bytes, arithmetic intensity)
- invariants: adjoint pairing, finite checks, residual monotonicity

In code: `tig.compute.vectorization_identities`, `tig.compute.layout_memory`, `tig.compute.complexity`, `tig.compute.benchmarking_contracts`.
