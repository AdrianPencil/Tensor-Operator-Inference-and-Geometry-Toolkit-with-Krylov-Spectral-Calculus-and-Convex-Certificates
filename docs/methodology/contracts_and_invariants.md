# Contracts and invariants

This repo encodes mathematical correctness as executable constraints.

## Core contracts

- **Adjoint pairing:** ⟨A x, y⟩ = ⟨x, A* y⟩
- **Finite outputs:** no NaN/Inf in intermediate states
- **Identity checks:** vec/unvec, Kronecker/Toeplitz consistency
- **Consistency checks:** dense reference comparisons on small problems

## Where enforced

- `tests/` for invariants
- `tig.compute.benchmarking_contracts` for reusable assertions
- hero pipelines specify which invariants should hold
