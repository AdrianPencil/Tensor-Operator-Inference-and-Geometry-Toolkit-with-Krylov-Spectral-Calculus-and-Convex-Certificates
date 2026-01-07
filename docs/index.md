# tensor-inference-geometry (tig)

A math-first library for **tensor inference** and **geometry-aware optimization**, with an emphasis on:

- tensor spaces and multilinear maps
- Fréchet derivatives, JVP/VJP, adjoint calculus
- matrix-free operator theory and Krylov solvers
- inverse problems, regularization, Bayesian framing
- matrix manifolds (Stiefel, Grassmann, fixed-rank)
- tensor networks (MPS/MPO), noncommutative operators
- stochastic processes (SDEs, filtering)
- convex duality, prox operators, SDP relaxations
- spectral calculus (expmv, resolvents, trace/logdet)
- compute contracts: vectorization identities, layout, complexity

Core implementation is **TensorFlow-first** (float64, vectorized). Optional hooks can be added for SciPy/JAX/PyTorch without changing the theory.

## Layout

- `src/tig/` - library code
- `docs/theory/` - math notes aligned with the code
- `experiments/` - YAML configs for “hero” pipelines
- `notebooks/` - narrative demos
- `tests/` - mathematical contracts and sanity checks

## What “math-first” means here

- operators are explicit objects (matvec, rmatvec, adjoint pairing)
- derivatives are framed as linear maps (JVP/VJP) before implementation details
- stability and identifiability appear as first-class diagnostics (conditioning, rank, sensitivity)
- tensor structure is preserved (einsum laws, Kronecker/Toeplitz, tensor networks) instead of flattening by default

## Where to start

- Quick start: `quickstart.md`
- Hero demos overview: `hero_pipelines.md`
- Theory spine: `theory/01_tensor_spaces.md` onward
- Glossary: `glossary.md`
