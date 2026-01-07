# tensor-inference-geometry (tig)

A **math-first** library for tensor/operator geometry, inverse problems, conditioning, and matrix-manifold optimization.

Core philosophy:

- **Mathematics defines the API**: spaces, multilinear maps, adjoints, norms, manifolds, identifiability.
- **TensorFlow-first backend** for tensors, `einsum`, linear algebra, autodiff, and (optionally) GPU acceleration.
- **Tooling serves theory**: SciPy as reference solvers and sparse baselines; SymPy for symbolic checks; JAX/PyTorch as parity/autodiff oracles; NetworkX for tensor-network contraction graphs; PyMC/Qiskit as scoped demos.

## Install

Base (core math + TF backend):

```bash
pip install -e .
