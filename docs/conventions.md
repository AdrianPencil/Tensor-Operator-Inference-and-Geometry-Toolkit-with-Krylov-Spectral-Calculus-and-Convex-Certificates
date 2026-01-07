# Hero pipelines

Hero pipelines are small end-to-end demonstrations that connect theory → code → tests.

Each YAML file in `experiments/` describes a canonical task.

## List of hero configs

- `hero_autodiff.yaml`
  - JVP/VJP consistency via adjoint identity
- `hero_operator_solvers.yaml`
  - matrix-free CG on SPD systems, residual tracking
- `hero_spectral_expmv.yaml`
  - expmv vs dense expm reference on small problems
- `hero_inverse_phase.yaml`
  - linear inverse + MAP objective + regularization
- `hero_manifold_opt.yaml`
  - Stiefel optimization with retractions
- `hero_tensor_networks.yaml`
  - MPS/MPO contraction sanity vs dense reconstruction
- `hero_stochastic_filtering.yaml`
  - Kalman filter baseline (linear Gaussian)
- `hero_sdp_certificates.yaml`
  - SDP framing + PSD certificates + gap proxies
- `hero_kernels_greens_convolution.yaml`
  - kernels, Green operators, convolution adjoint checks
- `hero_stiff_coupled_dynamics.yaml`
  - stiffness proxies, coupled linear dynamics
- `hero_filter_functions_psd.yaml`
  - PSD + filter weights + Parseval sanity
- `hero_optimal_control_adjoint.yaml`
  - discrete control, adjoint gradient checks via autodiff

## Contracts each hero should satisfy

- finite outputs (no NaN/Inf)
- adjoint pairing where defined: ⟨A x, y⟩ = ⟨x, A* y⟩
- convergence or stability proxy decreases (when applicable)
- identity checks match analytic baselines (linear models, dense references)