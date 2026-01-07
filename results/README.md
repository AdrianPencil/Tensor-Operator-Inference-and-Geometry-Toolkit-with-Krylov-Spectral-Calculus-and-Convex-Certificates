# Results

This folder stores **run artifacts** produced by pipelines, notebooks, and benchmarks.

The core repo remains math-first: results here are *outputs*, not implementation dependencies.

## Suggested contents

- small JSON summaries (loss curves, residual norms, timing summaries)
- CSV tables for microbench outputs
- simple PNG/PDF plots exported from notebooks (optional)

## Conventions

- keep artifacts small (commit-friendly)
- include metadata in each artifact:
  - git commit hash (if available)
  - platform (CPU/GPU), TF version
  - dtype, shapes, seed
  - time stamp
- avoid saving large raw tensors unless necessary; prefer derived summaries

## Baselines

See `baselines/` for reference outputs used to compare regressions in microbench or smoke pipelines.
