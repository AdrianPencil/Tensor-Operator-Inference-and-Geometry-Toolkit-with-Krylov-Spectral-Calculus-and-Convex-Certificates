# Reproducibility

## Determinism

- use explicit RNG objects with fixed seeds
- store experiment configs in YAML
- keep tests deterministic (no hidden randomness)

## Artifacts

- results summaries are dicts that can be serialized
- hero configs define a canonical parameter set for each demonstration

## Where

- RNG: `tig.core.random.Rng`
- pipelines: `tig.workflows.pipelines`
- dataset generators: `tig.workflows.datasets`
