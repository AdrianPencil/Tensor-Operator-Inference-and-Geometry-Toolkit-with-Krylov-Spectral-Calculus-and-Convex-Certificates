# Numerical stability

## Principles

- keep operations in float64 where stability matters
- treat linear maps as operators (avoid forming dense matrices unless small)
- monitor conditioning proxies (SVD, norm bounds)
- prefer stable factorizations (QR retractions, SPD CG assumptions)

## Typical failure modes

- catastrophic cancellation in spectral / resolvent expressions
- poor conditioning in inverse problems (flat directions)
- stiffness in time stepping (dt constraints)

## Diagnostics in code

- conditioning summaries: `tig.viz.conditioning_plots`
- stiffness proxies: `tig.diffeq.stiff`
- sensitivity probes: `tig.uq.sensitivity`
