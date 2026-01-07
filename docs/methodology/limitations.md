# Limitations

- Many algorithms are intentionally minimal to preserve theory-first clarity.
- Dense SVD/eigendecompositions are used only for small diagnostics.
- Optional tooling (e.g., CVXPY for SDP solves) is not required for core functionality.
- The YAML hero configs are specs; a full YAML runner can be added as a workflow extension.
