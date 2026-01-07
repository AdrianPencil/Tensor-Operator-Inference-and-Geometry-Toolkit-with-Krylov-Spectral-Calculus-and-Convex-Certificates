# 17 - Boundary constraints as operators

Boundary/interface conditions are linear maps:

C u = d

Residual viewpoint:
r(u) = C u - d

This makes constraints compatible with:
- least squares penalties ||r(u)||²
- constrained solvers
- verification checks

In code: `tig.operators_kernels.boundary_constraints`, `tig.operators_kernels.interface_conditions`.
