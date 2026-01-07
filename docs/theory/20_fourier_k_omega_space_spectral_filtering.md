# 20 - Fourier (k, ω) space and spectral filtering

Fourier transform maps convolution ↔ multiplication.

- frequency ω = 2π f
- wavenumber k = 2π/λ

Filters are ω-weights applied to PSDs:
Var ≈ ∫ S(ω) F(ω) dω

In code: `tig.fourier.transforms`, `tig.fourier.k_omega_space`, `tig.fourier.filter_functions`, `tig.fourier.spectral_density`.
