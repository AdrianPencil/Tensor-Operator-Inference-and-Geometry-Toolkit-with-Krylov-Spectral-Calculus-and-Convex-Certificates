# 21 - Transfer functions and system identification

For LTI state-space:
x' = A x + B u
y  = C x

Transfer:
H(ω) = C (iωI - A)^{-1} B

Identification tasks:
- estimate parameters (A,B,C) from observed frequency response
- fit kernels or impulse responses

In code: `tig.fourier.transfer_functions`, `tig.operators_kernels.response`.
