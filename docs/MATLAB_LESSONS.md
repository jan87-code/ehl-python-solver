# Prior MATLAB research — lessons, not source-of-truth results

The previous MATLAB solver used a DC-FFT elastic deformation kernel, nonlinear Reynolds coupling, Hertz initialization, load stepping, damping/relaxation, and attempts at dimensionless governing equations. Earlier experiments showed strong sensitivity to the kernel quadrature and convergence control.

Reference *historical* test parameters (not assumed validated EHL results):
- Full-load case: Q = 1525.5 N, E' = 226 GPa, equivalent radius R = 0.015 m, entrainment speed u = 0.85 m/s, eta0 = 0.04 Pa s, alpha_p = 2.21e-8 Pa^-1.
- Reported analytical dry Hertz check: a ≈ 0.533534 mm; p0 ≈ 2558.759 MPa, computed with the relevant reduced-modulus convention.
- Smaller experimental case: Q = 5 N, a ≈ 0.079253 mm, p0 ≈ 380.086 MPa.
- Grid exploration: odd 513 × 513 arrays and dx ≈ 3.126176 micrometers.
- Kernel audit: a cell-integrated kernel changed adjacent-cell coefficients by several percent relative to point samples, while FFT and direct convolution agreed at ~1e-15 relative Frobenius error for the audited kernel.
- Previous solver experienced erratic load residuals, non-convergence, and indexing / shape mismatches. Do not reproduce questionable code by rote.

Crucial: clarify the definition of E', E* and equivalent radius before translating formulas; factor-of-two differences are common. Compare independent analytical results with verified reference data.

The newer user-supplied N33 pressure-only continuation checkpoint has now been
independently evaluated: see [MATLAB_REFERENCE.md](MATLAB_REFERENCE.md) for the
50 N case, its passing reference gates, operator differences and reproduction
command. It supersedes the historical parameters above for immediate parity work.
