# Physics model and numerical conventions

## Ultimate target
Steady, isothermal 2D point-contact EHL, extended stage-wise to pressure-dependent viscosity and density, cavitation, shear thinning and thermal effects. The initial foundation is *not yet an EHL solver*.

## Coordinates
Array shape `(ny, nx)`. `x` follows lubricant entrainment; `y` is transverse. Use SI units at interfaces and explicit grid spacing `dx`, `dy`. Center odd-sized grids about x=y=0 for analytical comparisons.

## Film thickness (planned)
`h(x,y) = h0 + x²/(2 Rx) + y²/(2 Ry) + delta_elastic(x,y)` with a clearly documented sign convention. Rigid approach / reference indentation conventions must be validated before coupling.

## Reynolds residual (implemented foundation)
For steady, incompressible, isoviscous flow, the dimensional flux is
`q = -h**3 / (12 eta0) * grad(p) + Ue * h * e_x`, where
`Ue = (u1 + u2) / 2` is the mean of the two surface velocities in the
entrainment direction. Thus
`div(h**3 grad(p)) - 12 eta0 Ue * dh/dx = 0`.

The implemented residual uses `X = x/L`, `Y = y/L`, `H = h/h_ref`, and
`P = p/p_ref`. Its dimensionless form is
`div(H**3 grad(P)) - Lambda * dH/dX = 0`, with
`Lambda = 12 eta0 Ue L / (h_ref**2 p_ref)`. Here `eta0` is in Pa s,
`Ue` in m/s, `L` and `h_ref` in m, and `p_ref` in Pa. `Lambda` is
nonnegative for the documented positive entrainment direction. If the
entrainment speed is instead defined as `u1 + u2`, the coefficient is
`6 eta0 (u1 + u2) L / (h_ref**2 p_ref)`; do not mix these conventions.

`reynolds_residual` in `src/ehl_solver/reynolds.py` computes a conservative
finite-volume divergence on cell-centered arrays of shape `(ny, nx)`, with x
along entrainment and y transverse. `dx` and `dy` are dimensionless spacings;
the output contains only interior cells, shape `(ny - 2, nx - 2)`. Face
values of `H**3` use arithmetic interpolation. The optional dimensionless
source `S` forms the manufactured equation
`div(H**3 grad(P)) - Lambda * dH/dX - S = 0`; it is for discretization tests.
The routine does not impose boundary conditions or solve for pressure. This
validated residual does not implement cavitation, coupling, or load control
and is not a complete Reynolds or EHL solver.

## Elastic deformation
For combined elastic half-spaces, the normal displacement is
`w(x,y) = 1/(pi E*) integral[p(x',y') / r] dA'`, where
`r = sqrt((x-x')**2 + (y-y')**2)` and
`1/E* = sum((1 - nu_i**2) / E_i)`. This is the same `E_star` convention
used by the Hertz reference, not the alternative modulus that is twice as
large.

The implemented cell-integrated influence coefficient integrates `1/r` over
each rectangular pressure cell and evaluates displacement at cell centers.
The singular central cell is integrated analytically, not sampled at its
singular center. The resulting odd-sized lag kernel has axes `(ny_lag,
nx_lag)` with x along entrainment and y transverse; its center is the
zero-offset cell. Convolution uses zero-padded linear FFTs (no wraparound),
and its coefficients have units m/Pa, so a pressure field in Pa produces
displacement in m. Pressure values are piecewise constant within their
cells. The current deformation benchmark discretizes a dry Hertz pressure
field and checks convergence of its center displacement to the Hertz
approach `a**2/R` under grid refinement. It does not yet compare the full
displacement profile. This elastic component is validated independently and
does not constitute a coupled EHL solver.

## Validation
1. Hertz contact radius and peak pressure with known analytical values.
2. FFT convolution against direct loops, including corners and non-square arrays.
3. Cell-integrated kernel tests, direct-convolution parity, and Hertz center-displacement grid refinement.
4. Reynolds manufactured residual convergence (validated); full pressure
   solution, boundary-condition, mass-conservation and mesh-independence
   benchmarks remain planned.
5. Coupled load closure, positivity, residual decay, grid/domain studies (planned).
6. Reference film-thickness cases and GPU double precision parity (planned).

## Computational strategy
GPU: use PyTorch tensors for both NVIDIA CUDA and AMD ROCm. On ROCm, PyTorch also reports GPUs via `torch.cuda`. A single environment generally uses one GPU-enabled build (CUDA **or** ROCm), not both at once.
