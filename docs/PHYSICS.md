# Physics model and numerical conventions

## Ultimate target
Steady, isothermal 2D point-contact EHL, extended stage-wise to pressure-dependent viscosity and density, cavitation, shear thinning and thermal effects. The initial foundation is *not yet an EHL solver*.

## Coordinates
Array shape `(ny, nx)`. `x` follows lubricant entrainment; `y` is transverse. Use SI units at interfaces and explicit grid spacing `dx`, `dy`. Center odd-sized grids about x=y=0 for analytical comparisons.

## Film thickness (planned)
`h(x,y) = h0 + x²/(2 Rx) + y²/(2 Ry) + delta_elastic(x,y)` with a clearly documented sign convention. Rigid approach / reference indentation conventions must be validated before coupling.

## Reynolds equation (planned)
Implement and test the incompressible, isoviscous steady Reynolds equation in conservative flux form with an upwind / appropriate convective treatment and explicit cavitation treatment. Specify entrainment vs individual surface speeds (factor-of-two conventions). Validate flux conservation, boundary conditions and load equilibrium.

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
4. Reynolds manufactured solutions, mass conservation, mesh independence (planned).
5. Coupled load closure, positivity, residual decay, grid/domain studies (planned).
6. Reference film-thickness cases, full-profile elasticity, and GPU
   film-height parity (planned).

## Computational strategy
The NumPy implementation is the CPU reference. The existing optional PyTorch
backend supports PyTorch's `cuda` device label for NVIDIA CUDA or AMD ROCm. A
separate optional native AMD HIP backend uses C++/HIP and hipFFT directly and
does not import PyTorch. It computes complex128 transforms on a zero-padded
full-convolution grid; inverse normalization is by the total padded grid size,
then the output is cropped from the kernel center. Host arrays remain
float64/complex128 and are transferred for each convolution; no device-resident
iterative solver is implemented. The native convolution has been checked
against NumPy and direct convolution on an RX 6600 XT (`gfx1032`), including
rectangular, asymmetric, edge and randomized pressure fields. Benchmarks and
limits are recorded in `docs/ROADMAP.md`; this validates only the convolution/
elastic-deformation path, not EHL coupling or Reynolds physics.
