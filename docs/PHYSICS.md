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
Half-space displacement kernels require the correct reduced modulus convention; verify point-load, cell-integrated and Hertz comparisons. The starter contains *only a general convolution utility*, not a contact kernel.

## Validation
1. Hertz contact radius and peak pressure with known analytical values.
2. FFT convolution against direct loops, including corners and non-square arrays.
3. Unit tests of cell-integrated elastic kernel (planned).
4. Reynolds manufactured solutions, mass conservation, mesh independence (planned).
5. Coupled load closure, positivity, residual decay, grid/domain studies (planned).
6. Reference film-thickness cases and GPU double precision parity (planned).

## Computational strategy
GPU: use PyTorch tensors for both NVIDIA CUDA and AMD ROCm. On ROCm, PyTorch also reports GPUs via `torch.cuda`. A single environment generally uses one GPU-enabled build (CUDA **or** ROCm), not both at once.
