# Development roadmap

- [x] Python packaging, GitHub CPU CI, Codex guidance and optional GPU backend
- [x] Analytical Hertz reference and unit tests
- [x] FFT linear convolution parity with direct CPU reference (generic kernel)
- [x] Half-space elasticity: singularity-aware cell-integrated DC-FFT kernel
- [x] Dry-contact Hertz pressure discretization / displacement benchmark
- [ ] Nondimensional isoviscous-incompressible Reynolds residual and manufactured solutions
- [ ] Cavitation with explicit model and mass conservation verification
- [ ] Coupling of Reynolds + elasticity + film thickness + load control
- [ ] Stable continuation/load stepping and convergence / energy diagnostics
- [ ] Density- and pressure-dependent viscosity
- [ ] Shear thinning and thermal EHL when isothermal validation passes
- [x] Optional native HIP/ROCm complex128 convolution backend, GPU parity tests and grid benchmarks
- [ ] Optional CUDA / PyTorch GPU benchmarks and profiling

Native HIP verification on AMD Radeon RX 6600 XT (`gfx1032`), TheRock
10.3.0a20261023, Python 3.12:

- `hipcc --offload-arch=gfx1032` build completed using the SDK-only build script.
- GPU regression tests against NumPy and direct `scipy.signal.convolve2d`:
  7 passed, covering impulse centering, asymmetric/even kernels, rectangular
  and boundary fields, random pressure arrays, and dimensional displacement.
- Benchmark medians (3 repetitions; GPU event compute excludes transfers;
  convolution wall includes transfers; elasticity wall also includes
  cell-integrated influence generation):

  | Grid | GPU execution (ms) | Convolution wall (ms) | Elasticity end-to-end (ms) | Max abs error (m) | Max relative error |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 129 x 129 | 0.427 | 41.555 | 61.313 | 2.12e-22 | 1.04e-15 |
  | 257 x 257 | 2.822 | 28.582 | 138.211 | 1.06e-21 | 2.35e-15 |
  | 513 x 513 | 16.298 | 46.454 | 507.612 | 2.33e-21 | 2.88e-15 |

The extension reuses the validated Python cell-integrated coefficients and
returns displacement in meters for pressure in pascals. It creates and frees
HIP buffers/plans per call, transfers host arrays on each call, and has no
resident-device API. No Reynolds equation, coupled EHL solver, load balance, or
GPU film-height validation is implemented; these timings/parity results do not
validate those planned components.

The adaptive selector benchmarks the current full elasticity call (including
coefficient generation and host transfers), and separately measures FFT
execution where the backend exposes device timing.

Adaptive selection check on the RX 6600 XT / i7-960 at a 129 x 129 pressure
grid (3 repetitions): NumPy full deformation took 46.945 ms median; HIP full
deformation took 68.511 ms, with HIP device execution at 0.455 ms. The selector
chose NumPy because end-to-end HIP time includes coefficient generation and
host/device transfers. At this size HIP is numerically operational, but its
transfer cost outweighs the GPU compute advantage. Benchmark timings vary;
cached medians are keyed to the grid, backend/runtime versions, hardware, and
workload configuration.

Do not check off a box without reproducible evidence (tests or documented benchmark output).
