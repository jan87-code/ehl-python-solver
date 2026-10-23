# Development roadmap

- [x] Python packaging, GitHub CPU CI, Codex guidance and optional GPU backend
- [x] Analytical Hertz reference and unit tests
- [x] FFT linear convolution parity with direct CPU reference (generic kernel)
- [ ] Half-space elasticity: singularity-aware cell-integrated DC-FFT kernel
- [ ] Dry-contact Hertz pressure discretization / displacement benchmark
- [ ] Nondimensional isoviscous-incompressible Reynolds residual and manufactured solutions
- [ ] Cavitation with explicit model and mass conservation verification
- [ ] Coupling of Reynolds + elasticity + film thickness + load control
- [ ] Stable continuation/load stepping and convergence / energy diagnostics
- [ ] Density- and pressure-dependent viscosity
- [ ] Shear thinning and thermal EHL when isothermal validation passes
- [ ] Optional CUDA / ROCm benchmarks, parity tests and profiling

Do not check off a box without reproducible evidence (tests or documented benchmark output).
