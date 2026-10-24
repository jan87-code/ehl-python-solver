# Development roadmap

- [x] Python packaging, GitHub CPU CI, Codex guidance and optional GPU backend
- [x] Analytical Hertz reference and unit tests
- [x] FFT linear convolution parity with direct CPU reference (generic kernel)
- [x] Half-space elasticity: singularity-aware cell-integrated DC-FFT kernel
- [x] Dry-contact Hertz pressure discretization / displacement benchmark
- [x] Nondimensional isoviscous-incompressible Reynolds residual and manufactured solutions
- [x] Prescribed-film full-film Reynolds solver, parabolic point-contact verification, domain/grid sensitivity, and sparse-solver diagnostics
- [x] Prescribed-h0 weak Reynolds-elasticity coupling with compliance continuation and zero-compliance regression
- [x] Coupled pressure/film visualization, interpolated cross sections, convergence histories, and optional accepted-iteration PNG snapshots
- [ ] Cavitation with explicit model and mass conservation verification
  - A prescribed-film JFO active-set prototype and discrete wedge/conservation
    regressions exist in `src/ehl_solver/cavitation.py`; physical benchmark,
    front-continuity, starvation, and coupled verification remain incomplete.
- [ ] Coupling of Reynolds + elasticity + film thickness + physical load control
  - A signed full-film h0 root finder and bordered Newton-Krylov solver with
    dual line search/adaptive load stepping now converge on the documented
    weak numerical benchmark. This is mathematical verification only:
    moderate/strong coupling, arbitrary load branches, cavitation, and
    physical load validation remain incomplete.
- [ ] Stable continuation/load stepping and convergence / energy diagnostics
  - Adaptive Newton load steps, retry records, and Newton/continuation plots
    are implemented, but verified only over a short weak-coupling signed-load
    sequence; broad continuation robustness and energy diagnostics remain
    unfinished.
- [ ] Density- and pressure-dependent viscosity
- [ ] Shear thinning and thermal EHL when isothermal validation passes
- [ ] Optional CUDA / ROCm benchmarks, parity tests and profiling

Do not check off a box without reproducible evidence (tests or documented benchmark output).
