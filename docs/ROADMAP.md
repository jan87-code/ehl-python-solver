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
- [x] Prescribed-film JFO cavitation prototype with wedge mass-conservation,
  front-refinement and uniform inlet-starvation verification
  - Regressions cover local/global flux balance, the full-film limit, repeated
    rupture/reformation fronts on refined smooth wedges, and reduced inlet
    supply. This is numerical verification of a fixed-film model, not
    experimental validation; arbitrary front behavior, feed geometry, and
    elastic/load-controlled cavitation remain planned.
- [ ] Coupling of Reynolds + elasticity + film thickness + physical load control
  - A signed full-film h0 root finder and bordered Newton-Krylov solver with
    an analytic matrix-free discrete JVP, Schur correction, verified sparse
    preconditioners, directional-derivative Armijo line search, and adaptive
    load stepping converge on weak and one moderate numerical benchmark.
    The 1 MPa stress case still stalls with a substantial Reynolds residual,
    and the 100 kPa initial state requires an explicit offset adjustment just
    to become feasible. This is mathematical verification only: strong
    coupling, arbitrary load branches, cavitation, and physical load
    validation remain incomplete.
- [ ] Stable continuation/load stepping and convergence / energy diagnostics
  - Adaptive Newton load steps, retry records, and Newton/continuation plots
    are implemented and verified over short weak/moderate signed-load cases;
    frozen-film ILU/Jacobi/identity Krylov comparisons and failed-state
    diagnostics are available. Broad continuation robustness and energy
    diagnostics remain unfinished.
- [ ] Density- and pressure-dependent viscosity
- [ ] Shear thinning and thermal EHL when isothermal validation passes
- [ ] Optional CUDA / ROCm benchmarks, parity tests and profiling

Do not check off a box without reproducible evidence (tests or documented benchmark output).
