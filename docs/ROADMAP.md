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
    preconditioners, directional-derivative Armijo line search with bounded
    tighter-Krylov retries, and adaptive load stepping converge on the weak
    and moderate numerical benchmarks. A residual-based Jacobian audit and
    small-grid spectrum analysis find a near-turning region in the exploratory
    1 MPa branch. An experimental dense pseudo-arclength corrector follows the
    branch through a shallow local maximum, but advances maximum compliance
    only from 0.179175 to 0.179191 and does not reach full compliance. A bounded
    100 kPa probe still cannot accept compliance 0.05 from the rigid reference,
    while making the full-compliance initial film feasible requires a large
    explicit h0 raise. This remains a mathematical verification model: strong
    coupling, arbitrary load branches, cavitation, and physical load validation
    are incomplete.
- [ ] Stable continuation/load stepping and convergence / energy diagnostics
  - Adaptive Newton load steps, retry records, and Newton/continuation plots
    are implemented and verified over short weak/moderate signed-load cases;
    frozen-film ILU/Jacobi/identity Krylov comparisons and failed-state
    diagnostics are available. Small-grid Jacobian conditioning and
    pseudo-arclength branch diagnostics/plots are experimental and do not
    resolve strong coupling. Broad continuation robustness and energy
    diagnostics remain unfinished.
- [ ] Density- and pressure-dependent viscosity
- [ ] Shear thinning and thermal EHL when isothermal validation passes
- [ ] Optional CUDA / ROCm benchmarks, parity tests and profiling

Do not check off a box without reproducible evidence (tests or documented benchmark output).

Strong-coupling follow-up: fixed-load adaptive compliance continuation now
retains rejected trials and the last converged state, with independent
acceptance checks and a reproducible `run_compliance_continuation.py` probe.
The probe records per-stage nonlinear, load, Krylov, line-search, and
bordered-correction diagnostics. Armijo failure retries a correction only
when its independently measured true linear residual exceeds the configured
accuracy target. The 100 MPa and 10 MPa cases reach full compliance; the
1 MPa branch remains stalled near factor 0.179175 and the bounded 100 kPa
probe fails before factor 0.1. The new `run_pseudo_arclength_diagnostic.py`
audits the small-grid Jacobian, compares exact Schur and bordered corrections,
measures spectra along the accepted 1 MPa branch, and exercises a dense
pseudo-arclength corrector. The latter detects a tangent reversal at
gamma `0.179191378` but reaches only that maximum before following the branch
back down; its final gamma after 20 steps is `0.168947605`. ILU remains the
most effective of the tested frozen-film preconditioners at gamma `0.175`
(14 Krylov iterations versus 82 for Jacobi and 134 without preconditioning).
These diagnostics do not establish a physical fold or full-compliance
solution. Neither strong coupling nor physical load control is checked off
by this increment.


## Autonomous development increment (2026-10-25)

Audited `develop` at `0e8a44c` in an isolated worktree because the original
`feature/hip-backend` checkout contains pre-existing staged changes. Those
files are preserved. The current priority remains conditioning and robust
continuation of the signed full-film mathematical branch, before physical
load/cavitation coupling or GPU work. Baseline CPU verification: 123 passed,
4 optional GPU tests skipped.

Implemented an analytic compliance column and augmented state/compliance JVP,
and replaced the pseudo-arclength finite-difference parameter column.
Independent endpoint/interior residual-difference tests, a joint directional
check, and a near-film-floor regression pass. Added Copilot guidance pointing
to AGENTS.md rather than duplicating its requirements. Final checks: `ruff check .` clean; `pytest -q` 128 passed, 4 optional
GPU skips, with BLAS/OpenMP threads limited to one. No strong-compliance
or physical-validation milestone is marked complete.

Next action: audit pseudo-arclength acceptance and termination at compliance
bounds, then run a bounded small-grid continuation comparison. In particular,
target crossing currently terminates beyond the requested parameter rather
than solving exactly at it; gamma=1 is also a feasibility boundary. Preserve
failed-state diagnostics and distinguish reaching a target from crossing it.
