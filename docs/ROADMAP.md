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

## Exact continuation endpoint increment (2026-10-25)

Corrected pseudo-arclength termination: predictor crossings now use a
fixed-parameter endpoint corrector, and corrector-only crossings are
re-corrected before acceptance. Initial-target roots return without stepping.
Scalar increasing/decreasing, bounded-domain and curved-branch checks cover
these paths. The E*=10 MPa coupled regression reaches compliance exactly 1
and agrees with an independent fixed-compliance Newton solve; this remains
a signed mathematical benchmark. This fixes the target/boundary limitation
identified in the previous increment, without resolving the 1 MPa branch.
Final checks: `ruff check .` clean; `pytest -q` 133 passed, 4 optional
GPU skips, with BLAS/OpenMP threads limited to one.

Next development action: run a bounded grid/domain sensitivity study of the
1 MPa turning region using the analytic compliance column and exact endpoint
semantics. Compare residual roots, film feasibility, conditioning and tangent
reversal across meshes before drawing physical conclusions or changing
preconditioners. Continue to preserve the original staged HIP checkout.

## Grid/domain branch sensitivity increment (2026-10-25)

Added a bounded, reproducible 1 MPa branch study at one fixed signed load,
with direct-convolution elasticity and independently differenced face-flux
state audits. Three same-domain grids and one enlarged domain all detect
positive-film tangent reversals; largest sampled compliance factors are
0.182129572 (7x9), 0.179202822 (13x17), 0.178327919 (19x25) and
0.176400509 (17x21 expanded at baseline spacing). All 48 accepted states
pass independent Reynolds/load/coupling/edge/flux checks. See PHYSICS.md
for the physical inputs, negative target load, measured residuals and run
recipe. Attempt states, metadata and optional plots are preserved in new
untracked result directories. No existing baseline was overwritten.
Verification: `ruff check .` clean; `pytest -q` 140 passed, 4 optional
GPU skips; the four-case benchmark with all plots completed.

The observed mesh and domain dependence precludes claiming a converged
fold or physical instability. Next action: quantify the exact gamma/E*
compliance scaling before treating the 100 kPa case as a distinct failure;
then localize tangent reversals and extend the grid/domain study with a
common dimensional load. Strong coupling, physical load closure, and coupled
cavitation remain unfinished.

## Modulus-invariant compliance scaling increment (2026-10-25)

Verified the exact gamma/E* dependence of the current linear-elastic
full-film equations at fixed load and scales, including state fields,
residuals, state/parameter derivatives, bordered corrections and separately
converged Newton solutions. The 100 kPa small-grid branch can be followed
through a tangent reversal near gamma=0.0179; earlier rejected 0.05/0.1
trials were beyond the corresponding 1 MPa turning region.

Added an optional normalized-compliance arclength metric with SI reference
modulus, preserving default behavior and actual gamma in returned states.
Exact-fold and coupled regressions cover branch tracking, tangent orientation,
strict feasibility, exact endpoints and invalid weights. With E_ref=1 MPa,
the 13x17 1 MPa and 100 kPa runs match all 12 accepted state vectors to
6.67e-16 and normalized gammas to 1.95e-16. Independent state audits pass;
maximum sampled gammas are 0.179202822 and 0.017920282. Neither run reaches
its requested full compliance. See PHYSICS.md for reproduction and limits.
Verification: `ruff check .` clean; `pytest -q` 158 passed, 4 optional
GPU skips. Both default-metric and normalized modulus studies completed with
independent audits and plots; single-thread resource limits were retained.

Next action: localize the tangent reversal using the normalized parameter,
then quantify grid/domain changes with a fixed signed load and a tighter
sampling/localization error. Distinguish discretization dependence from
branch-conditioning and globalization failures before changing the primary
Newton preconditioner. Physical cavitation/load coupling and independent
film-height validation remain unimplemented.

## Bracketed turn localization increment (2026-10-25)

Implemented bounded localization using the existing pseudo-arclength
corrector, strict feasibility/Armijo checks, tangent-sign bracketing and a
contraction guard. Both bracket and normalized parameter-tangent tolerances
are required. Failed predictors, correctors, noncontracting probes and
budget exhaustion preserve the last bracket and full attempted history.
Analytical quadratic/cubic turns, reparameterization, failure recovery and
an independently audited coupled-state regression cover the implementation.

The fixed-load four-case study now localizes gamma at 0.182176253504 (7x9),
0.179225816365 (13x17), 0.178395632035 (19x25), and 0.176425633616 (expanded
17x21 at baseline spacing). All 64 refinement probes pass independent
Reynolds/load/elasticity/film/boundary/flux checks. All final weighted brackets
are below 7.64e-7 and normalized tangents below 4.38e-7. A tighter baseline
bracket changes gamma by only 1.67e-13. Reproducible settings, numerical
limits, timings and plots are described in PHYSICS.md; result directories
were newly created and remain untracked.
Verification: `ruff check .` clean; `pytest -q` 170 passed, 4 optional
GPU skips. The four-case localized study and tighter-baseline restart completed.

The mesh/domain shifts remain larger than continuation-sampling corrections.
No continuum fold or physical EHL instability is established; the model still
has signed pressure and a negative target load. Full-compliance convergence
and physical load control remain unresolved, and those boxes stay open.

Next development action: independently test nullity, transversality and
curvature at the localized discrete turn before labeling it a simple fold.
Use the retained brackets for such diagnostics rather than repeating the
expensive failed continuation. Then prioritize a positive-load prescribed-film
cavitation/elasticity benchmark; the negative signed-load stress branch cannot
validate nonnegative-pressure physical EHL. Preserve the original staged HIP
checkout and keep CPU verification ahead of GPU work.
