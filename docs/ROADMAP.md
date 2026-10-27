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
    experimental comparison remain planned. Separate coupled JFO/elasticity
    and positive-load numerical benchmarks are now recorded below.
- [ ] Coupling of Reynolds + elasticity + film thickness + physical load control
  - Separate monolithic JFO pressure/content/offset coupling now closes positive
    force on limited isoviscous numerical benchmarks. Independent physical
    film-height validation and broad domain/material robustness remain open.
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
- [x] Pressure-dependent density and viscosity: verified SI laws, conservative
  prescribed-film JFO and initial uniform elastic numerical cases.
  - Published classical-EHL agreement, resolved mesh accuracy and variable-
    property positive-load control remain open; see the 2026-10-27 milestones.
- [ ] Shear thinning and thermal EHL when isothermal validation passes
- [ ] Optional CUDA / ROCm benchmarks, parity tests and profiling
  - [x] Explicit PyTorch CUDA cached DC-FFT elasticity, full-film Reynolds
    residual and rigid film geometry with float64 CPU/direct/Hertz/flux checks.
  - [x] Device-resident signed full-film residual and complete analytic JVP
    evaluation audit, including compliance and force row; CPU solver unchanged.
  - [x] Synchronized 33/65/129/257/513 CUDA timings, transfers and PyTorch
    memory peaks on the Windows RTX A2000 Laptop GPU.
  - [x] Resident float64 GMRES, verified frozen-film surrogate/Jacobi,
    Schur/full-bordered recovery and complete signed full-film Newton/load
    stepping on documented mathematical cases, including 513x513 execution.
  - [ ] Device JFO/complementarity and physically validated GPU EHL;
    broad strong-coupling, ROCm coverage and Linux GPU execution remain open.

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

## Discrete turn nondegeneracy audit increment (2026-10-25)

Added a saved-root diagnostic for singular-value isolation, parameter
transversality and quadratic branch curvature. It checks the complete
analytic Jacobian against residual finite differences and tests second
residual differences at three step sizes. Scalar and multi-state analytical
regressions include degenerate counterexamples and failed feasibility.
The coupled integration test replays a saved root without repeating
continuation. Existing callback construction is shared with the benchmark.

All four localized 1 MPa roots pass the documented numerical evidence
thresholds: smallest/next singular ratios 5.03e-8 to 1.30e-7, normalized
transversality 0.307 to 0.347, stable predicted normalized parameter
curvatures -1.078 to -1.285, and analytic/FD Jacobian relative errors below
1.40e-11. These support nondegenerate discrete local maxima, not a continuum
fold proof or a physical EHL instability. Root/field audit tolerances remain
explicit, and full-compliance/physical-load milestones stay open.
Verification: `ruff check .` clean; `pytest -q` 176 passed, 4 optional
GPU skips. All four saved-root condition audits completed successfully using
single-thread CPU execution.

Next development action: stop treating the signed negative-load branch as a
physical target. Establish a positive-load prescribed-offset cavitation /
elasticity benchmark, beginning with zero-compliance and weak-compliance
limits, independently checked mass/complementarity/film residuals, and
feasibility. Retain the signed Newton, nested and Picard references; integrate
load control only after that coupled cavitation benchmark is verified.

## Positive-load prescribed-offset JFO/elasticity reference (2026-10-25)

Implemented and benchmarked a separate CPU reference coupling the existing
JFO active-set solver and half-space elasticity at prescribed h0. Only the
film iterate is relaxed; final pressure/saturation pairs retain
complementarity. Acceptance includes mass conservation on the exact
pressure-coupled film, feasibility, complementarity and update defects.
Independent direct-convolution and opposing-face flux audits pass. Failure
exceptions retain history and last candidates without claiming convergence.

Zero compliance matches the fixed-film JFO solver exactly. Small cases at
100, 10 and 1 MPa pass; the 10 MPa benchmark has positive force ranging from
0.00221020 N to 0.00235702 N on 9x13 through 25x37 grids, decreasing force
changes under refinement, and positive film with downstream reduced content.
The 0.99-inlet case reduces force and preserves mass. See PHYSICS.md for
settings, measured residuals, source limitations and reproduction.
Verification: `ruff check .` clean; `pytest -q` 195 passed, 4 optional
GPU skips; all four benchmark cases and plots completed under single-thread
CPU limits.

Next: retain this reference while implementing and independently verifying
an analytic monolithic pressure/saturation Jacobian, fixed-film sparse
preconditioning and strict globalized Newton iteration. Add simultaneous
positive-load/h0 control only after that pressure/saturation solve agrees
with the reference. Physical-load equilibrium and full EHL validation are
still unchecked.

## Monolithic cavitation and positive-load closure (2026-10-25)

Added analytic pressure/content/elasticity JVPs and a semismooth min(P,1-theta)
constraint, with optional simultaneous SI force/h0 closure. Frozen-film
active-set sparse LU preconditions matrix-free GMRES. Force mode uses an
explicit Schur correction, complete reconstructed-residual checks and full
bordered recovery. Fraction-to-boundary/Armijo globalization preserves stated
primal tolerances and positive film without clipping. Failed candidates,
true linear residuals, histories and Krylov traces remain inspectable.

Complete finite-difference Jacobians include load and h0; dense bordered
comparison and forced Schur/Krylov recovery regressions pass. Prescribed-h0
Newton agrees with the independent reference at 100/10/1 MPa. At a fixed
positive 0.002 N on 9x13/17x25/25x37 grids, h0 converges to approximately
20.9358/21.4899/21.5506 um, with independent mass/complementarity, direct
elasticity, film, boundary, inlet and force checks passing. An independent
nested fixed-point h0 root matches the coarse pressure to 8.25e-13 relative.
See PHYSICS.md for caps, parameter conventions, measured data and limitations.

This implements numerical positive-load closure for the documented low-load
cases. The broad physical-load/EHL validation box remains open: domain
sensitivity, independent film-height references, material/rheology range,
and general nonlinear/continuation robustness are unfinished. The earlier
signed full-film solvers are retained unchanged.
Verification: `ruff check .` clean; `pytest -q` 213 passed, 4 optional
GPU skips. Three fixed-force grid benchmarks, the nested reference comparison
and plots completed; raw Krylov trace recording was also exercised.

Next: adaptive positive-load continuation with independently checked accepted
states, explicit retries and preservation of the last converged root. Then
extend the physically admissible benchmark range before adding pressure-
dependent properties or any GPU work.

## Adaptive admissible positive-load continuation (2026-10-25)

Added bounded force continuation with independently recomputed root checks,
last-accepted-only warm starts, adaptive step growth/halving, explicit retry
budgets and full rejected-candidate/Newton/Krylov retention. Rejected shortened
final steps are genuinely reduced, and numerically identical targets do not
trigger roundoff-sized continuation. Tests cover both force directions,
recovery, exhaustion and false-success/corrupted-return rejection.

At 13x17 and E*=10 MPa, paths from 0.002 N reach 0.001, 0.005 and 0.020 N
in six accepted stages each. All 18 accepted roots pass independent direct
elasticity and conservation/film/complementarity/force checks. At 0.020 N,
h0 is 4.62029 um, minimum film 7.81880 um and peak displacement 4.10269 um;
a separate fixed-point solution matches pressure to 1.46e-12 relative.
PHYSICS.md records full inputs, iteration counts, residuals, plotting and
reproduction. Synthetic tests exercise rejection paths even though these
bounded benchmark paths require no rejected load stages.
Verification: `ruff check .` clean; `pytest -q` 221 passed, 4 optional
GPU skips. All three continuation paths and plots completed; the stronger
final root also passed the independent fixed-point reference comparison.

Next: domain sensitivity at fixed positive force, followed by more demanding
mesh/material ranges and independently sourced film-height references.
Pressure-dependent viscosity/density need separate constitutive benchmarks
before integration. Full EHL validation and GPU work remain open; the existing
signed reference implementations and original staged HIP work are preserved.

## Positive-force domain sensitivity and recovery (2026-10-25)

Added a fixed-force domain study with constant baseline spacing and matched
physical inputs. Direct cold starts on larger domains exhausted the existing
Newton cap; verified coupled reference roots plus adaptive force continuation
reach Q=0.002 N on 13x17, 21x25 and 29x33 without raising iteration limits.
All 20 accepted load roots pass independent state audits. Three real rejected
29x33 trials retain their candidates and diagnostics; smaller warm-started
steps recover and reach the target.

Offsets change substantially with the domain: 21.3005, 46.3684 and 76.9021 um.
Individual boundary flux budgets are now reported, making the changing
inlet/outlet/lateral flows visible alongside their small net imbalance.
The largest direct-audited grid has 957 nodes. PHYSICS.md records extents,
flow budgets, residuals, failure/recovery evidence and reproduction; no result
baseline was overwritten.
Verification: `ruff check .` clean; `pytest -q` 225 passed, 4 optional
GPU skips. All three domain cases reached the target with independent audits
and plots, using the existing bounded solver and single-thread CPU settings.

**Current physical limitation:** the coupled positive-load model is not
domain-independent on these extents. Same-domain grid convergence and exact
force closure must not be presented as validated physical film heights.
Next: independently audit inlet/far-field consistency using an analytical
rigid parabolic cavitation limit, then use controlled domain expansion or
appropriate verified far-field conditions before material/film comparisons.
Pressure-dependent properties need their own benchmarks and must not be used
to conceal this boundary/domain sensitivity. GPU work remains deferred.

## Desktop GPU implementation audit (2026-10-25)

Built and tested an isolated copy of the staged native HIP feature on the
RX 6600 XT (gfx1032), using the existing SDK. Host GPU access is required;
the sandbox hides the device nodes. Forty selected tests pass, including
all seven native HIP regressions; five optional PyTorch tests skip in the
CPU environment. Extended direct/invalid-buffer/repeated-call checks and
129/257/513/1025-grid parity/benchmarks complete. Native displacement errors
are below 4.88e-21 m. CPU and device/transfer-inclusive timings are separated.

Temporary native-convolution hooks in the current 9x13 positive-load Newton
solver reproduce NumPy roots at 10/1 MPa to roundoff, with independent audits.
They slow the small solver by about 35-39 times because every convolution
allocates/plans/transfers host arrays. Large isolated convolutions benefit,
but no device-resident iterative solver or automatic GPU enablement is
claimed. The staged selector chooses NumPy at 129 and HIP at 513 in fresh
full-workload checks. The staged GPU sources/index are preserved; binaries
and result data are not tracked.

The separate installed PyTorch GPU environment terminates its isolated probe
with SIGILL. Native HIP works independently; no driver/system/package changes
were made. A 48-shape sweep hit its 180 s bound (planning/compilation suspected); a checkpointed
12-case sweep and all subsequent benchmarks completed. See PHYSICS.md for
complete evidence and the reproducible validation command.

Continue the CPU inlet/far-field roadmap check. GPU optimization cannot
resolve the already observed domain dependence; any future cached-plan or
device-resident work needs separate numerical/performance verification.

## Independent rigid parabolic inlet/rupture benchmark (2026-10-25)

Following the desktop GPU audit (including the unusable PyTorch probe),
continued the CPU roadmap with an analytical finite-inlet, infinite-width
parabolic JFO reference. It returns SI pressure/film, mean-speed flux and
force per transverse width, with a smooth p=dp/dx=0 rupture. Independent
quadrature checks pressure, rupture and force; regressions also cover flux,
complementarity, admissibility, inlet sensitivity and invalid inputs.

The wide rigid 2-D JFO centerline approaches this independent limit on
17x33/65/129 grids: pressure errors fall from 3.174% to 0.642% to 0.163%,
and force-per-width errors from 3.341% to 0.646% to 0.165%. Local mass
residuals remain below 8.59e-13 and content error decreases under refinement.
The exact reference itself shows strong finite-inlet load sensitivity;
PHYSICS.md documents formulas, units, measurements, sources and reproduction.

This verifies the finite-inlet/smooth-rupture treatment in a distinct rigid
limit. It does not close the elastic point-contact domain problem. Next:
separate inlet/outlet/transverse extent studies at matched force/material
and interior resolution, using flow budgets and the new reference to guide
boundary verification. Do not transfer infinite-width boundary formulas to
point contacts without independent justification. Physical film-height
validation, pressure-dependent properties and resident-device GPU iteration
remain open.
Verification: `ruff check .` clean; `pytest -q` 233 passed, 4 optional
GPU skips. The analytical quadrature checks, three wide JFO grids and plotted
reference/inlet study completed with single-thread CPU settings.

## Separate boundary sensitivity at positive force (2026-10-25)

Completed seven matched-input cases: baseline plus inlet, outlet and both
transverse edges extended independently by four/eight original cells. All
original nodes and spacings are preserved. Each case has an independent
rigid/coupled prescribed-offset reference and adaptive continuation to
Q=0.002 N. All 42 accepted load roots pass direct elasticity and conservative
flux audits, with maximum local mass residual 6.68e-9; no retries are needed.

Inlet expansion changes h0 from 21.3005 to 32.9531/40.7640 um; transverse
expansion gives 27.0191/29.7132 um. Outlet expansion leaves h0 unchanged and
shared-node pressure differs by less than 4.90e-16 relative. Added outlet
cells remain cavitated with conserved row flux; a regression checks this
root invariance independently. Rigid reference forces also have strong
inlet/transverse sensitivity. The effect is therefore present before
elastic coupling. PHYSICS.md records matched inputs, limits, histories,
flow budgets and reproduction. New output directories preserve prior data.

The point-contact model remains domain-dependent; physical film heights are
not validated. Next: a bounded rigid point-contact far-field study that
separates physical extent and interior resolution, with explicit scaling
and thin-film applicability, before material/constitutive extensions.
The CPU reference, signed mathematical solvers and staged HIP sources are
preserved. GPU audit findings above remain applicable.
Verification: `ruff check .` clean; `pytest -q` 244 passed, 4 optional
PyTorch skips. All seven independent CPU runs and comparison plots completed.

## Rigid far-field linear accuracy repair (2026-10-25)

The next bounded rigid-domain probe exposed poorly scaled active-set sparse
systems: H cubed grows far from the minimum gap while constraints have unit
coefficients. Original unscaled local mass residuals reach 1.41e-6 on a
257x153 grid. Added positive row/column equilibration with exact recovery of
the original pressure/content coordinates. The unscaled direct path remains
available explicitly; diagnostics always evaluate original mass fluxes.
No stencil, physical boundary, clipping or regularization change is made.

The same largest probe now has max mass residual 4.70e-11 with integrated
pressure unchanged to 3.4e-14 relative. Independent dense-system and
whole-face divergence regressions cover recovery and original equations;
bad scales/invalid options raise. PHYSICS.md records formulas and measured
scaled/unscaled comparisons. Next: complete the controlled rigid extent /
resolution / dimensional-similarity study using this accurate linear path.
Verification: `ruff check .` clean; `pytest -q` 249 passed, 4 optional
PyTorch skips. Bounded single-thread rigid probes complete through 39,321
nodes; the regression also checks the independently recomputed mass stencil.

## Optional CUDA operator milestone (2026-10-26)

The explicit CUDA request supersedes earlier GPU deferrals for this increment.
Added object-owned cached DC-FFT elasticity, optional device Reynolds/rigid
geometry operations and a separate signed full-film evaluator. The latter
ports the complete scaled nonlinear residual and analytic state/compliance
JVP, including load quadrature. It does not bridge transferred vectors into
SciPy inner iterations or modify the reference Newton algorithm. Kernel
construction/upload occurs once; arrays remain resident between calls.
Workspace estimates are limited to 1 GiB and half of free VRAM by default.

Independent spatial convolution, edge/no-wrap/signed/zero cases, Hertz
refinement, whole-face flux divergence, full Jacobian columns, CPU residual
differences, strict feasibility, device residency and propagated FFT failures
cover the new paths. The complete CPU suite is also exercised with PyTorch
imports disabled. Windows CUDA execution is verified here; Linux execution
remains for CI/another host. Physical and domain limitations are unchanged.

All requested grids through 513x513 fit. Against cached NumPy at 513x513,
DC-FFT is 9.43x faster, Reynolds 16.65x, nonlinear residual evaluation 9.70x,
and JVP evaluation 9.97x. At 33x33 all are slower on GPU. Peak allocated/
reserved memory is 166.94/202 MiB. Maximum peak-normalized elasticity and
coupled-residual errors are 8.46e-16 and 3.51e-15. PHYSICS.md records full
timings, transfer/setup costs, methodology and reproduction. No full GPU
solver milestone is checked off.

Next CUDA work: profile scalar validation/launch overhead and design a
verified device preconditioner/Krylov path only with complete bordered,
Schur, true-residual, feasibility and dual-line-search regressions. Do not
replace SciPy with repeated full-vector transfers. The physical roadmap
still needs the controlled rigid far-field extent/resolution study before
material or physical film-height claims.

Final verification: `ruff check .` clean; `pytest -q` 279 passed with CUDA
available (31.17 s). With torch imports blocked, 259 passed and 20 optional
tests skipped (28.00 s). With CUDA hidden but torch installed, the selected
backend/elasticity/device suite has 30 passed and 10 skipped. The initial
sandbox run could not access pytest's existing Windows temporary directory;
the final complete CUDA run used approved host access. Generated results and
pre-existing tracked egg-info modifications are excluded from this milestone.

## Resident Newton/Krylov mathematical milestone (2026-10-26)

Starting from 97a35f1, added optional torch float64 restarted GMRES with
CGS2 reorthogonalization, device Hessenberg/Givens/triangular operations,
true unpreconditioned residual checks, explicit breakdown/budget/failure
diagnostics and a basis-memory guard. max_iterations counts Arnoldi steps,
unlike SciPy's maxiter restart cycles. No field/basis exports or CPU fallback
occur in inner iterations. A frozen state linearization reuses state fields.

GPU Jacobi exactly matches the reference frozen-film diagonal. The separate
DST inverse exactly solves a documented mobility-scaled Dirichlet Laplacian
surrogate, independently checked by dense solves. It is not an ILU clone or
the exact variable-coefficient Reynolds inverse. Device Schur reconstruction
and full-bordered recovery retain the CPU mathematical equations and accuracy
ceilings. The resident Newton path retains inexact forcing, strict film
feasibility, directional Armijo, bounded tighter-Krylov recovery and adaptive
signed-load retries, with failed diagnostics and last accepted states retained.
The original NumPy/SciPy numerical solver is unchanged.

Complete CUDA signed-load calculations converge through 513x513 in four
Newton corrections, with independently recomputed original CPU residuals.
Final pressure agreement on matched CPU-reference 33/65/129 runs is within
1.90e-11 relative L2; largest h0 difference is 2.29e-16 m. Both nonlinear/load
tolerances are 1e-10. At 513x513 the independent scaled Reynolds RMS is
9.82e-14 and scaled load error below 2.04e-14. This does not establish physical
nonnegative-pressure EHL or cure the existing branch/domain limitations.

Reference-solver setup-inclusive speedups at 65/129 are 7.32x/20.87x;
33 is slower on GPU. Failed bounded CPU 257/513 linear attempts have no
claimed speedup or full CPU Newton comparison. PHYSICS.md separates reference
and identical-algorithm comparisons, iteration counts, transfers and memory.
GPU preconditioning, kernel caching and state-linearization caching contribute
to the reference speedups; they are not hardware-only speedups.

Next: expand strong-coupling/failure and continuation coverage on larger
meshes, profile launch/scalar synchronization, test Linux CUDA/ROCm, and
independently benchmark device JFO before claiming GPU physical EHL support.
The controlled rigid far-field physical verification remains a separate open
roadmap task; this GPU increment does not close it.

The identical cached algorithm on PyTorch CPU/CUDA at 513x513 measures
6.947/2.103 s (3.30x solve time, 2.78x including device setup/transfers in
a warmed process); first tested setup-inclusive win is 257x257. Matched
pressure parity is within 3.11e-16 relative L2 and peak allocated/reserved
memory is 329.85/390 MiB. Reference-solver and identical-algorithm crossover
sizes are deliberately distinguished. Final checks: Ruff clean; complete
suite 321 passed, including CUDA. With torch imports blocked: 259 passed,
62 optional skips. With CUDA hidden, selected backend tests: 51 passed,
31 skipped. No drivers/packages were installed; pre-existing egg-info changes
and generated benchmarks remain outside the commit.

## Scientific audit, physical interpretation and reproducible figures (2026-10-26)

Audited the latest develop state, including the optional resident signed
full-film NVIDIA path, and retained all numerical physics unchanged.
`docs/VALIDATION_REPORT.md` now states the exact conservative SI/scaled JFO
equations, constraints, deformation/Hertz conventions, finite boundaries,
solver assumptions and reasons the weak-domain pressure/film shapes cannot
be compared directly with classical piezoviscous EHL. Accessible NASA target
tables are identified with precise model/readiness limitations; no reference
agreement is manufactured for missing viscosity/density laws.

Added a single CPU runner for the independent Hertz/parabolic references,
rigid point extent/resolution study, monolithic/nested load comparison,
three loaded grids and seven separate boundary cases. It generates annotated
field/centerline, conservation/load/Krylov, grid/domain and active-region
figures with portable SI NPZ fields and finite JSON histories/provenance.
Fresh-directory and failure checkpoints preserve existing data and distinguish
failure from converged domain-sensitive behavior. A plotting-only defect
(automatic contour levels omitting tiny negative roundoff values) is fixed
without clipping fields; the generic full-film title is corrected.

The 25x37 loaded case independently closes 0.002 N with max mass residual
2.96e-13, hmin=21.8446984 um at (0.0208333,0) mm and pmax=0.00792941974
MPa at (-0.222222,0) mm. Coarse monolithic/nested pressure discrepancy is
8.25e-13 relative. Dry Hertz force/displacement errors and wide analytical
JFO pressure/content errors decrease under their documented refinements.

The distinct fixed-gap rigid probe separates extent and resolution on nine
grids through 257x257, using a 70,000-node cap and one BLAS/OpenMP thread.
All states pass a separate SI flux audit (largest max residual 1.03e-10),
but doubling extent from 16 to 32 natural lengths still changes force by
5.51%. Similarity tests independently verify pressure/force/length units.
No domain-independent physical film-height result is claimed. The maximum
far-corner rigid slope reaches 0.1753, so unlimited parabolic-domain expansion
is not automatically a physical remedy.

Next scientific action: quantify the rigid and coupled far-field error
budget with independently justified boundary treatment and resolved inner
meshes. Then benchmark SI viscosity/density laws and their derivatives
before adding them to the conservative monolithic JFO system. Keep the
existing CPU references and signed full-film CUDA path; a device JFO port
remains a separate verification milestone. No new drivers/packages or
simulation output baselines are introduced.

Final acceptance: `ruff check .` clean; `pytest -q` 269 passed, 62 optional
PyTorch/device skips (63.77 s). The complete runner finishes all five stages
and produces 96 figures plus 75 SI NPZ files under
`results/scientific_validation_20261026_final/`; representative field,
convergence, boundary and matched-reference figures were visually checked.
No new GPU execution or full physical EHL validation is claimed.

## Far-field/domain error-budget milestone — partial, open (2026-10-26)

Starting at b0e8347, audited pressure/content/supply/outflow/lateral and
elastic free-space boundary conventions. The flooded west content is fixed;
the integrated inlet volume flow is an outcome that changes with inlet gap
and domain placement. No content parameter is altered to force agreement.
Density remains constant and unspecified, so flow budgets are SI volume
flows, with mass flows equal to rho0 times those values.

Added a separate rectilinear prescribed-film JFO CPU reference with the same
arithmetic face coefficients, upwind content and active constraints. It
matches uniform-grid flooded/starved roots and converges toward the independent
analytical parabolic limit on nonuniform grids. A bounded direct midpoint-
cell elasticity audit verifies source units/singular integration and uniform
FFT padding/cropping/zero-support embedding. Graded displacements are diagnostic
postprocessing; no graded coupled elasticity or GPU JFO solver is claimed.

The reproducible 27-case study separates four fixed-domain mesh resolutions,
outer grading, nested all-boundary extensions, individual inlet/outlet/side
extensions and five unchanged uniform coupled cases. Every prior graded
domain node is retained; actual snapped SI extents are recorded. The runner
declares 1% engineering changes, 1e-8 original max continuity, 1e-10 relative
local/global flow and complementary/admissibility criteria before calculation.
It preserves states that fail those tests instead of silently accepting them.

Observed progress and remaining limitations:

- Accepted fine-core domain load changes decrease to about 1.17%, compared
  with the previous 5.51%; the target is not reached. Larger fine candidates
  at labels 192/256 fail the original continuity threshold (1.53e-8/3.99e-8).
- A coarse sequence has final changes 0.393%/0.210%, but matched core mesh
  refinement shifts load by 1.73% and pressure by about 2.6%. No coarse
  plateau is treated as combined accuracy. Fixed-domain final mesh changes
  are 0.533% load and 0.646% peak pressure; outer-grading refinement changes
  load by another 0.366%. No asymptotic order or rigorous total bound is claimed.
- Extending only the inlet from label 32 to 128 shifts load by 2.074%,
  transverse extension by 0.333%, and outlet extension only by roundoff.
  Simultaneous effects need not add linearly.
- The largest uniform coupled case (129x129, prescribed h0=0.1 um, E*=1 GPa)
  has Q=0.002328762465 N, hmin=0.112047199 um, pmax=0.162668923 MPa,
  max mass residual 8.39e-12 and net volume flow 3.24e-27 m^3/s. Nevertheless,
  its load changes about 13.3% from extent 8 to 16. It is not a fixed-load
  domain-independent benchmark.
- A fine 65x65 cold Newton case exhausts the existing cap. Its failed fields
  and 30-step history are saved; the separate prescribed-offset Picard
  reference, with tighter mass convergence for the study's relative criterion,
  recovers it within existing per-stage budgets. No iteration caps or acceptance
  thresholds are relaxed. Larger-domain far-corner slopes are also inconsistent
  with blindly assuming a small-slope lubrication far field.

`docs/VALIDATION_REPORT.md` section 5 gives complete assumptions, grids,
boundary flows, source-truncation and error-budget tables, front brackets,
failure/recovery evidence and reproduction. The final run is in
`results/farfield_budget_20261026_final/`: 25 accepted states, 2 explicitly
residual-rejected candidates, 11 figures and portable SI NPZ/JSON data.
Representative load/error-budget figures were inspected; source hashes match.

Next action: resolve far-field linear accuracy against the original flux
residual, and independently justify coupled outer boundary / graded elastic
treatment with a physical applicability and mesh/domain error budget. Keep
this milestone open. Pressure-dependent viscosity/density are subsequent
work only after appropriate domain verification; none is added here.

Verification: `ruff check .` clean; `pytest -q` 282 passed, 62 optional
PyTorch/device skips (63.66 s). Optional signed full-film CUDA/HIP work and
the original uniform CPU numerical solver remain preserved. No drivers,
system packages, result baselines or published history were modified.

## Reservoir/flux conditioning milestone (2026-10-26)

Starting from 78f83143cdeaebf5d86029b19d8ac102d0615a89, the new
`examples.run_flux_conditioning` analyzes all 27 preserved far-field states
and runs bounded same-grid controls, physical-flux corrections and independently
matched boundary tests. No domain, tolerance or iteration cap is increased.

Completed and numerically verified:

- Separate SI Couette/Poiseuille/net inlet and outward outlet/side budgets,
  contribution maxima, cancellation, original continuity, wider evaluation
  and pressure-ULP sensitivity. Point inlet cancellation stays about 3.46,
  unlike the strongly cancelling analytical 1-D limit. Large h^3 contrast
  and pressure-state accuracy matter; wider reporting alone is insufficient.
- Exact nested-cell force identity: the old 1.174% rigid and 13.259% coupled
  changes comprise about 38%/39% added-area load and 62%/61% shared-domain
  readjustment, principally peripheral. A same-grid rigid control grows
  12.904%, showing the large sensitivity already without elasticity.
- Up to three optional physical-flux corrections lower label-192 continuity
  from 1.53e-8 to 4.76e-9 under the original 1e-8 criterion; label 256 still
  fails at 1.71e-8. Default behavior stays unchanged, and raw rejected states
  are retained separately. Wider residual precision is explicitly checked;
  states/corrections remain float64. Load changes are only at roundoff.
- Prescribed-film pressure-edge matching passes an exact nonuniform test,
  a mesh-refined independent 1-D analytical window and a finite rigid point
  restriction checked additionally against the original uniform solver.
  Matched 1-D force errors are below 0.1% on the three 129-node windows;
  the point trace recovers its reference to roundoff. Neither is a new
  standalone point far-field law; known exterior load is accounted for.
- Documented reservoir, supply/starvation and computational-cut distinctions.
  Preserved coupled CPU, free-space FFT and optional signed full-film GPU code.

Evidence: `results/flux_conditioning_20261026_complete/`, with 41 SI NPZ
files, 17 figures, raw source-budget JSON/input hashes and current source
provenance. Existing results are untouched. Report section 6 gives exact
units, conditions, corrections, controls and reproduction. The original
section-5 rejections remain historical evidence, not silently relabeled.

Still open: the combined mesh/domain target and physically justified point
outer response. One recovered sub-1% increment does not establish two
successive resolved increments; the next candidate remains residual-rejected.
Large far-field slopes also limit the physical interpretation. Known-trace
matching borrows exterior information rather than deriving it. No graded
coupled elasticity or infinite-domain physical validation has been added.

Next development action: derive and benchmark a point-specific outer response
for explicit physical reservoir/geometry assumptions, including exterior
pressure/load and free-space elastic sources. Verify mesh effects and
original SI flux independently before coupling it. Keep the label-256
algebraic limitation visible. Constitutive laws and GPU JFO remain deferred.

Final checks: `ruff check .` clean; complete `pytest -q` 294 passed,
62 optional PyTorch/device skips (67.36 s). Twelve new regressions and all
benchmark stages pass their stated checks; the largest corrected candidate
remains explicitly rejected rather than counted as domain-validation success.
Representative final figures and source provenance were inspected.

## Finite-reservoir response milestone (2026-10-26)

Implemented a prescribed-film 2-D exterior response beginning at e666ff9.
The full pressure/content constraints are condensed per active set; exterior
pressure, supply and cavitation are computed without importing the larger
reference trace. The fixed physical ambient reservoir is +/-8L,+/-8Ly,
65x65 with dX=0.25 and the established h0=0.1 um point geometry/fluid.
Cuts at labels 1/2/3 recover independent uniform JFO pressure/load to roundoff,
with maximum original mass defect 1.79e-12. Source-cell partitions retain
exterior force and selected-target elastic deformation, independently checked
against uniform FFT. No coupled solve or driver/system change is made.

Evidence: `results/exterior_response_20261026/`, four SI field files, four
figures and a provenance/audit/history JSON. Numerical cut independence is
verified for this finite problem; continuum/physical far-field validation
remains open. Cost is bounded by 20,000 nodes, 100 active sets, 256 response
columns and 64 MiB estimated dense response work, plus sparse factor storage.
No acceleration claim is made; an exterior system is still factored.

Next action: retain the same physical reservoir and inner nodes while testing
controlled graded exterior approximations against an independently resolved
uniform reference. Measure mesh, pressure/load, supply/front and exterior
source errors separately. Do not integrate into coupled JFO until these pass.

Milestone verification: Ruff clean; full pytest 298 passed, 62 optional
device skips (70.66 s). Four new regressions pass; all three benchmark cuts
pass original SI and independent-reference criteria. Source hashes and final
figures were checked. This verified finite-reservoir step precedes the next
outer-mesh approximation study.

## Fixed-reservoir graded exterior milestone (2026-10-26)

Continued after f5ab34f with independent uniform references at dX=.5,.25,
.125,.0625 on the same +/-8L,+/-8Ly physical reservoir, then nine graded
response candidates with separate core spacing and outer-growth controls.
The finest 139x139 candidate, within the unchanged 20,000-node/64 MiB dense
work caps, differs from the independent 257x257 reference by 0.129% load,
0.056% shared-core pressure, 0.065% total center deformation, 0.177% exterior
source force and 0.139% exterior center deformation. A fixed physical supply
section differs by 0.00611%; 65 shared core rows have overlapping matching
front brackets. Original SI mass/constraint checks pass on all nine cases,
but five fail the separate field-error criteria and remain explicitly flagged.

Evidence: `results/exterior_mesh_20261026_verified/`, 13 raw SI states,
five figures and complete audit/history/source/mesh data. Report section 8
separates original conservative faces from the fixed-section diagnostic and
records reference mesh uncertainty. The last uniform load change is 0.4905%,
not a rigorous continuum error bound. Existing results, including incomplete
development checkpoints, are preserved. No coupled JFO/FFT/GPU code changed.

Next development action: independently verify displacement at all required
inner targets and the film-dependent exterior response/derivatives against a
full uniform coupled finite-reservoir reference before integrating it. Keep
reference, exterior/core mesh and nonlinear errors distinct. A further uniform
halving needs 263,169 nodes, beyond the 70,000-node reference-study budget;
use a verified bounded alternative rather than increasing limits silently.
Physical infinite-domain validation and the old large-domain label-256 issue
remain open. Constitutive physics and GPU JFO remain deferred.

Milestone verification: Ruff clean; complete pytest 300 passed, 62 optional
device skips (70.68 s). Independent reference/outer-refinement and diagnostic
flow/front regressions pass. All source hashes and 13-state/five-figure
manifest checks pass; coarse field-error failures remain visible.

## Exterior coupling prerequisites (2026-10-26)

Continued from 97e38db. Added a conditional prescribed-film exterior tangent
with the actual recorded wet/dry set, independently checked by paired
nonlinear solves for inner and upstream exterior film perturbations on uniform
and graded grids. Analytic force directions are also independently integrated
and compared. Added bounded direct retained/exterior source evaluation at
every inner target: 2048-target and 20-million-pair caps, batches of 32,
no dense kernel and no unequal-cell FFT.

Evidence: `results/exterior_coupling_audit_20261026_verified/`, 13 raw SI
archives, two figures, perturbed roots/audits, source/runtime hashes and an
independently rerun unchanged full coupled reference with its histories.
Uniform direction pressure errors at the last amplitude are 8.29e-8 and
8.16e-11. At the full coupled reference's frozen film the exterior pressure
error is 4.36e-15 relative, load matches to reported precision, and all 289
inner direct displacements match coupled FFT within 7.28e-23 m.

Still unimplemented: a new nonlinear exterior-coupled root and a consistent
graded source/target film map throughout the exterior. Fixed-active-set
derivatives do not cover front switches. These are prerequisites, not a
claim of coupled domain independence or an infinite reservoir formulation.

Next justified action: first verify a bounded, globalized exterior nonlinear
iteration on uniform finite reservoirs against the full coupled reference.
Independently audit original flux, load, film positivity and full source
coupling through active-set changes. Only then port to a verified graded
full film map, including exterior-target displacements/derivatives. Preserve
existing coupled/GPU code and all prior numerical limitations. No new
constitutive law or GPU JFO work is justified by these partial checks.

Milestone verification: Ruff clean; full pytest 304 passed, 62 optional
device skips (79.99 s). Four new uniform/graded derivative configurations
pass. All final benchmark acceptance flags, original perturbed-root audits,
recorded active sets, source hashes and 13-state/two-figure checks pass.
No nonlinear condensed-exterior coupled root is claimed from this evidence.

## Uniform exterior nonlinear coupling milestone (2026-10-26)

Continued from 39b8703 with a separate bounded prescribed-h0 Picard
experiment. Every step recomputes exterior pressure/content and full-source
uniform FFT film throughout the physical reservoir. The original monolithic
solver is unchanged and independently cold-started on each physical mesh.
33x33 cut-1/cut-2 probes give identical load to reported precision; 65x65
and 129x129 probes independently converge within existing caps. All physical
local/global/continuity/complementarity and film-feasibility criteria remain
unchanged. Full histories, last-candidate failure retention and all-inner
source/FFT audits are retained. No performance acceleration is claimed.

Evidence: producer `results/uniform_exterior_coupling_20261026_refined/`,
eight SI states and two figures. Its fine monolithic reference passes the
old solver's stopping tolerances but fails the stricter independent local
flow audit (3.50e-10 vs 1e-10), and remains explicitly rejected. A separate
reaudit uses only that independent reference as a warm start, tightens its
mass solve to 1e-10, and recovers it with one additional Newton correction.
No exterior root is used to seed the reference, no candidate is recomputed
or altered, and no tolerance is relaxed. Original rejected data/histories
are preserved. Authoritative reaudit:
`results/uniform_exterior_coupling_20261026_checked/`, with nine SI states,
a reference-comparison figure and input/source hashes.

All four exterior roots pass comparison against the audited independent
references (pressure/load errors below 4e-10). Total outer iterations are
108/108/107/103 across five compliance stages, at most 28 in a stage.
The fine case remains within 20,000 nodes and response work limits; it is
computationally expensive, not a production acceleration. Mesh load changes
shrink from 6.06% to 1.58%; the 1% coupled mesh target is still not met.
Finite-reservoir cut independence does not establish a physical infinite
point-domain limit or classical EHL accuracy.

Next justified action: verify a bounded full graded source/target film map
and its nonlinear derivatives, preserving source-cell force and displacement,
before graded coupling. The finest graded rigid grid has 19,321 nodes;
a naive complete direct map would need 373,301,041 source-target pairs,
exceeding the 20-million-pair direct budget and 2,048-target cap. Develop and
independently benchmark a mapped/multilevel alternative instead of silently
raising limits or applying the uniform FFT to unequal cells. Coupled mesh
accuracy needs its own budget; another uniform exterior halving exceeds the
20,000-node cap. Keep physical-reservoir dependence, old label-256 precision
limitations and generalized front/globalization work visible. Piezoviscous
physics and GPU JFO remain deferred.

Final milestone verification: Ruff clean; complete pytest 306 passed,
62 optional device skips (75.11 s). Independent-root and failure-retention
regressions pass. All four final candidates and three audited references
pass original criteria; the original fine reference rejection, raw fields
and its one-correction recovery remain separately preserved. Producer and
reaudit hashes/field counts/figures were verified and representative final
plots inspected. No graded nonlinear or infinite-domain validation claim
is made from this uniform finite-reservoir experiment.

## Pressure-dependent properties: authorized priority change (2026-10-27)

The user explicitly reprioritized the next three milestones after abf2a25:
(1) independently benchmarked SI density/viscosity laws and derivatives;
(2) conservative pressure-dependent prescribed-film JFO with independent
mass/constraint/derivative checks; (3) initial coupled pressure-dependent
uniform finite-reservoir cases with continuation and independent field/load
checks. Graded elasticity remains deferred. The existing 1.58% coupled mesh
uncertainty and physical reservoir assumptions stay documented; constitutive
work does not claim to resolve them or to match classical EHL data already.

Current action: verify original-source constants/units, implement/test the
standalone laws first, then commit/push each independently verified milestone.
Use bounded single-thread NumPy/SciPy runs and preserve every baseline. If
interrupted, resume the first unfinished milestone here rather than restarting
the far-field investigation. Weekly account allowance is not visible to the
agent; no quota-fit guarantee is made.

Milestone 1 complete: SI constitutive laws, analytic derivatives, independent
high-precision/finite-difference checks and source-unit resolution are in
`docs/CONSTITUTIVE.md`. Evidence: `results/constitutive_20261027_verified/`.
Ruff clean; full pytest 308 passed, 62 optional device skips (85.23 s).
Next: conservative variable-property prescribed-film mass flow, including
an independent pressure-driven reference and constant-property JFO parity.

Milestone 2 implemented and benchmarked: pressure-dependent prescribed-film
JFO now conserves SI mass using rho in both mobility and entrainment. The
independent potential-integral channel converges with final pressure/flux
errors 6.41e-10/4.10e-8. Point cases reach 24.3 MPa and eta/eta0=1.70;
physical audits pass, while the last 8.63% load and 14.75% peak mesh changes
remain open. Evidence: `results/property_reynolds_20261027_verified/`.
Next action after final regression/commit: coupled uniform variable-property
cases, independent constant-property monolithic parity, full source/film and
mass/load checks. Keep publication matching as subsequent verification.

Milestone-2 final checks: Ruff clean; full pytest 314 passed, 62 optional
device skips (87.60 s). All channel/reference and original physical audits
pass; point mesh accuracy is separately flagged false. Proceed to milestone 3.

Milestone 3 implemented: uniform elastic/material continuation now updates
full-source FFT film around the pressure-dependent JFO solve. Constant-law
monolithic parity and a separate complete nonlinear small-grid reference
pass (pressure disagreement 1.60e-10 relative). Full pytest 316 passed,
62 optional skips (94.98 s); Ruff clean. Completed 17/33 coupled runs are
preserved in `results/property_coupling_20261027_verified/`; a bounded 65
case is being added in `results/property_coupling_20261027_fine/` before
final documentation/commit. No published-table or mesh-accuracy claim yet.
If interrupted, finish that benchmark's audits and record its limitations;
all three physics components are now implemented, independently checked.

All three authorized material milestones are complete as initial numerical
components. The final 65x65 coupled run in
`results/property_coupling_20261027_fine/` passes original physical audits:
pmax=19.7907 MPa, eta/eta0=1.54673, rho/rho0=1.011488,
Q=.2191838031 N and hmin=.113147175 um. Seven SI archives and seven
figures retain fields, histories, independent references and a failure probe.
The last load/peak mesh changes are 7.34%/10.19%; the 1% target is false.
No publication/strong-load accuracy claim is made. Source hashes/manifests
and representative final plots were verified. Full pytest 316 passed,
62 optional device skips (94.98 s); Ruff clean.

Next justified action: benchmark-matched material/geometry/supply selection,
then independently verified positive-load control and bounded continuation
for the pressure-dependent coupled model toward a published classical EHL
case. Keep finite-reservoir and mesh errors separate. Graded elasticity and
GPU JFO stay off this critical path until measured benchmark errors justify
them. Do not restart the completed isoviscous far-field investigation or
require another user prompt between already-authorized increments.
