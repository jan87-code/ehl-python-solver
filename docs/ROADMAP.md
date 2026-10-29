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
  - Monolithic JFO pressure/content/offset coupling closes positive force on
    limited isoviscous and pressure-dependent cases, including the registered
    200 N benchmark on three grids. Physical film-height validation, resolved
    mesh/domain accuracy and broad material robustness remain open.
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
  - Limited variable-property positive-load control now closes 200 N.
    Published classical-EHL agreement, resolved mesh accuracy and broad
    physical-load validation remain open; see the latest milestones below.
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

## Pressure-dependent load control (2026-10-27, complete)

Authorized next milestone: prescribe positive force and solve the gap offset
for the new pressure-dependent JFO/elastic model. Start with a bounded nested
scalar root around independently converged prescribed-offset solutions,
holding SI domain/scales/fluid fixed. Every force evaluation must pass fresh
mass, complementarity, film/source and force-integration checks; failures
must retain diagnostics and never become successful root evaluations.
Benchmark known-offset recovery against the separate complete nonlinear
reference, then a distinct imposed force and bracket/failure/budget checks.
Current prescribed-offset solver cold-starts from positive rigid gaps, so
this first load controller will explicitly require positive offset brackets;
negative-offset/strong-load branches remain subsequent work, not silently
clamped. Preserve previous results and commit/push after complete checks.

The milestone is complete: `solve_property_at_load` uses bounded bracketed
force control with fresh physical audits and retained failed candidates.
A separate complete nonlinear reference recovers the known 100 nm offset
(relative error 1.57e-10). Two brackets independently reach 0.10 N at
h0=123.885855 nm on 9x13; the same force on 13x17 gives h0=160.621945 nm.
All final relative force errors are below 5.77e-12. The 29.65% offset mesh
change remains an accuracy limitation; force closure is not mesh validation.
Evidence: `results/property_load_20261027/`, 35 accepted force probes,
41 SI NPZ archives, three figures and a separately retained failed probe.
Source hashes and representative plots were checked. Ruff is clean; full
pytest: 327 passed, 62 optional device skips (188.72 s).

Next justified action: use the user's selected physical benchmark to match
fluid law, geometry, load, entrainment and supply, then perform bounded
fixed-force mesh/domain studies. Positive offset brackets are currently
required; signed-offset or strongly loaded branches need separate verified
continuation if the benchmark requires them. No classical EHL or physical
film-height accuracy claim follows from the present load-control milestone.

## User circular 200 N benchmark readiness (2026-10-27, complete; target open)

Use the user's 113 GPa contact-modulus convention explicitly: U=eta0 Ue/(E R),
W=Q/(E R^2), G=alpha E. Physical inputs are Q=200 N, Rx=Ry=15 mm,
E=113 GPa (alternative E'=226 GPa), Ue=2 m/s, eta0=.04 Pa s and
alpha=2.21e-8/Pa. Supplied correlation estimates are hmin=.278495 um and
hcentral=.477353 um, not independent numerical reference fields.

`Circular200N` records SI inputs, both dimensionless conventions and exact
Hertz scales. The assumed Roelands index is derived from the supplied
ambient logarithmic viscosity slope (z=.6713666711); it is not silently
left at the previous .68. Density uses the existing Dowson-Higginson law
and assumed rho0=850 kg/m^3; these choices remain unconfirmed lubricant
assumptions. No temperature or Barus law is inferred from alpha alone.

Run bounded positive-offset readiness probes before attempting a 200 N
scalar root. Keep physical bounds +/-3a and Hertz-based scales fixed;
independently vary mesh and retain every failed nested state. Do not claim
200 N closure or compare film errors to the correlation unless the full
mass/film/complementarity/force gates pass. If the positive-offset probes
fail to bracket 200 N, implement verified feasible signed-offset continuation
next, without treating low-force film values as a physical comparison.

Readiness assessment completed and independently audited on 17x25 and 33x49
nested grids, fixed +/-3a domain. Seven prescribed-offset roots pass original
mass/complementarity/film checks. At h0=.5 um, Q changes 1.69805->1.97006 N
(16.02%) on refinement; at .1 um the coarse Q is 3.33222 N and the fine
cold rigid/material solve overflows. The latter's exception chain, last
available partial fields and 123 property iterations are retained and
explicitly rejected. No audited bracket for 200 N exists in these probes,
so no scalar force solve or film-height accuracy claim is made.

Evidence: `results/circular_200n_readiness_20261027_final/`,
`results/circular_200n_readiness_20261027_refined/` and
`results/circular_200n_comparison_20261027/`; eight SI archives and one
inspected comparison figure. Seven saved-state audits independently check
mass, full-source film, sampled direct displacement and SI load. Source
hashes match. Seven new regressions verify supplied Hertz numbers, both
modulus conventions, ambient-alpha derivative and fixed-domain scale/grid
behavior. The case-registration/readiness milestone is complete; reaching
200 N remains open.

Next development action: implement warm-start signed-offset continuation
with positive total film, bounded load/material steps and retained retries.
First verify independent small-case parity/recovery, then attempt this
200 N case and separate mesh and physical-reservoir effects. The current
cold positive-offset limitation must be removed by verified solver changes,
not pressure clipping, force scaling or relaxed acceptance criteria.

Final readiness checks: Ruff clean; complete pytest 334 passed, 62 optional
device skips (205.30 s). All retained accepted states and comparison inputs
were independently re-audited; the overflow candidate remains rejected.

## Warm signed-offset continuation (2026-10-27, verified reference complete)

Roelands is now confirmed by the user. Retain the explicitly matched ambient
slope and the 113 GPa convention. Density parameters remain assumptions.
Implement a separate bounded analytic full-material/full-compliance Newton
reference on at most 400 interior nodes; leave the cold solver unchanged.
Include pressure-law and full-source FFT derivatives, independently checked
linear corrections, feasibility and true directional Armijo globalization.
Require a fresh physical/film audit of every warm seed and accepted root.
Follow signed offsets with film-limited steps and bounded rollback/retries;
measure force rather than rescaling pressure. First positive-offset parity
and a numerical-Jacobian signed-root comparison pass. A verified branch has
crossed zero offset and exceeded 8 N; the next step encounters a retained
line-search limitation. Complete the bounded final run and full tests,
document exact accepted/rejected states and commit/push the verified scope.
If interrupted, the checkpoints under `results/signed_200n_20261027_final/`
retain the last accepted root and each failed trial; do not label this 200 N
closure or physical film-height validation.

Signed-offset reference milestone complete: analytic material/elastic Newton,
seed auditing, positive total film, true correction and directional Armijo
checks, bounded rollback/retry continuation and independent signed-root
agreement are implemented. Full pytest 341 passed, 62 optional device skips
(267.32 s); Ruff clean. The numerical-Jacobian reference differs by 1.737e-11
in relative pressure on a saved negative-offset root.

Final evidence: `results/signed_200n_20261027_final/` and
`results/signed_local_reference_20261027/`. Fifteen accepted steps, nine
retained failed/retried proposals, 26 total SI archives including the
independent reference and one inspected figure. Source hashes match.
The final root is h0=-.3014203907 um, Q=8.9655393334 N, hmin=.1142840090 um,
pmax=713.916834 MPa; original mass/constraint/source checks pass. The declared
24-attempt budget ends with seven consecutive accepted small steps after
line-search retries. It does not reach -5 um or close 200 N. No physical
branch limit, mesh accuracy or film-height agreement is inferred.

Next action: continue from the audited signed root, independently improve
active-constraint step handling and replace dense corrections with a bounded
matrix-free/sparse route for finer-grid checks. Then use audited load-directed
steps/brackets toward 200 N, keeping force, mesh and physical-domain errors
separate. The current dense reference is limited to 400 interior nodes;
33x49 and larger grids are deliberately rejected rather than allocated.

## Scalable load-directed continuation (2026-10-27, verified bounded path)

Build on commit 08d64ce using the preserved 8.9655393334 N signed root as
coarse-grid input and the preserved 1.9700562924 N refined prescribed-offset
root. Input SI axes and archive hashes are checked; existing arrays are not
modified. Implement matrix-free material/elastic/offset Jacobian products,
sparse frozen-film JFO preconditioning and verified bordered Schur/GMRES
corrections. Preserve the dense reference's default cap and all original
physical/force criteria. Add bound-aware Armijo and selected active-constraint
recovery without clipping or lowering the film floor.

Small-grid finite differences, independent dense correction parity, forced
Schur fallback, known-offset recovery and Krylov/retry failure retention pass.
The authoritative run is checkpointing under
`results/load_directed_200n_20261027_final/`. Both 17x25 and 33x49 grids close
10 and 20 N with fixed physical bounds; the refined grid continues higher.
The coarse branch approaches the film floor on larger proposals. Retain its
rejected histories and compare only independently force-closed shared loads;
no matched-offset comparison or failed state may substitute for these gates.
Finish the bounded run, independent saved-field audits and full regression
suite, then document exact progress, remaining resolution/continuation limits,
commit and push. An earlier exploratory run is preserved separately.

Scalable load-directed milestone complete as a bounded CPU numerical path.
Matrix-free bordered material/elastic JVPs, sparse frozen-film preconditioning,
verified Schur/GMRES plus fallback, affine pressure/content/film bound steps,
original-directional Armijo and bounded force-step rollback are implemented.
Full pytest 347 passed, 62 optional skips (263.68 s); Ruff clean. The old
dense reference remains available with its default 400-interior-node cap.

Authoritative evidence: `results/load_directed_200n_20261027_final/` and
`results/load_directed_200n_20261027_audit/`. All 41 accepted states pass
independent original mass/film/boundary and SI force audits; 14 rejected
partial states retain full available nonlinear/Krylov/globalization data.
Four actual bordered fallbacks and 116 accepted shortened steps pass
independent history checks. Input archives are unchanged and source hashes
match. Two unsmoothed figures were inspected.

Coarse 17x25 closes 10/20 N and reaches 20.62890625 N before larger proposals
approach the film floor and exhaust Newton budgets. Refined 33x49 closes
2/4/8/10/20/40/80 N and reaches an accepted adaptive target of
100.4423828125 N (h0=-2.76927325835 um, hmin=.1375437173 um,
pmax=1406.190560 MPa). Larger proposals fail the unchanged true bordered
Krylov check; smaller retries recover progress until the declared attempt
budget for 120 N ends. The 200 N target remains unachieved.

Matched-force comparisons reveal large resolution error: at 10 N, coarse/
refined hmin=.105371/.280550 um; at 20 N, .00546368/.254703 um. The 1%
mesh criterion is false. The strongest refined contour has visible cell-scale
pressure variation, retained without smoothing. No physical film-height or
pressure accuracy is inferred from equation convergence.

Next justified action: strengthen pressure-law/elastic preconditioning and
scaling against the archived high-load true-Krylov failures, and independently
verify finer-grid warm-transfer/repair. Resume from audited accepted roots,
compare grids at force-closed shared targets, and investigate the pressure
variation/resolution error before a physical 200 N comparison. Do not relax
force/mass/complementarity/film tolerances, raise the floor, or change fluid
laws to bypass the measured limitations.

## Pressure-law/local-elastic preconditioning (2026-10-27, complete)

Keep the earlier frozen-property option as the default and add explicit
material and local-elastic sparse pressure tangents. Independent SI face
finite differences and dense bordered parity pass; complete equations/JVPs
are identical across modes. An archived failed 106.375 N correction still
fails with the old preconditioner, but passes the same true-residual check
with both new modes. Full-model continuation, restarted only from the
preserved accepted 100.442 N root, closes 120, 160 and 200 N on 33x49,
with 11 accepted steps and no rejected force proposals. This establishes
200 N equation/force closure on that grid, not mesh/physical accuracy.
The independent saved-state audit passes; full regression verification is
running. Finish reporting, commit/push this milestone, then independently
verify finer-grid transfer/repair and a matched 200 N refinement.

Final milestone checks: Ruff clean; full pytest 352 passed, 62 optional
device skips (261.86 s). Eleven independent archive audits, recorded source
hashes and raw pressure/film/restart figures pass inspection. The 200 N
force target now closes; the coarse pressure variation and film-height
accuracy remain open. Continue with verified finer-grid initialization.

## Finer-grid warm initialization/repair (2026-10-27, verified on 65x97)

The 200 N sparse-tangent milestone is committed/pushed as f04a7c5. Add an
explicit unaccepted interpolation type, audit the source root, preserve its
SI force on nested fixed-reservoir grids, and solve the full target equations
before accepting any transferred state. Existing root-seed auditing remains
strict. Source pressure/content and signed offset are not clipped or lifted.

Direct 65x97 repair initially exhausts 30 corrections; its guess and failed
partial fields/histories are saved separately. A 10 N initialization attempt
also stalls with tiny feasible steps near active fronts. Safeguarded active
lookahead now keeps those feasible steps as backups while trying novel active
models under original merit/feasibility checks. It reaches 200 N on 65x97 in
nine corrections/18 linear models. Independent saved-field audit passes.
The original failures remain rejected and are reproducible with
`--no-active-lookahead`; their partial states are never reused as roots.

Evidence: `results/load_refinement_200n_20261027_final/` and its `_audit/`.
The same source is also being tested with factor-three nested refinement,
97x145 (14,065 physical nodes, below 20,000). Finish that bounded run, archive
audits, final full tests and matched-load reporting, then commit/push.
The minimum-film change from 33x49 to 65x97 is still +730.9%; equation
closure is not mesh accuracy. Pressure variation remains visible.

The independently verified transfer/active-repair milestone is complete on
65x97: 200 N closes in nine corrections/18 linear models, with max original
mass 1.188e-15, relative local defect 4.447e-15 and true linear error below
8.243e-10. Five new regressions pass; full pytest 357 passed, 62 optional
skips (299.96 s), Ruff clean. Source hashes and independent saved-state,
force/source/globalization audits pass; two raw figures were inspected.

At the same 200 N and physical reservoir, hmin changes .0196093->.162938 um,
central film .332082->.442995 um and peak pressure 1725.25->1741.10 MPa.
The 1% mesh target is false and cell-scale pressure variation persists.
No physical validation claim is made. Preserve the failed pre-lookahead
repairs and every unaccepted initialization; only the fully audited repaired
root may seed subsequent calculations.

The separately bounded factor-three 97x145 case remains in progress under
`results/load_refinement_200n_20261027_factor3/`; its initialization is saved,
but no final root is claimed yet. Complete its existing solver budget and
retain the result/failure history. Next assess that matched-load resolution
change and investigate conservative transport/discrete stability of the
pressure pattern before modifying physics or interpreting film correlation
agreement. Do not silently increase the node or iteration caps.

## Recovery and independent transport consistency (2026-10-27)

Recovered GitHub develop at b6daed0 in a separate worktree, preserving the
original staged HIP work. The production 200 N archives and pending 97x145
checkpoint are absent from this workspace; that run cannot be resumed or
its saved-state audit reproduced here. No third-grid root is claimed.

Added an independent smooth SI continuum-source check of the existing
pressure-dependent face-flux stencil and a uniform alternating-pressure
linear probe. Full-film convergence approaches order two; smooth variable
content approaches order one. Centered density transport has an alternating
null response, while diffusion matches its nonzero analytic symbol. This
is limited stencil evidence, not coupled stability or physical validation.
Four regressions cover both orders and the separate tangent responses.
See PHYSICS.md for inputs, evidence, reproduction and limits.

Next: recover the original source/checkpoint archives, complete the bounded
97x145 repair and audit all accepted matched-force fields. Then assess the
complete coupled transport/elastic tangent and active fronts at those roots,
and separate reservoir sensitivity from mesh error. Do not change transport
or infer pressure-pattern causality from the smooth probe alone.

Final verification: Ruff clean; full pytest 361 passed, 62 optional device
skips (269.22 s), with BLAS/OpenMP limited to one thread. The standalone
four-grid probe completed with runtime/source provenance under
`/tmp/ehl-transport-consistency-verified/`; no simulation output is committed.
Production refinement and archive-dependent audits remain unfinished.

## Fresh 200 N source regeneration (2026-10-27)

The previously missing 33x49 source and 65x97 refinement have now been
regenerated from a cold prescribed-offset solve, rather than inferred from
documented numbers. All 44 accepted load-continuation roots and the repaired
65x97 root pass independent saved-state and globalization audits. Fresh
200 N film/offset values reproduce the earlier reports; pressure variation
and the failed 1% mesh criterion remain. Source/field hashes and reproduction
scripts are copied to `results/recovered_200n_20261027/` in the original
project checkout, preserving the staged HIP work and existing baselines.
See VALIDATION_REPORT.md section 20 for the exact bounded path and scope.

The fresh 97x145 repair has now exhausted its unchanged 30-correction
budget and remains rejected: mass_maximum=.0643674, relative local mass
defect=.440414 and complementarity=.107005, despite exact 200 N force.
All 240 linear models pass the true correction check (max 2.607e-10),
but accepted steps repeatedly collapse toward roundoff. Its initialization,
failed partial fields, full history and independent failure audit are saved.
A third-grid mesh-convergence claim is therefore still unavailable.

Next: independently verify active-constraint/roundoff stagnation handling
against this saved failure before tuning preconditioning or transport.
Preserve the original reference and unchanged physical acceptance gates.
Only accepted force-closed roots may enter subsequent mesh/domain studies.
Final verification: Ruff clean; full pytest 361 passed, 62 optional device
skips (285.51 s). No production numerical changes were made.

## Active-lookahead candidate selection (2026-10-27)

The archived 97x145 failure reveals a candidate-selection defect: iteration
zero defers an original-model step reducing merit 4.413396e-5 -> 4.294239e-5,
then accepts a final model at 4.413396e-5 instead. That final acceptance
prevents the existing fallback from restoring the better verified candidate.
The best policy now compares both candidates, marks superseded acceptances,
and rejects state/merit equality caused by rounding. Six targeted cases
cover selection, no-progress rejection and the explicit legacy reference.
Original physical gates, Jacobians, correction accuracy and caps are retained.

The 65x97 200 N replay remains bitwise identical in pressure/film, with
nine corrections/18 models and a passing independent saved-state audit.
The 97x145 replay supersedes 16 worse final candidates, accepts 17
strictly merit-decreasing corrections and then rejects its partial state
when four tried active models have zero feasible fraction-to-boundary
steps. Its 112 linear models remain accurate (max true error 2.181e-10).
Final merit improves from the legacy 4.091610e-5 to 2.756063e-5, but original
mass_maximum=.0495271 and complementarity=.0941517 still fail; no third-grid
root or mesh accuracy is claimed. Input arrays are identical to the legacy
run. Both failures remain preserved; independent history checking verifies
the minimum verified merit candidate wins at every accepted correction.

Next: localize the pressure/content constraint blocking all final directions
and benchmark targeted active pivots at that bound, including roundoff and
cycling cases. Preserve the legacy and best-candidate references, original
acceptance gates and existing budgets; do not bypass the failure by clipping.
Final checks: Ruff clean; full pytest 367 passed, 62 optional device skips
(310.29 s). Evidence and code snapshots are saved under
`results/lookahead_selection_20261027/` in the original project checkout.

## Targeted content-bound recovery (2026-10-27)

The final best-candidate 97x145 failure is blocked at physical (y,x) index
(62,86): theta reaches -1e-10, P=.0167591, and the original accurate Newton
direction has dD=.354948 (outward across D<=1+1e-10). The old recovery only
handles proposed negative P/D and therefore misses this upper-D bound.
A diagnostic single-cell full-film pivot removes the zero step while
retaining the original slope/true-JVP checks; another nearby content bound
then becomes limiting. The saved failed state is used only for diagnosis.

Explicit repair now optionally prioritizes earliest affine content blockers
within the original content tolerance, solves the selected full-film model,
and retains the previous best-candidate/legacy policies via explicit flags.
Five analytic cases verify targeted root recovery, simultaneous blockers,
competing bounds, nonempty-content exclusion and failed-cycle retention.
No equation, floor, physical gate or iteration budget is changed.

The 65x97 replay remains bitwise identical, with nine corrections/18 models,
no content pivots and a passing independent archive audit. The 97x145 repair
now closes 200 N in 28 corrections/141 models within the unchanged budget,
with nine content-bound pivot proposals. Its independent field/history/
force/source audit passes: max original mass 2.643e-15, relative local
balance 6.225e-15, complementarity 1.493e-28 and max true correction error
9.754e-10. Input arrays are identical to both preserved failed baselines;
neither rejected partial was reused as an accepted seed.

Matched 65x97->97x145 changes are hmin .162938->.187512 um (+15.08%),
central film .442995->.396128 um (-10.58%) and peak pressure
1741.100->1716.684 MPa (-1.402%). Raw centerlines retain cell-scale pressure
variation. The 1% mesh gate is still false; no continuum order, physically
resolved pressure spike or film-height accuracy is claimed.

Next: benchmark conservative transport/discrete stability at these accepted
high-load roots before changing the stencil. Separate reservoir sensitivity
from mesh error and retain both older recovery policies. Physical validation
remains open. Evidence/code snapshots, both accepted replays, independent
audits and the unsmoothed three-grid plot are preserved under
`results/content_bound_pivots_20261027/` in the original project checkout.
Final checks: Ruff clean; full pytest 372 passed, 62 optional device skips
(307.78 s), one BLAS/OpenMP thread. Production equations and budgets are
unchanged; only bounded model selection has been extended.

## Windows recovery and loaded-root transport audit (2026-10-28)

Starting at 5c9d723, the missing archives were regenerated with the registered
R=15 mm, E*=113 GPa, G=2497.3, 200 N case. New reusable recovery orchestration
cold-starts the 33x49 prescribed-offset source and reproduces 44 accepted
local-preconditioned load steps. Every accepted state is checkpointed. The
existing factor-two/three repairs then recover 65x97 and 97x145 roots within
the original nine/28 correction and 18/141 model counts. Separate archive
audits pass for all 46 accepted roots. Reproduced film values match the
reported values; no missing root was reconstructed from a correlation or
accepted by interpolation. Production solvers, stencils and budgets are unchanged.

Added an independent seven-term SI directional flux decomposition, tested
against flux finite differences and an analytical alternating-pressure symbol.
At all three accepted 200 N roots, five zero-force, zero-edge pressure probes
exercise density/viscosity, diffusion, advection and full-source elastic film.
The independent mass action agrees with the analytic JVP within 1.62e-14
relative; best feasible centered residual differences are below 9.41e-9.
Both global and centerline raw alternating projections are reported because
opposite transverse phases can cancel in a global projection. The finer-grid
x-centerline projections remain .33518/.30067, versus .001987/.001042 for
the sampled dry Hertz reference. The latter is an aliasing reference, not an
EHL solution. These are diagnostics, not a stability or causality proof.

Elastic film contributions remain non-negligible: for alternating x probes,
their action norm relative to frozen-film material action is .889/.250/.109;
for smooth x probes it is 5.15/2.80/1.81. The earlier uniform frozen-film
density null mode therefore cannot stand in for the loaded coupled analysis.
The production pressure pattern remains unresolved; no transport change,
filtering, clipping or physical film-height claim is introduced.

The optional CUDA cross-check uses the RTX A2000 only for deformation of
accepted pressure fields and five pressure directions per grid. Float64 root
displacement differences are below 2.55e-21 m; direction differences below
1.45e-24 m. The production material/JFO continuation remains CPU-only.

Evidence: `results/recovered_200n_20261028*` and separate `_audit` directories;
`results/loaded_transport_20261028_final` contains all diagnostic arrays,
source hashes and the inspected raw three-grid plots. Recovery source
snapshots and SHA-256 manifests retain provenance; generated data are not
committed. New output directories are required. Validation report section 23
and README document reproduction without the lost archives.

Next: compare resolved pressure wavelengths and feasible active-front
directions at these accepted roots, then perform separately verified
fixed-spacing inlet/outlet/transverse extent changes. Do not infer temporal
stability from a steady Jacobian or change the stencil solely to suppress
the plotted pattern. The next integer nested mesh would exceed the existing
20,000-node cap; plan and verify resource/scaling changes before raising it.
Mesh/domain and physical validation remain open.

Checks: Ruff clean; complete Windows CPU/CUDA pytest 439 passed, one skipped
(237.00 s), BLAS/OpenMP threads one. The skip is the wider-longdouble diagnostic:
Windows does not provide extra precision over float64. Six new diagnostic
regressions pass. Pre-existing tracked packaging metadata changes are preserved.
An explicit simulated interruption at the accepted 186.375 N root resumes
through four newly audited steps to 200 N; final pressure differs by
6.65e-12 relative L2 and offset by 1.27e-18 m from the cold path. The
loaded-root audit also passes with torch imports deliberately blocked.

Recovery package: `results/recovery_20261028_verified.zip` (7,887,923 bytes),
343 files verified against its SHA-256 manifest and ZIP CRCs, including exact
source snapshot at 47f70c2ae19b64d5cc065846a3ce4698347bacb2. Archive SHA-256:
`10c55aab7b0f963d9713db798f57fc80bfa22ecf40924510c9c5a9abb39beaf6`.
The package is local/Git-ignored; only the reproducible code, evidence and
archive identity are committed. No off-machine synchronization is asserted.

## Resolved carriers, fronts and material reservoir budget (2026-10-28)

Starting at 9afbd68, added fixed physical sine/cosine pressure carriers at
wavelengths a, a/2 and a/4, with explicit four-cell sampling omissions and
separate Nyquist references. These are not a convergence fit.
Added one-sided bound-feasible rupture/content probes
and an exact min(P,D) directional derivative check. Production fields are
not rounded into ties, filtered or changed. All 43 accepted-root probe checks
pass: complete mass decomposition error <8.94e-15, best feasible residual
differences <9.66e-9, front-only <1.09e-10. No reformation/exact tie is present
in these roots; analytic/reflected tests cover those paths separately.

The checked x-Nyquist mass gains are .169/.225/.371 times the corresponding
windowed a-wavelength cosine gains on the three meshes. This quantifies the
selected static responses; it does not establish a Jacobian spectrum,
instability or the cause of the pressure pattern. All accepted fields and
the pressure pattern remain intact.

Added an explicitly unaccepted same-spacing reservoir extension initializer;
existing fixed-reservoir interpolation and source-audit rules remain strict.
Six independent 200 N repairs use four/eight-cell inlet, outlet and transverse
extensions of the same 65x97 root. All pass original field/force and independent
saved-state/history/source audits within existing 30-correction/eight-model
budgets. The node cap remains 20,000; maximum tested target is 81x97.
Initializer tests verify exact coordinate/force preservation and independent
cold-root parity; no offset lift, pressure/content projection or new boundary
condition is introduced.

Largest tested minimum/central-film changes are .058302%/.156815%, versus
the earlier 15.08%/-10.58% finer-grid changes. Largest peak-pressure change
is .062652%; outlet extensions are invariant to roundoff here. Individual
boundary mass flows vary substantially. Thus the tested finite-reservoir
range is a smaller inner-field effect than the measured mesh error, not a
proof of infinite-domain or physical film-height accuracy.

Evidence: `results/wavelength_front_20261028_final`,
`results/material_reservoir_20261028` and `_cells8`, with separate
`_audit_final` directories and raw shared-node difference figures. Source
hashes are unchanged. Production residual/JVP, force/field criteria,
constitutive laws, preconditioners and nonlinear budgets are unchanged;
GPU work remains secondary and the material/JFO model stays CPU-only.

Next: independently benchmark a conservative transport alternative against
smooth source/symbol/constant-law limits before any adoption; retain the
current stencil as the reference. Continue matched-force mesh comparisons
only with verified resource/scaling changes before exceeding the node cap.
These diagnostics prioritize the unresolved discrete pressure pattern; they
do not justify tuning material laws or interpreting correlation agreement
as physical validation. Larger-domain and additional-mesh verification remain open.

Verification: Ruff clean; complete Windows CPU/CUDA suite 453 passed, one
skipped (237.98 s). The existing wider-longdouble diagnostic skips because
Windows supplies no extra precision. Fourteen new CPU regressions cover the
diagnostics and reservoir initialization/repair; optional torch remains optional.

## Isolated transport alternative and resource measurements (2026-10-28)

Starting at 6ff85dc, independently checked a conservative donor-capacity
candidate and analytic smooth flux JVP. Production stencils and Newton paths
remain unchanged. Nonuniform telescoping, scalar faces, constant-capacity
parity, stationary h*theta, discrete symbols, full/zero-material JVPs and
manufactured continuum-source tests pass. Forty-three loaded elastic/front
directions pass; no candidate root or physical improvement is established.
The candidate adds first-order numerical diffusion and loses the reference's
second-order full-film smooth capacity transport. A nonzero alternating
density-advection response is insufficient grounds for adoption.

Measured the existing local preconditioner block and bordered fallback at
the three accepted meshes. Finest sparse LU payloads are about 14/60 MiB;
81 float64 state vectors require 16.8 MiB. Payloads exclude peak workspaces;
they do not validate larger meshes or CUDA memory use. Matrix-free storage
remains essential: the finest dense full Jacobian would require 5.50 GiB.
No node cap or nonlinear budget is raised.

The existing cached FFT-friendly ElasticityOperator reproduces accepted
deformation to <4.9e-16 relative error. Isolated timings identify reusable
spectra and FFT padding as a promising CPU improvement before a GPU port.
These timings are not an integrated solver speedup. Material/JFO remains
CPU-only, while existing optional CUDA tests retain their prior scope.

Next bounded milestone: integrate the already verified reusable elastic
operator behind an explicit reference/cached CPU comparison, verify full
bordered products/corrections/globalization and matched-force roots, and
measure actual peak process memory and Krylov work before any >20,000-node
trial. Separately investigate a conservative higher-order transport option
with smooth-source and front/positivity checks; retain the donor candidate
as a diagnostic baseline. Mesh/domain and physical film accuracy remain open.

## Optional cached CPU material elasticity (2026-10-28)

Starting from 69b6a54, reused the existing verified ElasticityOperator behind
an explicit `elasticity_mode="cached"` option in material force Newton,
load continuation and unaccepted-guess repair. Default reference execution,
the original flux, constitutive laws, sparse preconditioning, physical gates,
30-correction/eight-model budgets and 20,000-node cap are retained. This is
CPU optimization, not a GPU material/JFO port or a new transport scheme.

Independent small-grid tests check the full elastic/material bordered product,
dense-reference correction, Schur fallback, strict film rejection, both
globalization checks, rejected trials, active backup, Krylov failure and
continuation retry mode/seed preservation. Matched-force saved 200 N roots
on all three existing meshes undergo identical perturbed-offset repair in
both modes, then independent original field/force/history/direct-source audits.
See validation report section 26 for actual timings, parity and OS memory.

Three measured repairs per mode after warm-up give reference/cached speedups
2.93x, 2.44x and 2.27x on 33x49, 65x97 and 97x145. Pressure differences are
<1e-14 relative, total-film differences <6.8e-21 m, h0 <8.5e-22 m. Finest
worker-lifetime resident peaks are 162.32/164.27 MiB, including startup,
warm-up and audits. This is measured local repair performance, not a cold-
continuation speedup or peak estimate for a larger grid. Complete CPU/CUDA
suite: 470 passed, one Windows longdouble skip (208.07 s); Ruff clean.
Seventeen new tests also pass with torch blocked.

Remaining accuracy work: investigate a higher-order conservative transport
candidate with independent smooth/front tests before adoption; reproduce
matched-force mesh trends using explicitly bounded resource controls before
attempting the next 129x193 mesh above the current cap. Peak-memory evidence
from local repairs does not validate that larger mesh or cold continuation.
Physical film accuracy and full material/JFO CUDA solving remain open.

## Bounded fourth-mesh attempts and multigrid foundation (2026-10-28)

Starting at 4b1c55c, added an explicit 30,000-node allowance, preserving the
20,000-node default and every original physical/nonlinear gate. The 129x193
target retained the same 200 N case and +/-3a reservoir. Two independent
initializations (65x97 factor two and 33x49 factor four) passed resource/
factor preflights but exhausted 30 Newton corrections. Rejected states and
full histories are preserved; neither is a converged fourth mesh. Independent
audits find scaled mass defects .002841 and .079187 despite force closure.
No convergence order or new film-height prediction is established.

The accepted three-grid comparison quantifies shared-node pressure L2
differences 31.44% and 29.90%. Minimum film changes decrease, but central film
and peak pressure are nonmonotone over all three grids. Provisional unequal-
spacing fits are explicitly unqualified; no stencil is changed to smooth
the profiles. Numerical accuracy and the origin of the pressure pattern remain open.

Implemented and independently tested CPU geometric Galerkin V-cycles for
the prescribed/frozen-film SPD Reynolds operator: bilinear interpolation,
transpose restriction, Galerkin coarse matrices, symmetric damped Jacobi,
bounded exact Cholesky and true-residual checks. Smooth 33..513 square-grid
tests take eight preconditioned CG iterations across the range. Frozen
material-mobility surrogates converge in 21-22 iterations, with separate
known-pressure forward-error checks; full material derivatives are absent.

Optional MG also enters the existing signed full-film mathematical bordered
Newton solver through its pressure preconditioner, retaining nonsymmetric
GMRES, Schur/fallback, inexact forcing, dual line search and convergence
criteria. Zero/weak/moderate compliance and recovery tests pass. Small-grid
timings do not justify replacing default ILU: MG is slower on two tested
grids and only modestly faster on the largest. These are mathematical tests,
not physical 200 N material/JFO validation.

Next: develop and independently verify a pressure/content block preconditioner
that incorporates MG while preserving the material/JFO active constraints,
full elastic/material JVP and load border. Revisit 129x193 only with explicit
resource/iteration diagnostics and original gates. Nonlinear FAS, robust
anisotropic/high-contrast scaling, material/JFO MG and CUDA MG remain open.
Do not infer nonlinear convergence or mesh accuracy from elliptic MG speedups.

Verification: complete CPU/CUDA suite 490 passed, one Windows precision skip
(243.33 s); Ruff clean. Sixty-five resource/MG/monolithic tests also pass
with torch imports blocked. Validation report sections 27-28 retain actual
failed-refinement residuals and both useful and slower MG timings.

## Active pressure/content multigrid prototype (2026-10-28)

Continued from `1ab27ba` on `develop`.

Added a diagnostic-only `PropertyBlockMultigrid` inverse for the frozen
material/JFO pressure-content block. Boolean active pressure constraints enter
the SPD hierarchy before Galerkin coarsening. Known complementarity corrections
are eliminated exactly; projected pressure V-cycles and exact lower-triangular
cavitated donor-content solves form a fixed nonsymmetric block inverse. It
omits unknown cavitated-content coupling into full-cell mass rows. The complete
elastic/material Jacobian and force border remain outer Krylov operators.

Independent tests cover full, cavitated, front and disconnected active sets;
masked SPD/Galerkin structure; known-pressure recovery; linearity and exact
constraint rows; dense surrogate parity at the exact coarse level; material
JVP finite differences; and true-residual/dense-reference checks for both
Schur-reconstructed and full bordered GMRES corrections on a small material
case. This is algebraic verification of a prototype, not nonlinear JFO MG,
129x193 convergence, timing/scaling evidence or physical film validation.
Production solver selection, stencils, line searches, fallback paths and
resource/nonlinear acceptance budgets remain unchanged.

Next: compare this block inverse with existing frozen/material/local sparse
preconditioners on accepted and rejected loaded states, recording true Krylov
residuals, iterations, setup/application time and memory. Only after that
comparison should an explicit production option be considered, with complete
nonlinear feasibility, dual-globalization, recovery and continuation tests.
Do not infer material/JFO scalability from the masked SPD tests.

Verification on Linux CPU: repository-wide `ruff check .` and `git diff
--check` pass. With `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src`,
`.venv/bin/pytest -q` reports 440 passed, 62 optional GPU/backend skips
(332.47 s). The final targeted prototype run reports 13 passed (19.42 s),
including two transport-rejection tests added after full-suite collection.
The MG/prototype combined run before those last two cases reports 19 passed.
No GPU execution or loaded-state performance is asserted by these checks.

## Loaded-state block-MG comparison (2026-10-28)

Continued from `1432462`: added an isolated-process comparison runner for
frozen/material/local sparse full-border inverses and the active block-MG
prototype. It uses the complete material/elastic JVP, matched CPU caching,
independent true-residual checks and a known-direction probe. Accepted roots
and rejected partials retain separate source audits/classifications; reports
preserve hashes, raw vectors, Krylov failures, setup/action/solve timings and
worker-lifetime resident-memory measurements. Production equations, nonlinear
paths, acceptance gates and resource caps are unchanged.

Reproducible CPU command, from repository root with the optional GPU runtime
unneeded (the output directory must be new):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python \
  -m examples.benchmark_property_block \
  --input-directories \
    results/block_mg_inputs_20261028/accepted_33x49 \
    results/block_mg_inputs_20261028/accepted_65x97 \
    results/block_mg_inputs_20261028/accepted_97x145 \
    results/block_mg_inputs_20261028/rejected_97x145 \
  --output-directory results/block_mg_comparison_REPRODUCE \
  --repeats 3 --restart 60 --cycles 4
```

Input snapshots were extracted without alteration from preserved commit
`a4bee767fe01fc79793c20aec70ae359ea5d13c3`. The accepted 33x49 source is
`results/recovered_200n_20261027/regenerated_load_33x49` (last accepted state
`attempt_043/solution.npz`); the accepted 65x97 and 97x145 sources are
`results/content_bound_pivots_20261027/content_pivot_refinement_factor2` and
`content_pivot_refinement_factor3`; the rejected 97x145 source is
`results/lookahead_selection_20261027/best_candidate_refinement_factor3`.
Each input copy includes its original summary and selected state; the local
input manifest retains original Git paths and SHA-256 hashes. Inputs and
measurement outputs are Git-ignored, and no prior results are overwritten.

Measured outcome (three observations per worker, four states): material and
local full-border preconditioning pass both complete-JVP RHS checks in all
repeats. Local Newton iterations are 22/29/33 on accepted 33x49/65x97/97x145
roots and 31 on the rejected 97x145 partial; material takes 108/95/85/85.
Frozen and block MG exhaust 240 iterations on both RHS vectors throughout,
with true residuals near one. The current prototype is therefore not selected
for production. Lower MG payload/RSS does not compensate for failure to solve.
Exact timing, memory, forward-error and correction-parity results are in
validation report section 30; raw traces, source/vector hashes and inspected
plots are retained in `results/block_mg_comparison_20261028_final`.

Next bounded step: investigate a diagnostic pressure/content inverse that
retains material and local elastic tangent information, comparing it against
this measured reference with complete-JVP/force checks before considering
production integration. Keep existing local sparse preconditioning and every
nonlinear, physical and resource gate unchanged. No additional mesh is accepted
and no physical film-height or GPU material/JFO claim is added.

A separate original-FFT/SI-flux audit of all 32 saved directions confirms the
reported passing/exhausted classification; maximum independent mass-action
error is 5.22e-15 relative. Evidence and its reproduction script are retained
in `results/block_mg_comparison_20261028_audit`.

To recover the four input snapshots on a fresh clone, fetch the preserved
archive branch, then extract only the original summaries and selected states:

```bash
git fetch origin feature/hip-backend
python - <<'PY'
import json
import subprocess
from pathlib import Path

commit = "a4bee767fe01fc79793c20aec70ae359ea5d13c3"
root = Path("results/block_mg_inputs_20261028")
root.mkdir(parents=True, exist_ok=False)
paths = {
    "accepted_33x49": "results/recovered_200n_20261027/regenerated_load_33x49",
    "accepted_65x97": "results/content_bound_pivots_20261027/content_pivot_refinement_factor2",
    "accepted_97x145": "results/content_bound_pivots_20261027/content_pivot_refinement_factor3",
    "rejected_97x145": "results/lookahead_selection_20261027/best_candidate_refinement_factor3",
}
for name, source in paths.items():
    summary = subprocess.check_output(["git", "show", f"{commit}:{source}/summary.json"])
    data = json.loads(summary)
    accepted = [e for e in data["grids"]["refined"]["attempts"]
                if e.get("accepted") and abs(e.get("load_n", 0)/200-1) <= 1e-8]
    state = accepted[-1]["state_file"] if accepted else "failed_partial.npz"
    for relative, content in (
        ("summary.json", summary),
        (state, subprocess.check_output(["git", "show", f"{commit}:{source}/{state}"])),
    ):
        target = root / name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
PY
```

This reads preserved Git objects without switching branches or replacing source
code. The benchmark recomputes original field/force/film audits and records
input hashes before accepting any classification. No generated state is
reconstructed from correlations or promoted by interpolation.

Verification: repository-wide Ruff and whitespace checks pass. Complete Linux
CPU suite: 459 passed, 62 optional backend/GPU skips (312.52 s), with
BLAS/OpenMP/MKL threads one. Thirty benchmark/prototype regressions also pass
as a targeted run (19.95 s). The 17 new benchmark regressions independently
check dense full-border parity, exact active/force rows, true-JVP corrections,
false-success/exhaustion reporting, source mutation/film/force gates, setup
rejection and retained worker failures. No GPU execution is asserted.

Local evidence archive: `results/block_mg_evidence_20261028_verified.zip`,
12,078,003 bytes, 151 files; every manifest member SHA-256 and ZIP CRC verified.
Archive SHA-256:
`a73459777143a34dd204f2210ff0cd946054ee8310593d32be0495a0279a8b44`.
It contains exact original inputs/provenance, raw comparison vectors/traces,
measured source snapshot, inspected plots, independent audit and published
validation source. The archive is Git-ignored; no off-machine copy is asserted.

## Tangent-preserving active reduction and nonsymmetric MG (2026-10-28)

Continued from `9150695`: independently implemented exact active-row
elimination with all material/local pressure and content mass couplings.
The n-unknown reduced block supports two explicitly selected diagnostics:
exact sparse LU, and a fixed nonsymmetric geometric Galerkin V-cycle with
ILU smoothing and bounded coarse LU. Neither is wired into production Newton.
Original frozen block MG, SPD Reynolds MG, residual/JVP, material laws, force
row, nonlinear budgets/gates and physical node caps remain unchanged.

The diagnostic runner retains its four historical default modes. To compare
new tangent hierarchies and exact-reduction controls with both sparse references:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python \
  -m examples.benchmark_property_block \
  --input-directories \
    results/block_mg_inputs_20261028/accepted_33x49 \
    results/block_mg_inputs_20261028/accepted_65x97 \
    results/block_mg_inputs_20261028/accepted_97x145 \
    results/block_mg_inputs_20261028/rejected_97x145 \
  --output-directory results/tangent_block_comparison_REPRODUCE \
  --modes local material local-reduced material-reduced local-mg material-mg \
  --repeats 3 --restart 60 --cycles 4
```

Use the existing unchanged input recovery instructions above; every output
directory must be new. Failed setup, exhausted solves and rejected physical
input classifications remain explicit. No coarse correction is interpreted as
a new physical state, and no failing MG inverse invokes the reduced-LU control.

Measured outcome: exact reduced LU retains both sparse references' iteration
counts on all four states and both RHS vectors. At the accepted 97x145 grid,
local inverse setup/application are 3.93x/4.61x faster, worker-lifetime RSS falls
271.97 -> 164.32 MiB, and factor payload falls 60.31 -> 5.79 MiB. This is an
isolated full-border diagnostic, not an integrated nonlinear speedup or a
validated larger-grid memory allowance. All original force/mass/film gates
and the rejected input's classification are unchanged.

Local tangent MG passes the 33x49 case (47/41 Newton/known-vector iterations),
but stalls at 240 on 97x145 and has coarse ILU failure on 65x97 and the rejected
partial. Material-only MG stalls on 33x49 and has coarse setup failures on the
other inputs. No regularization or fallback hides those outcomes. Exact
active-row algebra is verified independently; the mixed geometric hierarchy
is not robust enough for production.

A separate audit rechecks all 38 saved directions, confirms 34 passing/four
exhausted classifications, and retains five setup rejections. Independent SI
mass-action error is <=7.05e-15 relative; eight full/reduced inverse comparisons
also pass with sparse-surrogate residual <=1.85e-16. See validation report
section 31 for timings, memory scope, action differences and limitations.
Raw outputs/source snapshots and audit scripts are retained in new
`results/tangent_block_comparison_20261028_final` and `_audit` directories.

Next: introduce exact reduced LU as an explicit optional field preconditioner
with complete Newton feasibility/globalization/recovery/continuation tests and
independent matched-force audits. Retain the current production default and
fallback behavior until those checks pass. Treat phase-aware coarse spaces
and robust nonsymmetric MG as separate research work; neither unresolved mesh
accuracy nor physical film-height validation is advanced by linear speedups.

Verification: complete Linux CPU suite 499 passed, 62 optional backend/GPU skips
(333.63 s), with BLAS/OpenMP/MKL threads one. The targeted tangent/benchmark
run reports 57 passed (15.86 s), covering 36 new tangent tests and extended
benchmark paths. Repository-wide Ruff and whitespace checks pass; both raw
cost/status figures were visually inspected. No GPU execution is asserted.

Local evidence archive: `results/tangent_block_evidence_20261028_verified.zip`,
13,233,407 bytes, 162 files; all SHA-256 manifest entries and ZIP CRCs verified.
SHA-256: `939061f061cf1679bf65a1e1c450ea1dce13cdd0d0b4e46972e39db4da48a514`.
It retains unchanged inputs/provenance, exact measured sources, all worker
reports/vectors, inspected plots, independent audit and validation source.
Generated data/archive remain Git-ignored; no off-machine copy is asserted.

## Optional reduced field LU in actual material Newton (2026-10-29)

Continued from `5726ff8`: integrated exact active-row reduction as
`field_factorization="reduced"` in the material force Newton, continuation and
unaccepted-guess repair APIs. The default remains `"full"`. Frozen/material/
local sparse tangent and reference/cached elasticity selections are independent.
One already assembled sparse mass/constraint matrix supplies the reduced
factorization. Experimental MG is not selected by production.

Original true-JVP Krylov checks, both Schur solves, full-border recovery,
complete bordered correction check, strict total-film feasibility, both
line-search stages, active backups and continuation seed/retry rules remain.
Factor failures and failed true residuals are now retained in success/error
and nonlinear model diagnostics, including recovery through full bordered LU.
No physical equation, load/field gate, nonlinear budget or node cap changes.

Reproduce matched-force repairs on the existing three accepted inputs (fresh
output directory required). Both modes use cached CPU elasticity and the same
unaccepted +0.01 Href offset perturbation of each 200 N source:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python \
  -m examples.benchmark_reduced_material \
  --input-directories \
    results/block_mg_inputs_20261028/accepted_33x49 \
    results/block_mg_inputs_20261028/accepted_65x97 \
    results/block_mg_inputs_20261028/accepted_97x145 \
  --output-directory results/reduced_field_repairs_REPRODUCE --repeats 3
```

Each isolated worker warms once and measures complete repair calls, not just
linear inverse actions. Every returned state/history is independently checked
against the original SI mass/force/film and direct-source deformation gates;
all measured histories and pressure/content/film hashes are retained. The
parent verifies output identity and matched-force parity. Timings exclude
independent post-call audits; worker-lifetime RSS includes them and warm-up.
Historical linear timings are not substituted for actual Newton measurements.

Measured actual full/reduced median repair ratios are 0.927/1.140/0.997 on
33x49/65x97/97x145: slower on the coarsest case, modestly faster in the middle,
and effectively unchanged on the finest. Original Newton normally solves two
field blocks plus a load Schur response; prior isolated full-border inverse
speedups do not predict whole-repair performance. The default stays unreduced.
The reduced finest run includes a 13.124 s sample; all samples/ranges are kept.
No integrated memory improvement is claimed.

Both modes pass original physical/numerical gates and take 4/4/5 Newton
corrections; pressure differences are <=6.12e-15 relative and film differences
<=6.78e-21 m. Finest Krylov totals differ by one (332/331). A separate audit
reloads all six saved states and 18 measured histories, checks unchanged
source/output hashes and original mass/force/film/direct-source diagnostics.
Maximum mass defect is 1.85e-13 and direct displacement discrepancy 4.24e-21 m.
See validation report section 32 and the new `_final`/`_audit` directories for
actual timings, scopes and inspected cost/convergence plots.

Next bounded work: return to nonlinear conditioning/globalization and rejected
mesh diagnostics with original gates and existing resource controls. Preserve
the default; use reduced LU only as an explicit comparison. Experimental MG,
unresolved 129x193 convergence and physical film-height validation remain open.

Verification: repository-wide Ruff and whitespace checks pass. Complete Linux
CPU suite: 525 passed, 62 optional backend/GPU skips (321.72 s), with
BLAS/OpenMP/MKL threads one. Targeted integration/benchmark/globalization checks:
37 passed (8.80 s), including 26 new regression cases. Both raw timing and
mass-convergence figures were visually inspected. No GPU execution is asserted.

Local archive: `results/reduced_field_evidence_20261029_verified.zip`,
3,041,195 bytes and 133 files; all manifest SHA-256 entries and ZIP CRCs verified.
SHA-256: `acf02d4af362a397bc666a11e3d90ce6962d8228435a93bd5b4aeb42cce7e872`.
It preserves source inputs/provenance, exact measured sources, all audits,
histories/output hashes, raw timings, inspected plots and validation source.
Data/archive remain Git-ignored; no off-machine copy is asserted.

## Bounded fourth-mesh Newton diagnosis (2026-10-29)

Continued from `79a2b74`: added isolated read-only state observation, exact
internal-state/selected-row snapshots, SI feasibility-limit diagnostics and
replay tools. The default numerical path has no observer. Public repair and
instrumented runs share original guess-field validation. Observer failures are
reported separately from convergence failures. No residual, JVP, line-search
choice, active-pivot rule, physical gate, node cap or nonlinear budget changes.

Reproduce from the existing accepted 65x97 source, using new output directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python \
  -m examples.diagnose_mesh_newton \
  --input-directory results/block_mg_inputs_20261028/accepted_65x97 \
  --output-directory results/mesh_newton129_REPRODUCE \
  --maximum-physical-nodes 30000

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python \
  -m examples.analyze_mesh_newton \
  --input-directory results/mesh_newton129_REPRODUCE \
  --output-directory results/mesh_newton129_REPLAYS --iterations 0 15 17 29

# Separately inspect every alternative at the smallest accepted step.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python \
  -m examples.analyze_mesh_newton \
  --input-directory results/mesh_newton129_REPRODUCE \
  --output-directory results/mesh_newton129_ALTERNATIVES --iterations 15 --all-models
```

The 129x193 run passes preflight (3.51e-13 factor residual; about 196 MiB RSS)
but exhausts the original 30 corrections. Force remains 200 N; mass defect
ends at .0028407992 versus the 1e-8 gate, with local/global balance also failing.
This is still NOT an accepted fourth mesh. It records 31 exact states,
118 active models and 8,236 Krylov iterations. All returned linear models pass
true residual checks (maximum 8.71e-10); no field setup failure, block Krylov
failure or full-border recovery is recorded in this run.

Of 118 directions, 79 are non-descending for the original merit and therefore
receive no line-search trial. The 39 descending models yield 30 accepted
steps plus nine deferred alternatives. Recorded nonaccepted trial reasons are
38 pressure/content infeasibilities and nine deferred feasible Armijo steps;
there are no recorded Armijo-only rejections. At iteration 15, a weak-pressure
outer-reservoir content-upper limiter restricts the chosen safe alpha to
8.70e-7. A 147-row alternative is more pressure-limited (alpha 1.19e-7) and
less effective; the remaining two alternatives are non-descending. The original
best verified backup is reproduced. Iteration 17 takes alpha .798 and the final
iteration takes a full step and makes substantial progress: this is budget
exhaustion with restricted intermediate progress, not proof of a nonexistent
root or complete stagnation.

All four accepted-step replays reproduce their stored next states and slopes
exactly on this environment. All four models at iteration 15 are separately
checked. Seven distinct directional products agree with the independent SI
flux decomposition within 1.89e-16 relative error. No field smoothing or
pressure/content clipping is introduced. Validation report section 33 records
full terminal gates, coordinates, alternative-model checks and limitations.

Next bounded hypothesis: independently prototype prioritizing only the earliest
content-upper (D-lower) blocker before broad whole-step active flips, retaining
the current reference/default and all true-JVP/merit/feasibility checks. Verify
manufactured active-bound cases and existing physical benchmarks before any
adoption. Do not increase the iteration budget or relax bounds to accept this
partial. Mesh/domain and physical film-height validation remain open.

Verification: complete Linux CPU suite 542 passed, 62 optional backend/GPU
skips (337.33 s), BLAS/OpenMP/MKL threads one. The 17 new CPU regressions cover
SI bounds, observational isolation, terminal/failure capture and raw model
replay/classification. Ruff and whitespace checks pass. Both raw convergence/
spatial figures were visually inspected. No GPU execution is asserted.

Local evidence archive: `results/mesh_newton_evidence_20261029_verified.zip`,
32,913,819 bytes and 477 files; every manifest SHA-256 and ZIP CRC verified.
SHA-256: `035f719ae23a2477362a8d2d81793f68cab1eea6e892e7fe412b034f9a4424ad`.
It retains unchanged inputs, exact solver/replay sources, all states/histories,
alternative checks, independent SI-flux directions/audit and inspected plots.
The archive is Git-ignored; no off-machine copy is asserted.

### Supplied MATLAB reference checkpoint (2026-10-29)

Prioritize same-case MATLAB reference comparison before further fine-grid
active-set tuning. The independently reconstructed 50 N N33 alpha=0.8978125
checkpoint passes its saved pressure-only FB/load gates. Full target viscosity,
Python JFO parity and physical/grid validation remain unverified. See
[MATLAB_REFERENCE.md](MATLAB_REFERENCE.md); the diagnostic evaluator is not a
new production solver or a MATLAB runtime reproduction.

### Pressure-only MATLAB Newton recovery reproduced (2026-10-29)

Implemented a separate diagnostic N33 finite-difference bordered Newton and
adaptive viscosity replay, retaining the supplied source's reduced active-set,
Schur/direct/LM competitors, pressure projection, relative-decrease line search
and unchanged gates/budgets. A 1.94359 nm offset perturbation is recovered in
11 corrections; pressure changes by 3.96e-6 relative L2 and minimum film by
3.69e-13 m from the original checkpoint. This is nontrivial solver recovery,
not just re-evaluation of an already accepted state.

The exact source continuation schedule rejects alpha 0.9 and 0.89890625 after
20 corrections apiece, then reaches its minimum increment stop; full viscosity
remains unachieved. An independent SI direct-convolution/face-flux verifier
accepts the recovered reference root and rejects both failed trials. It also
quantifies the pressure-only cavitation mass defect; none of these states is
promoted to a JFO/physical benchmark. See [MATLAB_REFERENCE.md](MATLAB_REFERENCE.md)
for reproduction commands, numerical results and known MATLAB-runtime limits.

Final verification of this increment: 556 passed, 62 optional skips (333.77 s),
Ruff/whitespace checks clean. Fourteen focused audit/reproduction checks pass.
Runtime-source snapshots, checkpoint, accepted/failed physical fields, histories,
independent SI audits and inspected plots are preserved locally under the
Git-ignored reproduction results directory; no simulation datasets are staged.

### Controlled reference globalization investigation (2026-10-29)

Isolated the first pressure-only MATLAB replay stall through a four-way
ablation: directional Armijo alone exactly reproduces the failed sequential
baseline, while competing the original active/Schur/LM candidates solves
alpha=0.9 in two corrections under either decrease test (baseline fails after
20). Added an analytic one-sided projected-merit derivative with nonlocal
elasticity, material laws, Reynolds scaling, FB kinks and force border; the
compatibility solver default remains unchanged.

Both competition policies advance the original schedule to verified
alpha=0.9189712524414063 (six accepted, four rejected stages), then stop at the
minimum increment. Full viscosity is still unachieved. Lower-damping probes
at a later stalled state do not establish a cure. SI direct-convolution/flux
checks audit every attempted state. The last reference-accepted state has
large mass imbalance, including at positive-pressure nodes: the coarse scaled
FB/load gates are insufficient for physical Reynolds acceptance. Do not relax
production mass gates or promote these reference states to physical/JFO roots.
See [MATLAB_GLOBALIZATION.md](MATLAB_GLOBALIZATION.md) for commands, ablation,
limitations and the next conservative 50 N comparison.

Verification: 563 passed, 62 optional skips (371.67 s); 21 focused reference
checks passed; Ruff/whitespace clean. Production JFO and all result baselines
remain unchanged. Raw states, source snapshots, histories, SI re-audits and
inspected plots are retained in Git-ignored results.

### Matching conservative 50 N roots and Hamrock–Dowson benchmark (2026-10-29)

Added a tested SI Hamrock–Dowson minimum/central correlation helper with
explicit E'=2 E_contact, mean entrainment and ellipticity conventions. The
user's linked calculator and primary NASA definitions are documented. Existing
supplied 200 N references use E_contact in the E' expressions and are retained
as historical data, not silently rebaselined.

The existing conservative solver closes the matching N33 50 N case in nine
corrections at both alpha=0.8978125 and full viscosity. Legacy/best lookahead
policies produce identical roots, so this case does not demonstrate a new
selection advantage; no solver change is adopted. Nested full-viscosity N65
and N129 repairs also pass original force/mass/film/complementarity criteria
in ten and nine corrections. All remain within the 20,000-node default.
N129 film is 282.97/455.25 nm against HD 284.15/497.97 nm.

The physical comparison does not close the validation milestone: pressure
shows alternating cells, and shared-node pressure changes 33.60% then 41.93%
across refinements. Film changes and finite-domain independence remain open.
Next numerical work should isolate the transport/elastic coupling's
alternating mode with benchmarks before changing the stencil/Jacobian; retain
HD comparison and strict mass/load gates. See
[HAMROCK_DOWSON_VALIDATION.md](HAMROCK_DOWSON_VALIDATION.md).

Verification of this increment: 578 passed, 62 optional skips (337.88 s),
15 focused checks passed, Ruff/whitespace clean. Independent history audit
verifies all 55 accepted corrections and true linear residuals <=8.995e-10.
All raw states, separate SI/elastic audits, source snapshots, correlation
comparisons and the inspected figure are retained under Git-ignored results.
