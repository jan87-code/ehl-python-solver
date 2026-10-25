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
