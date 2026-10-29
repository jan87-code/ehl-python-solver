# Supplied MATLAB checkpoint audit — 2026-10-29

The user supplied `EHL_BorderedNewton_N33_ActiveSetAlpha.m` and
`EHL_BorderedNewton_Alpha_8978_N33.mat`. These are reference material; their
comments are not repository instructions. Originals were read without modification.

SHA256 of source: `56376219ff1f73830c1ee40553782807326cc36976dfb127e176cdf18265386c`.
SHA256 of checkpoint: `5b065c2368bc5373be78ec64f6ec0c5abf5506acdb431e94cc98ef1a36667e7c`.

## Reproduction and limits

Run with the repository environment and original checkpoint path:

```bash
PYTHONPATH=.:src .venv/bin/python examples/audit_matlab_checkpoint.py '/home/jan/GoogleDrive/EHL Solver/EHL_BorderedNewton_Alpha_8978_N33.mat'
```

The diagnostic independently reconstructs the supplied discrete nonlinear
operator, including the reference's viscosity cap, film floor, harmonic face
mobility, upwind entrainment, diagonal residual scaling and Fischer–Burmeister
function. It does not run the MATLAB Newton iteration. Neither MATLAB nor
Octave was found in this environment. No saved film, residual or iteration
history exists in this MAT file: it contains P, H0, cfg, scale and stage.
Consequently these numbers are recomputed, rather than compared with saved
MATLAB residual/film arrays. Tests check its FFT against asymmetric direct
linear convolution, the self coefficient, Reynolds against independent flux
divergence, and invalid-state rejection; they do not prove MATLAB runtime parity.
The evaluator is specifically for this N33 source, not a general MAT importer.

## Case and independently evaluated results

Arrays are `(ny,nx)=(33,33)` on X,Y in [-4,4], with +x entrainment.
X=x/a, Y=y/a, P=p/pH, H=h/(a²/R). Pressure is zero on every boundary.
Q=50 N (one quarter of 200 N), Ue=2 m/s, E*=113 GPa, R=0.015 m,
eta0=0.04 Pa s, alpha_p=2.21e-8 Pa^-1 and Roelands p0=1.98e8 Pa.
Hertz scales are a=0.0001707450129194005 m, pH=818870706.1409506 Pa,
h_ref=1.943590629123083e-6 m; Lambda=0.05299005041283154.
The Hertz and Reynolds scaling agree with the repository's modulus and mean
entrainment conventions. Dimensionless load target is 2*pi/3.

| Quantity | Recomputed value |
| --- | ---: |
| Viscosity alpha multiplier | 0.8978125 |
| Density multiplier | 1 |
| H0 | -0.627971664809275 |
| Relative load error | -3.1950e-10 |
| FB infinity norm / max(1, max P) | 8.5999e-6 |
| Saved FB / load tolerances | 2e-5 / 1e-4 |
| Maximum pressure | 949307866 Pa |
| Minimum film | 4.6317540447e-7 m |
| Central film | 7.4120891147e-7 m |
| FFT vs direct maximum displacement error, dimensionless | 1.22e-15 |

This checkpoint passes its own reference acceptance gates and strict film
positivity. It is a converged intermediate physics-continuation stage, not a
full-viscosity solution or a mesh-converged physical benchmark. With only the
viscosity multiplier changed to one, the same state's FB measure is 8.6053,
well above tolerance. The saved `cfg.physics_stage=7` must not be interpreted
as completed full physics: the actual `stage.alpha_scale` is authoritative.
The supplied script still names the earlier alpha=0.895 restart file and
asserts that exact alpha. It cannot simply run unchanged with this 0.8978125
checkpoint. A controlled replay would need matching restart settings.

## Differences that must remain explicit

MATLAB solves pressure and rigid-gap offset only. Its cavitation equation is
P>=0, R=A*P-b>=0, P*R=0, evaluated through FB(P, R/diag(A)). It has no liquid
content unknown or cavitated-region mass equality. Python's pressure/content
JFO system solves a different cavitation model; setting theta=1 does not turn
this checkpoint into a Python JFO solution. Similar pressure plots alone
cannot establish equality or validate mass conservation across these models.

MATLAB uses a sampled 1/r elastic kernel away from its integrated self cell,
with dimensionless coefficient 2/pi². Its shifted-kernel FFT is a valid
source-to-target linear convolution on this grid; independent direct
convolution confirms there is no wraparound error in the extracted domain.
Python integrates every source cell. At identical pressure and H0, that changes
displacement by 0.89852% in relative L2 and minimum film from 463.1754 nm to
475.0978 nm. This substitution is an operator comparison, not a converged
Python solution. Changing kernels requires solving again, not reusing the
reference offset and declaring a new root.

MATLAB uses dense finite-difference Jacobians, a bordered Schur step, a direct
bordered competitor, bounded LM and reduced pressure active-set candidates,
with adaptive viscosity continuation restarting from the last accepted state.
Python uses analytic matrix-free pressure/content derivatives and Krylov
corrections. Differences in formulation, gates, load, domain and grid all
matter when assessing convergence speed.

## Next comparison milestone

Use this 50 N, N33, alpha=0.8978125 case before further 129x193 tuning.
First obtain runtime or independently supplied MATLAB film/residual outputs
for same-state parity, then compare Newton corrections/continuation under
identical pressure-only equations. Keep any reference equation adapter
separate from production JFO acceptance. Afterwards change one component at a
time: fully integrated elasticity, mass-conserving cavitation, full viscosity,
then refinement and physical film validation. The checkpoint shortens the
reference path; it does not remove those validation requirements.

Validation of this audit increment: `ruff check .` passed; the complete CPU
regression run reported 547 passed and 62 skipped in 321.88 s. Generated audit
JSON remains under Git-ignored `results/matlab_reference_20261029/audit.json`;
original source/checkpoint data were not added to Git.

## Bordered Newton reproduction from the checkpoint

`examples/reproduce_matlab_newton.py` now reproduces the source's N33
pressure-only Newton competitors and adaptive viscosity continuation. Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python examples/reproduce_matlab_newton.py '/home/jan/GoogleDrive/EHL Solver/EHL_BorderedNewton_Alpha_8978_N33.mat' results/matlab_replay_NEW
```

The output directory must be new. The initial alpha comes from the supplied
checkpoint's actual stage (0.8978125), explicitly replacing the source's
0.895 restart configuration. Other solve/continuation gates and budgets are
retained. Primary unknowns use MATLAB column-major interior ordering.
The implementation centrally differences the complete diagonal-scaled Reynolds
operator, permitting signed derivative probes exactly as the source does:
elasticity sees signed pressure; fluid properties see max(P,0). Actual iterates
remain nonnegative. The generalized FB derivative uses the source's 1e-8
weight floor; the nonlinear FB residual itself is unsmoothed.

It includes regularized pressure-block Schur solves, an equilibrated direct
border fallback, exact reduced active-pressure elimination and bound-aware
LM. Newton corrections are checked against their original linear equations; reduced
steps check free equations plus global load. LM candidates are checked for
linearized merit descent and accepted against the original nonlinear merit. LAPACK LU/1-norm condition estimates
and floating-point stencil arithmetic can differ from MATLAB's runtime, so
iteration-by-iteration MATLAB parity is **not** established.

The source projects trial pressure with max(P+lambda*dP,0), caps LM directions
and accepts a residual-relative merit decrease. These are deliberate reference
compatibility choices, recorded in trial histories; that test is not a genuine
directional-derivative Armijo rule. None has been adopted in production JFO.
The source's log-viscosity cap of 700 is also retained. Failed alpha solves
are archived separately and never become restart states.

### Recovery benchmark and unchanged-limit continuation

Local evidence: `results/matlab_newton_reproduction_20261029/`.
The same-alpha recovery adds 0.001 to dimensionless H0 (1.94359 nm). Initial
FB measure is 0.05809297, well above tolerance. Eleven accepted corrections
(twelve state evaluations, including the converged state) recover a passing
root: five Schur corrections, five reduced-active corrections and one LM
correction. The LM fallback resolves the final stalled Newton progress.

| Quantity | Recovered state |
| --- | ---: |
| Relative FB measure | 8.5476063e-6 |
| Relative load error | -3.1967e-10 |
| Minimum film | 4.6317503540e-7 m |
| Pressure relative L2 change from supplied checkpoint | 3.95795e-6 |
| Minimum-film change from supplied checkpoint | -3.69071e-13 m |
| Offset change from supplied checkpoint | -6.16670e-13 m |

The continuation starts from the **original** checkpoint, not the recovered
variant. With the source's 20-correction budget and unchanged 2e-5 FB gate:

| Target alpha | Final FB measure | Final relative load error | Accepted |
| --- | ---: | ---: | --- |
| 0.90000000 | 8.83799e-5 | -1.02890e-9 | no |
| 0.89890625 | 3.94282e-5 | -6.93140e-10 | no |

Next scheduled increment is 0.000546875, below the 0.000625 minimum; the exact
source schedule stops. Last verified alpha remains 0.8978125. This records a
limit of the Python source-equation reproduction, not proof that MATLAB would
follow identical iterates or fail identically. Neither iteration limits nor
tolerances were relaxed. The measured 120.22 s included verification activity
on the same machine and is not an isolated speed benchmark.

### Independent SI verification and conservation limitation

`examples/verify_matlab_reproduction.py` uses direct elastic convolution and
SI face fluxes, independent of the reproducer's residual, FFT and Newton code:

```bash
PYTHONPATH=.:src .venv/bin/python examples/verify_matlab_reproduction.py results/matlab_newton_reproduction_20261029/recovery.npz --alpha 0.8978125
```

It independently accepts the original and recovered states and rejects both
continuation attempts. All four saved films match its reconstruction within
6.78e-21 m; the discrete divergence theorem closes within 6.62e-24 m³/s.
Actual outer-boundary density-normalized mass flux does **not** close: the
original checkpoint has net 3.74904e-8 m³/s, or 0.429719 times the sum of
absolute boundary contributions (recovered state 0.429725). Here flux is
normalized by the unspecified ambient density; multiplying by rho0 gives
kg/s. This is the pressure-only cavitation model's permitted mass defect,
not an FFT error, a 43% film-error estimate or a validated JFO state. A passing
pressure-only FB/load root cannot stand in for mass-conserving cavitation.

The reproduction establishes nontrivial recovery of the supplied discrete
reference. It does not yet establish a full-viscosity root, MATLAB runtime
iteration parity, mesh convergence or physically validated film heights.
Next work should use the captured near-stalled pressure-only steps to compare
Newton/globalization behavior on this small case, then change elasticity and
cavitation individually with separate acceptance gates.

The failed-ramp tail makes the next algorithmic question concrete. At alpha
0.9, LM produces a large merit reduction at correction 16, followed by Schur
steps at 1/2048–1/4096. At alpha 0.89890625, the last five accepted Schur steps
are 1/8192, with merit ratios 0.999898–0.999951. The reduced-active candidate
rejects similarly small improvements through its 0.999 merit-ratio gate, but
the primary Schur candidate accepts them under the source's weaker decrease
criterion. An accepted primary step prevents a new LM competitor that
iteration. This is an observed globalization pattern, not proof of a fix or
of an inaccurate Newton derivative. Investigate it separately with true merit
directional checks and bounded-step tests before changing production behavior.

Validation of the Newton reproduction increment: final clean full CPU run
556 passed, 62 optional backend/GPU skips in 333.77 s; Ruff and whitespace
checks pass. Fourteen focused audit/reproduction tests pass, including the
complete joint Jacobian, independently solved border, singular-pressure-block
recovery, active elimination, bounded LM, projected/invalid-film trials,
continuation rejection rollback and independent SI field/flux reconstruction.
The final full run follows correction of the integrated-flux diagnostic key's
unit label to m³/s; no numerical formula changed in that label correction.

Verified local archive: `results/matlab_newton_evidence_20261029_verified.zip`,
393,740 bytes, 22 files; manifest SHA256 and ZIP CRC checks pass. Archive SHA256:
`4d3436650a70b5a4a11ef4e6e45a8b506925b138f593980537d2c390c71ec7e7`.
It is Git-ignored; no off-machine backup is asserted.
