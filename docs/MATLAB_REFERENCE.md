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
