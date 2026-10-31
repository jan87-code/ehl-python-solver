# Limited-flux frozen linear corrections

The experimental limited candidate now has a matrix-free frozen linear
correction model and a matching sparse preconditioner. Small mathematical
benchmarks verify Schur reconstruction, bordered fallback, true linear
residuals, failure diagnostics and feasible directional Armijo comparisons.
**A nonlinear candidate solve, active-front/load-range validation and accepted
physical EHL results remain unfinished.** Donor/arithmetic defaults and
existing numerical baselines are preserved.

## Scope and interfaces

`limited_system.LimitedLoadProbe` contains the mathematical residual and true
one-sided directional derivative previously implemented in the standalone
`ResidualProbe`. The example retains that name as an import alias; its
numerical definitions are unchanged. The new
`LimitedLoadLinearization(case,state,guide,preconditioner=...)` freezes one
state and guide. It copies and marks those arrays read-only, rejects another
state, and supplies the interface needed by the existing verified
`solve_cavitating_correction` routine. A new state/guide requires a new model.
It is deliberately not a nonlinear iteration driver.

All public case inputs are SI, arrays `(ny,nx)`, +x mean entrainment, NumPy
float64. Nondimensionalization is the same as
[LIMITER_SWITCHES.md](LIMITER_SWITCHES.md): P=p/p_H, D=1-theta and h0/h_ref;
mass rows use a/(100*q_ref), complementarity is min(P,D), and load is relative
to the target in N. The guide freezes limiter and complementary row choices.
`PointContactCase.transport` selects the reference geometry/kinematic helper;
the actual operator here is explicitly `limited_capacity`, recorded separately
in the benchmark. No prototype CLI transport choice is changed.

## Operator and preconditioner

The matrix-free action retains the complete nonlocal cell-integrated elastic
response, pressure-dependent density/viscosity, content and signed offset.
Copied east/transverse content directions match the residual's boundary
closure. The frozen action is linear and agrees with the true directional
derivative along its guide. For another correction it can differ; the merit
slope is always recomputed using the actual one-sided derivative.

The sparse preconditioner is assembled analytically from pressure/content
expansions, physical face gradients/averages, shared-face divergence and the
same guide-selected capacity reconstruction. No dense Jacobian is assembled
for solving. The wider reconstruction stencil and copied outlet content are
included. Pressure diffusion retains arithmetic mobility.

- `material`: freeze film in the field block, retaining the full local
  pressure derivatives of density and viscosity.
- `local`: additionally retain the cell-self elastic compliance in the field
  block. Nonlocal elastic couplings are still omitted from preconditioning.

Both keep the true full offset column and integrated-pressure force row.
Sparse matrix copies protect cached preconditioning data from caller mutation.
Independent tests compare the entire assembled action with an analytic flux
oracle using the declared frozen/local film approximation, separately from
the full nonlocal action. The sparse preconditioner is not presented as the
exact coupled Jacobian.

The existing correction engine is reused without changing its budgets or
checks: ten Krylov cycles, restart 60, krylov_rtol=1e-9, independently measured
subsolve residuals and reconstructed-correction residual <=5e-9, with full
bordered recovery for an unsafe/failed Schur path. Here a "true linear
residual" means residual of the **frozen analytic linear model**, independently
of its sparse preconditioner. It does not mean F'(z;d) equals B_g*d for every
new correction.

## Benchmarks, 2026-10-30

The primary cases are unaccepted weak full-film mathematical states: target
1 N, offset 2 micrometers, positive interior pressure near 0.02*p_H, theta=1,
registered full Roelands/Dowson–Higginson fluid. They do not close Reynolds or
load before the correction and are not representative physical-load or
active-cavitation validation. No MATLAB or accepted checkpoint is needed.

| Grid | Preconditioner | Independent correction residual | Total Krylov iterations |
| --- | --- | ---: | ---: |
| 5x7 | material | 2.83e-12 | 8 |
| 5x7 | local | 4.67e-10 | 6 |
| 9x13 | material | 2.94e-12 | 8 |
| 9x13 | local | 2.51e-9 | 7 |
| 17x25 | material | 3.08e-12 | 8 |
| 17x25 | local | 3.17e-12 | 8 |

All six use Schur reconstruction and pass the unchanged 5e-9 complete
correction gate. Independent tiny-grid dense references are used only for
verification (31 unknowns): relative correction differences are 1.58e-13 and
5.91e-13. Sparse payloads are 1,844/11,220/53,660 bytes by mesh; these exclude
LU, FFT, Krylov and allocator memory and are not total peak-memory claims.
Larger-mesh performance has not been measured.

Every primary correction has a negative recomputed actual directional merit
slope. Each full step is rejected for feasibility, then the affine-bound
stage finds a feasible decreasing trial satisfying the genuine directional
Armijo inequality. The passing amplitudes are about 0.5319, 0.6629 and 0.7314
by mesh for both preconditioners. These are **single mathematical correction
trials**, not accepted EHL states or evidence of converged nonlinear solves.

A forced unsafe Schur indicator exercises full bordered fallback on 9x13;
its independently checked residual is 3.03e-11. A deliberately exhausted
one-cycle/restart-one run fails and retains actual subsolve residuals
1.525e-3 and 1.267e-2 plus Krylov histories. Tests also inject a singular
preconditioner to verify preserved setup/fallback failure diagnostics.
The reduced failure budget is a rejection test, not a relaxed solve policy.

## What remains

The mathematical correction/preconditioning milestone is verified. It does
not resolve the loaded near-tie finite-step discrepancies described in
[LIMITER_SWITCHES.md](LIMITER_SWITCHES.md), prove global generalized-Jacobian
membership, supply active-front/model recovery or implement load-step retries.
A future nonlinear adapter must rebuild frozen models as state/guide changes,
recompute the true directional slope for each correction, preserve all failed
models/trials, and retain strict film and original pressure/content gates.
It must then independently pass final limited-flux mass/load/film/cavitation
acceptance before physical 50/200 N runs or any default promotion.

## Reproduction and retained evidence

Choose a fresh output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/benchmark_limited_linear.py --output results/my_limited_linear
```

The local final output is `results/limited_linear_final_20261030/`.
`summary.json` records SI inputs/scales, true residuals, complete subsolve
histories, line-search trials and intentional failures. Per-case NPZ files
retain exact state, guide, correction, residual and SI axes. Scope flags
explicitly remain `candidate_is_converged=false` and
`physical_validation_complete=false`. Existing output directories are rejected.

Verification: `ruff check .` and whitespace checks pass. The full suite reports
**652 passed, 62 skipped in 345.07 s**, and all 43 focused limited-flux,
directional and linear-model checks pass (2.36 s). Optional unavailable backend
tests skip gracefully; no GPU result is claimed. Regression checks preserve
state/guide immutability, matrix-copy isolation, true offset/load borders,
independent preconditioner actions, dense correction oracles and rejection
histories. Nonlinear/physical acceptance remains explicitly incomplete.

The local archive is `results/limited_linear_evidence_20261030.zip`
(743,186 bytes, 218 entries), with source/test snapshots, runtime versions,
exact weak states/guides/corrections, full trial/failure histories, logs and
a SHA256 manifest. SHA256:
`35f8839563ba3e260207d42a68a15c52d179f938409ceea4d11359fc89f75b4d`.
Every manifest digest and ZIP CRC was verified. The archive is local and
Git-ignored; no upload or accepted candidate root is claimed.

The subsequent experimental nonlinear adapter and bounded continuation are
documented in [LIMITED_NONLINEAR.md](LIMITED_NONLINEAR.md); the linear checks
above alone do not establish nonlinear or physical convergence.
