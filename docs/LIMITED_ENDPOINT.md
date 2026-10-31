# Exact MC endpoint through guarded secant initialization

This follow-up establishes an accepted **finite-grid** limited-capacity root
for the registered 50 N case on 33x33, 65x65 and 129x129. The 33x33
path starts from a cold Hertz guess; finer paths use re-audited donor seeds. It does not
establish mesh convergence, physical accuracy, a robust load/material envelope,
or completion of the GUI/release gate. The prototype CLI default remains donor.
The earlier failed path in [LIMITED_NONLINEAR.md](LIMITED_NONLINEAR.md) is
preserved as a regression and diagnostic baseline.

## What changed

The exact MC Jacobian/preconditioner can become singular or ill-conditioned
in nearly constant-capacity cavitated regions. In the previous near-endpoint
run, tiny residuals produced disproportionately large content corrections,
rejected Armijo trials and failed true Krylov residual checks. Probing a
preconditioner with retained donor transport improved some linear solves but
did not remove nonlinear stagnation. That experiment is retained as evidence;
it does not change the production linear model or its preconditioner.

Instead, the continuation path now tests one secant prediction after each
accepted stage for which there are two independently audited stage roots.
For accepted states z_old, z_last at ordered fractions tau_old < tau_last < 1:

```text
factor = (1 - tau_last) / (tau_last - tau_old)
direction = factor * (z_last - z_old)
z_prediction = z_last + direction
```

All states use the same physical case, grid and scales: float64 interior
`P=p/pH`, `D=1-theta`, signed `h0/Href`, C order `(ny,nx)`, +x entrainment.
Tau is a numerical transport parameter, not time. Physical case interfaces
remain SI. Both source roots are re-audited for their own explicit equation.
A failed partial state is never an input to prediction or retry.

This is **initialization by continuation**, not a bordered Newton correction.
It does not claim a Krylov/Newton residual for the secant vector. Before a
prediction can initialize final verification, it must pass all of:

- The original pressure/content bounds and strict film feasibility.
- Strict decrease of the **exact tau=1** residual merit.
- Directional Armijo with the actual one-sided MC derivative at z_last,
  `phi(prediction) <= phi(last) + 1e-4 * Dphi(last; direction)`, with
  strictly negative directional slope. No residual-relative surrogate.
- Independently reconstructed SI mass/force/complementarity/boundary gates
  for the exact MC equation, with the original unchanged tolerances.

A separate tau=1 stage then calls the ordinary solver with **zero allowed
Newton corrections**. It must pass the direct full-source elastic reconstruction
and independently recomputed final SI audit before it can return an accepted
root. A promising prediction, low merit or nearby blended root is insufficient.
Final-audit failure restores the last accepted blended state before bisection.
At most one prediction probe is made per accepted stage and the existing
24-stage/30-correction/eight-model/GMRES/line-search budgets are unchanged.

Every prediction diagnostic records source fractions, source/predicted-state
hashes, factor, actual slope, merits, feasibility and provisional SI audit.
The final attempt identifies the secant initialization explicitly. Individual
stage bundles retain raw states and full Newton histories; the path summary
retains prediction diagnostics, including rejected probes. The research API
accepts `endpoint_prediction=False` to reproduce the previous strategy.

## Registered 50 N cold-start result

`results/limited_secant_cold_50n_20261031` reaches the exact endpoint in 11
stage attempts, including a rejected tau=0.99 step and its rollback/retry.
The eligible prediction uses accepted tau=0.99999 and tau=0.999999 roots.
Its exact MC merit decreases from 6.530e-22 to 9.385e-33, with true directional
slope -1.306e-21, satisfying the original Armijo inequality.

The final, direct-elasticity audited tau=1 state has:

| Check | Measured value |
| --- | --- |
| Force | 50.0 N |
| Scaled maximum mass residual | 1.599e-13 |
| Relative local mass balance | 1.872e-12 |
| Relative global mass balance | 1.117e-14 |
| Maximum complementarity | 2.673e-30 |
| Direct/FFT film reconstruction difference | 3.39e-21 m |
| Minimum film | 159.402 nm |
| Central film | 296.593 nm |

The final state solves the full limited equation; it is not a relabeled
blended stage. The corresponding HD estimates remain 284.15/497.97 nm
minimum/central. This coarse result is substantially lower and has not been
shown to be physically more accurate than the donor baseline. A discrete
root is the prerequisite for a refinement study, not its conclusion.

## Reproduction and regression checks

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src .venv/bin/python examples/run_limited_transport_path.py \
  --case cases/circular_50n.json --output results/my_mc_endpoint
```

Use a fresh directory. `--no-endpoint-prediction` disables the new initialization;
`--seed` may instead specify an independently audited matching donor prototype
run. Read `summary.json`, especially operator/fraction, stage acceptance,
`initialization`, prediction diagnostics and final audit. Iteration histories
and NPZ arrays are preserved for rejected as well as accepted stages.

Self-contained tests repeat the actual cold path, independently reconstruct
all predictions from their accepted source states, recompute true slopes and
SI/direct-elasticity acceptance, reject damaged/reversed sources, and inject
a final direct-audit failure. A separate injected path verifies restoration
of the last accepted root after failed endpoint verification, and prediction
can be disabled without changing retry seeds. No MATLAB checkpoint or ignored
result bundle is required by these regressions.

Further 50 N mesh and 200 N continuation evidence is recorded below. Broad
MC robustness and physical validation remain open; the accepted donor results,
case baselines, main branch and planned GUI are unchanged.

## First controlled refinement: 50 N 65x65

The identical case/domain on 65x65 also reaches an exact, independently audited
MC root (`results/limited_secant_50n_65_20261031`), in 13 stage attempts.
The final secant again uses tau=0.99999 and 0.999999 accepted roots, followed
by separate zero-Newton direct verification. Force is 50.0 N; scaled maximum
mass is 3.054e-13, local balance 1.399e-12 and global balance 2.534e-15.
Minimum/central films are 257.963/428.588 nm, versus 159.402/296.593 nm on
33x33. The large refinement change prevents a mesh-convergence claim despite
better proximity to the HD estimates. Pressure/minimum locations and finer
mesh evidence must also be assessed before assigning physical accuracy.

## Coarse 200 N limit remains open

`results/limited_secant_200n_20261031` exhausts 24 stages: 12 accepted blended
stages and 12 rejected attempts on 33x49. The last accepted tau is
0.9689062500000001, with exact 200 N force and local balance 2.276e-13
**for that blended equation**. Its minimum film is only 3.329 nm and central
film 277.367 nm. Failed larger-fraction steps can approach the strict film
floor and fail convergence. No floor/clipping/tolerance change is made.
Neither prediction nor ordinary final verification establishes a full MC
root for this case. The path is correctly saved as `rejected_path`; this
partial state is not a physically validated 200 N prediction. Fine-grid
50 N success must not be generalized to this load or geometry/resolution.

## Third mesh: 50 N 129x129

`results/limited_secant_50n_129_20261031` reaches an independently audited
exact MC root in 14 attempts (11 accepted stages, three rejections).
The final prediction uses tau=0.9999 and 0.99999, with merit decreasing from
7.550e-21 to 7.020e-29 and true directional slope -1.510e-20.
Zero-Newton direct verification gives exact 50 N force, scaled maximum mass
2.155e-11, relative local balance 5.658e-11, global balance 5.587e-14,
complementarity 3.445e-28 and direct/FFT film difference 8.05e-21 m.
All original numerical gates pass without relaxation.

The fixed-domain, identical-fluid comparison is:

| Grid | Minimum [nm] | Central [nm] | Minimum vs HD | Central vs HD |
| --- | ---: | ---: | ---: | ---: |
| 33x33 | 159.402 | 296.593 | -43.90% | -40.44% |
| 65x65 | 257.963 | 428.588 | -9.22% | -13.93% |
| 129x129 | 281.506 | 462.814 | -0.93% | -7.06% |

HD uses the same SI force/radius/fluid inputs and `E_prime=2*E_contact`;
its minimum/central estimates are 284.150/497.969 nm. No parameter or
acceptance threshold is fitted to those estimates.

**This sequence is not mesh converged.** The last refinement changes minimum
and central film by +9.126% and +7.986% relative to 65x65. Pressure on shared
nodes changes by 4.770% in relative L2 norm and 20.828% in maximum norm
(normalized by the finer pressure maximum). The sampled minimum shifts from
(x/a,abs(y/a))=(0.375,0.75) to (0.25,0.8125). Close minimum-film agreement
with HD alone cannot establish pressure accuracy or resolved minimum location.
Further spatial/domain checks and independent reference evidence remain R1
requirements, alongside the incomplete 200 N case above. The next grid/resource
step needs an explicit cost assessment within the research node limits;
this increment does not silently raise those limits.

Verification: full suite **681 passed, 62 optional skips** (400.96 s), Ruff
and whitespace clean. The first concurrent full run ended with SIGTERM and
was not counted as passing; the fresh complete run above covers the final
code, including all seven new prediction regressions. GPU physical validation
is not added. Full raw-state/correction/prediction replay and local provenance
are retained in `results/limited_endpoint_evidence_20261031/` and its sibling
ZIP/receipt; these datasets remain outside Git.

Independent history replay reconstructs every raw state exactly on all four
paths: 11/13/14 stage attempts and 101/98/79 accepted Newton steps
for the 33/65/129 meshes; the incomplete 200 N path retains all 24 stages
and 317 accepted Newton steps. Selected full correction residuals, actual
directional slopes, Armijo trials, prediction hashes and direct final audits
are independently recomputed. Replaying a rejected stage does not promote it.

## Fourth mesh: 50 N 141x141

The fixed +/-3a donor case was continued to an exact MC root on 141x141
(19,881 physical nodes, within the existing 20,000-node limit) using the
independently audited 141x141 donor prototype state as a seed. The path has
10 attempts: nine accepted stages and one rejected stage; the exact fraction-one
root passes the independent force, mass, complementarity, film and direct
elasticity checks. Its minimum/central films are 283.271/464.741 nm.

The comparison against the exact 129x129 root is archived in
`results/limited_comparison_50n_129_to_141_20261031/`. These grids are not
nested, so pressure is compared by bilinear sampling of the 141 grid at the
129 grid's coordinates. The relative pressure L2/max differences are 1.805% /
8.911%; minimum/central film change by +0.627% / +0.416%. The minimum location
moves from `(x/a, abs(y/a))=(0.25, 0.8125)` to approximately
`(0.2857, 0.8)`.

The Hamrock–Dowson references remain 284.150/497.969 nm minimum/central.
The 141-grid minimum is 0.309% below the minimum correlation, while central
film remains 6.673% below its correlation. These estimates are not acceptance
thresholds: close minimum-film agreement does not establish pressure accuracy,
minimum-location resolution, mesh convergence or physical validation. The
central-film offset and pressure differences leave further resolution and
domain checks open. This mesh is already near the current node budget; do not
raise it implicitly. A larger uniform mesh needs a separate cost/resource
assessment.

The portable two-path comparison now independently audits both roots, reports
the case-matched SI Hamrock–Dowson film estimates and per-grid relative film
differences, and keeps `mesh_convergence_assessed=false`. It does not infer a
pass/fail threshold from the correlation. Focused tests cover these outputs,
unit/checksum tampering, partial-root rejection and non-nested interpolation.
