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

## Opt-in 50 N high-resolution mesh refinements

The regular case budget remains 20,000 physical nodes. `PointContactCase`
accepts explicit 30,000, 40,000 or 50,000 node budgets only for research runs;
the budget is provenance/resource metadata and does not change the physical
case. Prepared square meshes are 173x173 (29,929 nodes), 199x199 (39,601
nodes) and 223x223 (49,729 nodes). The experimental path runner requires an
independently audited exact MC seed for any larger budget, transfers it
bilinearly without promoting it, and preflights the frozen sparse factors and
measured Krylov/memory estimate before starting continuation. The preflight
requires a known-vector factor residual below 1e-8, estimated peak below both
2 GiB and 75% of currently available RAM. The 2 GiB ceiling was introduced
in tracked commit `1f3f1a1` for the 50k mesh study. An unsuccessful preflight is
recorded as `rejected_preflight`; it does not silently fall back or raise the
normal default.

The continuation attempt cap is 48; the default remains 24. A larger cap only
allows additional bounded continuation stages and retries. It does not change
Newton corrections per stage, equation tolerances, or root acceptance gates.

Reproduce the refinement and independent comparisons from a fresh output
directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python examples/run_limited_transport_path.py \
  --case cases/circular_50n_173.json \
  --limited-seed results/limited_mesh_50n_141_20261031 \
  --output results/limited_mesh_50n_173_20261031

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python examples/compare_limited_paths.py \
  results/limited_secant_50n_129_20261031 \
  results/limited_mesh_50n_173_20261031 \
  --output results/limited_comparison_50n_129_to_173_20261031

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python examples/compare_limited_paths.py \
  results/limited_mesh_50n_141_20261031 \
  results/limited_mesh_50n_173_20261031 \
  --output results/limited_comparison_50n_141_to_173_20261031

# After 173x173 is accepted and measured, try 199x199 (39,601 nodes).
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python examples/run_limited_transport_path.py \
  --case cases/circular_50n_199.json \
  --limited-seed results/limited_mesh_50n_173_retry_20261031 \
  --output results/limited_mesh_50n_199_20261031

# Attempt 223x223 (49,729 nodes) only after inspecting the 40k result and a
# fresh preflight; the runner refuses estimates above its 2 GiB / RAM ceiling.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python examples/run_limited_transport_path.py \
  --case cases/circular_50n_223.json \
  --limited-seed results/limited_mesh_50n_199_20261031 \
  --output results/limited_mesh_50n_223_20261031
```

The initial transfer is only an unaccepted guess; every continuation stage
and the final fraction-one root retain the existing unmodified gates and
direct-elasticity audit. The comparisons are non-nested and report interpolation
error explicitly. Passing these finite-grid checks would measure sensitivity,
not certify asymptotic convergence or physical accuracy.

## 40,000-node 50 N mesh: 199x199

The 39,601-node 199x199 case passed its sparse-factor/resource preflight:
known-vector relative residual 3.172e-16, estimated peak 803,666,448 bytes
(below 1 GiB), and 2,193,317,888 bytes available at preflight. The exact MC
path reached `accepted_numerical_root` in 12 stage attempts, with two rejected
near-endpoint attempts. Independent direct elasticity, exact force, local and
global mass, complementarity, and strict film checks passed:

| Check | Measured value |
| --- | ---: |
| Force | 50.000000000000014 N |
| Relative local mass balance | 2.902e-11 |
| Relative global mass balance | 1.240e-14 |
| Maximum complementarity | 2.608e-30 |
| Direct/FFT film reconstruction difference | 1.228e-20 m |
| Minimum film | 286.690 nm |
| Central film | 469.711 nm |

Compared with the exact 173x173 root, minimum/central film change +0.029% /
+0.346%; pressure L2/max differences are 2.017% / 10.444%, and exit-region
pressure L2/max differences are 4.587% / 11.567%. The sampled minimum shifts
from `(0.2326,0.8372)` to `(0.2828,0.8081)` in
`(x/a,abs(y/a))`. Minimum film is +0.894% relative to Hamrock–Dowson; central
film remains 5.675% below. The pressure and minimum-location changes do not
support a mesh-convergence claim despite small film changes.

The 40k run's sampled process RSS reached about 935,512 KiB during continuation,
larger than the preflight's 803,666,448-byte estimate; this is a sampled value,
not a recorded maximum-RSS measurement. This underprediction matters when
assessing the 50k case; retain extra headroom and do not interpret the
preflight estimate as a peak-memory guarantee. Full run and comparison records
are preserved in `results/limited_mesh_50n_199_20261031/` and
`results/limited_comparison_50n_173_to_199_20261031/`.

## 50,000-node 50 N preflight: 223x223

The historical 49,729-node 223x223 sparse-factor preflight passed its
known-vector check (relative residual 3.446e-16), but its estimated peak of
1,083,328,872 bytes exceeded the then-current 1 GiB cap by 9,587,048 bytes.
Available memory was 1,831,874,560 bytes; no nonlinear stage was started and
no 50k root was produced. The tracked ceiling was later raised to 2 GiB in
commit `1f3f1a1` (2026-11-02). The old rejection is historical evidence under
the 1 GiB policy, not evidence that the current 2 GiB preflight rejects the
case. No 223x223 preflight or nonlinear run has been repeated under the
current policy. The estimate remains uncertain because sampled RSS in the
40k process exceeded its preflight estimate; do not infer a 50k peak guarantee
from the larger ceiling.

The first 173x173 attempt on 2026-10-31 transferred the accepted 141x141 MC
state and passed the independent full bordered-factor check (relative residual
3.10e-16). Its conservative peak estimate was 545,394,608 bytes, but only
502,837,248 bytes were available at preflight time; the required 75% headroom
ceiling was 377,127,936 bytes. It was correctly recorded as
`rejected_preflight` before any nonlinear stage. At that point no 173x173 root
or mesh comparison existed. The retry below was run only after the recorded
memory preflight passed; do not lower the margin or force the run on a
memory-pressured host.

The retry passed preflight when 2,170,900,480 bytes were available and completed
as `accepted_numerical_root` at the exact MC endpoint. It used the unchanged
24-attempt/30-correction/10-Krylov-cycle budgets; 12 stages include two rejected
near-endpoint trials, and the final fraction-one direct audit passed. The
independently recomputed results are:

| Check | Measured value |
| --- | ---: |
| Force | 50.0 N |
| Relative local mass balance | 1.978e-11 |
| Relative global mass balance | 1.006e-14 |
| Maximum complementarity | 7.526e-32 |
| Direct/FFT film reconstruction difference | 1.186e-20 m |
| Minimum film | 286.605 nm |
| Central film | 468.093 nm |
| Hamrock–Dowson minimum/central | 284.150/497.969 nm |

The minimum is 0.864% above the Hamrock–Dowson correlation; the central film
remains 6.000% below. This correlation agreement is not an acceptance gate.
The portable comparison reports:

| Coarse -> fine | Pressure L2 | Pressure max | Exit pressure L2/max | Minimum film change | Central film change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 129 -> 173 | 2.173% | 13.395% | 5.046% / 15.104% | +1.811% | +1.141% |
| 141 -> 173 | 2.307% | 11.619% | 5.180% / 12.950% | +1.177% | +0.721% |

All three grids have the same physical domain and case; these grid pairs are
non-nested and pressure is bilinearly sampled at the coarse nodes. The minimum
moves from `(0.25,0.8125)` on 129x129 and `(0.2857,0.8)` on 141x141 to
`(0.2326,0.8372)` on 173x173 in `(x/a,abs(y/a))`. Minimum and central film
changes are getting smaller, but pressure changes remain material and the
minimum position still moves. The 173-grid result therefore does not establish
mesh convergence, a continuum film prediction, or full physical validation.
At this point it was the largest square grid tested within the explicit
30,000-node budget; larger 40k/50k opt-ins are documented above. The
machine-specific preflight estimated 545,394,608 bytes against a 1 GiB cap;
this is an estimate, not a measured peak-RSS guarantee.

Comparison files and source hashes are retained outside Git under
`results/limited_comparison_50n_129_to_173_20261031/` and
`results/limited_comparison_50n_141_to_173_20261031/`. The retry's stages,
full history and independent final audit are under
`results/limited_mesh_50n_173_retry_20261031/`.

## Provisional 50 N mesh sensitivity gate and 173/199 follow-up

Numerical root acceptance and mesh sensitivity are separate questions. For
this study, use the following **provisional engineering sensitivity gate**;
it is not a physical-accuracy tolerance:

* Each compared limited-capacity endpoint must independently pass its existing
  exact-load acceptance and final audit.
* On two consecutive refinements of the same physical case and domain, the
  relative changes in minimum film, central film, full-field pressure
  relative-L2 norm, and exit-region pressure relative-L2 norm must each be at
  most 1%.
* The minimum-film location must move by no more than one fine-grid spacing
  in either coordinate. Report pressure maximum-norm changes as diagnostics,
  but do not use a pointwise maximum as a stand-alone gate.
* For domain sensitivity, keep grid spacing fixed and extend the inlet/outlet
  and transverse boundaries independently. Compare the common physical region
  using the same film, pressure, and minimum-location measures and 1%/one-cell
  limits. Each domain variant must independently pass the exact-load audit.

The 1% tolerance is a provisional numerical-change target, not a claim about
continuum error, experimental accuracy, or agreement with Hamrock-Dowson.
Correlation comparisons remain informative only and are not an acceptance
criterion. A mesh result cannot substitute for the separate domain test.

A fresh 173x173 donor root was accepted and used only to initialize the exact
MC continuation; the latter passed its preflight and was independently
accepted at transport fraction 1.0. Its final audit reports 50.0 N, relative
local/global mass balances of 6.088e-11/3.051e-14, and minimum/central films
of 286.605/468.093 nm. The donor and exact MC histories are retained locally
under `results/prototype_50n_173_validation_20261105/` and
`results/limited_mesh_50n_173_validation_20261105/`; the comparison reports
are under `results/mesh_gate_50n_141_to_173_20261105/` and
`results/mesh_gate_50n_173_to_199_20261105/`.

| Refinement | Minimum film change | Central film change | Full pressure L2 / max | Exit pressure L2 / max | Minimum movement (fine cells, x/y) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 141x141 -> 173x173 | +1.177% | +0.721% | 2.307% / 11.619% | 5.180% / 12.950% | 1.52 / 1.07 |
| 173x173 -> 199x199 | +0.029% | +0.346% | 2.017% / 10.444% | 4.587% / 11.567% | 1.66 / 0.96 |

Both adjacent comparisons fail the provisional gate: the full- and exit-field
pressure L2 changes exceed 1%, and the minimum shifts more than one fine cell
in x. The coarser pair also exceeds the minimum-film limit. The finer pair
shows small changes in the two scalar film heights, but that alone does not
establish mesh convergence. The comparison grids share the same domain and
are non-nested; fine pressure is bilinearly sampled at coarse nodes. All three
roots remain numerically accepted, but these results establish neither
physical accuracy nor a converged continuum solution.

### Fixed-spacing domain-extension comparison

The accepted 173x173 exact MC root was compared with two one-axis domain
extensions. Each axis was extended symmetrically from a half-width of 4a to
5.0233a while preserving the original spacing of 0.04651163a; the other axis
and its spacing were unchanged. The resulting grids contain 37,541 nodes
(173x217 for x extension and 217x173 for y extension). The donor roots were
transferred by the existing fixed-spacing embedding, but each limited-capacity
endpoint was solved and independently audited.

| Extended axis | Exact endpoint force | Relative local/global mass balance | Minimum / central film change | Full pressure L2 / max | Exit pressure L2 / max | Minimum movement (fine cells, x/|y|) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x: +/-4a -> +/-5.0233a | 50.000 N | 6.379e-11 / 2.193e-14 | +0.273% / +0.333% | 0.532% / 3.063% | 0.991% / 3.310% | 0.00 / 0.00 |
| y: +/-4a -> +/-5.0233a | 50.000 N | 3.391e-11 / 1.074e-14 | +0.075% / +0.023% | 0.134% / 0.712% | 0.259% / 0.774% | 0.00 / 0.00 |

Pressure and film changes are measured on the common original-domain nodes;
the exit region retains its definition above. The symmetric minima use
`(x/a, abs(y/a))` for location comparison. Both one-axis extensions satisfy
the provisional 1%/one-cell sensitivity gate. The x-extended exit-pressure L2
change is close to the limit (0.9908% before rounding), so it should not be
treated as a robust margin. The pressure maximum changes are reported as
diagnostics and do not independently fail the gate.

The preflight estimates were 853,240,669 bytes (x) and 853,043,605 bytes (y),
against the tracked 2 GiB ceiling; both are also below 1 GiB. These estimates
are not measured peak-RSS guarantees.

Exact run histories are retained locally under
`results/limited_domain_x_50n_173x217_20261105/` and
`results/limited_domain_y_50n_217x173_20261105/`; the independently
recomputed common-domain comparison is under
`results/domain_gate_50n_173_fixed_spacing_20261105/`. The matching case
inputs and accepted prototype donors are retained locally as
`results/case_domain_x_extension_50n_173x217.json`,
`results/case_domain_y_extension_50n_217x173.json`,
`results/prototype_domain_x_50n_173x217_20261105/`, and
`results/prototype_domain_y_50n_217x173_20261105/`.

### Simultaneous fixed-spacing x/y extension

The 173x173 base case was also extended in both axes to 217x217 (47,089 nodes),
again preserving spacing at 0.04651163a and increasing each half-width from
4a to 5.0233a. The 50 N donor root was accepted. An initial limited-capacity
continuation with the default 24-attempt budget stopped at fraction
0.9999478125. The stage-attempt cap was therefore raised from 24 to 48
(default remains 24; equations, per-stage correction limits, and acceptance
tolerances were unchanged). The retry reached fraction 1.0 and passed an
independent direct-elasticity/final limited-equation audit after 40 attempts:
force 50.000 N, relative local/global mass balance 9.552e-11/2.093e-14,
minimum/central film 287.655/469.833 nm. Preflight estimated 1,088,983,557
bytes against the 2 GiB ceiling, with a known-vector residual of 3.098e-16.
This estimate is not a peak-RSS guarantee.

| Extension | Minimum / central film change | Full pressure L2 / max | Exit pressure L2 / max | Minimum movement (fine cells, x/|y|) |
| --- | ---: | ---: | ---: | ---: |
| Both axes: +/-4a -> +/-5.0233a | +0.366% / +0.372% | 0.720% / 4.105% | 1.362% / 4.425% | +2 / -1 |

Fields are compared on the common original-domain nodes; the symmetric
minimum-location convention is `(x/a, abs(y/a))`. Minimum and central film
changes and full-field pressure L2 meet the provisional 1% criterion, but
exit-pressure L2 exceeds it and the x minimum shifts by two fine cells. The
simultaneous extension therefore **fails** the provisional domain-sensitivity
gate. The one-axis results do not establish insensitivity when both boundaries
are extended together. The maximum pressure changes are diagnostics, not
stand-alone gates.

The first 24-attempt history is under
`results/limited_domain_xy_50n_217x217_20261105/`; the successful 48-cap retry
is under `results/limited_domain_xy_50n_217x217_retry48_20261105/`. Its
independent common-domain comparison is under
`results/domain_gate_50n_173_xy_fixed_spacing_20261105/`, with case input
`results/case_domain_xy_extension_50n_217x217.json` and accepted donor under
`results/prototype_domain_xy_50n_217x217_20261105/`. L1 was not added to the
comparison report. This sensitivity study does not establish physical
accuracy, mesh convergence, or validated EHL predictions.

### Incremental larger-domain check at 223x223

The simultaneous extension was increased from 217x217 to 223x223 while
preserving the same spacing and exact 50 N case. The 49,729-node target has
half-width 5.1628a on each axis (three added grid cells per side relative to
217x217). Its accepted MC endpoint passed the independent direct-elasticity
audit in 12 stage attempts: force 50.000 N, relative local/global mass balance
2.097e-11/4.435e-15, minimum/central film 287.720/469.966 nm. Preflight passed
with known-vector residual 3.337e-16 and adjusted estimate 1,201,899,684 bytes
against the 2 GiB ceiling; this is not a peak-RSS guarantee.

| Comparison | Minimum / central film change | Full pressure L2 / max | Exit pressure L2 / max | Minimum movement (fine cells, x/|y|) | Provisional gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 217x217 -> 223x223 | +0.023% / +0.028% | 0.070% / 0.391% | 0.132% / 0.422% | 0 / 0 | Pass |
| 173x173 -> 223x223 (cumulative) | +0.389% / +0.400% | 0.789% / 4.497% | 1.494% / 4.843% | +2 / -1 | Fail |

The 217-to-223 increment passes the provisional gate, but it extends each
half-width by only 0.1395a. The cumulative comparison from the original 4a
domain still fails on exit-pressure L2 and the x minimum location. Thus, the
small incremental change does not establish domain independence and does not
erase the larger observed domain sensitivity.

The accepted root, case, and donor are retained locally under
`results/limited_domain_xy_50n_223x223_20261105/`,
`results/case_domain_xy_extension_50n_223x223.json`, and
`results/prototype_domain_xy_50n_223x223_20261105/`; the independently
recomputed comparisons are under
`results/domain_gate_50n_217_to_223_fixed_spacing_20261105/`. A materially
larger simultaneous extension at this spacing would exceed the documented
50,000-node benchmark budget and requires a separate resource review. The
mesh-sensitivity gate also remains unmet.

## Fifth mesh: exact 50 N 223x223 endpoint

The 49,729-node 223x223 case was rerun on 2026-11-06 from the independently
audited 199x199 exact MC endpoint, using the same physical case, fixed +/-4a
domain, equations and acceptance thresholds. Its coarse resource preflight
passed: adjusted estimate 1,259,911,478 bytes, against the 2 GiB ceiling and
3,249,704,960 bytes available when checked. The known-vector factor residual
was 3.446e-16. The run deliberately used `--no-measured-peak`; the recorded
adjusted estimate is not a measured peak-RSS value or a guarantee. Improved
factor-fill and peak-memory estimation is deferred to a later optimization
round.

The exact fraction-one endpoint was independently accepted after eight
continuation attempts, with no rejected stages. Force is 49.99999999999999 N;
relative local/global mass balance is 6.093e-11/1.802e-14, maximum
complementarity is 1.481e-28, and direct-elasticity film reconstruction
difference is 1.736e-20 m. Minimum/central film is 287.867/470.732 nm.

The independent same-domain comparison with 199x199 uses bilinear fine-at-
coarse sampling (the grids are non-nested):

| Comparison | Minimum / central film change | Full pressure L2 / max | Exit pressure L2 / max | Minimum movement (fine cells, x/|y|) | Provisional gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 199x199 -> 223x223 | +0.411% / +0.217% | 1.772% / 10.397% | 4.530% / 11.790% | 0.85 / 0.58 | Fail |

Both roots pass their independently recomputed force, mass, complementarity,
film-feasibility and direct-elasticity audits. The pressure L2 changes exceed
the provisional 1% sensitivity limit, despite small film changes and a
sub-cell minimum movement. The result therefore does not pass the provisional
mesh gate and does not establish a continuum limit or physical film accuracy.
The approximate Hamrock-Dowson minimum/central films are 284.150/497.969 nm;
they are comparison references, not acceptance thresholds.

Reproduce in fresh result directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python examples/run_limited_transport_path.py \
  --case cases/circular_50n_223.json \
  --limited-seed results/limited_mesh_50n_199_20261031 \
  --output results/limited_mesh_50n_223_20261106 \
  --no-measured-peak

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python examples/compare_limited_paths.py \
  results/limited_mesh_50n_199_20261031 \
  results/limited_mesh_50n_223_20261106 \
  --output results/limited_comparison_50n_199_to_223_20261106
```

The run and comparison bundles are local under `results/`; the comparison
records source-summary checksums and both independently recomputed audits.

## Directional fixed-domain refinements: 199x223 and 223x199

To separate x and y resolution effects without changing physical boundaries,
two cases retain the exact 50 N material/fluid inputs and the same +/-4a
domain. Arrays are `(ny,nx)`: `199x223` refines x, while `223x199` refines y.
Each has 44,377 nodes and is below the existing 50,000-node cap. Both use the
accepted 199x199 root as an unaccepted interpolated seed and unchanged exact
MC equations and acceptance thresholds. Coarse preflights passed with
adjusted estimates 1,042,314,243 and 1,097,789,678 bytes respectively; no
measured peak RSS was requested or recorded.

Both exact endpoints pass independently recomputed force, mass,
complementarity, film and direct-elasticity audits:

| Grid `(ny,nx)` | Accepted/rejected attempts | Force [N] | Local/global mass balance | Minimum / central film [nm] | Direct film discrepancy [m] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 199x223 (x refined) | 9 / 1 | 50.00000000000000 | 1.561e-11 / 1.063e-15 | 287.077 / 470.678 | 1.482e-20 |
| 223x199 (y refined) | 9 / 1 | 49.99999999999999 | 4.492e-11 / 1.259e-14 | 287.478 / 469.764 | 1.525e-20 |

Independent bilinear fine-at-coarse audits give:

| Comparison | Pressure L2 / max | Exit pressure L2 / max | Minimum / central film change | Minimum movement (fine cells, x/y) |
| --- | ---: | ---: | ---: | ---: |
| 199x199 -> 199x223 (x refinement) | 1.357% / 9.929% | 3.588% / 10.884% | +0.135% / +0.206% | 0.15 / 0.00 |
| 199x199 -> 223x199 (y refinement) | 1.096% / 9.572% | 2.427% / 10.575% | +0.275% / +0.011% | 1.00 / 0.58 |
| 199x223 -> 223x223 (then refine y) | 1.140% / 11.208% | 2.563% / 12.210% | +0.275% / +0.011% | 1.00 / 0.58 |
| 223x199 -> 223x223 (then refine x) | 1.350% / 10.019% | 3.577% / 10.642% | +0.135% / +0.206% | 0.27 / 0.00 |

The x-refinement changes pressure more than y-refinement in these comparisons,
while both retain pressure L2 changes above the provisional 1% gate. Repeating
each directional increment against the square 223x223 root shows the same
ordering. The scalar minimum/central film changes are small, but they do not
establish field convergence. These non-nested comparisons include bilinear
sampling effects; they isolate practical directional sensitivity, not a formal
truncation-error decomposition or continuum limit. There is no physical
validation claim.

Inputs are [circular_50n_199x223.json](../cases/circular_50n_199x223.json)
and [circular_50n_223x199.json](../cases/circular_50n_223x199.json). Exact
paths and four independent comparison bundles are preserved locally under
`results/limited_mesh_50n_199x223_20261106/`,
`results/limited_mesh_50n_223x199_20261106/`, and
`results/limited_comparison_*_20261106/`. Improved resource/peak-memory
estimation remains deferred to a separate optimization round.

## Nested fixed-domain grid sequence

The next same-physics sequence uses 27x27, 53x53, 105x105 and 209x209 grids
on the same +/-4a domain. Each refinement doubles the interval count, so
coarse nodes are nested in the next grid; the final grid has 43,681 nodes and
remains within the existing 50,000-node experimental budget. Exact fraction-one
roots were independently accepted on 27x27 (18 accepted stages in 27 attempts),
53x53 (16 in 23) and 105x105 (12 in 16). The 27->53 and 53->105 interpolated
root transfers each failed at their initial correction; cold runs then reached
the exact roots. Thus these are independently solved grid roots, not a
demonstration that warm-start refinement is robust.

The first 209x209 run, initialized from the accepted 105x105 root, reached
fraction 0.9999 after six accepted stages but was interrupted before the
fraction-one endpoint. Its local summary still says `running`; it is partial
diagnostic data, not a completed or accepted root, and is preserved unchanged.
A fresh retry is required before any 105->209 comparison. Even if all roots are
obtained, the sequence is evidence for observed grid sensitivity, not physical
validation or asymptotic convergence by itself.

The later heartbeat-enabled retry started from 105x105, passed the 209-grid
resource preflight and recorded a durable accepted 0.999945 stage after an
initial rejection at 0.99999. The checkpoint independently reports exactly
50 N, local mass balance 7.350e-13 and direct-elasticity verification. Its
following 0.99999 Newton attempt was still active at iteration 30 after about
3,611 seconds. No completed stage archive or terminal status followed; the
summary remains `running`, so the stop cause is unknown and the fraction-one
endpoint was not established. The resumable state is the accepted stage, not
the in-progress Newton iterate. It is the source for the fresh-output resume
command below.

The first restart attempt loaded and checksum-verified that checkpoint, then
was safely stopped by the resource preflight before any nonlinear stage. The
adjusted coarse estimate was 1,153,223,608 bytes, while only 887,304,192 bytes
were available at the post-factorization check; the 75% memory ceiling was
665,478,144 bytes. The factor residual passed at 3.085e-16. This is a resource
rejection, not a solver convergence failure; the attempt is archived at
`results/limited_mesh_50n_209_nested_retry_20261107/`.

After memory was freed, the next restart passed preflight (adjusted estimate
1,153,223,608 bytes; 2,297,647,104 bytes available at the check; 1,723,235,328
byte ceiling). Starting from the accepted 0.999945 checkpoint, it accepted
0.9999725 and then the exact fraction-one root in two attempts. The endpoint
passes the independent exact-MC audit: force 50 N, relative local/global mass
balance 9.332e-11/3.121e-14, complementarity 2.101e-30, and direct film
reconstruction discrepancy 1.906e-20 m. Its minimum/central films are
287.062/470.179 nm. The path used one CPU thread; no peak-RSS guarantee is
implied by its coarse preflight.

All adjacent grids in the 27x27 -> 53x53 -> 105x105 -> 209x209 sequence now
have independently accepted exact fraction-one endpoints. The fine-at-coarse
shared-node comparisons are:

| Comparison | Pressure L2 | Exit pressure L2 | Minimum film change | Central film change | Minimum location `(x/a, |y|/a)`, coarse -> fine |
| --- | ---: | ---: | ---: | ---: | --- |
| 27x27 -> 53x53 | 6.931% | 8.167% | +99.633% | +63.799% | (0.000, 0.923) -> (0.308, 0.769) |
| 53x53 -> 105x105 | 5.640% | 13.968% | +19.953% | +12.586% | (0.308, 0.769) -> (0.231, 0.846) |
| 105x105 -> 209x209 | 4.839% | 10.975% | +2.866% | +2.922% | (0.231, 0.846) -> (0.269, 0.808) |

The pressure L2 differences decrease, but exit-pressure changes are not
monotone and remain large; minimum locations also move. The coarsest grid's
minimum film nearly doubles on the first refinement. Therefore, the observed
sequence does not meet the provisional 1% field-sensitivity gate and does not
establish mesh convergence, continuum accuracy, or physical validation. All
three comparisons use shared nodes on the same physical domain, with
independently recomputed exact-root audits. The JSON reports and source
checksums are preserved under `results/limited_comparison_50n_*_nested_20261107/`.

## 209x209 to existing 223x223 endpoint

To check whether the nearby 50k-node root changes the nested-grid picture
without another nonlinear solve, the accepted 209x209 endpoint was compared
against the existing 223x223 endpoint. Both use the same 50 N physics and
fixed +/-4a domain and independently pass the exact MC, mass, force,
complementarity and direct-elasticity audits. The grids are non-nested, so the
fine pressure is bilinearly interpolated to coarse nodes:

| Comparison | Pressure L2 / max | Exit pressure L2 / max | Minimum / central film change | Minimum location `(x/a, |y|/a)`, coarse -> fine |
| --- | ---: | ---: | ---: | --- |
| 209x209 -> 223x223 | 1.808% / 11.647% | 4.008% / 12.338% | +0.281% / +0.118% | (0.269, 0.808) -> (0.252, 0.829) |

This nearby higher-resolution comparison shows smaller scalar-film changes,
but pressure and exit-region differences remain above the provisional 1% gate.
The minimum moves by about 0.017a in x and 0.021a in |y|. Since this is a
non-nested comparison and not an independent truncation-error estimate, it
does not establish convergence or explain the larger nested-grid exit
sensitivity. A localized support/front and boundary-condition diagnostic is
still warranted before deciding on a much larger grid. The independently
audited comparison bundle and source-summary checksums are preserved under
`results/limited_comparison_50n_209_to_223_20261107/`.

The bounded runner now supports timestamped progress heartbeats. By default it
reports every 300 seconds, including elapsed time, current continuation
attempt/fraction, Newton iteration and available mass/force audit values.
Override the interval with `--progress-interval SECONDS`; the normal final
accepted/rejected status is still emitted when the run exits.
Every completed stage is archived atomically; the latest accepted state and
its fraction are also recorded as a restart checkpoint. `Ctrl+C` and `SIGTERM`
write `status="interrupted"`, the interruption reason and the last progress
snapshot. A fresh run can resume from the latest completed accepted stage
without changing the original results directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python examples/run_limited_transport_path.py \
  --case cases/circular_50n_209.json \
  --resume-from results/limited_mesh_50n_209_nested_retry_20261106 \
  --output results/limited_mesh_50n_209_nested_retry2_20261107 \
  --no-measured-peak --progress-interval 300
```

The resume loader checks the case, completed-stage status, state checksum and
state shape; continuation independently re-audits the checkpoint before using
it. A hard kill or power loss cannot write a terminal reason, but the latest
atomic accepted-stage checkpoint remains available. In that case the source
summary may still say `running`; the new run records that source status and
does not guess at the stop cause. An in-progress Newton iterate is not an
accepted state and is not promoted or resumed. The successful resumed endpoint
and adjacent-grid comparisons are archived under
`results/limited_mesh_50n_209_nested_retry2_20261107/` and
`results/limited_comparison_50n_*_nested_20261107/`, respectively.

## Localized nested-grid pressure and front diagnostics

The comparison command now adds `nested_resolution_diagnostics` to its JSON
when the fine grid doubles the interval count on both axes. It compares fine
fields at shared coarse nodes, partitions the existing exit window, reports
pressure-error locations and squared-error shares, compares cavitation states
using `theta < 1 - 1e-10`, and reports native-grid first-cavitated sample shifts
on shared transverse rows. These extra metrics are diagnostics, not acceptance
gates; first-node shifts are not subcell front estimates. Non-nested
comparisons retain their existing interpolation-based metrics and omit this
nested-only analysis.

The 50 N 105x105 -> 209x209 exact roots use the same +/-4a domain, with
Hertz radius `a=0.170745 mm`. The root audits independently pass force, mass,
complementarity and direct-elasticity checks. Shared-node pressure L2 changes
4.839%; the existing exit-window L2 change is 10.975%. Spatial localization
shows:

| Region | Share of total squared pressure difference | Regional pressure L2 | Cavitation-state flips |
| --- | ---: | ---: | ---: |
| `0.5 <= x/a < 1.0`, `|y/a| <= 1.2` | 71.954% | 10.974% | 2 |
| `1.0 <= x/a <= 1.5`, `|y/a| <= 1.2` | 0.025% | 25.852% | 0 |
| `0.5 <= x/a < 1.0`, `|y/a| <= 0.5` | 70.757% | 11.221% | 0 |

The maximum shared-node pressure difference is 210.791 MPa at
`(x/a,y/a)=(0.84615,0)`: 438.612 MPa on 105x105 versus 649.403 MPa on 209x209.
Both cells have `theta=1`, so this largest pressure discrepancy is not an
active-set classification flip. On the centerline, the 105 grid has 694 MPa
at `x/a=0.769` and 439 MPa at `x/a=0.846`; the 209 grid has 633 and 649 MPa
at those shared nodes and 665 MPa at the intervening `x/a=0.808` node. This
shows a steep, poorly represented outlet-side pressure feature on the coarse
grid, but does not establish the continuum peak or its physical accuracy.

Four of 11,025 shared nodes change cavitation classification; two are in the
inner exit half. Across 31 shared transverse rows with `|y/a|<=1.2`, native
first-cavitated sample positions shift by a maximum of `0.076923a` and a mean
absolute `0.024814a`. The sampled pressure is zero on all outer edges; the
largest pressure on the first interior x and y rows is respectively
`5.68e-5` and `1.80e-5` of the global peak. This makes a large edge pressure
value an unlikely explanation for the localized discrepancy, but it does not
prove domain independence or exclude nonlocal boundary effects.

The earlier 53x53 -> 105x105 comparison is similarly concentrated: 76.748% of
its squared pressure difference lies in `0.5 <= x/a < 1.0`, `|y/a|<=1.2`,
with a 139.266 MPa maximum difference at `(0.76923,0.30769)`. Across the
27->53, 53->105 and 105->209 sequence, global pressure L2 changes decrease
from 6.931% to 5.640% to 4.839%, but the local pressure differences remain
large. The diagnostic bundles, each with fresh root audits and source
checksums, are under `results/limited_sensitivity_diag_50n_*_verified_20261107/`.

Reproduce the final comparison and localization audit with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -m examples.compare_limited_paths \
  results/limited_mesh_50n_105_nested_cold48_20261106 \
  results/limited_mesh_50n_209_nested_retry2_20261107 \
  --output results/limited_sensitivity_diag_50n_105_to_209_verified_20261107
```

The next targeted experiment is a fixed-domain x-only refinement in this
inner exit region (for example 105x209, about 22k nodes), with y spacing held
at the 105 grid. This can test whether x resolution captures the sharp
pressure feature before spending the substantially larger memory budget of a
uniform 417x417 solve. The result will still require a separate y/domain
study and must not be used as physical validation.
