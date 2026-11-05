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
