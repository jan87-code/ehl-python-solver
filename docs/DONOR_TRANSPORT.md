# Coupled donor-capacity transport experiment — 2026-10-30

This extends the previously benchmarked `transport_candidate` flux to an
**experimental**, CPU-only pressure/content/elasticity/load Newton system.
The production arithmetic-capacity default is preserved. The first physical
comparison is the full-viscosity 50 N circular case in
[HAMROCK_DOWSON_VALIDATION.md](HAMROCK_DOWSON_VALIDATION.md).

## Isolated numerical change and benchmark

Write c=rho*h and m=rho*h³/(12*eta). The reference x-face flux is
`-average(m)*dp/dx + Ue*average(c)*theta_left`.
The candidate is
`-average(m)*dp/dx + Ue*c_left*theta_left`.
The transverse diffusion flux is unchanged. Units are SI: pressure Pa,
film m, face mass flux kg/(m s), interior divergence kg/(m² s), integrated
boundary flow kg/s; array order `(ny,nx)`, +x entrainment.

With theta=1 and exactly alternating c_i=c0+epsilon*(-1)^i, the reference
advective face capacity is constant and its divergence response is zero.
The donor divergence response has alternating amplitude 2*Ue*epsilon/dx.
The new regression tests check this directly before any coupled comparisons.
This isolates a full-film transport blind spot; it is not a proof of nonlinear
or temporal stability. The original independent donor tests also cover
nonuniform conservation, scalar faces, complete flux derivatives, two-cell
symbols and manufactured continuum-source consistency.

The donor term adds first-order numerical diffusion to smooth transport.
That tradeoff is explicit: the reference is second order for prescribed
smooth full-film capacity, while donor capacity is first order. Pressure
smoothing, filtering, constitutive clipping and artificial field clamps are
not used. Reported fields solve the changed conservative equations.

## Coupled implementation and checks

`ehl_solver.donor_load.DonorWarmSystem` selects the existing donor flux and
analytic flux direction. Film directions retain the complete zero-padded
nonlocal elastic response plus signed-offset direction. The complementarity
selection, supply/outflow handling and pressure boundaries are inherited.
A separate SI face reconstruction provides the donor audit; the reference
mass audit is not used to accept a donor root.

`DonorLoadSystem` keeps the load border, active recovery, fraction-to-boundary
and genuine directional Armijo logic of `PropertyLoadSystem`. Its sparse
preconditioner corrects the frozen content block and material/self-film
pressure tangent to match donor transport. The offset column uses the complete
true donor JVP. A base-class warm-system factory hook permits the adapter;
its default remains `PropertyWarmSystem`. All old solver entry points retain
the arithmetic stencil. Accepted arithmetic roots cannot be silently relabeled
donor roots or used as accepted donor-transfer seeds.

New checks cover the full material/elastic/offset JVP against joint finite
differences, bordered corrections against an independent dense solve,
forced full-border recovery, mixed active rows, frozen/material/local content
blocks, material/self-film pressure tangents against independent perturbed
fluxes, strict film feasibility, zero-budget rejection and a genuine small
nonlinear donor repair with original mass/force/Armijo gates. Existing
callback, line-search and retry tests remain in the full suite. SI audit
conservation is checked on nonuniform grids.

## Controlled full-viscosity 50 N comparison

The domain [-4a,4a]², fluid, load, elasticity, boundaries, initialization,
30-correction/10-Krylov-cycle budgets and original acceptance gates are the
same as the arithmetic benchmark. N33 starts from the supplied MATLAB
pressure/H0 with unaccepted theta=1. Finer meshes use force-preserving
bilinear pressure/content guesses; each source must be an accepted same-fluid,
same-transport root, and each target must be repaired. No offset adjustment,
field projection, relaxed tolerance or node-cap increase is added.

| Scheme / grid | Minimum film, nm | Central film, nm | Shared-node pressure L2 mesh change |
| --- | ---: | ---: | ---: |
| Arithmetic 33² | 171.5854 | 371.6841 | — |
| Arithmetic 65² | 265.5510 | 479.9217 | 33.60% |
| Arithmetic 129² | 282.9728 | 455.2518 | 41.93% |
| Donor 33² | 241.7382 | 484.8184 | — |
| Donor 65² | 300.9506 | 513.6006 | 11.71% |
| Donor 129² | 302.6991 | 504.2935 | 7.106% |

Pressure change uses common nodes and the finer pressure norm. The last donor
film changes are +0.581% minimum and -1.812% central, versus +6.56%/-5.14%
for arithmetic transport. All donor grids pass strict mass, load, film and
complementarity acceptance. The N129 direct reconstruction has maximum film/
displacement error 1.19e-20 m, scaled maximum mass residual 5.7833e-14,
relative local balance 1.5572e-13, net boundary mass flow zero to reported
roundoff, force 50 N and complementarity 4.1573e-26. Zero reported global net
flow is floating-point closure, not exact-arithmetic proof.

At N129, Hamrock–Dowson predicts 284.1497/497.9692 nm; donor differences are
+6.53% minimum and +1.27% central. These are quantitative comparisons, not
calibration targets or an automatic validation threshold.

The fixed-window raw centerline-x alternating projection falls from 0.21913
(arithmetic N129) to 0.002721 (donor N129). Unfiltered maps and profiles show
the dominant alternating pattern greatly reduced. These projections are
phase/window-dependent diagnostics; a physical sharp feature can have a
nonzero value, so they do not independently prove accuracy. No filtered field
is used for any result, comparison or convergence check.

A fresh arithmetic N33 solve matches the previous baseline's pressure,
content, film, displacement, offset, coordinates and raw state **bitwise**.
This verifies unchanged default numerical behavior for the matched benchmark;
all earlier result files remain untouched.

## Reproduction and limits

Use new output directories:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/compare_conservative_50n.py '/home/jan/GoogleDrive/EHL Solver/EHL_BorderedNewton_Alpha_8978_N33.mat' results/donor33_NEW --alpha 1 --transport donor
```

Refine with `--nodes 65 --seed results/donor33_NEW`, then `--nodes 129` and
the accepted N65 directory. The experimental transfer is explicitly nested
factor two at identical physical boundaries; it cannot extrapolate. Omit
`--transport donor` to retain the arithmetic default. Records include all
exact states, SI fields, audits, histories, source snapshots and hashes.

This is promising evidence for the transport change in one full-viscosity
case, not full EHL validation or production-default adoption. Pressure still
changes 7.1% on the last refinement, central film is nonmonotone, donor
numerical diffusion is first order and the finite-reservoir extent is not
independently converged. Next checks should isolate domain sensitivity at
fixed physical spacing, further resolution under explicit resource limits,
and another load. Fluid-law/correlation uncertainty and the sharp exit region
also need assessment. No run above the default 20,000-node budget, new GPU
implementation or performance speedup is claimed.

## Fixed-spacing domain check

A bounded additional experiment embeds the accepted donor N65 [-4a,4a]²
state in N97 [-6a,6a]² at **identical** spacing 21.3431266 micrometers.
The exterior is initialized with p=0, theta=1 as an explicitly unaccepted
guess; source pressure/content in the old rectangle and signed offset are
preserved. New-domain force is unchanged before repair. It uses the same
30-correction/10-cycle limits and only 9,409 physical nodes. No extrapolation,
clipping or changed inlet/outlet rule is introduced.

The expanded-domain solve passes original donor mass/load/film/constraint
gates. Minimum/central film become 302.2005/516.2570 nm, changes of
**+0.415%/+0.517%** from the same-spacing smaller domain. Pressure relative L2
change on the common rectangle is **0.7823%**. Direct film reconstruction
error is 6.78e-21 m; scaled mass maximum is 1.5569e-13, relative local/global
balance 7.7814e-13/1.8919e-16 and complementarity 4.7358e-25.

This is a two-domain sensitivity result at the N65 spacing, not proof of an
infinite reservoir or domain independence at the N129 spacing. The sharper
exit region, further resolution and another load remain open. The complete
standalone domain script, initialization and states are preserved under
`results/donor_transport_analysis_20261030/`; no production boundary behavior
was changed. This advances, but does not complete, the domain-validation
priority listed above.

Validation of this coupled increment: full CPU suite **588 passed, 62 optional
backend/GPU skips** (359.78 s), Ruff and whitespace checks clean. The 21 focused
donor-component/coupled tests pass, including the existing manufactured
consistency benchmarks. Five loaded N129 pressure probes with an independent
1 nm offset direction agree with central finite differences to better than
5.74e-9 at the best tested step. All 23 accepted corrections in the three
new-grid solves and domain solve pass independently rechecked original
Armijo/true-linear-residual gates. These timings include concurrent benchmark
work and are not isolated performance measurements.

Verified local archive: `results/donor_transport_evidence_20261030_verified.zip`,
13,488,985 bytes, 122 files; all manifest SHA256 and ZIP CRC checks pass.
Archive SHA256:
`67f056a08d37481cc87095bca5a91d3655b065bb2317d805a9171c4597ec78bc`.
It retains arithmetic baselines, three donor grids, fixed-spacing domain
initialization/solve, exact source versions, raw states and histories, loaded
JVP checks, independent audits and the inspected unfiltered figure. It is
Git-ignored; no off-machine backup is asserted.

## Follow-up: standalone prototype and 200 N (2026-10-30)

The second-load, mesh, exit and fixed-spacing directional domain study is now
recorded in [PROTOTYPE_VALIDATION.md](PROTOTYPE_VALIDATION.md). The bounded
SI-input [prototype](PROTOTYPE.md) reproduces 50 N without MATLAB. At 200 N,
minimum film remains discretization sensitive despite strict equation/load
closure, so donor remains a candidate and the library default is preserved.
