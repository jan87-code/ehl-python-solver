# Limiter switches: directional derivatives and merit probes

This increment supplies an analytic one-sided derivative of the isolated
limited flux and a separately frozen linear action. It adds mathematical
full-residual and Armijo probes. **A coupled candidate Newton solve,
preconditioner and physically validated candidate root remain unfinished.**
Existing donor/arithmetic solvers, default choices and baselines are unchanged.

## Two different maps

The SI conventions and capacity reconstruction are those of
[LIMITED_TRANSPORT.md](LIMITED_TRANSPORT.md): arrays `(ny,nx)`, +x mean
entrainment, x/y/h in m, p in Pa, theta dimensionless and mass flux kg/(m s).
Directions dh,dp,dt have units m,Pa,1 per unit dimensionless amplitude.

`limited_flux_directional_derivative` computes the analytic derivative of the
real-arithmetic discrete flux along a specified direction as amplitude tends
to zero **from above**. Minmod branches are selected lexicographically by
(base slope, slope direction). Positive slopes select the smallest base
candidate, then its smallest directional value at a tie; negative slopes use
the corresponding maxima. Zero slopes enter or leave a monotone branch
according to the direction. Mixed signs give zero slope. The bounded
one-sided west closure follows the same rule.

Comparisons use exact zeros/ties. No tolerance merges near-ties, and no film,
pressure, capacity or content is snapped, filtered or clipped. The original
flux function is unchanged.

This derivative is positively homogeneous and generally **non-additive** at
switches. It must never be passed directly to GMRES as a `LinearOperator`.
`freeze_limited_flux_jvp` instead copies the base fields and selects branches
once using a guide direction g. Its returned map B_g is linear for arbitrary
subsequent directions and satisfies:

```text
B_g g = F'(z; g)                 (along its selection guide)
B_g d may differ from F'(z; d)   (for another direction d)
phi'(z; d) = F(z)^T F'(z; d)/N  (phi = ||F||²/(2N))
```

Consequently, solving a frozen linear problem does not justify using its
product as the merit derivative for an arbitrary resulting correction.
The actual directional derivative must be recomputed for that correction;
a nonnegative slope requires rejection/recovery. The new map is a component
for testing, not a proof of global generalized-Jacobian membership or a
complete coupled Newton convergence theorem.

The existing `limited_flux_direction` remains the earlier fixed branch
reference. Its values and tie policy are preserved for comparison.

## Independent checks

Corner tables cover positive/negative ties, zero-to-monotone transitions,
mixed signs, all-zero plateaus and a nonzero 1e-100 slope. The latter explicitly
checks that near-zero slopes are not silently treated as exact switches.
Material/film/content directions at an exactly constant capacity match global
one-sided finite differences in both constant- and full-material limits.
An independent plotted plateau probe has relative flux-derivative error
9.63e-12 at amplitude 1e-3.

Tests verify positive homogeneity and non-additivity of the true derivative,
additivity of the frozen map, agreement on its guide, and isolation from later
mutation of the input base fields. A prescribed-film manufactured-source merit
test independently checks the directional slope and a descent Armijo inequality
at an exact plateau switch, and identifies the opposite ascent direction.

The probe uses the original explicit nondimensionalization: a is dry Hertz
radius, p_ref is Hertz peak pressure and h_ref=a²/R. Unknowns are interior
P=p/p_ref and D=1-theta in C order, followed by h0/h_ref. With
q_ref=rho0*h_ref³*p_ref/(12*eta0*a), mass rows are
a*div(q)/(100*q_ref), complementarity rows min(P,D), and the load row is
(Q-Q_target)/Q_target. N=2*(ny-2)*(nx-2)+1 in the merit normalization.
The row factor 100 is the existing solver scale, not a new tolerance.

The read-only `ResidualProbe` in `examples/benchmark_limiter_switches.py`
assembles the limited mass residual with the original scales, min(P,D)
complementarity and relative force, D=1-theta. Full-source elastic/material
and signed-offset directions are retained. Complementarity ties use
min(vP,vD) for the true derivative, while the frozen map selects a fixed row
from its guide. This is a diagnostic adapter, not a production solver.

The reconstruction reaches the copied east content boundary. Differentiating
that boundary copy is necessary, unlike the original left-donor advection's
stencil. The probe now copies outlet/transverse content directions consistently
with `PropertyWarmSystem.unpack`. Regression tests verify the complete copy
and strict film feasibility, while preserving the original dimensionless
pressure tolerance -1e-10. A -1e-16 Pa rounding artifact in the accepted seed
is not clamped or mistaken for a change to the solver's physical gates.

An independent 9x13 smooth mathematical state checks the complete frozen
product against centered differences. At amplitude 1e-3, relative errors are
5.23e-10 for mass, 2.75e-13 for complementarity and 2.14e-11 for load. Blocks
are checked separately so a large block cannot conceal a bad smaller one.
This weak unaccepted state is not physical-load validation.

## Loaded 200 N results and remaining uncertainty

The accepted 97x145 donor prototype is independently audited, then its exact
raw dimensionless state is used for limited-residual probes. That state is
not a root of the limited equations. Its small averaged scaled merit,
8.8831e-13, must not be confused with passing the original pointwise mass gate.

For six signed pressure/elastic, content and 1 nm offset directions, guide
agreement is exact in the computed products and frozen linearity errors are
<=2.10e-16. Three directions have negative analytic merit slopes:

| Direction | Analytic merit slope | First passing alpha | Trial merit |
| --- | ---: | ---: | ---: |
| Negative pressure/elastic probe | -8.948e-14 | 0.0625 | 8.8768e-13 |
| Positive content probe | -2.331e-15 | 0.125 | 8.8818e-13 |
| Negative 1 nm offset probe | -9.244e-15 | 1 | 8.7912e-13 |

Each passing trial satisfies the analytic directional Armijo comparison with
coefficient 1e-4, lowers the actual merit and respects the original pressure/
content tolerance and strict film floor. The three nonnegative slopes are
not line-searched. Full failed trials are retained. The trial directions are
prescribed diagnostic probes, not computed Newton corrections; passing this
comparison does not make any trial an accepted EHL state or validate the
solver's two-stage Newton line search.

The loaded finite-step derivative problem is **not fully resolved**. On these
near-flat numerical capacity plateaus, many slopes are close to, but not
exactly at, switches. Moderate steps cross branches; sufficiently small steps
encounter floating-point cancellation. Over amplitudes 1e-2 to 1e-7, the best
global full-residual discrepancies still range from about 0.87% to 50.2%
across the signed probes. Relative merit-slope discrepancies also remain
large for some tiny slopes. All estimates, signs, amplitudes and absolute
slopes are recorded rather than replaced by a favorable regional norm.

Thus the exact-switch derivative and frozen-map semantics are implemented
and benchmarked, while independent uniform loaded derivative verification,
Krylov/Schur corrections, matching sparse preconditioning, both nonlinear
line-search stages, recovery/retries and final candidate mass/load/film
acceptance remain open. Next coupled work must test those items explicitly;
these probes establish no new minimum film, mesh limit or HD agreement.

## Reproduction

Use a new output directory and an accepted donor prototype seed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/benchmark_limiter_switches.py --seed results/prototype_200n_97x145_20261030 --output results/my_switch_probes
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/plot_limiter_switches.py --report results/my_switch_probes/summary.json --output results/my_switch_figure
```

The saved donor seed can be generated without MATLAB using
[PROTOTYPE.md](PROTOTYPE.md). Outputs retain the complete base residual, exact
state, full trial histories, finite-difference diagnostics and explicit scope
flags. The final inspected local evidence is under ignored results:
`limiter_switches_final_20261030` and
`limiter_switches_final_figure_20261030`. Earlier diagnostic outputs are
preserved; the final report includes the corrected original pressure gate
and copied-content derivative.

Verification of this increment: `ruff check .` and whitespace checks pass.
The final full suite reports **645 passed, 62 skipped in 342.75 s**; all 36
focused limited-flux/directional checks pass (2.23 s). Optional unavailable
backend tests skip gracefully. No GPU or coupled-candidate accuracy claim is
added. The loaded near-tie discrepancies are recorded research limitations,
not omitted failures or substituted acceptance checks.

The local archive is `results/limiter_switches_evidence_20261030.zip`
(5,467,978 bytes, 276 entries). It contains final probe/figure data, the
accepted donor seed, full source/test snapshots, runtime versions, logs and
a SHA256 manifest. SHA256:
`166168d139024279efa1707eceb448c04d9cac908bb6e9a07028434dc5ced4b3`.
Source/state and manifest hashes and ZIP CRCs were verified. This archive
remains local and Git-ignored; no upload is claimed.

## Follow-up: coupled frozen linear correction model

The mathematical probe now lives in `limited_system.LimitedLoadProbe`, with
the example preserving `ResidualProbe` as an alias. A matching sparse
preconditioner and verified frozen-model Schur/bordered correction benchmarks
are recorded in [LIMITED_LINEAR_CORRECTIONS.md](LIMITED_LINEAR_CORRECTIONS.md).
This advances the linear component; nonlinear/physical validation remains open.
