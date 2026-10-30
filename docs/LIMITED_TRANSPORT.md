# Bounded capacity reconstruction: isolated benchmark

This milestone implements and benchmarks an experimental less diffusive flux
primitive. **It does not add a coupled higher-order EHL solver or change any
prototype/default result.** The current donor and arithmetic references remain
available unchanged. The loaded diagnostics below explicitly prevent treating
this candidate as a drop-in Newton replacement.

## Discrete equations and units

`limited_transport.py` uses CPU NumPy float64. Public axes/film are m, pressure
Pa, density kg/m³, viscosity Pa s and positive-x mean entrainment m/s. Arrays
are `(ny,nx)`. The transported capacity is A=rho*h*theta [kg/m²]. Shared face
mass flux is kg/(m s), and midpoint-volume divergence is kg/(m² s).

Pressure diffusion retains the reference arithmetic face mobility
m=rho*h³/(12 eta). Only x advection changes:

```text
qx[i+1/2] = -average(m)*delta_x(p) + U*A_reconstructed[i+1/2]
A_reconstructed[i+1/2] = A[i] + (x[i+1]-x[i])*slope[i]/2
```

Interior slopes use the monotonized-central (MC) minmod of twice the backward
slope, the nonuniform quadratic-centered slope and twice the forward slope.
At the west node a one-sided quadratic derivative from the first three nodes
is bounded against twice the first forward slope. This supplies a consistent
closure without external ghosts or altering prescribed reservoir values.
The same flux is used by both adjacent volumes, so conservation telescopes
also on nonuniform axes. The y flux and full-source elastic operator are
unchanged.

The reconstructed capacity lies between the adjacent nodal capacities,
including at the west face. Positive nodal capacity therefore gives a
nonnegative advective face capacity. This is a reconstruction property, not
a positivity or TVD theorem for a coupled nonlinear solver or a time step.
No pressure, film or content field is clipped or filtered. The limiter itself
reduces reconstruction near extrema/fronts; that accuracy impact is explicit.

## Analytic linearization and its limitation

`limited_flux_direction` includes the material pressure derivatives,
independent film/content directions and all reconstruction terms on the
selected limiter branch. A caller can supply a full-source elastic film
and signed-offset direction. The selected map is linear in the direction.
At equal candidate slopes, the deterministic preference is centered, backward,
then forward; zero/extremal slopes select a zero slope derivative.

On smooth branches this is an analytic flux JVP and passes independent centered
finite differences. **At a switching point it is a selected linearization,
not necessarily the true directional derivative of the residual or merit.**
A two-sided finite difference there need not match it. In particular, neither
this primitive nor a small smooth manufactured error establishes the full
bordered Jacobian, preconditioner, Newton correction or Armijo slope required
by the repository's coupled-solver gates.

## Independent manufactured benchmarks

The monotone source uses linear SI pressure, exponential x film and quadratic
y film variation, with either theta=1 or a smooth exponential content. The
constant-material limit has independently derived continuum derivatives.
Every interior cell is included, including the west-adjacent column; the
boundary closure is not removed from the error norm. These are prescribed
fields with a source, not complementary EHL roots.

For varying content, relative divergence L2 errors are:

| Grid | Donor | Limited candidate | Candidate observed order |
| --- | ---: | ---: | ---: |
| 33² | 8.041e-3 | 3.997e-5 | — |
| 65² | 4.028e-3 | 1.009e-5 | 1.986 |
| 129² | 2.016e-3 | 2.534e-6 | 1.993 |
| 257² | 1.008e-3 | 6.350e-7 | 1.997 |

Donor orders are about 0.997–0.999. The theta=1 monotone benchmark likewise
approaches second order for the candidate and first order for donor.

A separate trigonometric source includes capacity extrema and grid-dependent
limiter switches. With varying theta, candidate orders are 2.696, 0.517 and
1.815 over successive doublings: the errors decrease, but a blanket
second-order claim is unsupported. At 257² the candidate error is 1.140e-3
versus donor 1.266e-2. Smooth-field accuracy does not establish sharp-exit
accuracy or a uniform nonlinear convergence rate.

Other checks independently reconstruct scalar faces on a nonuniform grid,
verify weighted divergence/boundary-flow cancellation, preserve a stationary
variable-film state with h*theta constant, check bounded sharp-front capacity
including zero downstream content, and verify the selected JVP stays linear
at plateaus with a nonzero alternating-mode response. Sharp-front tests
prescribe fields; they do not solve cavitation or demonstrate a sharper EHL
exit.

## Loaded 200 N diagnostics and adoption decision

The accepted 97x145 donor prototype root is independently re-audited before
flux-only tests with pressure/full elastic, content and 1 nm offset directions.
All plus/minus diagnostic fields retain positive film, nonnegative pressure
and bounded content. They are **unaccepted perturbed fields**, not solutions.

Evaluating the new flux on the donor root gives scaled mass residual
1.7715e-3, well above the original 1e-8 gate: that state is not accepted for
this candidate. Over steps 1e-2 through 1e-7 the best global relative
finite-difference discrepancies from the fixed selected linearization are
about 1.22% (pressure/elastic), 25.25% (content) and 11.02% (offset).
Full-film-only errors are also retained separately; they never replace the
global result or an acceptance criterion. The script records limiter branch
changes to distinguish smooth-branch checks from switching behavior. On cells
whose two x-face limiter labels stay fixed for both perturbations, the best
relative errors are 1.19e-9, 7.48e-11 and 1.43e-9 respectively. Roughly
5,400 cells still switch branches in the small-step loaded probes. These
restricted checks support the smooth-branch implementation; they do not
remove the global directional-derivative issue or establish a globally valid
generalized Newton matrix.

The candidate is therefore retained **only as an isolated benchmark**.
The next implementation must establish a consistent treatment of limiter
switches and a genuine directional-derivative merit/Armijo test, then add the
complete coupled Jacobian, matching sparse preconditioner, reconstructed
bordered correction, strict feasibility, failed-solve diagnostics and original
independent final gates. Only after those pass should 50/200 N candidate
roots and grid/exit comparisons be attempted. This milestone provides no new
minimum-film prediction and assigns none of the remaining HD discrepancy to
the limiter, fluid or correlation.

## Reproduction and evidence

From the repository with the existing environment, choose new output paths:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/benchmark_limited_transport.py --output results/my_limited_benchmark
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/audit_limited_transport_loaded.py --seed results/prototype_200n_97x145_20261030 --output results/my_limited_loaded_audit
```

The loaded audit needs the accepted donor prototype output; it does not need
MATLAB. See [PROTOTYPE.md](PROTOTYPE.md) to generate that seed independently.
Both scripts reject existing output directories. Raw summaries and the
inspected benchmark figure are retained locally under ignored results:
`limited_transport_benchmark_complete_20261030` and
`limited_transport_loaded_branches_20261030`.

Verification: `ruff check .` and whitespace checks pass. The full suite reports
**620 passed, 62 skipped in 358.80 s**. All 11 focused candidate tests also
pass (2.82 s). Optional unavailable backend tests skip gracefully; no GPU
accuracy/performance claim is added. The global loaded switching discrepancies
remain explicit research limitations, not skipped passing assertions.

The local evidence archive is `results/limited_transport_evidence_20261030.zip`
(5,305,321 bytes, 272 entries), including full source/test snapshots, runtime
versions, the accepted donor seed, manufactured/loaded diagnostics, figure,
logs and a SHA256 manifest. SHA256:
`58db7c0cc163f0d8bceafefc8c064897b03df2f4b7688576e7fed474793c06eb`.
Source/state hashes, every manifest digest and ZIP CRCs were verified. The
archive remains local and Git-ignored; no upload or coupled success is claimed.
