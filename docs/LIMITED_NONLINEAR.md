# Bounded limited-flux nonlinear experiments

This increment implements an experimental nonlinear/load adapter for the
limited-capacity candidate. It does not change the prototype CLI default,
establish a validated physical EHL envelope, or complete R1 of the
[product plan](PRODUCT_ROADMAP.md). The arithmetic and donor references remain
available. See [LIMITED_LINEAR_CORRECTIONS.md](LIMITED_LINEAR_CORRECTIONS.md)
for the underlying correction engine and [LIMITER_SWITCHES.md](LIMITER_SWITCHES.md)
for the distinction between an actual directional derivative and a frozen
linear model.

## Equations, coordinates and scales

Public case quantities are SI, CPU float64 arrays have order `(ny, nx)`,
and positive x follows mean entrainment speed Ue [m/s]. The state contains
interior `P=p/pH`, `D=1-theta`, then the signed offset `h0/Href`, in C order.
`pH`, `Href` and the Hertz length scale are those of the resolved case.
The actual gap is `h=h0+geometry+elastic_deformation`, in meters, using
full-source, zero-padded cell-integrated half-space elasticity.

The target uses MC-limited reconstruction of `A=rho*h*theta` [kg/m²].
The optional continuation equation uses
`qx(tau)=qx_donor+tau*(qx_MC-qx_donor)`, `0<=tau<=1`, with the same
pressure diffusion and transverse flux throughout. Mass flux has units
kg/(m s). Tau is a numerical continuation parameter, not physical time.
Only tau=1 identifies the complete limited equation. Tau=0 reproduces the
existing donor operator, including its complete Jacobian and load border.
The residual comprises conservative mass divergence, `min(P,D)`, and
relative force error; the original flux scaling is retained.

## Bounded nonlinear implementation

`limited_newton.solve_limited_load` rebuilds frozen guide-selected linear
models at accepted states. Each model retains the exact nonlocal elastic
Jacobian, with a sparse material or cell-self elastic preconditioner.
Candidate corrections are independently reconstructed and must have true
relative residual <=5e-9. Linear/Krylov failures retain their diagnostics.

The bounds remain 30 nonlinear corrections, eight distinct models per
correction, ten GMRES cycles with restart 60, and 24 full/affine line-search
trials per model. Every merit test uses the actual original residual and
its true one-sided directional derivative, including at limiter switches.
A full step is attempted first; the second stage enforces the original
pressure/content bounds and strictly positive film by a fraction to the
boundary. Accepted trials satisfy genuine directional Armijo decrease.
There is no pressure clipping, content projection or near-zero slope snapping.

Recovery may select complementary linear rows and pivot unmatched free-content
columns found by sparse structural matching. These choices change a correction
model, not the nonlinear residual or final acceptance gates. Duplicate models
are skipped. An optional nearby reference point for the linear model is
implemented and consistency-tested, but **disabled by default**: the loaded
probe did not demonstrate that it resolves the observed stagnation.

## Independent acceptance and continuation

`limited_audit.py` reconstructs the dimensional limited face flux separately
from the candidate flux/JVP routines, including the west reconstruction and
copied-content boundaries. It checks cell and global flux balance, force by
independent trapezoidal quadrature, complementarity, bounds and boundaries.
Final acceptance also reconstructs elasticity by direct full-source convolution.
The original gates are unchanged: scaled maximum mass <=1e-8; relative local
and global balance <=1e-10; complementarity <=1e-10; relative force <=1e-8;
pressure >=-1e-10*pH; theta within [-1e-10,1+1e-10]; film strictly above
1e-12 m. Boundary pressures are exactly zero and inlet theta is one.

Transport continuation begins with donor closure, then approaches tau=1 through
explicit stages with bounded bisection. Intermediate stages request a tighter
local balance <=1e-12 to leave headroom for the final MC audit. This only
tightens acceptance; requesting a tolerance above 1e-10 is rejected. At most
24 stages are attempted, with a minimum retry interval of 1e-12 in tau.
Failed partial states never seed retries. After each accepted intermediate
stage, the exact tau=1 equation is independently audited. Only if that passes
is an audit-only tau=1 stage scheduled, including the final direct elasticity
check. A nearby blended root is never relabeled as an MC root.

A separate SI force path preserves the physical grid, pressure, content and
signed offset while changing load scales. Failure halves the load increment
and rolls back to the last accepted root. Its regression exercises a small
50 to 50.05 N donor path with an injected failed step. This validates rollback
and scale handling; it does not establish a loaded MC continuation envelope.
Observer callbacks receive isolated snapshots; rendering/checkpoint failures
stop separately from numerical failures. Compact path histories retain scalar
and Krylov diagnostics and hashes; the example driver archives full vectors
and raw states for each stage before compaction.

## Reproduction

Use a new output directory for every run. Direct MC experimentation from a
cold Hertz guess:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src .venv/bin/python examples/run_limited_case.py \
  --case cases/circular_50n.json --output results/my_limited_direct
```

Bounded transport continuation from a cold guess:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=.:src .venv/bin/python examples/run_limited_transport_path.py \
  --case cases/circular_50n.json --output results/my_limited_path
```

The path driver also accepts `--seed` pointing to an accepted, re-audited
prototype run with exactly matching physical inputs, grid and domain. Seed
acceptance for the donor/reference equation does not imply MC acceptance.
Each stage saves `state.npz` with SI coordinates/pressure/film/displacement,
content and the explicit dimensionless state, plus a JSON history/audit.
Top-level `summary.json` distinguishes `rejected_path` from
`accepted_numerical_root`; read the exact transport fraction and audit.
Source snapshots and checksums preserve both successful and rejected attempts.
These research drivers do not implement the planned GUI or application API.

## Evidence and remaining limitations

The self-contained tests cover the complete blended derivatives and sparse
operator, donor endpoint equivalence, selected complementary rows, optional
reference-point consistency, actual nonlinear history replay, both line-search
stages, strict feasibility, independent final audits, Krylov failure retention,
structural pivots, force-step retries and observer isolation. They require no
MATLAB checkpoint or ignored result bundle.

Cold donor endpoints close at 50 N and 200 N. The cold 200 N 33x49 case takes
eight corrections, recovers the existing donor film (104.878 nm minimum,
436.669 nm central), and passes independent mass/force/direct-elasticity checks.
That verifies the adapter endpoint, not a new physical result.

Direct limited solves at the existing loaded seeds encounter singular
free-content blocks in cavitated plateaus. Structural pivots and nearby
linearization probes retain diagnostics but do not establish direct robustness.
Transport continuation advances the 50 N 33x33 case close to tau=1; failed
stages and exact MC endpoint audits remain explicitly distinguished from roots.
The detailed final run outcome is recorded below.

The nearly limited coarse film is approximately 159.4 nm minimum and 296.6 nm
central, well below this case's HD estimates of 284.15 and 497.97 nm. It is
not evidence of improved physical accuracy. A full MC root, if obtained,
still needs refinement, domain/side-lobe/exit checks and independent physical
comparison. No tolerance was adjusted to match HD. The next numerical task is
to resolve the limiting cavitation-front robustness and then assess controlled
mesh convergence before selecting a final transport scheme or claiming R1.

## Final bounded 50 N run (2026-10-31)

`results/limited_transport_endpoint_50n_20261031` exhausts the unchanged
24-stage budget: 15 accepted blended stages and nine rejected attempts.
The last accepted fraction is 0.9999999992499999, with exact 50 N force,
relative local balance 1.802e-14 for its own equation, and direct/FFT film
reconstruction error 3.39e-21 m. Its raw minimum/central films are
159.402/296.593 nm on 33x33.

Re-evaluating that same state with the **exact tau=1 equation** gives local
balance 1.405e-10, above the original 1e-10 gate. The exact endpoint is
therefore rejected even though its scaled mass and load checks pass.
Continuation proximity is not acceptance. Smaller retry intervals and tighter
intermediate headroom have not resolved the remaining stagnation within these
budgets. The result remains `rejected_path`, with the accepted blended state
and every rejected stage preserved. No full loaded MC root is claimed.

Independent archive replay reconstructs all 24 raw stage states exactly and
verifies 143 accepted nonlinear steps, original feasibility, complete selected
correction residuals and actual directional Armijo decrease. The largest
measured correction residual is 4.811e-9, below the unchanged 5e-9 gate.
Final direct SI audits are recomputed separately for both each stage equation
and the exact MC equation. Passing these implementation checks does not
reverse the endpoint rejection above.

Local reproducibility evidence is retained under ignored
`results/limited_nonlinear_evidence_20261031/` and its sibling ZIP/receipt.
It includes source snapshots, the accepted donor seed, cold 200 N endpoint,
failed direct/recovery/reference probes, all bounded paths, raw stage vectors,
independent endpoint audit and complete replay report. The manifest records
file sizes and SHA256 values; the receipt records archive hash/CRC verification.
This local archive is not included in Git or claimed uploaded to GitHub.

Verification: Ruff and whitespace checks pass; full CPU suite **674 passed,
62 optional skips** (356.72 s). The final focused adapter/flux/linear suite
passes all 65 checks (8.46 s). GPU physical validation is not added.

## Subsequent guarded endpoint initialization

[The secant follow-up](LIMITED_ENDPOINT.md) now establishes an exact 50 N
33x33 MC root through guarded initialization and a separate final audit.
The failed run documented above remains unchanged. This subsequent discrete
closure does not establish physical mesh convergence or broad MC robustness.
