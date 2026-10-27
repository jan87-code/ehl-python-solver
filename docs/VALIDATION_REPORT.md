# EHL scientific verification report

Audit history: 2026-10-26 through 2026-10-27. Sections 1–10 retain the
original incompressible/isoviscous audit; sections 11 onward document
pressure-dependent NumPy/SciPy CPU developments. The latest bounded
matrix-free force path now closes 200 N on 33x49 and 65x97 nodes;
mesh/domain accuracy remain open. This report records numerical verification,
**not complete physical validation of classical piezoviscous EHL**.

## 1. Diagnostic audit before further physics changes

The full-film mathematical reference solves

```
q = -h^3 grad(p)/(12 eta0) + Ue h e_x,       div(q) = 0,
Ue = (u1 + u2)/2,
h = h0 + x^2/(2 Rx) + y^2/(2 Ry) + w[p].
```

Pressure is signed and no cavitation condition is enforced in that reference.
Its negative-pressure tensile load is mathematical; it cannot represent a
ruptured liquid film. The separate mass-conserving CPU model solves

```
q = -h^3 grad(p)/(12 eta0) + Ue h theta e_x, div(q) = 0,
p >= 0,  0 <= theta <= 1,  p (1-theta) = 0,
integral_domain p dx dy = Q > 0.
```

Here p is gauge pressure relative to pc=0, theta is transported fractional
film content, eta0 is constant dynamic viscosity [Pa s], and q is volume
flux per transverse width [m^2/s]. Constant density cancels from mass
balance; no numerical density value or compressibility law is implied.
Saturation contributes no separate mechanical pressure or load. Positive
pressure increases the gap through linear elastic displacement w [m].

All solution fields use `(ny,nx)`, x along positive entrainment, west xmin
inlet and east xmax outlet. SI interfaces use x,y,h,w [m], p [Pa], Q [N],
eta [Pa s], Ue [m/s] and E* [Pa]. Internal X=x/L, Y=y/L, H=h/h_ref and
P=p/p_ref give `q_hat=-H^3 grad(P)+Lambda H theta e_x`, with
`Lambda=12 eta0 Ue L/(h_ref^2 p_ref)` and
`q_ref=h_ref^3 p_ref/(12 eta0 L)` [m^2/s]. Thus q=q_ref q_hat;
integrated boundary flux has scale q_ref L [m^3/s].

The conservative uniform-grid stencil averages H^3 arithmetically on faces,
averages H arithmetically in the advective flux and upwinds theta from the
west. Divergence applies only to interior control volumes; their outer faces
lie halfway between interior and boundary nodes. Pressure is zero on every
edge, west theta=1 is prescribed, east theta is copied for output, and there
is no transverse advective content flux. A flooded west boundary does not
prove that the finite domain represents the fully flooded far-field limit.

Elasticity uses the source-cell integral of `1/(pi E* r)` [m/Pa],
`1/E*=sum((1-nu_i^2)/E_i)`. Linear FFT convolution is zero-padded to full
support with backward FFT normalization and centered cropping; there is no
circular wraparound. The Hertz circular reference uses
`a=(3 Q R/(4 E*))^(1/3)`, `p0=3 Q/(2 pi a^2)` and approach `a^2/R`.
The common alternative modulus E'=2 E* must be converted explicitly.
Positive dry-contact displacement equals approach at the center; this sign
does not prescribe h0 in a fluid load-controlled calculation.

The cavitating Newton unknowns are interior P, deficit D=1-theta and h0/h_ref.
Mass rows use div(q_hat)/100, complementarity uses min(P,D), and the force
row uses `(Q_calc-Q)/Q`. The analytic active-set generalized derivative,
directional derivative at ties, verified Schur/GMRES corrections, feasibility
and Armijo globalization account for nonsmooth constraints. The separate
prescribed-film active-set solve and prescribed-offset Picard solve remain
independent numerical references. Linear equilibration changes coordinates
only; reported mass diagnostics always evaluate the original flux.

### Why existing pressure/film plots differ from classical EHL

Inspection of saved positive-load fields shows an inlet pressure hump and
a predominantly parabolic bowl, with gap-increasing deformation small
relative to the prescribed geometry/offset in the weak case. The model has
constant viscosity/density, very soft numerical material E*=10 MPa, low
pressure and finite nearby inlet/side boundaries. It has neither demonstrated
a Hertz-like strongly loaded plateau nor independently verified outlet film
constrictions or a piezoviscous spike. Their absence in this regime alone is
not a solver failure. Signed full-film plots additionally contain unphysical
tension. Regimes, constitutive laws and boundaries must match before making
shape comparisons; plotting must not impose an expected EHL profile.

The old generic pressure title incorrectly says full-film even when reused
for a JFO result. Automatic contour levels can also exclude tiny negative
roundoff pressures, producing white polygons in a cavitated zero-pressure
region. That rendering defect must be corrected using the entire supplied
data range, without clipping fields or hiding the signed pressure minimum.

### Suitability and unresolved limitations

At fixed Q=0.002 N, moving only the inlet changes h0 from 21.3005 to
40.7640 um; moving only the transverse edges changes it to 29.7132 um.
Moving an already cavitated outlet leaves the shared pressure unchanged to
roundoff on these particular cases. Therefore the small-domain film heights
are not domain independent, even though nonlinear, force and mass residuals
pass. At the baseline minimum film the ratio h_min/sqrt(2 Rx h_min) is about
0.033; this and the gap/extent must be reviewed as domains and offsets change.
No arbitrary thin-film validity cutoff is used to claim physical accuracy.

Missing classical-EHL ingredients/evidence include benchmark-matched
pressure-dependent viscosity and density, adequately resolved inlet/outlet
and side domains, independent film-height data, and coverage of realistic
loads/materials. The model also assumes Newtonian lubricant, smooth surfaces,
small-slope parabolic geometry, linear elastic half spaces, steady flow and
isothermal conditions. Thermal effects, shear thinning, transient squeeze,
roughness, mixed contact and dissolved-gas physics are absent. These are
stated model assumptions, not reasons to alter fields to resemble a figure.

The optional NVIDIA resident signed full-film Newton/Krylov implementation
is preserved. Its existing mathematical parity/performance evidence does
not establish device JFO support. This report adds no GPU port and makes
no new CUDA/ROCm execution claim on a host where those tests skip.

## 2. Evidence and reproducible benchmark figures

The following sections are populated from independently rerun benchmarks,
not inferred from successful code execution or figure appearance.

### A. Dry circular Hertz contact

Inputs: Q=5 N, R=0.015 m and E*=113 GPa. The analytical radius is
79.2528145 um, p0=380.086113 MPa and approach=0.418733907 um. Pressure
has its analytical Hertz support; it is not obtained by clipping a fluid
solution. Grids span x,y=+/-4a. A separate SI trapezoidal load integration
and the DC-FFT center displacement are checked against the analytical values.

| Grid | Integrated load [N] | Relative load error | Relative center displacement error |
| --- | ---: | ---: | ---: |
| 33x33 | 4.856600873 | 0.0286798 | 0.0166626 |
| 65x65 | 4.957580249 | 0.00848395 | 0.00484126 |
| 129x129 | 4.992816573 | 0.00143669 | 0.000922867 |

The independent analytic radial load integral is exactly Q. Sampled Cartesian
integration and cellwise pressure approximation produce the errors above;
these are not nonlinear load-closure errors. The sampled dry gap can have
small negative discretization errors and is explicitly not labeled a positive
lubricant film. Direct spatial versus FFT tests, including singular source
cells and no-wrap cases, independently cover the deformation operator.

### B. Prescribed-film isoviscous cavitation

The matched analytical benchmark uses h=20 um+x^2/(2 R), R=0.01 m,
eta=0.04 Pa s, Ue=0.425 m/s and a finite flooded inlet at x=-3L,
L=sqrt(2 R hmin)=0.632455532 mm. The wide 2-D comparison has x=+/-3L,
y=+/-8L, 17 transverse nodes, p=0 at every edge and prescribed film with
no deformation or load controller. It approaches an exact infinite-width
smooth-rupture solution, not an exact finite-width reference.

With X=x/L, H=1+X^2 and p_ref=12 eta Ue L/hmin^2, Lambda=1. Primitives
I2'=H^-2 and I3'=H^-3 give
`P=I2(X)-I2(Xin)-Hc*(I3(X)-I3(Xin))` upstream. The positive rupture Xc
satisfies `I2(Xc)-I2(Xin)=Hc*(I3(Xc)-I3(Xin))`, Hc=1+Xc^2. Downstream
P=0 and theta=Hc/H; both sides have constant SI flux Ue*hmin*Hc. Independent
adaptive quadrature tests pressure and load to relative 1e-10, and the
analytical flux identity to relative 2e-14. These are derived references,
not values manufactured from the numerical solver.

| Grid `(ny,nx)` | Relative pressure L2 error | Relative force-per-width error | Maximum content error |
| --- | ---: | ---: | ---: |
| (17,33) | 0.0317444 | 0.0334057 | 0.0721956 |
| (17,65) | 0.00641594 | 0.00645935 | 0.0370292 |
| (17,129) | 0.00163204 | 0.00165019 | 0.0185179 |

The exact rupture is 0.293551858 mm; the last grid's first reduced-content
node is 0.296463531 mm. Exact force per transverse width is 32.380268794 N/m;
the last centerline integrates to 32.326835053 N/m. These units are N/m,
not point-contact force N. Maximum original dimensionless mass residual is
9.82e-13; complementarity is zero at printed precision. The separate
converging/diverging and repeated-wedge regressions also cover admissible
content, three rupture/reformation transitions, front refinement, starvation
and local/global conservation. Arbitrary reformation and experimental gas
physics are not thereby validated. No matched published field dataset was
available for these particular generated wedge/point inputs; they are
analytical-limit and self-consistency verification cases.

### C. Elastic, cavitating positive-load numerical case

Exact shared inputs: Q=0.002 N, E*=10 MPa, Rx=0.01 m, Ry=0.02 m,
eta=0.04 Pa s, Ue=0.425 m/s; initial h0=20 um, L=0.5 mm,
h_ref=20 um, p_ref=21250 Pa and Lambda=12. Coordinates span
x=[-0.5,0.75] mm, y=[-0.5,0.5] mm. West content is one; every pressure
edge is zero. All solves use float64 NumPy/SciPy, full compliance, at most
30 Newton corrections, maximum mass/relative local tolerance 1e-8,
complementarity/admissibility tolerance 1e-10 and relative force tolerance
1e-8. The prescribed-film reference is tested before use in coupled solves.

| Grid `(ny,nx)` | h0 [um] | Sampled minimum film [um] | Peak pressure [MPa] | Maximum mass residual | Newton corrections |
| --- | ---: | ---: | ---: | ---: | ---: |
| (9,13) | 20.9357776900 | 21.2310978562 | 0.00826372104 | 1.32e-11 | 11 |
| (17,25) | 21.4898604871 | 21.7850065175 | 0.00791291610 | 3.83e-10 | 7 |
| (25,37) | 21.5505758045 | 21.8446983918 | 0.00792941974 | 2.96e-13 | 19 |

The finest sampled film minimum is at (0.020833333,0) mm and peak pressure
at (-0.222222222,0) mm. These are grid extrema, not a subgrid extremum
estimate. Independently trapezoid-integrated load is 0.002 N on all three
grids (zero error at reported precision). Finest net dimensionless flux is
2.04e-14, maximum P(1-theta)=1.68e-23, theta range [0.4815039935,1],
direct spatial film-coupling discrepancy 6.78e-21 m, and maximum gap-increasing
deformation 0.407830633 um at (-0.222222222,0) mm. The maximum local
displacement/film ratio is 0.017213, quantifying this case's weak coupling.
Tiny negative pressure roundoff
(finest -9.84e-19 Pa) is reported, retained and compared with the declared
admissibility tolerance; no claim of exact floating-point positivity is made.

At 9x13 a separate prescribed-offset Picard solver nested inside a bracketed
scalar force root matches monolithic pressure to 8.25e-13 relative L2 and
h0 to 4.13e-17 m. The reference has its own direct audit; its fields and force
evaluations are saved and overlaid. This checks two algorithms solving the
same discrete physical assumptions; it is not independent experimental
validation. Successive same-domain h0 changes shrink from 0.554083 to
0.0607153 um, but domain error is much larger:

| Separate boundary change | Grid | x interval [mm] | y interval [mm] | h0 [um] |
| --- | ---: | --- | --- | ---: |
| Baseline | 13x17 | [-0.5,0.75] | [-0.5,0.5] | 21.3005243 |
| Inlet +4 cells | 13x21 | [-0.8125,0.75] | [-0.5,0.5] | 32.9530689 |
| Inlet +8 cells | 13x25 | [-1.125,0.75] | [-0.5,0.5] | 40.7639950 |
| Outlet +4 cells | 13x21 | [-0.5,1.0625] | [-0.5,0.5] | 21.3005243 |
| Outlet +8 cells | 13x25 | [-0.5,1.375] | [-0.5,0.5] | 21.3005243 |
| Both y edges +4 | 21x17 | [-0.5,0.75] | [-0.833333,0.833333] | 27.0190576 |
| Both y edges +8 | 29x17 | [-0.5,0.75] | [-1.166667,1.166667] | 29.7131725 |

All original nodes remain fixed: dx=78.125 um and dy=83.333333 um. The
seven independently started reference/continuation cases reach the same
target through 42 audited accepted roots. The inlet/side changes remain
substantial even for the rigid reference. See PHYSICS.md and the generated
JSON for individual boundary budgets, accepted/rejected histories and
conservation. An already cavitated outlet happens to be insensitive here;
that is not a guarantee for other domains or regimes.

### Controlled rigid extent versus resolution

A distinct rigid point-contact study keeps the physical minimum gap fixed
at 0.1 um, Rx=0.01 m, Ry=0.02 m, eta=0.04 Pa s, Ue=0.001 m/s.
It uses L=sqrt(2 Rx h0)=44.7213595 um, Ly=sqrt(2 Ry h0)=63.2455532 um,
p_ref=2.14662526 MPa, Lambda=1, x=+/-A L and y=+/-A Ly. Extents
therefore vary without solving a different h0. dx=dX L, dy=dX Ly;
`W=Q/(p_ref L^2)` uses the **same** L for both nondimensional integration
coordinates, so the transverse Jacobian Ly/L is retained.

| A | dX | Grid | W |
| ---: | ---: | ---: | ---: |
| 2 | 0.5 | 9x9 | 0.179096380659 |
| 4 | 0.5 | 17x17 | 0.349960118810 |
| 8 | 0.5 | 33x33 | 0.466972938676 |
| 16 | 0.5 | 65x65 | 0.530974979074 |
| 32 | 0.5 | 129x129 | 0.563747480350 |
| 8 | 0.25 | 65x65 | 0.498109443073 |
| 8 | 0.125 | 129x129 | 0.507565861780 |
| 16 | 0.25 | 129x129 | 0.562387946786 |
| 32 | 0.25 | 257x257 | 0.595191474007 |

Every case passes an independent **SI** face-flux calculation before
conversion to dimensionless divergence: largest max mass residual 1.03e-10,
relative local flux imbalance 6.53e-14 and absolute net dimensionless flux
6.04e-9. The largest grid has 66,049 nodes; a 70,000-node preallocation cap
and the existing 100-active-set iteration limit bound this study.

At A=8, resolution-induced load changes decrease (0.0311365 then 0.00945642
in W). At dX=0.25, doubling A from 16 to 32 still changes W by 5.51% of
the larger result. The largest case has Q=0.00255530610 N, pmax=0.188843594
MPa at (-0.022360680,0) mm and hmin=0.1 um at (0,0). Maximum rigid slope
grows to 0.1753 at the far corners; this warns against treating unlimited
parabolic-domain expansion as automatically valid thin-film physics.
No far-field limit or asymptotic convergence order is claimed.

Independent similarity regressions multiply h0 by four and both natural
lengths by two: pressure scales by 1/8 and force by 1/2, while content is
unchanged. Multiplying eta by two and speed by three multiplies physical
pressure/force by six. A constant-film transport test verifies dimensional
boundary-flow units and the interior control-volume extent; a corrupted
content field is detected by the independent flux audit.

## 3. Classical EHL benchmark readiness

Candidate targets are Table I of [Hamrock and Dowson, NASA TN D-8317,
Fully Flooded Results (1976)](https://ntrs.nasa.gov/api/citations/19770004457/downloads/19770004457.pdf),
printed p.17 (PDF page 19):

| k | U | W | G | Reported numerical Hmin=hmin/Rx |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1683e-11 | 0.1106e-6 | 4522 | 3.367e-6 |
| 1.25 | 0.1683e-11 | 0.1106e-6 | 4522 | 4.105e-6 |

Their U=eta0 Ue/(E' Rx), W=Q/(E' Rx^2), E'=2 E*, and G=E'/p_iv,as
differ from this code's Lambda. These are numerical table values, not the
least-squares correlation. For an explicitly chosen spherical scaling
Rx=0.015 m, E'=226 GPa, eta0=0.04 Pa s, conversion gives Ue=0.14263425
m/s, Q=5.62401 N and the first target hmin=50.505 nm. This is a derived
parameterization, not a claim that the report specified those SI inputs.

The model must first match [NASA TN D-8049, Theoretical Formulation,
equations 23/25](https://ntrs.nasa.gov/api/citations/19750022492/downloads/19750022492.pdf):
pressure-dependent density and Roelands viscosity, including their original
unit conversions and fluid constants. A Barus exponential is not an exact
substitute. Benchmark-matched far-field/cavitation treatment and a resolved
elliptic geometry are also required. No comparison is executed against
these classical targets by the current isoviscous solver.

For soft-contact follow-up, [NASA TN D-8528](https://ntrs.nasa.gov/api/citations/19770022577/downloads/19770022577.pdf)
provides a separate low-modulus benchmark family; its formulation still
retains compressibility/viscous effects. It must not be assumed identical to
our constant-property case merely because both use soft materials.

The next justified milestone is a controlled rigid/coupled far-field error
budget, followed by independently benchmarked SI constitutive laws and their
derivatives before integrating them into the conservative monolithic JFO
system. No existing low-pressure result is evidence of classical EHL accuracy.

## 4. Reproduction and outputs

From the repository root, with a separate local environment installed on
each platform:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_scientific_validation \
  --output-directory results/scientific_validation_new
```

PowerShell equivalent (uses a local writable Matplotlib cache):

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:PYTHONPATH="src"
$env:MPLCONFIGDIR=Join-Path $env:TEMP "ehl-mpl"
python -m examples.run_scientific_validation --output-directory results/scientific_validation_new
```

The runner refuses existing output directories, checkpoints each completed
stage and preserves failure status rather than labeling incomplete runs
successful. Source hashes and runtime versions identify the actual code
under test, including an uncommitted runner during development. It uses
pathlib, finite portable JSON and SI NPZ arrays, and requires no PyTorch.

The verified run is under `results/scientific_validation_20261026_final/`:
`hertz/`, `analytical_reference/`, `rigid_farfield/`, `grids/` and `boundaries/`.
Root `summary.json` contains exact parameters, audited results, histories,
sampled extrema/coordinates and a figure manifest. Per-case NPZ files retain
raw pressure, film, deformation and content; the matched nested root has its
own NPZ. Load histories retain exact zeros rather than flooring them for log
plots. No field is smoothed or clipped; negative roundoff values remain
visible in numerical diagnostics and the full contour range.

Mandatory figure types are generated: pressure [MPa], film [um], deformation
[um], content, streamwise pressure/film, transverse film, original mass and
force error histories, Krylov histories, grid/domain sensitivity, active
regions and matched analytical/nested comparisons. Field figures include
operating parameters, domain/grid, orientation, extrema and solver status.
Dry Hertz plots explicitly mark lubricant quantities as not applicable.
The active-region threshold is documented as theta<1-1e-8; it is a diagnostic
label applied after solving, never a change to pressure or content.

These figures and numerical files are generated artifacts, intentionally
excluded from Git. The scripts, report, regression tests and reproduction
commands are tracked. Existing result directories are preserved.

Final verification on this CPU environment: `ruff check .` clean;
`pytest -q` **269 passed, 62 skipped** (63.77 s). The skips are optional
PyTorch/device coverage; this run does not repeat the earlier Windows NVIDIA
execution. All five runner stages complete, producing 96 figures and 75 SI
NPZ files. Representative contours, extrema labels, Newton/load histories,
domain comparisons and matched-reference overlays were visually inspected.
The physical discrepancy remains open even though these numerical checks pass.

## 5. Far-field error budget follow-up (2026-10-26)

**Outcome: domain independence remains open.** The previous 5.51% rigid
domain-load change decreases to about 1.17% between the last two accepted
fine-core domains (1.16% using the later load as denominator). A coarser
sequence has two successive changes below 1%, but is not sufficiently
mesh-resolved. Larger fine-core candidates violate the original continuity
criterion. Neither those candidates nor a coarse plateau establish the
requested engineering accuracy. No constitutive physics or GPU solver is
changed in this follow-up.

### Boundary audit and supply interpretation

The imposed pressure is zero gauge on every boundary node. West content is
exactly one; positive-x face transport uses that upstream value. East content
is copied only for output and has no upstream influence on the upwind flux.
North/south content is copied from the nearest interior row for output,
without imposing an additional normal advective flux. There is no advective
transverse content flux, but the zero side pressure
allows diffusive lateral outflow. Boundary budgets surround the **interior
midpoint control volumes**, not the rectangle through the outer node centers.
The reported outward west flow is negative for incoming fluid; east, north
and south flows are positive for outgoing fluid in these cases.

This is a finite ambient-pressure, flooded-reservoir/content boundary
approximation. It prescribes content, **not integrated mass feed**. Increasing
the inlet distance changes its gap, pressure gradient and resulting volume
flow. The study retains the identical content and pressure assumptions and
reports those changing flows; it does not describe them as a constant-feed
experiment. Constant density cancels from the model, so flows are reported
in m^3/s. Multiplication by a specified constant rho0 gives kg/s; no density
value or density-pressure law is invented here.

Elastic displacement is free-space half-space response with pressure taken
as zero outside the source domain; displacement is not fixed to zero at a
fluid boundary. Kernel coefficients integrate 1/(pi E* r) over source cells
and have units m/Pa. For uniform `(ny,nx)` arrays the kernel has shape
`(2ny-1,2nx-1)`, full FFT padding is at least `(3ny-2,3nx-2)`, inverse
normalization is the padded element count and the same crop begins at
`(ny-1,nx-1)`. A zero-pressure embedding regression and independent spatial
summation confirm that extending empty support leaves shared displacements
unchanged. Thus circular FFT wraparound is excluded as the cause of the
observed field/domain differences on those checks.

### Declared criteria and new prescribed-film verification path

Before running, the budget runner records these provisional criteria:

- Successive load, sampled minimum-film and peak-pressure changes below 1%,
  on more than one domain increment, with separately checked mesh effects.
- Maximum original dimensionless continuity defect <=1e-8, relative local
  flow defect <=1e-10, and `abs(sum(F))/sum(abs(F)) <=1e-10` globally.
- Dimensionless complementarity <=1e-10 and pressure/content feasibility to
  the existing 1e-10 dimensionless tolerance; exactly zero pressure edges and
  flooded west content are independently checked.
- Front comparisons use node brackets at theta<1-1e-8, with a provisional
  tolerance of two locally resolved core cells. This is a resolution-based
  comparison criterion supported by the separate analytical rupture checks,
  not a proven continuum error bound for arbitrary 2-D fronts.

Uniform extension to hundreds of natural lengths would exceed the resource
budget. `rectilinear_cavitation.solve_rectilinear_cavitation` therefore adds
a separate **prescribed-film** CPU verification path. Face mobility/film
retain the arithmetic interpolation and content retains west upwinding.
Each face-flow difference is divided by its own midpoint control width.
The equilibrated active-set system keeps the original constraints and
1e-10 tolerances; no pressure/content/film clipping or regularization is used.
Independent SI whole-face fluxes audit the sparse row assembly.

Uniform-grid parity covers flooded/starved cases, and nonuniform x/y grids
converge toward the independent finite-inlet parabolic analytical solution.
Iteration exhaustion raises explicitly. The original uniform NumPy/SciPy
solver and monolithic coupled solver are unchanged. In particular, the
translation-invariant FFT operator is **never** applied to unequal cells.
Graded elasticity here is only postprocessing by direct cell integration;
no coupled graded-grid solver is claimed.

The inner normalized coordinates retain an exactly uniform [-2,2] core.
Outside it, successive cell widths grow by a fixed factor of 1.1. Domain
expansion snaps outward to existing nodes: every prior node/cell is retained,
and actual SI bounds, rather than requested labels alone, are saved. Fixed-
domain mesh studies instead terminate exactly at the same physical bounds.
Inputs are h0=0.1 um, Rx=10 mm, Ry=20 mm, eta=0.04 Pa s, Ue=0.001 m/s,
L=44.7213595 um, Ly=63.2455532 um, p_ref=2.14662526 MPa and Lambda=1.
All runs use one BLAS/OpenMP thread, at most 70,000 nodes and the existing
100 active-set / 30 Newton correction budgets.

### Separate mesh and outer-grid error contributions

The physical mesh-study domain is x=+/-1.431083506 mm and
y=+/-2.023857703 mm, with the same h0/material/flow inputs throughout.
Rigid minimum and central film are prescribed at 0.1 um at the origin;
their invariance is not an independent film-height validation.

| Core dX | Grid | dx,dy in core [um] | Integrated load [N] | Peak pressure [MPa] |
| ---: | ---: | --- | ---: | ---: |
| 0.5 | 51x51 | 22.3607,31.6228 | 0.002389522816 | 0.172222135 |
| 0.25 | 71x71 | 11.1803,15.8114 | 0.002532770891 | 0.188578567 |
| 0.125 | 101x101 | 5.59017,7.90569 | 0.002578992883 | 0.193414769 |
| 0.0625 | 147x147 | 2.79508,3.95285 | 0.002592733490 | 0.194663315 |

Successive relative load changes (relative to the preceding grid) are
5.995%, 1.825% and 0.533%; peak-pressure changes are 9.497%, 2.565%
and 0.646%. The peak remains at (-0.022360680,0) mm. The finest rupture
bracket is [0.013975425,0.016770510] mm; no centerline reformation occurs.
The coarse front bracket is wider and is retained rather than interpolated
to an expected location. Sampled pressure extrema and full front brackets
are recorded for every case.

Keeping the finest core and the **same physical domain**, reducing outer
growth from 1.1 to 1.05 gives a 197x197 grid, load 0.002602223079 N and
peak 0.194746019 MPa: another 0.366% load and 0.0425% pressure change.
Outer grading thus contributes to the error budget independently of core
resolution. These observed changes are not a rigorous summed error bound
or an established asymptotic order.

### Domain changes and rejected fine-core candidates

The following graded sequence keeps core dx=5.59017 um and dy=7.90569 um
fixed. Requested labels are in units L/Ly; snapped SI extents are explicit.

| Label | Grid | x half-extent [mm] | y half-extent [mm] | Load [N] | Max continuity defect | Accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 101x101 | 1.461699 | 2.067155 | 0.002581786825 | 6.47e-11 | yes |
| 64 | 117x117 | 3.094925 | 4.376885 | 0.002653662258 | 8.28e-10 | yes |
| 128 | 131x131 | 5.999312 | 8.484309 | 0.002684826537 | 4.22e-9 | yes |
| 192 | 139x139 | 8.768026 | 12.399862 | 0.002695311553 | 1.53e-8 | **no** |
| 256 | 145x145 | 11.659141 | 16.488515 | 0.002700948000 | 3.99e-8 | **no** |

Accepted 32->64 and 64->128 load changes are 2.784% and 1.174% relative
to the preceding state. Peak-pressure changes are already tiny, and all
these core rupture brackets are [0.011180340,0.016770510] mm. The last
two apparent load changes are below 1% but their continuity defects fail;
their raw states are saved and red crosses distinguish them in the plots.
Small updates, an apparently stable pressure maximum and excellent global
balance do not override that failed local criterion.

On core dX=0.25 the accepted 64/128/192/256 loads are
0.002607925736/0.002639121803/0.002649500444/0.002655051122 N.
The final two relative changes are 0.393% and 0.210%, with all local defects
below 6.75e-9. However the matched fine-core 128 load is 1.73% higher and
its peak pressure is about 2.6% higher. The coarse plateau therefore cannot
establish the combined 1% target.

The increasingly large H^3 contrast and far-field linear accuracy are an
unresolved numerical limitation at the rejected fine-core sizes. Bounded
experiments with linear iterative refinement and altered sparse pivoting did
not resolve the original face-flux defect; a constraint-eliminated trial was
less accurate and was not retained. The trusted full constrained system
remains in use. No acceptance tolerance or iteration cap was relaxed.

Far-corner rigid slope grows from 0.179 at label 32 to 0.735 at 128 and
1.428 at 256. Thus the very large formal parabolic domains cannot be assumed
to satisfy the small-slope lubrication approximation physically. Better
algebra alone will not justify a physical far-field model at those slopes.

### Boundary attribution, flux budget and omitted pressure

Starting from the same label-32 grid, extending only the inlet to label 128
changes load by +2.0742%; extending only both transverse boundaries changes
it by +0.3334%; extending only the outlet changes it by roundoff (4.45e-16
relative). Every old node is retained. The inlet is the largest separate
boundary influence tested. Simultaneous effects are not additive: the
pressure field and transverse leakage readjust together.

For accepted fine-core label 128, outward west/east/south/north volume flows
are [-1.590940919e-8, 9.634576726e-9, 3.137416234e-9, 3.137416234e-9]
m^3/s. Their net is 1.252e-21 m^3/s; normalized global defect is 3.94e-14
and maximum relative local defect is 5.37e-12. Complementarity is zero at
printed precision; theta spans [7.08336e-5,1]. These excellent global metrics
are reported separately from the local absolute criterion.

Direct source-cell integration at selected targets confirms the uniform
FFT convention, including singular source cells and physical cell dimensions.
For a fixed, already-solved label-128 rigid pressure field, deleting sources
outside +/-16 L, +/-16 Ly removes 0.000141763 N (about 5.28% of load) but
only 2.878e-11 m (0.2205%) of center displacement at diagnostic E*=1 GPa.
This is a **source-truncation diagnostic** on the same pressure field, not
a re-solve or a coupled elastic solution. Re-solving the smaller domain
also redistributes interior pressure and imposes new fluid boundaries;
that distinct error must not be confused with FFT padding or omitted-source
displacement. No source cropping is applied to the solved fields themselves.

### Unchanged coupled reference and recovery

Separate uniform monolithic cases use E*=1 GPa and the same physical h0,
geometry, viscosity and speed. These are **prescribed-offset** comparisons:
load is measured, not forced to match across domains. The existing positive-
load benchmark in section 2C is preserved, not replaced by these cases.

| Domain label | Core dX | Grid | Load [N] | Minimum/central film [um] | Peak pressure [MPa] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.5 | 17x17 | 0.001447190554 | 0.109918528 | 0.149572264 |
| 4 | 0.25 | 33x33 | 0.001558660470 | 0.110772113 | 0.160992199 |
| 4 | 0.125 | 65x65 | 0.001589763765 | 0.111025027 | 0.164019187 |
| 8 | 0.25 | 65x65 | 0.002056143756 | 0.111774059 | 0.162732294 |
| 16 | 0.25 | 129x129 | 0.002328762465 | 0.112047199 | 0.162668923 |

The 65x65 cold Newton mesh case exhausted 30 corrections (last max mass
3.56e-8). Its full failed fields/history are retained. The existing independent
Picard reference, resolved to the study's tighter 1e-10 mass tolerance,
recovers a root in 121 total iterations spread across the unchanged
compliance stages. Warm Newton confirms it with no further corrections;
the separate SI max residual is 5.41e-11. No per-stage iteration cap is raised.
Regressions verify both the failure retention and bounded recovery.

At the largest coupled domain x=+/-0.715541753 mm,
y=+/-1.011928851 mm, dx=11.18034 um and dy=15.81139 um. Its film minimum
is at (0,0), peak pressure at (-0.022360680,0) mm, theta range
[0.005194471,1], max mass residual 8.39e-12, relative local defect 2.20e-14,
normalized global defect 5.25e-17 and net flow 3.24e-27 m^3/s.
Complementarity is 1.01e-24. Direct spatial checks at the center and pressure
peak agree with FFT displacement; the full fields and sample errors are saved.
This is sampled spatial verification on the large grid, supplemented by
complete small-grid spatial/embedding regressions, not a full direct audit
at every large-grid node.

Despite that convergence and positive film, label 8->16 still increases
coupled load by about 13.3%; minimum film changes by about 0.244% and peak
pressure by about -0.039%. A stable centerline/minimum is therefore not enough
to infer load/domain independence. The current mismatches arise from boundary
placement, discretization and unresolved far-field numerical accuracy within
the restricted model; they do not justify changing plots to resemble
piezoviscous EHL or adding constitutive laws to conceal the discrepancy.

### Reproduction and next development action

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_farfield_budget \
  --output-directory results/farfield_budget_new
```

PowerShell uses the same environment-variable setup as section 4 and
`python -m examples.run_farfield_budget --output-directory results/farfield_budget_new`.
The verified run is `results/farfield_budget_20261026_final/`: 27 candidate
states (25 accepted, 2 residual-rejected), complete histories/failed-candidate
arrays, source/runtime provenance, SI flux budgets, mesh/domain changes and
11 figures. Centerline overlays share the identical [-2L,2L] physical x
range; complete domains remain in NPZ data. Existing result directories
are not overwritten and generated files remain outside Git.

Next: improve and independently verify the far-field linear accuracy, retain
the declared acceptance criteria, then verify an appropriate coupled outer
boundary or graded elastic treatment with a physical applicability/error
budget. Pressure-dependent viscosity/density remain subsequent milestones;
domain independence has not been established and no such physics is added.

Final follow-up checks: `ruff check .` clean; complete `pytest -q` has
282 passed and 62 optional PyTorch/device skips (63.66 s). Thirteen new
regressions cover uniform/starved parity, a graded analytical limit,
midpoint SI flow units, nested node preservation, free-space FFT embedding,
direct cell integration, invalid inputs, iteration exhaustion and retained
failure/reference recovery. All 27 benchmark configurations finish; 25
states pass the declared audits and two remain explicitly rejected.
The maximum direct-versus-FFT sampled displacement discrepancy in accepted
coupled cases is 2.32e-23 m. Representative load and error-budget figures
were inspected; recorded Python source hashes match the tested sources.

## 6. Reservoir boundaries and physical-flux conditioning (2026-10-26)

Starting revision: `78f83143cdeaebf5d86029b19d8ac102d0615a89`.
**Outcome: the two reported load changes are quantitatively accounted for;
one original continuity rejection is recovered, but point-contact domain
independence remains open.** No new domain enlargement, iteration-cap change,
constitutive law, nonuniform coupled elasticity or GPU implementation is used.
The original 27 states, including both rejected candidates, are retained.

### What the existing boundary problem represents

Zero gauge inlet pressure places an ambient reservoir at a specified finite
distance. West theta=1 prescribes flooded content, not its integrated supply.
The inlet gradient and gap determine the unrestricted resulting volume flow.
Moving that reservoir changes the physical boundary problem. The east pressure
is also ambient; copied east content is output extrapolation for west-upwind
transport. Ambient side pressures permit lateral pressure-driven leakage;
there is no transverse advective flux. Boundary budgets use interior midpoint
control volumes, as in section 5. Constant density remains unspecified;
reported m^3/s become kg/s only after multiplication by a specified rho0.

A finite reservoir with explicit physical dimensions is a legitimate model,
but convergence on its mesh does not demonstrate an infinite-domain limit.
Likewise, prescribing a feed or reducing inlet content defines a separate
supply/starvation experiment. Neither is a truncation-error correction to the
same flooded problem. Existing starved-inlet tests are preserved.

For the independent one-dimensional parabolic reference, smooth rupture
selects hc and a constant flux q=Ue hc [m^2/s]. In the full region,
q_C=Ue h and q_P=Ue(hc-h), hence the cancellation condition is
`(|q_C|+|q_P|)/|q| = 2 h/hc - 1`. The analytical finite-inlet solution
approaches a finite flux as the inlet retreats: at -256L it is approximately
1.22575e-10 m^2/s for the section-5 h0/Rx/eta/Ue. Its cancellation condition
grows from 105 at -8L to 106933 at -256L; the direct float64 component sum
then differs from the analytical flux by 3.68e-12 relative. This is a
mathematically meaningful **infinite-width** reference, with no lateral
leakage. It neither supplies a point-contact boundary law nor establishes
physical validity of arbitrarily large parabolic gaps/slopes.

### Point-contact inlet flow and numerical sensitivity

`flux_diagnostics.py` separately evaluates arithmetic-face Couette and
Poiseuille terms in SI, on the saved original `(ny,nx)` fields. The original
independent `flux_audit` remains the acceptance test. Positive inlet values
below denote incoming flow; all other boundary budgets retain the outward
west/east/south/north convention. Full component maxima, outlet/side flows,
local/global defects and roundoff diagnostics are saved for every original
case, including the coupled and residual-rejected cases.

| Fine-core label | Couette inlet [m^3/s] | Poiseuille inlet [m^3/s] | Net inlet [m^3/s] | Inlet cancellation condition | h^3 max/min |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5.144316e-10 | -2.833173e-10 | 2.311143e-10 | 3.45175 | 9.7669e9 |
| 64 | 4.871920e-9 | -2.685294e-9 | 2.186627e-9 | 3.45611 | 8.7910e11 |
| 128 | 3.545495e-8 | -1.954554e-8 | 1.590941e-8 | 3.45710 | 4.6628e13 |
| 192 | 1.106519e-7 | -6.100228e-8 | 4.964962e-8 | 3.45731 | 4.5439e14 |
| 256 | 2.601301e-7 | -1.434120e-7 | 1.167181e-7 | 3.45741 | 2.5119e15 |

Cancellation is moderate in these 2-D fields: the maximum individual x-face
condition at label 256 is 9.8783. Its maximum Couette, x-Poiseuille and
y-Poiseuille magnitudes are 1.30055e-5, 5.06721e-6 and 2.99279e-6 m^2/s.
The large changing inlet flow and lateral leakage distinguish this problem
from the constant-flux one-dimensional limit. Catastrophic subtraction of
Couette and Poiseuille terms is therefore not supported as the main cause
of the observed 2-D load changes.

Large mobility contrast still makes pressure/linear accuracy consequential.
At label 256, reevaluating the **same float64 state** with wider arithmetic
changes the maximum continuity defect only from 3.98654e-8 to 3.98651e-8;
the maximum evaluation difference anywhere is 1.40052e-10. Merely reporting
a wider residual does not repair the root. At the largest-defect node
(x=-9.641475 mm,y=0.102774 mm), changing stored pressure by one ULP changes
the dimensionless defect by 1.03894e-8. This is a sensitivity experiment,
not a proof that no better float64 state exists. The face-balance epsilon
estimate is a heuristic, not a rigorous error bound or a sparse condition
number. Existing diagonal equilibration remains in use; the h^3 range is
reported explicitly rather than mislabeled as the matrix condition number.
Separate section-5 core/outer-grading mesh errors remain part of the budget.

### Bounded correction against the original physical flux

The prescribed-film rectilinear solver gains an opt-in
`flux_refinement_steps=0..3`, default zero. After its active set settles,
it evaluates the physical SI flux divergence in explicitly wider arithmetic,
solves a correction using the same equilibrated float64 constrained matrix,
and stores float64 pressure/content. Each trial must preserve the existing
admissibility/complementarity tolerances and strictly reduce the original
**float64 independent SI** maximum continuity defect. Rejected trials and
the raw starting fields remain recorded; no clipping or regularization is
introduced. The active-set limit stays 100. This is distinct from the earlier
matrix-residual refinement experiments in section 5.

| Same original grid | Original max continuity | After correction | Existing 1e-8 criterion |
| --- | ---: | ---: | --- |
| all_128 | 4.22176e-9 | 1.54903e-9 | passes |
| all_192 | 1.53078e-8 | 4.75990e-9 | now passes |
| all_256 | 3.98654e-8 | 1.70676e-8 | still rejected |

The last label-256 trial does not improve the audit and is rejected. Relative
local/global defects and complementarity retain the existing criteria. For
the recovered label 192 they are 4.22e-12, 1.33e-16 and zero respectively.
Relative load changes caused by the corrections are below 8.3e-15, far too
small to explain the percent-level domain sensitivity. The original rejected
label-192 state is not retroactively relabeled: the corrected variant is a
separate saved state. The recovered 128->192 load change is about 0.391%,
but the preceding change remains 1.174% and label 256 still fails. The
combined multi-increment mesh/domain target is not met.

Here `np.longdouble` has 63 fraction bits versus float64's 52; it is not
assumed to be IEEE binary128. Platforms without wider precision explicitly
reject a forced correction request; the runner records its unavailability
and the precision-specific regression skips. Default float64 operation and
all other independent tests remain available on those platforms.

### Exact accounting of the 1.17% and 13.3% load changes

On nested grids with unchanged shared cells, SI trapezoidal integration gives
the exact discrete identity
`Q_new-Q_old = (Q_new-Q_new_on_old_rectangle) +
(Q_new_on_old_rectangle-Q_old)`.
The first term is added-area load; the second is pressure readjustment over
the old rectangle. There is no field interpolation or continuum-tail estimate.

| Comparison | Old/new Q [N] | Change [N] | Added area | Shared-area readjustment |
| --- | --- | ---: | ---: | ---: |
| Rigid fine 64->128 | 0.002653662258 / 0.002684826537 | 3.116428e-5 | 37.816% | 62.184% |
| Coupled uniform 8->16 | 0.002056143756 / 0.002328762465 | 2.726187e-4 | 39.459% | 60.541% |

These are respectively 1.17439% and 13.25874% relative to the old force.
Only 0.0575% and 3.3277% of the respective **force changes** arise from
readjustment within the common inner |x|<=2L, |y|<=2Ly core. Most change
is peripheral pressure/load, explaining why core extrema alone look stable.
The identity defects are zero at the reported floating-point precision.
This is a spatial accounting, not an additive causal decomposition of
nonlinear boundary effects.

An additional rigid solve on each **identical** coupled 8/16 uniform grid,
with the same h0, fluid, geometry and boundary assumptions but no displacement,
gives Q=0.002138508624/0.002414472343 N: growth is already 12.90449%.
Both rigid controls pass independent SI audits. Elasticity reduces absolute
load by about 3.85%/3.55%, and alters the growth to 13.26%; it is not the
principal origin of this sensitivity. The previously validated free-space
FFT, direct source-cell/embedding checks and separate source-truncation
diagnostic remain unchanged. None of these controls discards an elastic
tail in a purported coupled subdomain solution.

### Matched boundaries: verified scope and remaining limitation

The separate rectilinear reference now accepts optional nonnegative SI
`boundary_pressure_pa` edge data. Positive-pressure edges have theta=1;
a positive-pressure west inlet cannot be combined with starved content.
The original ambient-edge defaults are unchanged. An exact constant-film,
linear-pressure nonuniform test checks all boundary RHS contributions and
physical face fluxes independently before the following comparisons.

1. **Independent analytical infinite-width limit.** Fix the original physical
   reservoir at -8L, hmin=20 um, Rx=0.01 m, eta=0.04 Pa s, Ue=0.425 m/s
   and L=0.632455532 mm. Compute windows starting at -0.5L, -L and -2L,
   ending at +3L, with nine y rows spanning +/-4L. Their inlet and transverse
   pressure traces come from the original analytical solution. Independent
   quadrature restores the omitted upstream force per width. On 129 x nodes,
   reconstructed force errors are 0.00646%, 0.01840% and 0.09046%.
   At inlet -L, 65/129/257-node pressure errors decrease from 0.1466% to
   0.02199% to 0.00833%; force errors decrease from 0.1116% to 0.01840%
   to 0.00648%. Every case passes the original physical audits. Force units
   here are **N/m**, not point-contact N. Replacing the inlet by ambient
   pressure and using its own consistent analytical transverse traces
   instead changes the reservoir; differences from the original total force
   are 96.93%, 78.59% and 38.65%, not discretization errors of the old problem.
2. **Finite rigid point-contact numerical reference.** Take the audited
   129x129 uniform rigid +/-16L,+/-16Ly reference and solve its exact
   65x65 inner +/-8L,+/-8Ly window with its known pressure trace. Relative
   pressure discrepancy is 4.97e-16. Window load is 0.002306848668 N;
   adding the known outside 0.000107623675 N recovers the full
   0.002414472343 N. The independent window max continuity defect is
   3.27e-12. Ambient pressure on that same inner boundary instead yields
   0.002138508624 N. A regression additionally obtains the larger reference
   from the original uniform cavitation solver, independently of the new
   boundary assembly.

These establish that matching a computational cut to the **same** outer
problem reduces boundary error, and that moving an ambient reservoir is
different. They do not provide a standalone analytical point-contact outer
solution: the finite point test borrows its trace and outside load from a
larger finite reference. A coupled version would also need exterior elastic
displacement/source accounting. The one-dimensional formula cannot simply
be copied into a laterally leaking point model. Graded meshes control cost
and retain separate mesh errors; they supply neither missing exterior
pressure nor a physical infinite-domain law. No translation-invariant FFT
is applied to unequal cells, and no coupled boundary model is replaced.

Consequently the remaining uncertainty separates into: finite-reservoir
placement and peripheral load/readjustment; independently measured core and
outer-grading discretization; large-mobility algebraic/state accuracy;
exterior elastic sources (separately checked, not the same as fluid boundary
error); and the small-slope/parabolic model's applicability. The already
reported far-corner slopes of 0.735/1.428 at labels 128/256 prevent treating
unlimited formal expansion as verified lubrication physics. These results
do not prove that a physically justified point far-field limit is impossible;
they show why the current computations do not establish the requested target.

### Reproduction and retained evidence

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_flux_conditioning \
  --input-directory results/farfield_budget_20261026_final \
  --output-directory results/flux_conditioning_new
```

On a fresh checkout, first run the section-5 budget into a new directory
and pass that directory with `--input-directory`. PowerShell uses the same
environment setup as section 4. The verified output is
`results/flux_conditioning_20261026_complete/`: 27 preserved original states,
two new same-grid rigid controls, three corrected variants, eight analytical
window cases and one finite point matching case, totalling 41 SI NPZ files
and 17 figures. Source-budget JSON, input archive hashes, runtime versions
and current Python source hashes identify all reused/new data. Intermediate
development runs are also preserved. Generated results remain outside Git.
The figure manifest includes component/net flow, cancellation, original
continuity/ULP sensitivity, correction traces, force accounting, boundary and
mesh comparisons, common-window pressure/film, saturation and extrema.

Next action: derive and independently verify a point-specific outer response
with a stated physical reservoir/geometry, accounting for exterior load and
elastic sources, before integrating it into coupled solves. Retain separate
mesh and original SI-flux acceptance checks; label 256 remains a numerical
blocker. Do not advance to pressure-dependent constitutive laws or GPU JFO
on the strength of these diagnostic successes alone.

Verification: `ruff check .` clean; full `pytest -q` **294 passed,
62 skipped** (67.36 s). Twelve added regressions cover pressure-edge SI
fluxes, known and exact cancellation, nested-cell force identity, unchanged
root/physical-defect improvement, invalid inputs, analytical matched-boundary
mesh refinement, unavailable wider precision and independent finite point
matching. The skips are optional PyTorch/device coverage on this host; no
new GPU execution claim is made. All benchmark stages finish; original and
corrected residual-rejected states remain explicit. The 17-figure manifest,
41 NPZ files and current source hashes were checked. Representative boundary,
force-accounting and physical-defect figures were visually inspected.

## 7. Finite point-reservoir exterior response (2026-10-26)

Starting revision: `e666ff96dcbccd9399aa7a6c384017e8b035ddd0`.
**Verified scope: a rigid, prescribed-film discrete exterior response for
an explicitly dimensioned finite reservoir.** It computes exterior pressure
and content without supplying the larger reference's pressure trace. It
removes numerical-cut dependence for that same finite physical problem;
it does not remove physical reservoir dependence or establish an infinite
point-contact limit. The coupled JFO/FFT and signed GPU solvers are unchanged.

### Formulation and physical data

`exterior.solve_exterior_response` assembles the conservative midpoint
mass equations independently of the original uniform JFO implementation.
It uses the existing arithmetic mobility/film faces, west-upwind content,
ambient pressure on the physical outer edges and prescribed west content.
The prescribed-film, constant-property model and SI conventions of section 1
apply. The only new inner rectangle is a computational partition: no pressure
or incoming-content value is imposed there.

For each trial wet/dry active set, retain the **full** pressure/content
constrained system `A z=b`, partitioned into inner I and exterior E equations
and unknowns. Factor the equilibrated sparse exterior block and compute

```
z_E = A_EE^-1 b_E - A_EE^-1 A_EI z_I,
S = A_II - A_IE A_EE^-1 A_EI,
S z_I = b_I - A_IE A_EE^-1 b_E.
```

Only interface columns communicate with the exterior, so the dense response
is restricted to those columns. The response is affine for a fixed active
set, not a globally linear cavitation law. Start with wet nodes, reconstruct
both pressure and transported content over the whole exterior, update active
sets there and inside, and reject cycling/exhaustion/nonfinite states.
Exterior cavitation is solved, not guessed from a reference. Keeping the
content constraints and their upstream coupling prevents an artificial
flooded reservoir from appearing at the inner inlet. Flooded and 0.9-content
regressions independently agree with the original uniform JFO solver; the
latter is a separate supply problem, not a truncation-error adjustment.

States, factorization and response use float64. Positive row/column
scalings alter only linear coordinates. There is no clipping, regularization
or widened stopping tolerance. Limits are 20,000 total nodes, 100 active
sets, 256 response columns and a 64 MiB estimated dense-work cap checked
before response allocation. RHS columns are solved in batches of at most 16.
Sparse factors incur additional storage; the recorded budget is not a claim
of total process peak memory. Rebuilding an exterior factor per active set
also costs work: condensation is not claimed to accelerate the present solve.

### Independent rigid benchmark and supply/conservation

Fix h0=0.1 um, Rx=10 mm, Ry=20 mm, eta=0.04 Pa s, Ue=0.001 m/s,
L=44.72135955 um, Ly=63.24555320 um, p_ref=2.14662526 MPa and Lambda=1.
Use a **fixed** physical reservoir x=+/-8L, y=+/-8Ly, i.e.
+/-0.357770876 mm and +/-0.505964426 mm. The full 65x65 grid has
spacing 0.25L,0.25Ly. Every physical pressure edge is ambient; west theta=1.
The larger reference is independently solved by `solve_cavitating_reynolds`;
its fields are compared after solving, never passed into the exterior solver.
The computed load is 0.002138508623897257 N. This finite reservoir has
maximum rigid corner slope approximately 0.0438; physical applicability is
still an explicit model assumption, not certified by a chosen slope cutoff.

| Inner cut +/-label L,Ly | Retained pressure/content unknowns | Reference pressure relative L2 discrepancy | Original max continuity | New ambient reservoir at cut: Q [N] |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 162 | 5.13e-16 | 1.68e-12 | 0.000210919740 |
| 2 | 578 | 6.92e-16 | 1.54e-12 | 0.000861158114 |
| 3 | 1250 | 5.58e-16 | 1.78e-12 | 0.001332591641 |

All three response solves recover the full load to roundoff, with maximum
content discrepancy 1.67e-15. All take seven active sets, independently
finding over 1,400 dry exterior nodes. The largest dense-work estimate here
is 9.30 MB. Original SI relative local/global defects are below 1.88e-14
and 2.47e-15; complementarity is zero at printed precision. Pressure edges,
west supply and feasibility retain the section-5 tolerances. Comparison
criteria were set before solving to 1e-8 relative pressure/load error, in
addition to original max continuity 1e-8 and relative local/global and
complementarity/admissibility 1e-10. These are discrete reference-parity
criteria, not continuum mesh accuracy claims.

The outer reservoir volume flow is approximately 3.833948386e-12 m^3/s
incoming, with east/south/north outflows 2.353192609e-12,
7.403778885e-13 and 7.403778885e-13 m^3/s. For the label-2 cut the
incoming internal west flow is instead 1.113328581e-13 m^3/s; its outgoing
east/south/north flows are 7.784944859e-14, 1.674170477e-14 and
1.674170477e-14 m^3/s. This smaller inner supply is computed by the
exterior response. It is not prescribed equal to the physical inlet supply.
The independent interior-control-volume flux budget closes to 1.46e-28
m^3/s. Both outer and internal budgets are saved. Front brackets and raw
pressure/content fields are retained for each cut.

### Omitted load and exterior elastic sources

Reconstructed exterior pressure is used in the load, rather than adding a
borrowed scalar tail. `exterior_source_budget` partitions **source cells**
by their retained/exterior node masks. With zero physical pressure edges,
interior midpoint-cell areas equal nonzero trapezoidal nodal weights. Cells
centered at the cut straddle its node rectangle; this disjoint source
partition is explicitly different from integrating up to that rectangle.
For diagnostic E*=1 GPa, direct rectangular-cell integration evaluates the
same half-space 1/(pi E* r) operator at two targets. No FFT is used on unequal
cells, and this calculation does not change the rigid film.

| Cut | Exterior source load [N] | Exterior center displacement [nm] | Total center displacement [nm] |
| ---: | ---: | ---: | ---: |
| 1 | 0.001338361443 | 3.584026139 | 12.328255651 |
| 2 | 0.000689488603 | 1.213645841 | 12.328255651 |
| 3 | 0.000361483741 | 0.487461024 | 12.328255651 |

The retained plus exterior loads reproduce total force; displacements add
with maximum partition discrepancy below 5e-24 m. A separate regression
compares the complete uniform-grid displacement at both targets to the
validated zero-padded FFT. Thus a numerical cut neither discards exterior
mechanical pressure nor silently fixes its deformation to zero. This is
selected-target source verification, **not an integrated coupled exterior
elastic model**. Coupling would require film-dependent exterior response,
all required target displacements and their consistent nonlinear derivatives.

### Reproduction, limitations and next action

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_exterior_response \
  --output-directory results/exterior_response_new
```

`results/exterior_response_20261026/` contains four raw SI NPZ files,
source/runtime provenance, active-set histories, internal/outer flux audits,
load/deformation partitions and four figures. Existing directories are
refused; checkpoints retain incomplete/failure status. The default reference
requires no previous generated dataset. The experiment is a partition test
on the same physical mesh, deliberately separate from continuum refinement.

The response is point-specific through the full 2-D geometry, lateral
pressure flow and exterior cavitation. It provides a finite physical
formulation without a guessed pressure trace, but still computes an
exterior discretization. Next test: approximate that exterior on graded
meshes at **fixed physical reservoir bounds**, retaining the same inner
nodes, and separately quantify pressure, load, supply, front and diagnostic
elastic-source errors against a resolved larger uniform reference. Require
those errors and mesh changes to pass before considering coupled integration.

Milestone checks: `ruff check .` clean; complete `pytest -q` **298 passed,
62 optional device skips** (70.66 s). Four added regressions cover independent
flooded/starved reference parity, moving cuts and complete elastic-source
partition/FFT parity, and invalid-core/iteration-exhaustion behavior. All
three benchmark cuts pass the declared original SI and reference criteria.
Source hashes match; source-budget and cut-comparison figures were inspected.

## 8. Graded finite-reservoir exterior approximation (2026-10-26)

Starting revision: `f5ab34f89522518f6e3ee5887151f5c71b1001dd`.
**Outcome: a bounded graded prescribed-film exterior response meets the
specified 1% errors against an independently refined finite-reservoir
reference.** This improves a computational exterior approximation at fixed
physical boundaries. It establishes neither an infinite-domain limit nor
coupled nonlinear EHL validation.

### Independent reference resolution versus outer grading

Keep exactly the physical section-7 reservoir and operating parameters:
x=+/-8L, y=+/-8Ly, h0=0.1 um, Rx=10 mm, Ry=20 mm, eta=0.04 Pa s,
Ue=0.001 m/s, ambient physical pressure edges and west content one.
The original uniform JFO solver supplies references at four resolutions.
Each has independent SI mass/constraint audits. Uniform FFT supplies
reference diagnostic displacements at the center and (-0.5L,0), E*=1 GPa.
It does not alter the prescribed rigid film.

| Uniform dX | Grid | Q [N] | Peak p [MPa] | Center displacement [nm] |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 33x33 | 0.002004831810 | 0.171983644 | 11.335207793 |
| 0.25 | 65x65 | 0.002138508624 | 0.188248017 | 12.328255651 |
| 0.125 | 129x129 | 0.002179107398 | 0.193009914 | 12.687998177 |
| 0.0625 | 257x257 | 0.002189795408 | 0.194208742 | 12.776453129 |

Successive load changes are 6.668%, 1.898% and 0.4905% relative to the
preceding mesh. The final pressure and center-displacement changes are
0.6211% and 0.6972%. The finest original SI max mass defect is 3.10e-11.
The uniform reference has 66,049 nodes, within the previous 70,000-node
benchmark budget. Its last centerline rupture bracket is
[13.975425,16.770510] um. These observed changes quantify reference
uncertainty; they are not a proven continuum error bound. Another uniform
halving would require 263,169 nodes, outside this study's resource budget.
No such calculation or stronger continuum-accuracy claim is made.

The new study retains a uniform normalized [-2,2] core with dX=0.25,
0.125 or 0.0625. Outer widths grow geometrically by 1.2, 1.1 or 1.05;
the last cell ends at the **same physical** +/-8 boundaries, rather than
snapping the reservoir outward. These are distinct fixed-domain mesh
approximations. Condensation retains the inner +/-L,+/-Ly nodes; exterior
film, pressure and content are solved afresh, with no reference trace.
Every comparison uses the same model and supply assumption.

Before solving, the runner declares original max mass 1e-8, relative
local/global defects and complementarity/admissibility 1e-10. Separate
comparison tolerances are 1% for full load, shared-core pressure L2, peak
pressure, total center/upstream displacement and a diagnostic fixed-section
volume flow. Front kinds/counts must match on shared physical core y rows;
node-bracket separation must be within two local core cells, using the
unchanged theta<1-1e-8 diagnostic. Core pressure and fronts use actual
shared coordinates, without field interpolation. Rigid minimum film remains
prescribed at 0.1 um at the origin; its invariance is not physical film-height
validation.

| Core dX | Outer growth | Grid | Load discrepancy [%] | Shared-core pressure discrepancy [%] | Center displacement discrepancy [%] | Field criteria |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.25 | 1.2 | 37x37 | 3.8593 | 3.7424 | 4.2918 | fail |
| 0.25 | 1.1 | 43x43 | 3.0017 | 3.4318 | 3.8514 | fail |
| 0.25 | 1.05 | 51x51 | 2.6337 | 3.3024 | 3.6626 | fail |
| 0.125 | 1.2 | 59x59 | 1.7578 | 1.1048 | 1.3103 | fail |
| 0.125 | 1.1 | 71x71 | 0.9487 | 0.8233 | 0.9264 | pass |
| 0.125 | 1.05 | 85x85 | 0.6759 | 0.7256 | 0.7894 | pass |
| 0.0625 | 1.2 | 99x99 | 1.0119 | 0.4074 | 0.4940 | fail |
| 0.0625 | 1.1 | 115x115 | 0.3546 | 0.1465 | 0.1746 | pass |
| 0.0625 | 1.05 | 139x139 | 0.1287 | 0.0559 | 0.0648 | pass |

All nine cases pass the physical mass/constraint audits. Their largest max
mass, relative local and relative global defects are 1.27e-11, 1.08e-13
and 2.90e-14; complementarity is zero at printed precision. Low residuals
therefore do **not** rescue the five field-error failures. All front
comparisons pass; the finest matches transition count/kind on 65 shared
y rows, and its brackets overlap the reference brackets on every such row.
These are bracket comparisons, not a subcell front error estimate.

At fixed finest core, reducing outer growth from 1.2 to 1.1 and 1.05
changes Q from 0.002167636852 to 0.002182030760 to 0.002186978056 N.
At fixed growth 1.05, refining the core separately changes Q from
0.002132121840 to 0.002174994814 to 0.002186978056 N. The final respective
outer/core changes are approximately 0.227%/0.551%. Independent controls
show why grading and core discretization must both be budgeted. They do not
justify adding observed errors as a rigorous total bound or asserting an
asymptotic convergence order.

### Supply, exterior force and deformation

The native conservative outer boundary budgets lie on midpoint faces.
Their physical positions change when the last mesh cell changes even though
the reservoir nodes stay fixed. Those flows and exact face positions are
saved, but differences are not mislabeled as supply at an identical section.
For a common physical comparison the runner separately evaluates flux at
x=-L across y=[-Ly,Ly], using central pressure differences, nodal h/theta
and SI trapezoidal integration. This **diagnostic** section flux is not a
replacement for conservative face acceptance. Its stencil is uniform on all
studied cores and its units/sign are independently tested.

The finest uniform diagnostic net flow is 2.336034794e-14 m^3/s; the finest
graded result differs by 0.00611%. Its Couette/Poiseuille components are
saved separately. Inlet content is never tuned to obtain this agreement.
Physical outer and internal-cut conservative budgets also remain available.

At the finest matched core spacing, retained/exterior source-cell supports
at the cut are identical between uniform and graded models, allowing a
separate exterior comparison. The uniform exterior source force is
0.001450091928 N; the finest graded value is 0.001447523972 N, a 0.1771%
discrepancy. Exterior center displacement is 4.120756304 nm versus
4.115025068 nm, a 0.1391% discrepancy. These also pass the declared 1%
comparison. For coarser core spacings the cut cells have different widths;
the runner records exterior partitions but deliberately does not claim
like-for-like component errors on those different supports. Total load
and total target displacement comparisons remain meaningful throughout.

The finest graded peak is 0.194161953 MPa (0.0241% discrepancy); its total
center displacement is 12.768173992 nm and upstream displacement differs
by 0.0602%. Complete half-space response at the selected targets includes
both retained and exterior pressure sources. Reference totals use the
uniform FFT; graded totals use direct physical cell integration. No
translation-invariant FFT is applied to the nonuniform pressure field.
No deformation is fed back into the rigid prescribed film.

### Resource evidence, reproduction and next action

The 139x139 approximation uses 19,321 total nodes versus 66,049 reference
nodes, within the exterior solver's unchanged 20,000-node limit. The last
response has 161 columns, 2,178 retained pressure/content unknowns and
35,360 exterior unknowns. Its estimated dense work is 60,135,424 bytes,
within 64 MiB; sparse factor storage is additional. Eight active sets are
needed, below the unchanged cap of 100. Timings are recorded for provenance
and bounded-cost review, not as a general acceleration claim.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_exterior_mesh \
  --output-directory results/exterior_mesh_new
```

The authoritative run is `results/exterior_mesh_20261026_verified/`:
four independent uniform states, nine graded candidates, 13 SI NPZ files,
five figures, source/runtime hashes, histories, source partitions, front
checks and explicit physical-audit versus field-error acceptance flags.
Earlier development output, including a checkpoint interrupted by a fixed
NumPy-boolean JSON serialization error, is preserved. The final runner
completes; regression coverage verifies portable JSON front diagnostics.

This is an independently verified exterior discretization for a **finite**
point reservoir, providing pressure/supply/load and diagnostic exterior
elastic-source accounting. The finer reference still has measurable mesh
uncertainty. Physical reservoir dependence and the section-6 large-domain
label-256 limitation are not erased by changing the problem to this explicit
finite reservoir. No coupled model has been replaced.

Next justified action: verify all inner-target elastic displacements and
film-dependent exterior response/derivatives against a full uniform coupled
finite-reservoir reference, with controlled source/mesh errors. Only then
consider integrating the response into the nonlinear JFO solver. Further
reference refinement needs a bounded alternative to the next 263,169-node
uniform mesh, rather than silently raising resource limits. Piezoviscous
physics and GPU JFO remain deferred.

Milestone checks: `ruff check .` clean; full `pytest -q` **300 passed,
62 optional device skips** (70.68 s). Two added regressions independently
check fixed-reservoir outer refinement against the original uniform solver,
load/pressure/displacement convergence, common-section SI flow units,
front-mismatch detection and portable diagnostic JSON. All four uniform
and nine graded states finish; four graded candidates meet every stated
reference comparison. Source hashes, manifests and representative error
and saturation figures were inspected.

## 9. Film-response derivative and all-inner source prerequisites (2026-10-26)

Starting revision: `97e38db03f7c9fa3e5306f66c91f1961fbab755d`.
**Verified preparatory scope: prescribed-film exterior derivatives on a fixed
wet/dry set, and elastic-source accounting at every retained target.** The
unchanged uniform coupled solver supplies an independent reference. No new
nonlinear condensed-exterior coupling, graded coupled root, or physical
infinite-domain validation is claimed.

### Conditional film derivative

The exterior result now retains the actual pressure/content active set.
`differentiate_exterior_response` accepts SI film h [m] and direction v [m],
with `h(epsilon)=h+epsilon v` and dimensionless epsilon. Scales, reservoir,
west supply and active set remain fixed. With H=h/h_ref, V=v/h_ref, the
explicit film derivative of the original conservative face flux is

```
dq_x = -avg(3 H^2 V) diff(P)/dX + Lambda avg(V) theta_west,
dq_y = -avg(3 H^2 V) diff(P)/dY.
```

Divide the flux differences by the original midpoint control widths and
solve `A dz = -(div(dq), 0_constraints)` with the same full constrained
exterior condensation. Reconstruct both pressure [Pa per unit epsilon]
and content directions everywhere, including the exterior. The physical
inlet content direction is zero. Return the true rowwise linear backward
error separately from the physical root audit. Reject missing active-set
records and films that do not independently reproduce a feasible,
mass-converged base root. Tangent pressures may be signed; they are linear
directions, not physical pressure fields to clip.

On a 33x33 uniform +/-8L,+/-8Ly rigid problem with dX=0.5 and section-7
parameters, use smooth SI directions
`v=H0 exp(-((X-X0)^2+(y/Ly)^2)/2)` for X0=-0.5 and -4. The first acts
primarily near the contact; the second probes upstream exterior influence.
For epsilon=0.01,0.003,0.001,0.0003, independently solve both perturbed
nonlinear JFO problems and compare central differences. Their wet/dry sets
are explicitly checked against the recorded base set; no change occurs in
these benchmark pairs. All perturbed roots pass original SI mass/constraint
criteria, rather than being accepted only for a small derivative difference.

| Direction center X0 | Pressure error at eps=0.01 | Pressure error at eps=0.0003 | Content error at eps=0.0003 | Force-direction error at eps=0.0003 |
| ---: | ---: | ---: | ---: | ---: |
| -0.5 | 9.2067e-5 | 8.2857e-8 | 3.7196e-9 | 7.5355e-9 |
| -4 | 6.9622e-8 | 8.1586e-11 | 3.0977e-9 | 4.4980e-11 |

Errors are relative norms (force is a scalar relative error). Intermediate
values and every raw plus/minus pressure/content pair are saved. Force
directions are -0.0004570968134 and -0.00003144694870 N per unit epsilon,
obtained from the analytic pressure direction by SI quadrature and checked
against the independently perturbed loads. The exterior content error
reaches a finite-difference roundoff plateau at the smallest amplitude;
monotone convergence is not asserted for every quantity. A 1e-5 relative
last-pair comparison criterion is declared before running.

Four regression configurations repeat both directions on uniform and
nonuniform graded grids. Pressure/content finite differences retain the
same active sets and agree within 1e-6. Graded displacement-direction checks
use direct cells on both perturbed pressure fields; they do not apply a
uniform FFT to unequal cells. These checks do not establish a derivative
through a rupture/reformation switch. Generalized directional derivatives,
feasibility and globalization still need independent coupled verification.

### Every inner target, including the exterior sources

`retained_source_displacements` evaluates the existing exact rectangular-cell
half-space integral at every retained node, returning separate retained,
exterior and total contributions. It batches at most 32 targets, caps total
targets at 2,048 and source-target pairs at 20 million before evaluation,
and stores no dense influence matrix. The original source/node and response
budgets remain in effect. This is a bounded direct operator evaluation,
not a fast nonuniform coupled elasticity algorithm.

For the two uniform tangent benchmarks all 81 retained targets agree with
the independent full-domain FFT pressure-direction response to maximum
8.28e-24 m and 2.59e-25 m. Signed pressure directions retain the linear
half-space convention. Separately, rerun the unchanged uniform monolithic
coupled benchmark on the fixed 65x65 +/-8L,+/-8Ly reservoir, h0=0.1 um
and E*=1 GPa, with the original 30-correction cap. It converges in five
Newton corrections; Q=0.002056143756 N. Its **converged film is then frozen**
and supplied to the exterior prescribed-film solve, without supplying its
pressure trace or active set.

The exterior solve independently recovers this frozen-film pressure to
4.36e-15 relative L2, content to 1.34e-15 maximum discrepancy, and force
to the recorded quadrature precision. Its original SI max mass defect is
1.67e-12, relative local/global defects 1.12e-14/2.83e-15 and
complementarity zero at printed precision. At all 289 inner targets the
direct retained-plus-exterior source displacement differs from the reference
coupled FFT field by at most 7.28e-23 m; the largest source-partition
addition discrepancy is 3.31e-24 m. Exterior pressure/deformation is thus
retained throughout this frozen-root restriction check.

This is agreement at a known coupled solution's film, **not evidence that
a new exterior nonlinear iteration can reach that solution**. The uniform
coupled reference still solves its full physical mesh, and its established
SI and failure/history checks are preserved. No nonuniform elastic field
is silently substituted into the coupled solver.

### Reproduction and remaining work

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_exterior_coupling_audit \
  --output-directory results/exterior_coupling_audit_new
```

`results/exterior_coupling_audit_20261026_verified/` contains 13 raw SI NPZ
files, two figures, exact perturbation amplitudes, conditional active-set
checks, analytic/finite-difference load directions, all original physical
audits, the independently rerun coupled reference and its complete history,
all-inner target coordinates/source contributions, and source/runtime hashes.
Previous runs are retained. The final runner completes and every declared
preparatory comparison passes.

Next: verify a bounded nonlinear finite-reservoir iteration that updates
exterior film as pressure changes, initially on uniform grids against the
unchanged full coupled reference. Keep load, original SI flux, complementarity,
film feasibility and all-source coupling errors independent. Handle active-set
changes with verified globalization rather than reusing this conditional
derivative across a front switch. Graded coupled integration additionally
requires a verified full source/target film map and its consistent derivatives
at all required nodes, including exterior targets. Current inner-target
checks alone do not supply that map. No piezoviscous or GPU JFO work is added.

Milestone verification: `ruff check .` clean; full `pytest -q` **304 passed,
62 optional device skips** (79.99 s). Four new uniform/graded direction
regressions verify pressure/content finite differences, nonzero exterior
response, signed elastic-direction/FFT parity where applicable, direct
source finite differences on unequal cells, partition closure and rejection
of absent active sets or mismatched base films. All preparatory benchmark
criteria, 13-state/two-figure manifest and final source hashes pass. Both
final diagnostic figures were visually inspected.

## 10. Separate uniform exterior nonlinear coupling (2026-10-26)

Starting revision: `39b87033f99dbf485d61559ad6aaaa2ccd9f2639`.
**Verified scope: a separate uniform-grid, prescribed-offset finite-reservoir
exterior/Picard experiment**, compared against the unchanged monolithic
coupled solver. The new experiment starts from rigid film and zero pressure;
no reference field, pressure trace, active set or known coupled film seeds it.
It is a restricted isoviscous numerical benchmark, not a classical EHL or
infinite-domain validation.

### Nonlinear pressure, supply and full exterior film

`exterior_coupled.solve_uniform_exterior_coupling` takes the existing
uniform SI problem, explicit eta [Pa s], Ue [m/s] and a computational cut.
It verifies that the SI fluid data reproduce the problem's Lambda and
rejects incompatible flow data or a nonuniform physical source grid.
For compliance factors 0,0.25,0.5,0.75,1, each outer iteration:

1. Solves the current prescribed-film problem by exterior condensation,
   updating wet/dry sets and reconstructing pressure/content **everywhere**.
2. Applies the original full-source zero-padded uniform FFT to all reconstructed
   pressure, including exterior sources, and updates film at every physical
   node, including exterior nodes and boundary-node displacement.
3. Computes original independent SI flux and complementarity on the **exact
   pressure-coupled film**, not only on the previous inner-solve film.
4. Accepts a stage only when relative pressure/film iteration defects are
   <=1e-8, original max continuity <=1e-8, relative local/global balance
   <=1e-10, existing pressure/content/complementarity/supply tolerances
   <=1e-10, physical pressure edges are exactly zero and film is feasible.
   Otherwise relaxes film by the fixed factor 0.5 and continues.

The stage cap remains 100; inner active sets remain capped at 100 and all
20,000-node/response-memory limits remain unchanged. This is damped Picard
with compliance continuation, not an Armijo-globalized Newton method. It
does not reuse the conditional derivative across a front switch. Exhaustion
or an inner failure raises with complete outer/exterior histories and the
last candidate; a candidate is not silently labeled converged. A deliberately
one-iteration stage probe demonstrates this failure retention. No pressure,
content or film clipping is introduced.

The default old CPU coupled and signed GPU paths are unchanged. The new
experiment still factors an exterior discretization every step and computes
full-domain uniform FFT film; it is not advertised as faster or as a reduced
physical-domain model. Load is measured at prescribed h0, not controlled
against a target Q. Agreement of force here is an independent reference-error
check, not a new fixed-load closure result.

### Same finite reservoir and independent reference

Keep x=+/-8L,y=+/-8Ly with section-7 SI inputs, h0=0.1 um, E*=1 GPa,
ambient outer pressure and flooded west content. Independently rerun the
old full monolithic solver on every physical mesh before comparing results;
its cold failures/reference recovery, if any, remain in the saved reference
history and fields. The exterior iteration is independently cold-started.
The 33x33 grid is tested with cuts +/-L,+/-Ly and +/-2L,+/-2Ly; the 65x65
case uses the latter cut. Relative pressure, film and load comparison
criteria are declared at 1e-8, separately from all physical residual tests.
All-inner direct source integration independently checks the uniform FFT.

The two 33x33 cut choices give the identical reported load
0.001938705636271269 N, minimum film 0.110918916722 um and peak pressure
0.151611589 MPa. Both take 108 outer iterations across five stages, never
more than 28 in a stage. Their relative monolithic pressure/load errors are
3.27e-10/1.80e-10, and relative film error 1.09e-13. SI max continuity is
6.42e-10, relative local/global defects below 8.86e-11/2.36e-12 and
complementarity zero at printed precision. The arbitrary numerical cut
therefore no longer changes the coupled answer for this same finite mesh
and physical reservoir.

At 65x65, the experiment takes 107 total outer iterations, at most 27 per
stage. Q=0.002056143756122417 N, minimum film=0.111774059248 um and
peak p=0.162732294 MPa. Relative pressure/load/film discrepancies are
1.53e-10/7.46e-11/4.99e-14. Original SI max continuity is 6.96e-10,
relative local/global defects 6.51e-11/1.04e-12 and complementarity zero
at printed precision. All 289 retained target displacements agree with
complete direct source integration to 1.99e-23 m. Agreement with a discrete
root does not remove mesh error: 33->65 still increases coupled load by
6.0576%. Further refinement is audited separately below.

### Bounded 129x129 refinement and rejected-reference recovery

The optional fine case has 16,641 physical nodes and an inner +/-L,+/-Ly
cut. It converges in 103 outer iterations, at most 26 in a compliance stage.
Q=0.002088608691325601 N, minimum film=0.112026973190 um and peak
p=0.165680049 MPa. Original max continuity is 1.68e-9, relative local/global
flow defects 7.94e-11/2.05e-12 and complementarity zero at printed precision.
All 289 retained target displacements agree with complete direct source
integration to 1.99e-23 m. This case is expensive: a uniform full exterior
response is rebuilt for every active-set/film iteration. Single-thread CPU
and algebraic allocation/iteration caps control the experiment; there is no
production performance claim.

Its initial independent monolithic reference converges in 27 corrections
under the original defaults, but the separate SI audit finds max mass
7.97e-9 and relative local defect 3.50e-10. The latter violates this study's
1e-10 criterion. Accordingly the producer's fine reference and comparison
are explicitly rejected and its root summary says `failed acceptance`.
Small pressure/load differences do not override the failed reference audit.
The candidate fields and original rejected reference/history are all saved.

`examples.reaudit_exterior_coupling` copies and hashes those raw states into
another fresh output directory. It warms the **unchanged monolithic solver
only from its own original reference**, tightens mass tolerance to 1e-10,
and takes one further Newton correction. It never seeds a reference from an
exterior root. The tightened reference passes: max mass=4.28e-12, relative
local/global defects=1.98e-14/5.14e-17. No cap or acceptance criterion is
relaxed. This process repairs the independently computed reference precision,
not the already saved exterior candidate.

Recomputing the comparisons gives fine exterior pressure/load/film errors
3.22e-10/1.53e-10/1.07e-13. All four candidates pass against the audited
independent references. The entire original rejection remains available in
the copied producer summary and raw NPZ; no failed state is retroactively
renamed. Exact reconstructed film agrees with the archived candidate film
at every physical node at reported floating-point precision. These checks
separate independent-reference quality, physical residuals and field parity.

The coupled same-reservoir load changes shrink from 6.0576% (33->65) to
1.5789% (65->129), relative to the preceding result. The 1% coupled mesh
target therefore remains open despite excellent discrete reference parity
and cut independence. The next uniform exterior halving would have 66,049
nodes and exceed the current 20,000-node response cap. A naive complete
elastic source/target map on the finest 19,321-node graded rigid mesh would
involve 373,301,041 pairs, exceeding the bounded direct evaluator's
20-million-pair/2,048-target budgets. A verified mapped or multilevel
source treatment is required before graded nonlinear coupling; the uniform
FFT must not be applied directly to those unequal cells. No limits are
silently increased to make a success claim.

### Reproduction and retained evidence

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_uniform_exterior_coupling --include-fine \
  --output-directory results/uniform_exterior_coupling_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.reaudit_exterior_coupling \
  --input-directory results/uniform_exterior_coupling_new \
  --output-directory results/uniform_exterior_coupling_audited_new
```

The full producer is `results/uniform_exterior_coupling_20261026_refined/`:
three independent reference states, four exterior-coupled roots and an
intentional budget-one failure, totalling eight SI NPZ files and two figures.
The producer deliberately retains the fine reference rejection described
above. `results/uniform_exterior_coupling_20261026_checked/` is the
successful independent-reference reaudit: nine SI states, one additional
comparison figure, all original producer histories/data/input hashes and
fresh source/runtime provenance. Earlier coarse and diagnostic runs are
preserved. The failure probe's raw last candidate and complete outer/exterior
histories demonstrate bounded failure retention without raising any cap.
Exact zero film defects are retained and shown with a symmetric-log axis.

Next action: implement and independently verify a bounded complete graded
elastic source/target map, including exterior-target film and consistent
nonlinear derivatives. Maintain separate coupled mesh and exterior-model
error budgets; verify front changes/globalization before claiming broad
robustness. Finite physical-reservoir dependence, the old large-domain
label-256 precision limitation and classical EHL physical validation remain
open. No pressure-dependent properties or GPU JFO were added.

Final milestone checks: `ruff check .` clean; full `pytest -q` **306 passed,
62 optional device skips** (75.11 s). Two new regressions verify independent
monolithic/exterior nonlinear root parity, original physical acceptance,
film feasibility, iteration caps, preserved last-candidate failure/history
and rejection of inconsistent SI flow data. All four final candidates and
three independently audited references pass; the original fine reference
rejection remains explicit. Producer/reaudit source hashes, copied input
hashes, raw field counts and figure manifests were checked. Fine pressure/
film overlays, iteration/zero-defect histories and audited comparison
figures were visually inspected. The unresolved coupled mesh target and
full graded source-map cost remain documented rather than hidden.

## 11. Constitutive milestone, 2026-10-27

The user authorized moving pressure-dependent laws onto the next development
path while retaining the finite-reservoir and 1.58% coupled-mesh limitations.
`constitutive.py` implements independently tested SI density/viscosity and
analytic pressure derivatives. Source details, the 1975 printed viscosity
unit inconsistency and the explicit SI resolution from NASA RP-1255 are in
`docs/CONSTITUTIVE.md`. This is verification of empirical law evaluation,
not validation of a measured fluid or a classical EHL solution.

Independent 60-digit Decimal references and finite-difference derivative
checks cover 0..1 GPa, along with the original GPa density conversion and
constant-property limits. `results/constitutive_20261027_verified/` holds a figure,
raw SI NPZ values/derivatives and source/runtime JSON. The curve parameters
are illustrative at assigned fixed T=313.15 K. Existing result baselines,
flow solvers and GPU implementations are preserved. Integration into JFO
and initial elastic solves remain subsequent milestones.

Milestone-1 checks: Ruff clean; complete pytest 308 passed, 62 optional
device skips (85.23 s). Source hashes match the verified property curves;
the generated figure was inspected. At 1 GPa the illustrative laws give
rho=1038.888889 kg/m^3 and eta=242724.7165 Pa s; these large model values
are retained, not capped or presented as experimental measurements.

## 12. Pressure-dependent prescribed-film JFO, 2026-10-27

Milestone 2 adds a separate conservative variable-property CPU path; the
original constant-property solver remains an independent reference. Equations,
SI mass-flux units, face interpolation, coefficient damping, true-property
acceptance and bounded failure retention are specified in
`docs/CONSTITUTIVE.md`. No film feedback, GPU port or thermal law is added.

An independent continuum pressure-driven channel uses adaptive integration
of rho/eta and scalar potential inversion. With constant h=20 um, Ue=0,
x=[0,1] mm, p_w=5 MPa and p_e=1 MPa, exact transverse pressure data give
an independent 2-D Dirichlet reference. Pressure relative L2 errors on
17/33/65/129 streamwise nodes are 4.11e-8,1.05e-8,2.62e-9,6.41e-10;
mean centerline mass-flux errors are 2.62e-6,6.56e-7,1.64e-7,4.10e-8.
The final errors pass predeclared 1e-6 pressure and 1e-5 flux criteria.
Separate tests verify the original flooded/starved constant-property limit,
complete analytic smooth flux directions, constant high-pressure transport
with liquid density in SI kg/s, and explicit iteration-budget failure.

Rigid cavitating point cases retain h0=.1 um, Rx=10 mm,Ry=20 mm,
x=+/-8L,y=+/-8Ly, but use Ue=.1 m/s so pressure dependence is significant.
For 17/33/65 grids, Q=0.152191317/0.212848236/0.231211825 N and peak
p=8.74852/21.1658/24.2872 MPa. Finest eta/eta0=1.70475 and
rho/rho0=1.0139945. Original max mass defects stay below 7.52e-10;
relative local/global conservation and complementarity pass the unchanged
criteria. These are converged discrete physical-model roots. The last
load/peak mesh changes are 8.63%/14.75%, so the 1% point mesh target is
explicitly NOT met; no resolved EHL benchmark is claimed.

`results/property_reynolds_20261027_verified/` contains seven SI NPZ
states, two figures, exact source/runtime/parameter provenance, constitutive
continuation/active histories, mass budgets and front brackets. Fresh-dir
reproduction: `python -m examples.run_property_reynolds_verification
--output-directory results/property_reynolds_new` with the documented
single-thread environment. Raw pressure/content are not clipped, viscosity
is not capped, and existing baselines are preserved. Next: initial coupled
uniform cases with independent equation, elastic-source and force checks.

Milestone-2 checks: Ruff clean; full pytest 314 passed, 62 optional device
skips (87.60 s). Six added configurations cover limit parity, continuum
channel refinement, complete smooth-flux derivatives, density/transport SI
units and retained failure. Verified source hashes match; the independent
reference-convergence figure was inspected. Graded coupling remains deferred.

## 13. Initial pressure-dependent elastic cases, 2026-10-27

Milestone 3 adds a separate uniform, prescribed-offset CPU coupling path.
Full constitutive strength and full elastic compliance define its returned
state. Material continuation first solves the rigid pressure-dependent
problem; compliance continuation then updates film from every source on the
stated finite physical grid. Acceptance reevaluates original SI mass flow
on exact coupled film. The old monolithic, isoviscous references and GPU
implementations are unchanged. Full equations, bounds and failure semantics
are in `docs/CONSTITUTIVE.md`.

Independent checks precede field interpretation:

- Material-strength zero agrees with the unchanged monolithic JFO/elastic
  reference: relative pressure/force errors 1.05e-9/3.25e-10.
- A separate complete nonlinear hybrid-root algorithm on 9x13 cells,
  x=[-4L,6L],y=[-4Ly,4Ly], starts only from the old isoviscous reference.
  It converges in 161 evaluations, with original max mass defect 1.66e-13
  and relative local/global imbalance 4.71e-14/3.25e-16. The new material/
  elastic Picard result agrees to 1.60e-10 pressure and 8.20e-11 force.
  This is two algorithms solving the same discrete assumptions, not an
  independent experimental field comparison.
- Complete smooth flux directions, law derivatives, channel limit and
  constant-property parity remain separately verified. The hybrid reference
  is a bounded small verification algorithm, not a new globally convergent
  production Newton implementation. No bordered/load-controller change is made.
- Original SI mass, complementarity and feasibility use the existing
  tolerances. Exact full-source film and selected direct source-cell
  displacement are independently checked; force is integrated using both
  trapezoidal and rectangular-cell arithmetic. The budget-one failure probe
  retains the actual last candidate and outer/inner histories.

Larger initial grids share h0=.1 um,Rx=10 mm,Ry=20 mm,E*=100 GPa,
Ue=.1 m/s and the illustrative T=313.15 K fluid, with x=+/-8L,y=+/-8Ly.
Outer gauge pressure is zero and west content one. These are physical finite
reservoir assumptions; the study does not move them or claim infinite-domain
accuracy. At 17/33 nodes the coupled loads are 0.151876716/0.204189265 N,
minimum sampled films 0.107102987/0.111885032 um and peak pressures
8.38583/17.9606 MPa. The 33 grid reaches eta/eta0=1.48641 and
rho/rho0=1.0104571. Both pressure-dependent laws are active in the actual
coupled mass equations. Original max mass stays below 3.71e-10; relative
local/global imbalances stay below 8.50e-11/1.07e-12, and complementarity
is zero at printed precision. Direct peak displacement matches FFT within
1.66e-23 m; exact field coupling discrepancy is zero at reported precision.

The large initial 17->33 load/peak changes (34.4%/114%) demonstrate that
these are initial discrete roots, not resolved engineering predictions. The
bounded 65 grid and its separate mesh/error flags are reported next. Load is
measured at prescribed offset, not forced to a target, and current results
do not establish classical EHL film-height agreement or strong-load validity.

The bounded 65x65 coupled case completes in 107 outer iterations across
five compliance stages, at most 27 in one stage. Q=0.219183803068 N,
minimum sampled film=0.113147175105 um and peak p=19.790722834 MPa.
Viscosity rises to 1.5467298 times ambient and liquid density to 1.0114879.
Original max mass is 9.05e-10, relative local/global mass imbalance
7.65e-11/9.60e-13 and complementarity zero at printed precision. Independent
force quadrature differs by 1.65e-15 relative; direct peak deformation differs
by 1.66e-23 m, with zero reported exact-film coupling discrepancy.

The last 33->65 load/peak changes shrink to 7.34%/10.19%. The independent
1% mesh target remains false. These results complete the requested initial
coupled milestone, not mesh-converged or published classical-EHL validation.
Pressure dependence was added to the equations and independently verified;
it was not used to relabel the earlier far-field error budget as resolved.

The authoritative run is `results/property_coupling_20261027_fine/`:
seven SI archives and seven figures, with raw density/viscosity/pressure/
film/deformation/content, complete outer/material/active-set histories,
independent monolithic data, original mass budgets and retained failure.
Earlier completed coarse runs are preserved. Reproduce with
`python -m examples.run_property_coupling_verification --include-fine
--output-directory results/property_coupling_new` using single-thread setup.
Source hashes, acceptance flags, file counts and figure manifest were checked;
pressure/viscosity contours and pressure/film mesh profiles were inspected.

Milestone-3 checks: Ruff clean; full pytest **316 passed, 62 optional device
skips** (94.98 s). Two added regressions verify old monolithic limit parity,
retained failure, full coupled pressure/film/force agreement with a separate
nonlinear reference and its independent SI admissibility. No drivers, system
software, old solver implementation or result baseline was modified.

Next: obtain benchmark-matched fluid/geometry/supply data and introduce
verified positive-load control for the variable-property coupled model,
with controlled resolution and continuation toward classical pressures.
Retain the known finite-reservoir/mesh error budget. Published film-height
agreement is a subsequent milestone; it is not inferred from these roots.

## 14. Pressure-dependent positive-load control, 2026-10-27

Starting revision: `b491970e5a9b07e6c6ed1f7f77eb0bf03e8a0f0d`.
**Outcome: a bounded nested positive-force controller closes the new
pressure-dependent JFO/elastic equations on verified positive-offset cases.**
This is numerical verification of load control, not an experimental or
mesh-converged classical EHL result. No published test case is assumed.

`property_load.solve_property_at_load` changes h0 only. SI grid/domain,
geometry, E*, fluid, entrainment, flooded inlet and all nondimensional
scales stay fixed. Each force probe is a cold material/elastic continuation
solve; independently recomputed physical/film/force audits precede scalar
root acceptance. Initial samples must show a decreasing force response.
A sampled tolerance match is valid; otherwise a bracketed Brent solve is
required. Relative force error <=1e-8 remains mandatory independently of
scalar offset stopping. No pressure scaling, clamps, automatic domain or
bracket expansion, tolerance relaxation or GPU execution is used.

The public API bounds uncached force evaluations at 24, scalar iterations
at 24 and monotonicity samples at 3..7. Existing 20,000-node and 100-iteration
nonlinear stage/active-set caps remain. Memory retains scalar probe records
and best/last full roots; a callback streams complete probe arrays/histories.
Failed inner states are separately retained and never treated as a force
value. Signed offsets require a further verified initialization/branch
extension; the present cold-start model explicitly supports positive
brackets. Failure here does not establish global physical nonexistence.

### Independent recovery and imposed force

Use eta0=.04 Pa s,rho0=850 kg/m^3,z=.68 at assigned T=313.15 K,
Rx=10 mm,Ry=20 mm,E*=100 GPa,Ue=.1 m/s and the prior fixed scales.
The first 9x13 physical grid spans x=[-4L,6L],y=[-4Ly,4Ly]. An independent
complete nonlinear reference at h0=100 nm supplies its force; no reference
field seeds the load controller. Its template offset is deliberately 110 nm,
and the controller's bracket is [80,130] nm, so the known offset is not an
initial sample. That recovery tests pressure/content, elastic film and force
against another nonlinear algorithm, not solely scalar root convergence.

| Numerical case | Target Q [N] | Solved h0 [nm] | Sampled hmin [nm] | Peak pressure [MPa] | Relative force error | Force probes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent known-offset recovery | reference force | 100.000000016 | 107.968092 | 12.448450 | 8.62e-16 | 9 |
| Imposed force, 9x13, bracket [80,160] nm | 0.10 | 123.885854559 | 131.125194 | 9.944147 | 9.44e-15 | 9 |
| Same force/grid, bracket [100,150] nm | 0.10 | 123.885854559 | 131.125194 | 9.944147 | 4.17e-16 | 8 |
| Same force/domain, refined 13x17 | 0.10 | 160.621945032 | 174.083051 | 7.561251 | 5.77e-12 | 9 |

Known-offset relative error is 1.58e-10 and independent pressure relative
L2 discrepancy 3.81e-11. Different brackets for the imposed 0.10 N target
agree in h0 to 1.56e-14 relative. All returned states pass original mass,
local/global balance, complementarity and film feasibility. Final max mass
is below 4.19e-10; direct spatial peak displacement differs from full-source
FFT by at most 1.66e-24 m. Force values are recalculated from raw pressure,
not imposed by rescaling it. Reported numerical digits are reproducibility
information, not physical uncertainty estimates.

### Fixed-force mesh uncertainty and retained failures

Refinement holds the same physical bounds and force but changes h0 by
29.65%, minimum sampled film by 32.76% and peak pressure by -23.96%.
The sampled extrema also depend on these nonnested coarse grids. Thus force
closure alone cannot establish mesh accuracy or domain independence. These
large mesh effects remain explicit while the load-control equations pass.
The user's subsequent physical test case should supply matched operating
conditions; its own force/film/pressure resolution must be established.

The benchmark deliberately reduces the outer stage budget to one for a
probe. Its partial-compliance fields/history are saved, with zero accepted
force evaluations. Regressions additionally cover unbracketed targets,
force-budget exhaustion, a real failed inner solve after one accepted probe,
corrupted returned film/cached force, endpoint matching within tolerance,
invalid requests and scalar-offset stopping without force closure. No
failed partial root is labeled a converged load solution.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_property_load_verification --include-refinement \
  --output-directory results/property_load_new
```

The verified run is `results/property_load_20261027/`: 35 streamed accepted
force probes, their complete coupled/material/active histories, four final
solutions, an independent reference and a rejected partial candidate (41 SI
NPZ files total), plus three figures and finite JSON with source/runtime
provenance. Existing directories are refused; per-case checkpoints preserve
completed evaluations if interrupted. Force-error plots retain exact zeros.
Generated results remain outside Git. No existing baseline or old numerical
solver is overwritten.

Final checks: `ruff check .` clean; complete `pytest -q`: **327 passed,
62 skipped** (188.72 s). Skips are optional PyTorch/device coverage; no GPU
execution is claimed. Recorded Python source hashes match the tested
sources. The 41 SI archives and three-figure manifest were checked;
force/offset and matched pressure/film figures were visually inspected.

## 15. User circular 200 N benchmark readiness (2026-10-27)

Starting revision: `dc2e17ad19f00e845d8adfa416c5955bea060898`.
**Outcome: the operating case and modulus conversion are verified; the
current cold positive-offset path does not yet solve the 200 N case.**
This is a benchmark-readiness assessment, not physical validation or a
force-controlled comparison with Hamrock-Dowson film estimates.

### Explicit input conventions and assumed lubricant

The user's convention is E_contact=113 GPa, U=eta0 Ue/(E_contact R),
W=Q/(E_contact R^2), G=alpha E_contact. Set Q=200 N, Rx=Ry=15 mm,
Ue=2 m/s, eta0=.04 Pa s and alpha=2.21e-8/Pa. This gives
U=4.7197640118e-11, W=7.8662733530e-6 and G=2497.3. The alternative
E'=226 GPa convention has U/2, W/2 and 2G; the elastic kernel still uses
113 GPa. The modulus definition is 1/E_contact=sum((1-nu_i^2)/E_i).
Independent supplied Hertz quantities agree with the circular formulas:
a=.2710408131 mm, p0=1299.876220 MPa, approach=4.897541492 um.
The rounded supplied 1299 MPa differs by less than .1%.

The user's supplied correlation estimates hmin=.278495 um and
hcentral=.477353 um are retained verbatim. They are not exact field data,
and no independent source/formula match is asserted here. The constitutive
law was not specified by the user. This assessment assumes the existing
Roelands law, with z=alpha*gamma/log(eta0/eta_star)=.671366671110627,
gamma=196 MPa and eta_star=6.31e-5 Pa s. Its ambient logarithmic viscosity
derivative exactly matches alpha, independently checked by finite differences.
Matching that slope does not establish that this curve reproduces the user's
lubricant at GPa pressures. The assumed density model is Dowson-Higginson,
rho0=850 kg/m^3, a=.6e-9/Pa and b=1.7e-9/Pa. No temperature was supplied
or assigned; no thermal model is used.

### Bounded forward probes and separate mesh evidence

Both centered, nested uniform meshes use x,y=[-3a,3a], zero gauge pressure
on all edges, west content one, +x entrainment and full-domain free-space
elastic sources. L=a, href=Hertz approach and pref=Hertz peak are fixed for
every offset and both meshes; Lambda=12 eta0 Ue L/(href^2 pref). Neither
the physical reservoir nor the scales are regenerated from the trial gap.
These finite flooded reservoirs are not evidence of fully flooded far-field
convergence. BLAS/OpenMP use one thread; at most 1617 nodes are allocated
per state. Existing stage, active-set and admissibility limits are unchanged.

| Prescribed h0 [um] | Q on 17x25 [N] | Q on 33x49 [N] | Relative mesh load change |
| ---: | ---: | ---: | ---: |
| 5 | .2719783959 | .2786685967 | 2.460% |
| 2 | .6459915517 | .6801215833 | 5.283% |
| .5 | 1.6980452655 | 1.9700562924 | 16.019% |
| .1 | 3.3322223742 | rejected | not compared |

At .5 um, peak pressure changes from 32.6244 to 41.3295 MPa (26.68%),
and sampled minimum film changes from .537148878 to .546175885 um
(1.681%). No mesh-convergence criterion is met. At .1 um the coarse
solution has peak pressure 123.138891 MPa and minimum film .220393477 um,
but its force is only 3.33 N. Proximity to a supplied film-height estimate
at the wrong force is not physical agreement with the 200 N benchmark.

The seven accepted forward states pass independent re-evaluation of saved
SI mass fluxes, pressure/content admissibility and complementarity. Maximum
dimensionless continuity defect is 1.305e-11; maximum relative local defect
is 9.716e-11. Fresh full-source FFT film/displacement agree exactly with the
saved arrays at printed precision. Independent direct source-cell integration
at the center and pressure peak differs from FFT by at most 5.294e-23 m.
Separate SI trapezoidal force integration agrees with every saved force.
These checks verify the returned low-force roots, not a 200 N solution.

The fine .1 um cold solve overflows the viscosity exponential during its
initial rigid-film material continuation. No compliance stage is accepted.
The nested exception, 123 property-history entries and last available
pressure/content state are saved. That last partial state has mass defect
.00142434 and relative local defect .198866, so it explicitly fails the
original criteria. It is never included as a force evaluation or plotted
as a converged state. No exponential clipping, pressure cap, additional
iteration allowance or acceptance relaxation is introduced.

None of the accepted offsets brackets 200 N. The scalar load controller is
therefore not invoked; target_achieved, mesh_verified and domain_verified
remain false. This finite sample does not prove that every positive-offset
branch is insufficient. It shows that the current cold-start path and these
meshes cannot deliver the requested benchmark. The dry approach (4.90 um)
is much larger than the supplied central film (.477 um), motivating a
signed-offset elastic continuation. Their difference alone is not an EHL
solution or a justified fixed fluid offset.

### Reproduction and next justified solver change

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_200n_readiness \
  --output-directory results/circular_200n_coarse_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_200n_readiness --ny 33 --nx 49 \
  --output-directory results/circular_200n_fine_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.plot_200n_readiness \
  --inputs results/circular_200n_coarse_new results/circular_200n_fine_new \
  --output-directory results/circular_200n_comparison_new
```

Verified outputs are `results/circular_200n_readiness_20261027_final/`,
`results/circular_200n_readiness_20261027_refined/` and
`results/circular_200n_comparison_20261027/`. Together they retain seven
converged archives, one failed partial-state archive, complete available
histories, finite JSON, original Python source hashes, comparison input
hashes, seven independent saved-field audits and one inspected figure.
A preliminary three-offset run is separately preserved. No generated
results are committed; existing output directories are refused.

Next: add a separately verified warm-start continuation that maintains
positive *total film* while allowing signed rigid offsets. Advance load and
material strength through accepted roots with bounded retries, retaining
original SI mass, complementarity, source coupling and force gates. Verify
small-load parity and known-offset recovery before attempting the 200 N
branch. Once it closes, establish mesh/domain accuracy before interpreting
film differences from the supplied correlation. No GPU or constitutive-law
change is justified as a substitute for that solver work.

Final checks: `ruff check .` clean; full `pytest -q`: **334 passed,
62 skipped** (205.30 s). Skips are optional PyTorch/device coverage.
No new device execution is claimed. All seven numerical regressions and
seven independently re-audited accepted forward states pass. The 200 N
force target and mesh/domain accuracy remain explicitly unachieved.

## 16. Verified warm signed-offset reference (2026-10-27)

Starting revision: `24e3849fb4f00709ca8e0c98d3413fc8817b4805`.
**Outcome: full-material JFO roots can now cross zero rigid offset with
positive total film. This removes the cold-start sign restriction on a
verified small-grid path; 200 N closure and physical accuracy remain open.**
Roelands is now confirmed by the user. The ambient-alpha matching and 113
GPa convention remain explicit; density parameters are still assumptions.

### Separate bounded solver and original physical acceptance

`property_warm.PropertyWarmSystem` is a full-compliance, full-material
prescribed-offset system with interior P=p/pref and deficit D=1-theta. It
uses the original SI mass flux and full-domain uniform free-space FFT.
The signed h0 override changes only the rigid offset, leaving geometry,
fluid, physical reservoir, scales and source support fixed. The existing
positive-offset problem constructor, cold solver and positive-load controller
are unchanged. There is no pressure rescaling, field clipping or floor
replacement. Total film must exceed the original floor on every trial.

The analytic JVP differentiates rho(p), eta(p), full-source elastic film,
arithmetic face mobility rho h^3/(12 eta), capacity rho h, and west-upwind
content. Mass rows are SI divergence divided by rho0 href^3 pref/(12 eta0
L^2) and by 100; complementarity rows are min(P,D). Random smooth-state
finite differences independently test the complete material/elastic action.
Directional min derivatives at ties are checked separately.

The explicit dense reference is capped at 400 interior nodes (800 unknowns).
It uses row/column equilibration and verifies every reconstructed correction
against a fresh analytic JVP to relative 5e-9. Newton permits at most 30
corrections and 24 backtracks, with true directional Armijo decrease and
pressure/content feasibility to the existing 1e-10 tolerance. These are
bounded small-grid verification tools, not a scalable production Newton or
bordered force solver. Intermediate iterates may have mass/complementarity
defects; only a fresh independent SI mass/constraint audit accepts a root.
Every seed is checked at its declared offset against exact film, displacement,
original physical residual and boundary data; failed seeds cannot be reused.

`continue_warm_property_offset` proposes at most half the previous minimum
total film as an offset increment. Rejected Newton trials halve the step;
accepted steps grow by 1.5. All retries start from the last independently
audited root. The API caps attempts at 60; the benchmark explicitly uses
24. Scalar ledgers and callback-streamed fields/histories bound memory and
preserve both successful and failed work. This is offset continuation with
measured load, not a new imposed-force closure algorithm.

### Independent numerical references

A positive-offset 7x9 case agrees with the existing independently initialized
complete nonlinear hybrid reference. Seed identity returns without corrections;
a changed offset is separately solved. Regression tests cover full derivative
actions, directional ties, true correction errors, accepted merit descent,
seed/boundary rejection, strict film feasibility, iteration/resource limits,
injected retry rollback and retained attempt-budget failure.

On the circular 17x25 case, continuation reaches h0=-.2 um with positive
film and original mass/complementarity acceptance. A separate finite-difference
hybrid root, with no analytic JVP from the new solver, agrees locally. It starts
from a perturbed warm root: this verifies local algorithm agreement, not
independent branch discovery or uniqueness. This 17x25 signed branch can
be followed without approaching the film floor. No mesh adequacy or
convergence is presumed from this single grid.

A saved branch root at h0=-.1158467673 um is independently rerun by
`examples.verify_signed_reference`. The numerical-Jacobian solve converges
in 696 evaluations, maximum scaled residual 9.03e-19. Relative pressure
L2 discrepancy is 1.737e-11, relative film L2 discrepancy 1.174e-14,
maximum content difference 5.474e-12 and relative force difference
2.623e-12. Its original maximum dimensionless mass defect is 9.025e-17,
relative local defect 1.831e-15 and complementarity 9.261e-33. The independent
reference fields, input archive hash and runtime/source provenance are saved.

### Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_signed_200n --output-directory results/signed_200n_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.verify_signed_reference \
  --input-directory results/signed_200n_new \
  --output-directory results/signed_local_reference_new
```

The runner checkpoints every attempted step and refuses existing directories.
Its initial .5 um root is freshly solved, not borrowed from an earlier output.
All accepted roots retain raw SI p, theta, h, displacement, density and viscosity,
complete Newton/line-search histories and center/peak direct cell-integration
audits. Failed trials retain available partial fields/history under separate
names and never enter the accepted ledger. Generated arrays remain outside Git.

### Final bounded branch and remaining work

The authoritative run is `results/signed_200n_20261027_final/`, with a
fresh .5 um seed (Q=1.6980452655 N), 15 accepted offset steps and nine
rejected/retried trials in the declared 24-attempt budget. Selected roots:

| Signed h0 [um] | Measured Q [N] |
| ---: | ---: |
| .4 | 1.9232494000 |
| .25 | 2.4272709980 |
| .09126873782 | 3.4049380542 |
| -.01704943477 | 4.5510306771 |
| -.1158467673 | 5.9803798328 |
| -.2070056887 | 7.4824419007 |
| -.2806234909 | 8.6576112442 |
| -.3014203907 | 8.9655393334 |

The last root has hmin=.1142840090 um, central film=.3749396587 um and
peak pressure=713.916834 MPa. These are sampled low-force quantities,
not a 200 N comparison with the supplied correlation. Across accepted
steps the largest original dimensionless mass defect is 1.779e-13,
relative local defect 4.526e-11 and complementarity 8.358e-23. Largest
verified linear correction residual is 4.804e-12. Direct source-cell
center/peak displacement discrepancies stay below 4.236e-22 m.

The nine rejected trials exhaust line search near pressure/content bounds;
recorded trial reasons distinguish infeasibility from Armijo rejection.
They are not accepted merely because the previous step or a linear solve
succeeded. Smaller retries subsequently recover seven consecutive accepted
steps, so these rejected proposals do not establish a physical branch limit.
The run ends at its declared attempt budget, not loss of positive film or
proof that larger loads are impossible. It does not reach the requested
-5 um offset or close 200 N. Physical mesh/domain criteria remain false,
and the current dense cap prevents applying this reference directly to the
33x49 mesh. No tolerance or cap is raised to disguise these limitations.

The final branch retains 25 SI archives (seed, 15 roots and nine failed
partial states) and an inspected branch figure. The separate
`results/signed_local_reference_20261027/` adds one independent reference
archive. All recorded source hashes match; its input NPZ hash identifies the
exact signed root compared. An earlier exploratory run was interrupted after
checkpointing its trials and is preserved separately, without being labeled
complete or used as the authoritative final run.

Next justified action: resume from an audited accepted signed root, improve
active-constraint step handling with independent globalization regressions,
and implement a bounded matrix-free/sparse correction for finer-grid parity.
Introduce a load-directed continuation/force bracket only through audited
roots; retain the actual 200 N force gate before film comparisons. Density,
modulus conventions and the confirmed Roelands law should not be changed to
compensate for unresolved numerical continuation or resolution errors.

Final checks: `ruff check .` clean; full `pytest -q`: **341 passed,
62 skipped** (267.32 s). Seven new regressions and the independent saved-root
comparison pass. Skips are optional PyTorch/device coverage. No GPU execution,
200 N force closure, mesh convergence or physical EHL accuracy is claimed.

## 17. Matrix-free load-directed material JFO continuation (2026-10-27)

Starting revision: `08d64ce8693d2ff921370932539f9b00b17e4518`.
This milestone replaces the fine-grid dense correction bottleneck with a
separate matrix-free bordered force path. The existing dense signed-offset
reference, cold solvers, previous accepted archives and original physical
acceptance criteria are preserved. Roelands and the 113 GPa convention
remain fixed; density data remain the explicitly stated model assumptions.

### Scalable correction and independent algebraic verification

`PropertyLoadSystem` adds h0/href and the force equation to the previously
verified full-material/full-compliance JFO operator. Its complete analytic
action includes pressure-dependent density/viscosity, full-source elastic
film, the signed offset column, upwind content and SI load quadrature.
No dense Jacobian is formed on the scalable path. A sparse frozen-film,
frozen-property constrained JFO operator preconditions Schur/GMRES solves;
the offset column and force row are consistent with the full operator.
The material/elastic terms omitted by preconditioning remain in every true
matrix-free product. No constitutive law is simplified to make the solve pass.

The existing independently checked `solve_cavitating_correction` computes
pressure/content and offset responses, reconstructs the bordered step and
checks its complete analytic-JVP residual. Unsafe Schur reconstruction uses
a complete bordered GMRES fallback. Krylov recurrence histories, block/whole
true residuals, fallback and Schur indicators are retained. A small-grid
explicit dense Jacobian independently agrees with the matrix-free correction;
forcing an unsafe Schur threshold verifies the full fallback as well.
Finite differences independently check the complete bordered action,
including its offset and force terms. A separate independently initialized
nonlinear reference provides known-offset/pressure recovery at its measured
force. Injected Krylov failures retain their diagnostics and partial state.

The production path caps physical nodes at 20,000, Newton corrections at
30, GMRES restart at 60 and cycles per block at ten. Sparse LU fill consumes
additional memory; these limits are not a measured peak-process memory claim.
The old dense solver still defaults to at most 400 interior nodes; the shared
analytic system explicitly opts into the larger cap only for the scalable
path. A 33x49 sparse construction regression independently verifies that the
bordered matrix remains sparse beyond the old dense limit. CPU benchmarks
use one BLAS/OpenMP thread, NumPy/SciPy float64 and the existing uniform
zero-padded free-space FFT. No GPU or unequal-cell FFT is introduced.

### Feasibility, active constraints and original merit

Each Newton correction first tries its full step. Recovery starts from a
safe affine fraction to the original pressure/content bounds and exact
positive-total-film floor, then backtracks. Film is affine along a joint
pressure/offset direction, so its bound uses the full-source FFT direction,
not a guessed gap limiter. Bound ratios are evaluated only when the full
step could cross that bound, avoiding meaningless overflow from tiny
unconstraining direction components. No pressure/content projection,
film-floor lift, viscosity cap or acceptance relaxation is applied.

Failed searches can try up to eight selected active-constraint models.
Violating P_new or D_new rows are pivoted to their zero constraints, and
each selected linear system is independently checked. A switched constraint
row is explicitly a selected active-set model, not falsely described as the
old min derivative away from a tie. Regardless of that model, Armijo merit
and its directional slope use the original min(P,D) equations, including
true directional min derivatives at ties. Every line trial records step,
merit, acceptance and feasibility/Armijo rejection reason. Physical
admissibility and relative force <=1e-8 are independently required at return.
Selected-system dense parity is verified; this does not establish universal
active-front robustness for arbitrary loads or meshes.

### Force steps, preserved warm roots and matched-grid protocol

`continue_property_load` advances the imposed SI normal force. Failed solves
halve the requested force increment, accepted roots grow it by 1.5, and only
an independently audited accepted root becomes a continuation seed. Each
attempt starts from that root, never a failed partial Newton iterate. The
API caps attempts at 60 per call; this benchmark declares 12 attempts for
each requested force checkpoint and a 1e-5 N minimum increment. All loops
and failure exits remain bounded. Zero force errors are retained exactly.

Coarse input is the preserved 17x25 signed root at 8.9655393334 N. Refined
input is the preserved 33x49 prescribed-offset root at 1.9700562924 N.
Raw SI axes and input NPZ hashes identify these states. Fresh original SI
mass, pressure/content, boundary and full-source film audits validate them
before force stepping. No accepted input archive is modified or overwritten.
Both grids retain the identical physical +/-3a reservoir, Hertz-based scales,
fluid, geometry and fully flooded west-content assumption of section 15.
The refined path first closes 2, 4 and 8 N, then joins the same 10, 20, 40,
80, 120, 160 and 200 N checkpoints proposed on the coarse path. Once a grid
stalls its later checkpoints remain unachieved. Comparisons include only
checkpoints independently force-closed on both grids, not matched offsets
or failed candidates. Mesh accuracy and finite-reservoir placement remain
separate questions from equation/force convergence.

Every attempt checkpoints its requested force and status. Accepted SI fields
retain pressure, content, film, displacement, density and viscosity plus
complete nonlinear/Krylov/globalization histories and direct center/peak
source-cell displacement checks. Rejections retain last available partial
fields, signed offset, nonlinear/model/line histories and linear diagnostics.
Nonfinite diagnostic values are explicitly tagged in portable JSON, never
silently dropped or emitted as invalid numeric literals. Final bounded-run
status distinguishes successful force checkpoints from a continuation stall.

### Reproduction and independent archive audit

With the section-15 and section-16 source runs available (or freshly
reproduced under their documented commands):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_load_directed_200n \
  --coarse-source results/signed_200n_20261027_final/attempt_23/solution.npz \
  --refined-source results/circular_200n_readiness_20261027_refined/offset_0.5um/solution.npz \
  --output-directory results/load_directed_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.audit_load_directed \
  --input-directory results/load_directed_new \
  --output-directory results/load_directed_audit_new
```

The coarse source is specifically the section-16 attempt-23 root at signed
h0=-.3014203906597183 um; the refined source is the section-15 .5 um root.
For differently named reproduction directories, substitute those paths.
A wrong declared seed offset fails its exact-film audit instead of being
silently accepted. The runner refuses existing output directories and never
writes into either source directory. On a fresh checkout first generate
these two source runs; no large datasets are stored in Git.

The separate auditor reloads every accepted SI archive and independently
recomputes original mass/constraint/boundary acceptance, exact full-source
film/displacement and SI trapezoidal load against the actual requested force.
It also checks the saved complete correction errors and true directional
Armijo inequalities, retains source/archive hashes and plots the strongest
accepted refined field with its actual operating force. All pressure contour
levels span the raw signed range; no field is clipped or smoothed.

### Verified bounded results and matched-load mesh uncertainty

**Outcome: the matrix-free path reaches a fully accepted 100.4423828125 N
state on 33x49 nodes, using preserved warm roots and unchanged physical gates.
The requested 200 N state remains unachieved.** The old dense bottleneck no
longer prevents refined-grid force solves. This is numerical verification
of a restricted discrete model, not mesh-converged physical validation.

The authoritative run is `results/load_directed_200n_20261027_final/`:
17 coarse attempts (eight accepted/nine rejected) and 38 refined attempts
(33 accepted/five rejected). Coarse requested checkpoints 10 and 20 N close;
refined checkpoints 2, 4, 8, 10, 20, 40 and 80 N close. Additional accepted
adaptive targets reach 20.62890625 N coarse and 100.4423828125 N refined.
These last adaptive loads are actual force-closed targets, not claims that
the next requested 40/120 N checkpoints or 200 N were reached.

| Shared target Q [N] | Coarse hmin [um] | Refined hmin [um] | Coarse central film [um] | Refined central film [um] | Coarse/refined peak p [MPa] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | .1053712134 | .2805497782 | .3956703655 | .4061321004 | 820.0550 / 775.6412 |
| 20 | .005463677393 | .2547030083 | .2710452750 | .4935463323 | 756.4411 / 1058.9447 |

At 10 N refinement changes sampled minimum film by +166.25%, central film
by +2.644% and peak pressure by -5.416%. At 20 N the corresponding changes
are +4561.75%, +82.09% and +39.99%. The declared 1% mesh target is false.
These are genuinely matched-force comparisons with the same physical
reservoir; offset is independently solved on each mesh. Stable force or a
relatively stable central film cannot override the unresolved minimum film
and pressure errors. No convergence order is inferred from two grids.

The strongest refined root has Q=100.4423828125 N,
h0=-2.76927325835 um, hmin=.1375437173 um, central film=.3060045115 um
and peak p=1406.190560 MPa. Original max mass is 1.956e-15, relative local
imbalance 1.564e-13 and complementarity 2.205e-34. Direct source-cell
center/peak displacement differs from FFT by at most 4.236e-22 m there.
The inspected pressure contour shows visible cell-scale variation. Neither
that field nor its film values are presented as resolved classical EHL
predictions or compared to the supplied **200 N** correlation as if loads
matched. Further resolution and branch/discretization investigation are
required; no smoothing or pressure adjustment is used to conceal the pattern.

### Distinct stalled-step diagnostics

Coarse proposals above the last accepted 20.62890625 N root drive total
film toward the unchanged floor and exhaust the 30-correction budget.
The last accepted coarse minimum is 1.29650e-11 m, still above the original
1e-12 m floor. No physical-contact onset, branch nonexistence or engineering
accuracy is inferred from this unresolved coarse path. Its rejected partial
fields and every model/line trial remain distinct from accepted force roots.

The refined path rejects larger proposals when the true bordered Krylov
residual check fails. For example, the 106.375 N proposal reports a true
linear residual 3.072e-7 after 1200 total linear iterations across block/
fallback work, above the unchanged 5e-9 criterion. Smaller retries recover
98.78125, 99.73046875 and 100.4423828125 N. The bounded continuation ends
at its 12-attempt budget for the requested 120 N checkpoint. This is a
linear/preconditioning limitation of the tested route, not evidence that
the physical 200 N solution is impossible. No failed correction or partial
nonlinear iterate becomes a force value for the next accepted step.

### Independent audits, artifacts and next development action

`results/load_directed_200n_20261027_audit/` independently verifies all
41 accepted archives. Largest original dimensionless mass defect is
1.353e-12 and largest independent relative force error is 4.441e-16.
The audit checks 286 accepted-solve linear models, four actual bordered
fallbacks and 116 accepted shortened line steps; all recorded true linear
errors are below 9.780e-10 and every original directional Armijo inequality
passes. Alternative selected active-constraint recovery models are verified
by small-grid algebra tests but are not exercised in these accepted-run
histories; no broad empirical robustness claim is made for that optional
recovery route. Primary min active sets are updated during Newton solves.

The benchmark retains 55 SI NPZ files (41 accepted and 14 failed partial
states), complete histories, unchanged input hashes, runtime/source hashes,
explicit checkpoint/force flags and a branch figure. The independent audit
adds per-state/archive hashes, original residual/force/film/boundary audits,
globalization checks and a raw strongest-state pressure/film figure. Both
figures were inspected and recorded Python hashes match the final sources.
Earlier exploratory checkpoints are separately preserved, including an
interrupted pre-final run; they are not substituted for the authoritative run.

Next: improve material/elastic preconditioning and numerical equilibration
against the retained high-load Krylov failures, keeping true residual
criteria fixed. Independently verify a finer-grid warm-start transfer/repair
before extending matched-force resolution studies beyond 33x49. Investigate
the cell-scale pressure variation and minimum-film mesh sensitivity before
claiming physical accuracy or interpreting a 200 N film comparison. Changes
to the discretization require separate consistency/conservation benchmarks;
changing viscosity, density or plots is not an error-control strategy.

Final verification: `ruff check .` clean; full `pytest -q`: **347 passed,
62 skipped** (263.68 s). Six new regressions verify complete bordered
finite differences, dense/Schur/fallback correction parity, independently
known offset recovery at matched force, film directions, sparse fine-grid
construction, accepted merit/force gates, seed rejection, bounded retry
rollback and retained Krylov failure diagnostics. Optional skips are device
coverage; no GPU implementation or execution claim is added.

## 18. Verified sparse tangents and first 200 N force closure (2026-10-27)

Starting revision: `1e4f69fac7e5e1a00e903ba89c5fbacd99491bba`.
**Outcome: the full-material, full-compliance JFO equations close at 200 N
on the existing 33x49 grid. Physical/mesh accuracy remains unverified.**
The improvement changes only preconditioning; laws, reservoir, scales,
source support and all original acceptance criteria remain fixed.

### Independent sparse approximation and archived failure replay

Two explicit preconditioner options supplement the unchanged `frozen`
default. `material` differentiates rho/eta and rho transport capacity at
fixed film. `local` also includes only the source-cell self displacement
in that approximate pressure tangent. The resulting five-point correction
is assembled sparsely. Full-source nonlocal elasticity and all constitutive
derivatives remain in the unchanged true matrix-free operator. This is
not a replacement of physical elasticity by independent local cells.
Equations and exact coefficients are documented in CONSTITUTIVE.md.

Independent SI central differences check both approximate tangents. Complete
residual/JVP arrays are identical across all three preconditioners. Dense
small-grid correction parity checks confirm the same bordered Newton
problem is solved, with the unchanged true residual criterion <=5e-9.
No nonlinear/Krylov cap, film floor or acceptance tolerance is relaxed.

`examples.replay_load_linear` loads the archived unconverged linear state
from section-17's rejected refined attempt 30 at proposed Q=106.375 N.
The exact archive hash and offset are recorded; it is not used as a physical
warm root. On this identical state, with ten GMRES cycles/restart 60:

| Preconditioner | True correction accepted | True relative linear residual | Wall time [s] |
| --- | --- | ---: | ---: |
| Frozen material/film | no | original failure retained | 9.413 |
| Material tangent | yes | 6.802e-10 | 1.091 |
| Material + local self film tangent | yes | 9.713e-10 | .3644 |

These are single-state, single-host timings, not a general performance claim.
The original failed correction remains rejected; a separate corrected linear
solve does not retroactively validate its unconverged nonlinear state.
Full residual/Krylov histories and successful directions are retained under
`results/load_linear_replay_20261027/`.

### Restart from an accepted root and original physical audit

`examples.run_preconditioned_load` uses `preconditioner=local` explicitly.
It validates and hashes the last accepted 100.4423828125 N section-17 root,
then follows bounded force steps to 120, 160 and 200 N. Eleven accepted
force targets reach those checkpoints with no rejected load attempts. The
restart uses the same fixed 33x49 physical +/-3a reservoir, SI scales,
113 GPa convention, 2 m/s entrainment, .04 Pa s ambient viscosity, confirmed
Roelands law and assumed Dowson-Higginson density. No prior archive is changed.

At 200 N the sampled result is h0=-4.63055890822 um, hmin=.01960927772 um,
central film=.3320824786 um and peak pressure=1725.253140 MPa. Independent
SI force integration gives exactly 200 N at printed precision. Original
maximum dimensionless mass defect is 3.060e-16, relative local defect
1.256e-14, global defect zero at printed precision and complementarity
1.403e-30. Full-source film/boundary checks and independent selected-target
source-cell integration pass. Fields, rather than pressure scaling, supply
the normal force.

`results/preconditioned_load_20261027/` retains 11 SI archives, complete
nonlinear/Krylov/globalization histories, input/source/runtime hashes and
an unsmoothed restart figure. `results/preconditioned_load_20261027_audit/`
independently reloads all 11 roots, checks original SI fluxes, constraints,
full-source film, pressure edges/supply, raw force integrals and recorded
Armijo inequalities. Its 119 linear models have maximum true linear error
1.036e-9; maximum force error is 2.221e-16 relative. Recorded source hashes
match the final tested Python sources. No device runtime is involved.

The sampled minimum is far below the supplied .278495 um correlation
estimate and the central film differs from .477353 um. Neither agreement
nor disagreement at this resolution is physical validation: section 17
already demonstrated very large matched-load mesh errors and cell-scale
pressure variation. Reaching the correct force removes one blocker; it does
not establish a resolved film or pressure field. No field smoothing, film
lift or constitutive-parameter change is used to fit the correlation.

### Reproduction and next action

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.replay_load_linear \
  --input-directory results/load_directed_200n_20261027_final \
  --output-directory results/load_linear_replay_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_preconditioned_load \
  --input-directory results/load_directed_200n_20261027_final \
  --output-directory results/preconditioned_load_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.audit_load_directed \
  --input-directory results/preconditioned_load_new \
  --output-directory results/preconditioned_load_audit_new
```

Existing directories are refused. All reused states are hashed; no raw input
is overwritten. Next: verify interpolation as an explicitly unaccepted finer-
grid initial guess, repair it with the original full equations and force gate,
then compare independently converged 200 N roots at fixed physical bounds.
An interpolated pressure/content field alone must never become an accepted
warm root. Preserve the existing 20,000-node and iteration limits.

Final milestone checks: `ruff check .` clean; full `pytest -q` **352 passed,
62 skipped** (261.86 s). Five new configurations cover independent sparse
tangents, complete operator invariance and dense bordered correction parity.
Restart and raw 200 N pressure/film figures were visually inspected;
cell-scale pressure variation remains visible and is not hidden. Optional
skips are device coverage; no GPU execution is claimed.

## 19. Explicit finer-grid initialization and active repair (2026-10-27)

Starting revision: `f04a7c5742d26c6144b6fb3eef29c767700cc9bf`.
This milestone separates transferring a field from accepting a physical
root. It retains source archives and compares independently force-closed
solutions at 200 N on fixed physical reservoir bounds and identical laws.

### Force-preserving interpolation is not a solution

`PropertyLoadGuess` holds only an unaccepted pressure/content initialization,
signed offset and diagnostics. `prolongate_property_guess` first independently
audits the source root, then requires identical reservoir bounds, geometry,
modulus, SI scales, supply and floor, plus nested uniform target axes. Physical
(y,x) bilinear interpolation preserves the composite SI trapezoidal force on
the refined grid. No field clipping, pressure rescaling, extrapolation or
rigid-offset adjustment is applied. The target's own cell dimensions/kernel
and full-source FFT determine its total film. Changed target displacements
reflect changed source approximation, not circular FFT wraparound.

For the 33x49->65x97 200 N transfer, force remains exactly 200 N at printed
precision, but initial maximum mass defect is .03491795, relative local
imbalance .409726 and complementarity .122326. Initial sampled total film
is .03165391353 um. This interpolation is explicitly rejected as a root;
force preservation and positive film cannot substitute for mass/constraint
closure. The unaccepted raw initialization is retained separately.

`repair_property_load_guess` shares the original force-Newton engine but
has an explicit initial-guess contract: pressure/content bounds, physical
pressure edges, constitutive domain and exact positive total film must hold,
while initial mass/complementarity may fail. Final return requires all the
same original physical/force gates as warm continuation. Existing audited-
root continuation does not silently accept an interpolated non-root.
Independent small-grid tests compare repaired results against a separately
cold-initialized force solve and verify source immutability, load preservation,
wrong-domain/material rejection, strict film feasibility and zero-budget
failure retention.

### Retained failed repairs and bounded active lookahead

The initial 65x97 200 N repair exhausts 30 corrections. Its last available
partial state still has maximum mass defect .0197340, relative local defect
.224103 and complementarity .0597978. All remain failed. An independent
10 N transfer/repair also exhausts that budget; its history reveals repeated
steps shrinking through 4.29e-4, 4.29e-6 and 4.29e-8 near active constraints.
Those states are not reused as accepted continuation seeds. Both raw
initializations and failed fields/histories are preserved.

The repair path now enables safeguarded active lookahead. When a feasible
Armijo step is shorter than .1 and a novel selected active model exists,
try that model while retaining the short step as a fallback. Every model
keeps the true selected-system linear residual check; original min-equation
merit/slope and all physical bounds govern nonlinear acceptance. On cycling,
exhaustion or a failed alternative linear solve, the best retained feasible
Armijo step can be restored. Histories distinguish deferred steps, accepted
alternatives and restored fallbacks, including failed-model diagnostics.
No original tolerance or iteration/backtrack/model cap is raised. Existing
warm-root continuation keeps its previous default; explicit initialization
repair enables the new strategy. This heuristic is not a global convergence
proof.

Regression tests exercise real recovery from a flooded initial content field
and force a selected-model linear failure to verify restoration of the already
verified feasible Armijo step. The latter test retains a failed final root
when its correction budget ends; a successful fallback is not mislabeled as
full nonlinear convergence. The pre-lookahead experiments remain separately
reproducible with `--no-active-lookahead`.

### Independently accepted 200 N refinement

With active lookahead, the same 65x97 initialization converges in nine Newton
corrections and 18 linear models, including nine selected active models.
The independent saved-state auditor verifies the original field/force gates,
true correction errors and original-directional Armijo inequalities. Maximum
true linear error is 8.243e-10. The final original max mass defect is
1.188e-15, relative local defect 4.447e-15, global defect zero at printed
precision and complementarity 3.133e-29. Direct center/peak source-cell
integration differs from target FFT by at most 1.695e-21 m.

| Grid `(ny,nx)` | Independently integrated Q [N] | h0 [um] | Sampled hmin [um] | Central film [um] | Peak pressure [MPa] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 33x49 | 200 | -4.630558908 | .01960927772 | .3320824786 | 1725.253140 |
| 65x97 | 199.99999999999994 | -4.521772796 | .1629378195 | .4429945326 | 1741.100421 |

At matched force, minimum film changes +730.9%, central film +33.40% and
peak pressure +.9185%. The combined 1% mesh target remains false despite
excellent force/mass closure. Sampled extrema can move between nodes. The
raw pressure field still shows cell-scale variation. No asymptotic order,
physically resolved pressure spike, or film-height accuracy is established.
The supplied correlation estimates remain reference context, not exact fields
to which solver output is adjusted.

Authoritative evidence is `results/load_refinement_200n_20261027_final/`
and `results/load_refinement_200n_20261027_audit/`: the unaccepted guess,
fully repaired SI solution, complete 18-model history, input/source/runtime
hashes, independent archive/globalization audit and inspected matched-
centerline and raw pressure/film figures. The earlier no-lookahead 200 N
and 10 N failures and the first lookahead development run remain separately
preserved, never retroactively relabeled. Recorded source hashes match.
The source 200 N archive remains unchanged.

### Reproduction, bounds and remaining work

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_load_refinement \
  --input-directory results/preconditioned_load_20261027 \
  --output-directory results/load_refinement_200n_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.audit_load_directed \
  --input-directory results/load_refinement_200n_new \
  --output-directory results/load_refinement_200n_audit_new
```

`--no-active-lookahead` reproduces the previous globalization strategy;
failures remain failures. `--load-n` selects an already closed source load.
The default refinement factor is two; explicit `--refinement-factor 3` maps
33x49 directly to 97x145 at the same reservoir. This has 14,065 physical
nodes and stays below the unchanged 20,000-node cap. A further uniform
doubling of 65x97 would have 24,897 nodes and is rejected before allocation
by the registered case helper. No unequal-cell FFT or source truncation is
used to evade that limit. The third-grid bounded assessment is recorded
separately while it runs; no root is claimed before full repair/audit.

Next: complete the bounded third-grid assessment and compare only accepted
matched-force fields. Investigate the persisting grid-scale pressure pattern
with independent discretization consistency/stability evidence before any
change to conservative face transport. Finite physical-reservoir sensitivity
must also be assessed at matched force. More accurate linear solves alone do
not validate the unresolved film/pressure field.

Final 65x97 milestone checks: `ruff check .` clean; full `pytest -q`
**357 passed, 62 skipped** (299.96 s). Five new regressions cover force-
preserving initialization, explicit non-root/zero-budget handling,
independently cold-initialized repaired-root parity, physical-data/boundary/
film rejection, actual active lookahead recovery and safeguarded fallback
under an injected selected-model linear failure. Skips are optional device
coverage; all production calculations here are NumPy/SciPy CPU float64.

## 20. Regeneration after missing archives (2026-10-27)

Starting revision: bb01c3264e8e8821a0cb1e1bbc6f22d4723dae3b.
The original production results and pending factor-three checkpoint were
not present in the recovered Linux workspace. They were not relabeled or
reconstructed by interpolation. Fresh runs use the registered Circular200N
case and unchanged NumPy/SciPy float64 equations, physical +/-3a reservoir,
Roelands/Dowson-Higginson laws and original acceptance tolerances.

A fresh prescribed-offset 33x49 solve at h0=.5 um returns
Q=1.9700562924117646 N. Local-elastic-preconditioned force continuation
then targets 2/4/8/10/20/40/80/120/160/200 N, with at most 12 attempts per
target, initial increment min(2,max(.25,.2*Q_seed)) N, and the existing
30-Newton/10-Krylov-cycle limits. All 44 accepted states pass the separate
saved-field mass/complementarity/boundary/force/full-source-film audit and
original-directional Armijo/history checks. Maximum true linear residual
is 1.749e-9. The final 200 N offset is -4.630558908223391 um, reproducing
the earlier published root. This fresh path explicitly uses the existing
local preconditioner at all loads; it does not replay the earlier frozen-
preconditioner failure histories.

The existing `examples.run_load_refinement` then generates explicitly
unaccepted nested guesses and repairs them with the unchanged default
30-correction budget and safeguarded active lookahead. The regenerated
65x97 root again closes 200 N, with nine corrections/18 linear models.
Its independent saved-state/globalization audit passes; maximum true linear
error is 8.243e-10. Minimum/central film and peak pressure reproduce the
section 19 values. The raw field and matched-centerline figures were
inspected: alternating pressure variation is still visible, and the 1%
mesh criterion remains false. Equation convergence is not physical accuracy.

Recovery artifacts, exact reproduction scripts for the fresh seed and
load path, runtime/source hashes, NPZ fields and audit figures are preserved
under `results/recovered_200n_20261027/` in the original project checkout.
A `sha256.json` manifest covers copied artifacts; outputs remain untracked
and no existing baseline was overwritten. The source scripts refuse to
reuse their output directories. Run them only with fresh output paths, using
one BLAS/OpenMP thread and `PYTHONPATH=src:.`. Subsequent refinement/audit
commands use the existing examples from section 19 with the regenerated
load directory as their input. Paths recorded inside original summaries
refer to the generation directory, before the archival copy.

Pointwise density-tangent cell Peclet diagnostics use
Ue*h*abs(d(rho)/dp)*dx/(2*mobility), mobility=rho*h^3/(12*eta),
on cells with p>.1*pmax. Median values are 42020.4 (33x49) and
8889.33 (65x97); maxima are 4.560e7 and 2.172e7. These are coefficient
ranges, not a full coupled stability analysis: they omit nonlocal elastic
film derivatives, viscosity-gradient terms and active-front effects. They
show that the earlier smooth low-pressure probe is not representative of
the loaded local transport regime. No stencil or fluid-law change follows
from this diagnostic alone. The saved JSON retains the exact state paths
and explicitly restricted scope.

A complete bordered discrete-JVP check at the accepted 33x49 and 65x97
200 N roots uses alternating x pressure, 1 MPa base amplitude, supported
where p>.1*pmax, with zero content/offset perturbation. Both signs retain
exact nonlocal-film feasibility. Centered residual differences at amplitudes
.01/.001/.0001 have best relative complete-JVP errors 1.127e-9 and
5.802e-10 respectively; smaller steps increasingly show cancellation.
The archived reproduction script and JSON record all three errors and
source/runtime hashes. This tests those derivative directions, not full
spectra, conditioning, cavitation-front accuracy or coupled stability.

### Bounded 97x145 outcome: rejected, not a third-grid solution

The fresh factor-three repair exhausted the unchanged 30-correction budget.
All 240 linear models have true residuals <=2.607e-10, and the independently
recomputed full-source film equals the saved film. Direct center/peak source
integration differs from FFT by <=2.542e-21 m. However, original mass_maximum
is .0643674, relative local balance .440414, relative global balance .0294992
and complementarity .107005. Force is 200 N, but force preservation alone
cannot make this a root. The unaccepted initialization, failed partial NPZ,
complete history, final offset and independent failure audit are all retained.
No third-grid film value enters a matched-load convergence comparison.

Every correction tried eight active models. Accepted steps repeatedly shrink
from about 3e-10 toward 1e-26, interrupted by only three substantial steps
(.01359, .001095 and .02281). The original-directional Armijo inequalities
pass when recomputed, but repeated near-roundoff steps provide almost no
nonlinear progress. This distinguishes accurate linear corrections from
successful active-constraint globalization; it does not prove that no
97x145 root exists. All original limits and tolerances remain unchanged.

Next: replay the preserved rejected initialization/history to benchmark
active-constraint feasibility, model selection and roundoff/stagnation
handling, with explicit regressions for accepted steps that do not change
the state or merit meaningfully. Keep the current method as a reference,
retain failed states, and require original field/force gates before claiming
third-grid recovery. Then compare accepted matched-force meshes and audit
conservative transport/pressure-pattern stability. Reservoir and physical
film-height verification remain separate and open. Do not increase budgets
or alter material laws to bypass this failure.

Final checks: Ruff clean; full pytest 361 passed, 62 optional device skips
(285.51 s). No production solver, tolerance, floor or fluid law changed in
this regeneration/assessment milestone.
