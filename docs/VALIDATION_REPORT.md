# EHL scientific verification report

Audit date: 2026-10-26. Scope: NumPy/SciPy CPU stationary, isothermal,
incompressible, isoviscous JFO/elastic point contact with positive load
closure. This is numerical verification of a restricted model, **not complete
physical validation of classical piezoviscous EHL**.

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
