# EHL scientific verification report

Audit history: 2026-10-26 through 2026-10-28. Sections 1–10 retain the
original incompressible/isoviscous audit; sections 11 onward document
pressure-dependent NumPy/SciPy CPU developments. The latest bounded
matrix-free force path now closes 200 N on 33x49, 65x97 and 97x145 nodes;
mesh/domain accuracy remain open. This report records numerical verification,
**not complete physical validation of classical piezoviscous EHL**.

## Verification and validation evidence levels

Keep the evidence hierarchy explicit:

1. **Equation closure:** For each returned solution, independently check
   integrated pressure/load, the original Reynolds mass-flux residual and
   local/global conservation, JFO pressure/content feasibility and
   complementarity, reconstructed elastic film coupling, and nonlinear
   convergence. Existing reduced-model and coupled tests exercise these
   properties (including the cavitation/complementarity checks in
   `tests/test_cavitating_coupled.py` and
   `tests/test_cavitating_newton.py`). A test passing at one case or grid does
   not establish closure for all regimes, nor numerical accuracy.
2. **Numerical convergence:** Independently study (a) mesh refinement at a
   fixed physical domain and (b) domain expansion while holding physical grid
   spacing approximately fixed. Track central/minimum film and the main
   pressure field on common physical coordinates. Do not combine domain
   truncation error and mesh error into one sequence or infer physical
   accuracy from small equation residuals.
3. **Experimental validation:** Only after Levels 1 and 2 are demonstrated
   for the same declared model and case, compare its converged film against
   independent measurements. Ciulli Figure 13 is the first prepared
   reference, not yet a solver-vs-experiment result. Its digitization and
   uncorrected pressure-dependent optical effect remain part of the
   experimental uncertainty.

### Pressure comparison and outlet-spike protocol

An outlet-side Petrusevich-type pressure spike is an expected EHL feature, not
automatically a numerical error. Do not use a global pointwise pressure L2
error that mixes it with the main contact: a reference may not resolve the
spike, refinement can change its shape, and a small peak shift can dominate
that norm. Compare pressure on shared physical coordinates, not array indices.

For each fixed-domain mesh study, state physical bounds for three regions:
the main contact, a separately excluded outlet-spike window, and the
pressure-release/cavitation region, with bounds in physical coordinates (or
explicitly normalized by a fixed physical scale). Define and record the
spike window from the finest field being assessed using the secondary
downstream peak and its bracketing shoulders; freeze those dimensional bounds
when sampling the coarser meshes. Reuse the same physical bounds for the
corresponding fixed-spacing domain study. If the peak or shoulders are not
unambiguous, report that and do not claim a unique spike width.

Compare the main pressure profile/field only in the main-contact region,
excluding both the spike window and release/cavitation region. In the
pressure-release region, report the zero-pressure/cavitation front and its
mass-content/flux behavior rather than smoothing or scoring it as a
positive-pressure main-contact profile.

Analyze the outlet feature separately: record both the field-global maximum
and outlet-window peak pressure and `(x,y)` location; pressure at the nearest
samples on each side of the outlet peak and at the window shoulders; width at
half prominence using a linearly interpolated baseline between shoulder
pressures along the downstream line through the peak; the physical width and
number of grid cells across that width; and the spike-region integrated
pressure/excess-pressure quantity. Report physical grid spacing with the cell
count. Track peak location, profile shape and integrals across grids/domains.
Peak pressure and width need not vary monotonically; a changing spike can
indicate added resolution but is not evidence of accuracy by itself. Do not
impose a convergence tolerance without case-specific evidence. The Ciulli
Figure 13 dataset contains film thickness, not a pressure profile, and
therefore does not validate spike pressure.

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
is preserved. Its mathematical parity/performance evidence does not establish
device JFO support. A separately scoped hybrid CPU-preconditioned/CUDA-JVP
donor/JFO path is documented below; it does not provide a GPU-resident
preconditioner or imply physical validation. Optional GPU tests still skip
gracefully on hosts without an available device.

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

## 21. Correct active-lookahead candidate selection (2026-10-27)

Starting revision: 4f140886f9643de14f30656e1988734f9efcb020.
The archived failed factor-three history identifies a concrete logic defect,
independent of mesh accuracy: a better verified backup is restored only if
the final model fails. Thus a successful but much worse final model can
supersede a substantially better original-model step. In iteration zero,
the deferred alpha=.01359187449 candidate has merit 4.294238637e-5, while
the accepted final alpha=2.946659925e-10 candidate has merit 4.413395998e-5.
Repeated final steps eventually leave computed merit unchanged at roundoff.

The best policy compares the final trial with the lowest-merit retained
backup and marks a displaced acceptance as superseded. Original-directional
Armijo, exact film feasibility and original field/force criteria remain
mandatory. A trial must also change state and strictly decrease computed
merit; rounded equality is rejected. No field or pressure rescaling, clipping,
floor change, fluid-law change, extra active model or budget increase is used.
The explicit legacy policy preserves previous selection/roundoff behavior,
and lookahead-disabled warm continuation is unchanged.

Six small globalization regression cases reproduce the incorrect choice
and roundoff acceptance before the fix, verify the corrected choice and
rejection, and preserve the explicit legacy behavior. Existing independent
physical repaired-root parity, failed-model fallback, complete JVP, bordered
correction, Krylov and warm/load-continuation regressions remain applicable.
The failed partial is not reused as an accepted root: both numerical replays
start from the original audited 33x49 source and its unaccepted interpolation.

### Bounded replays and independent audits

The 65x97 replay closes 200 N with nine corrections and 18 linear models.
Pressure and film arrays are bitwise identical to the previously independently
audited root. The new saved-field/globalization audit passes as well; original
mass/force/film criteria are unchanged. No candidate supersession occurs on
this successful case, confirming its prior trajectory is retained.

The 97x145 replay begins with arrays identical to the archived legacy
initialization. It supersedes 16 worse accepted final trials and accepts 17
strictly merit-decreasing corrections before a rejected eighteenth correction.
There are 112 accurate linear models (max true error 2.181e-10). Independent
history recomputation verifies each accepted trial has the minimum merit of
the evaluated feasible Armijo candidates. The final four models have zero
feasible fraction-to-boundary steps, with the failure recorded as all feasible
Armijo/active recovery models exhausted. No correction or active-model limit
was increased. The legacy failure and its partial fields remain rejected.

| Failed 97x145 partial | Legacy | Best candidate |
| --- | ---: | ---: |
| Final original merit | 4.091609880e-5 | 2.756063378e-5 |
| Original max mass defect | .0643674331 | .0495270775 |
| Relative local mass defect | .440413535 | .337983675 |
| Relative global mass defect | .0294992174 | .0226399192 |
| Complementarity | .107004731 | .0941517332 |
| SI force [N] | 200 | 200 |
| Accepted root | no | no |

The new failed-partial audit independently recomputes the SI flux, exact
film/source relation, force and candidate histories; sampled direct source
integration differs by at most 3.389e-21 m. Source hashes match. No film
value from either failed 97x145 partial enters a mesh-convergence comparison.
Improved merit is not physical convergence. Strict decrease removes exact
rounded equality but does not guarantee a useful positive feasible step.

Evidence: `results/lookahead_selection_20261027/` in the original checkout
contains both new replays, the accepted-root audit, independently assessed
failed partial, exact source snapshot, source/input hashes and a SHA-256
manifest. The archived source in `results/recovered_200n_20261027/` remains
unchanged. A fresh replay uses:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src:. \
  python -m examples.run_load_refinement \
  --input-directory results/regenerated_load_33x49 \
  --output-directory results/best_candidate_refinement_new \
  --refinement-factor 3 --lookahead-selection best
```

Use `--lookahead-selection legacy` to reproduce the preceding policy;
use factor two for the accepted 65x97 comparison. Always use fresh output
directories. Generation summaries retain their original paths before the
archival copy. Existing field/force acceptance and audit commands are retained.

Next: independently identify the pressure/content lower-bound blockers in
the final zero-step models and benchmark targeted active pivots, including
cycling and roundoff cases. Keep both reference policies and original
acceptance criteria. Third-grid recovery, mesh/domain convergence and
physical film-height validation remain unresolved.

Final checks: `ruff check .` clean; full `pytest -q` 367 passed, 62 optional
device skips (310.29 s), with one BLAS/OpenMP thread. The accepted raw
centerline figure was inspected; persistent cell-scale variation is not
smoothed or treated as physical accuracy.

## 22. Targeted near-empty-content model pivots (2026-10-27)

Starting revision: 77510d6789647d4ec3d73303fdec9e968632ed40.
Diagnosis of the reconstructed rejected 97x145 partial locates the zero
step at physical (y,x) index (62,86), x=.158107141 mm, y=.237160711 mm.
Its P=.0167591019 and D=1.0000000001 correspond to theta at its negative
feasibility tolerance. The original direction has dD=.354948159, making
the affine feasible step zero. True correction error is 8.903e-11. Selecting
a full-film constraint only at that cell produces dD=-1.00000000028,
true error 1.843e-10 and a negative original merit slope. A feasible
7.792e-12 step passes strict original Armijo; a different nearby content
constraint then limits progress. This diagnostic is not an accepted root.
Complete states/directions, selected models and source hashes are retained.

The new recovery identifies the earliest affine pressure/content blocker
among near-empty-content cells (D>=1-1e-10) and selects D_new=0 at tied
upper-D blockers. It prioritizes that selected model before unrelated
full-step lower-bound violations. The true complete correction and original
nonsmooth directional Armijo remain required, along with exact film
feasibility and unchanged physical/force gates. No pressure, content, film,
load or property is projected or rescaled. The eight-model/30-correction
caps and seen-model cycling guard remain in force.

Five analytic bounded-complementarity cases establish exact feasible-root
recovery, reference-policy failure, simultaneous blockers, competing bounds,
nonempty-content exclusion and retained infeasible-cycle failure. Dense
corrections are independently checked in the toy problems. Existing full
JVP, bordered Schur/GMRES, load stepping and physical repaired-root parity
regressions remain applicable. `--no-content-bound-pivots` selects the
preceding best-candidate method; combine it with `--lookahead-selection
legacy` for the older final-model/roundoff policy. New outputs record both
settings and the interior flattened pivot indices.

### Accepted third-grid recovery and independent verification

The 65x97 replay closes 200 N with nine corrections/18 linear models and
no content-bound pivot proposals. Pressure and film arrays are bitwise
identical to the earlier accepted root, and its fresh independent audit passes.

The 97x145 replay now closes 200 N in 28 corrections and 141 linear models,
within the unchanged 30-correction/eight-model budgets. Nine content-bound
pivot proposals occur at iterations 14 and 19, involving four distinct
interior cells. Independent history checks verify strict original merit
decrease, true directional Armijo, one accepted trial per correction and
lowest verified candidate merit. Maximum true correction error is 9.754e-10.
The seed pressure/content/film arrays are bitwise identical to the preserved
failed factor-three initialization; neither failed partial is promoted to a
warm root. Full material laws and compliance remain active.

The separate saved-state audit independently recomputes SI flux/mass,
complementarity, exact full-source film, pressure edges, inlet content,
physical positivity and integrated force. Its original max mass defect is
2.643e-15, relative local balance 6.225e-15, relative global balance
5.079e-17 and complementarity 1.493e-28. Integrated force is 200 N at
printed precision. Center/peak direct source-cell integration differs from
FFT by at most 2.542e-21 m. Recorded benchmark source hashes match the
archived source snapshot; a later edit only clarified the recovery comment.

### Three accepted grids, unresolved mesh accuracy

All cases have the same finite physical +/-3a reservoir, SI fluid laws,
geometry and 200 N target. Grid order is (ny,nx).

| Grid | h0 [um] | Sampled minimum film [um] | Central film [um] | Peak pressure [MPa] |
| --- | ---: | ---: | ---: | ---: |
| 33x49 | -4.630558908 | .01960927772 | .3320824786 | 1725.253140 |
| 65x97 | -4.521772796 | .1629378195 | .4429945326 | 1741.100421 |
| 97x145 | -4.498402416 | .1875122800 | .3961280370 | 1716.683723 |

At the 1.5 refinement ratio between the latter two grids, minimum film
changes +15.082%, central film -10.579% and peak pressure -1.402%. The
combined 1% scalar mesh gate remains false. Sampled minimum locations move
and central values depend on the persistent grid-scale pressure pattern.
No asymptotic order or full-field convergence is inferred from these three
scalar comparisons. The raw three-grid centerlines and independent fine-grid
pressure/film fields were inspected without smoothing. Force/equation closure
does not establish a physically resolved pressure spike, mesh accuracy,
infinite-domain validity or agreement with a film-height correlation.

Evidence is preserved in `results/content_bound_pivots_20261027/` in the
original project checkout: the initial blocker diagnostic with full states/
directions, both accepted replays, separate saved-field audits, three-grid
comparison with hashes and minimum locations, raw figures, exact source
snapshot, reproduction scripts and SHA-256 manifests. The initial incomplete
checkpoint and earlier failed baselines remain distinct and unchanged.
A fresh factor-three run uses:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src:. \
  python -m examples.run_load_refinement \
  --input-directory results/regenerated_load_33x49 \
  --output-directory results/content_pivot_refinement_new \
  --refinement-factor 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src:. \
  python -m examples.audit_load_directed \
  --input-directory results/content_pivot_refinement_new \
  --output-directory results/content_pivot_refinement_audit_new
```

Use `--no-content-bound-pivots` for the preceding best-candidate method.
Also select `--lookahead-selection legacy` for the older final-candidate
policy, or combine `--no-content-bound-pivots --no-active-lookahead` for
the no-lookahead reference. Keep fresh output directories and the original
budgets. Generation summaries retain paths from before the archival copy.

Next: independent conservative-transport/discrete-stability benchmarks at
the accepted high-load roots, followed by separate fixed-spacing reservoir
sensitivity. Preserve the current stencil as a reference, benchmark any
alternative before adoption, and do not infer physical film accuracy while
pressure oscillations and mesh error remain unresolved.

Final checks: `ruff check .` clean; full `pytest -q` 372 passed, 62 optional
device skips (307.78 s), with BLAS/OpenMP threads set to one. No GPU execution
or physical EHL validation is claimed by this CPU model-selection milestone.

## 23. Recovered Windows archives and loaded-root directional transport (2026-10-28)

Starting revision: 5c9d7239e16a75f6be2f99e954468585648e707f. The benchmark
definition and numerical code survived in Git, but the October 27 raw root
archives were not present on this machine. This increment regenerates them
from the registered Circular200N physics; it does not reconstruct pressure
from reported film values, a correlation or a failed partial. Production
equations, acceptance criteria, sparse preconditioners, iteration/node caps
and the two older active-recovery policies remain unchanged.

### Recovery and independent field/history audits

`examples.regenerate_200n` cold-starts the 33x49 h0=0.5 um prescribed-offset
source, then uses the documented local-preconditioned force targets
2/4/8/10/20/40/80/120/160/200 N. Each target retains at most twelve load
attempts and the original 30-correction/eight-active-model solver budgets.
Every accepted force step and its full history are checkpointed immediately.
Failures and any unaccepted partial remain separately named. Optional resume
reads an accepted source, independently re-audits it and writes a new output
directory; an already completed 200 N source is used directly, not relabeled
as a new resumed solve. Output directories are never reused.

The cold path reproduces 44 accepted load steps. The two existing nested
repairs reproduce nine corrections/18 models on 65x97 and 28 corrections/
141 models on 97x145, all within their unchanged budgets. Independent
`examples.audit_load_directed` passes for all 46 accepted states: original SI
mass, local/global balance, complementarity, pressure/content boundaries,
positive exact total film, full-source displacement, integrated force and
original directional Armijo/linear-correction histories. Source production
files were not changed during these runs.

| Grid (ny,nx) | h0 [um] | Sampled minimum film [um] | Central film [um] | Peak pressure [MPa] |
| --- | ---: | ---: | ---: | ---: |
| 33x49 | -4.6305589082 | .01960927772 | .3320824786 | 1725.253140 |
| 65x97 | -4.5217727964 | .1629378195 | .4429945326 | 1741.100421 |
| 97x145 | -4.4984024160 | .1875122800 | .3961280370 | 1716.683723 |

These match the previously reported scalar roots; bytewise equality with
the missing original arrays is not claimed. Across the recovered final
roots, maximum original scaled mass defect is 3.186e-15, maximum relative
local mass defect 1.620e-14 and maximum complementarity 1.487e-28. Sampled
direct source-cell integration differs from FFT by at most 4.236e-21 m.
All force targets close 200 N. Maximum true correction error across the
44-step source is 1.749e-9 and across the repaired 97x145 history 9.757e-10.
Cross-platform last-bit differences in those diagnostics are retained.

Resume was tested separately by constructing an incomplete checkpoint from
the accepted 186.375 N root (no failed partial). Four further accepted steps
close 200 N and pass the separate archive/globalization audit. Final pressure
differs from the cold-path endpoint by 6.650e-12 relative L2 and offset by
1.263e-18 m. The source root/arrays are unchanged. These extra resumed roots
are separate from the 46-state cold/refinement audit count above.

The 65x97->97x145 minimum/central/peak changes remain +15.08%, -10.58%
and -1.402%. The combined 1% mesh gate is false. Fixed physical bounds are
still +/-3a in both axes; this regeneration is not a reservoir-convergence
study, a physical pressure-spike validation or an experimental film comparison.

### Independent complete directional mass actions

`transport_audit.mass_direction_components` independently implements the
seven SI contributions described in PHYSICS.md: pressure, density,
viscosity and film diffusion; density, film and content advection. Face
arithmetic averages, established content upwinding and original units are
preserved. Tests compare their sum with physical-flux finite differences
when pressure, film and content all change; an analytical uniform
alternating-pressure symbol separately checks diffusion and the density-
advection null response. No production residual/JVP function is reused to
assemble the decomposition.

For each newly audited 200 N root, five pressure probes use alternating x,
y, xy and smooth x/y directions. Peak direction is 1 MPa, pressure edges
and force direction are zero, and support is restricted to p>0.1*pmax with
the fixed physical exp(-2*((x/a)^4+(y/a)^4)) window. Content and rigid-offset
directions are zero. Elastic film directions retain the full-domain source
convolution, including effects outside the pressure-probe support.

The summed SI divergence is scaled with the original flux/mass scales and
compared with the complete analytic material/elastic JVP, including unchanged
complementarity and force rows. Its relative discrepancy is below 1.62e-14.
Centered residual differences at factors .01/.001/.0001 retain both-sign
pressure/film feasibility and all measured errors. The largest best-step
relative complete-JVP discrepancy is 9.406e-9. These are nonzero loaded,
nonuniform, full-material/full-elastic actions, rather than the earlier
uniform frozen-film observation alone.

| Grid | Alternating-x film / frozen-film material action norm | Smooth-x film / frozen-film material action norm |
| --- | ---: | ---: |
| 33x49 | .888662 | 5.154743 |
| 65x97 | .249511 | 2.803467 |
| 97x145 | .109440 | 1.809639 |

Frozen-film material action here retains pressure diffusion, density/viscosity
pressure derivatives and density transport. The separate film action is
elastic diffusion plus elastic entrainment. These ratios quantify static
directional contributions, not eigenvalues, energy growth, linear-solver
condition numbers or a temporal-stability test. All tested complete actions
are nonzero; the density-only alternating null is not the whole loaded
Jacobian. The result does not establish why the production pressure varies.
Feasible active-front/tie directions are not covered by these supported
pressure probes and remain a separate next experiment.

### Raw alternating projections and inspected figures

No filtering, smoothing or detrending is applied to the production arrays.
The dimensionless diagnostic is abs(sum(window*p*parity))/sum(abs(window*p)).
It is a projection, not a peak-to-peak oscillation amplitude. The fixed
physical window is identical across grids. Both full 2-D and sampled y=0
centerline values are reported; transverse phase cancellation can strongly
reduce the global number. An independently sampled dry Hertz pressure
provides a finite-sampling reference, not an EHL solution.

| Grid | Global x projection | Centerline x projection | Sampled Hertz centerline x |
| --- | ---: | ---: | ---: |
| 33x49 | .130938 | .112640 | .00611515 |
| 65x97 | .0167268 | .335179 | .00198702 |
| 97x145 | .0950790 | .300669 | .00104169 |

Thus the smaller intermediate-grid global projection does not establish
disappearance of the pressure pattern. Raw pressure and total-film centerlines,
including a contact-region film zoom, were inspected on the three grids.
They retain visible alternating pressure and film variation. Neither that
observation nor the sampled Hertz comparison establishes continuum
instability, causality or the correct resolved EHL spike.

### Secondary CUDA elasticity parity

The optional `--cuda-elasticity-check` reuses the existing float64 cached
cell-integrated elastic operator on the RTX A2000. It checks deformation
of each accepted GPa pressure field and the five signed probe directions.
Maximum root displacement discrepancy is 2.542e-21 m and maximum direction
discrepancy 1.448e-24 m. The largest diagnostic allocated/reserved peaks are
6,347,264/8,388,608 bytes. These are component parity checks outside the
production solve; no material/JFO GPU execution or speedup is claimed.
Omitting the flag keeps the audit on NumPy/SciPy and does not require torch.

### Reproduction and remaining work

Use one BLAS/OpenMP thread, a noninteractive plotting backend and new paths:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python -m examples.regenerate_200n --output-directory results/source_200n_new
python -m examples.run_load_refinement --input-directory results/source_200n_new \
  --output-directory results/refined_200n_new --refinement-factor 2
python -m examples.run_load_refinement --input-directory results/source_200n_new \
  --output-directory results/fine_200n_new --refinement-factor 3
python -m examples.audit_load_directed --input-directory results/source_200n_new \
  --output-directory results/source_200n_audit_new
python -m examples.audit_load_directed --input-directory results/refined_200n_new \
  --output-directory results/refined_200n_audit_new
python -m examples.audit_load_directed --input-directory results/fine_200n_new \
  --output-directory results/fine_200n_audit_new
python -m examples.audit_loaded_transport --input-directories results/source_200n_new \
  results/refined_200n_new results/fine_200n_new --cuda-elasticity-check \
  --output-directory results/loaded_transport_new
```

Equivalent PowerShell thread settings are in README. The retained source,
refinements and separate audits are in `results/recovered_200n_20261028*`;
diagnostic directions/component arrays, source hashes, exact inputs and raw
figures are in `results/loaded_transport_20261028_final/`. These generated
archives remain outside Git. Cold recovery stores production source snapshots
and a SHA-256 manifest, and runtime records identify source-file hashes.
Previously existing outputs and pre-existing packaging changes are preserved.

Next: compare resolved pressure wavelengths and feasible active-front
directions before testing an alternative conservative flux. Then separate
fixed-spacing inlet/outlet/transverse reservoir sensitivity from mesh error.
Do not use a steady directional Jacobian as a time-stability proof, alter
material laws to fit the supplied film estimates, smooth away the pressure
pattern, or raise the 20,000-node cap without a verified resource/scaling plan.

Verification: `ruff check .` clean; complete Windows CPU/CUDA `pytest -q`
439 passed, one skipped (237.00 s). The wider-longdouble correction diagnostic
skips because Windows longdouble has no additional mantissa precision. Six
new independent diagnostic regressions pass; physical/model accuracy remains open.
The loaded-root audit additionally passes with torch imports deliberately blocked.

The local `results/recovery_20261028_verified.zip` packages the three cold/
refined roots and audits, loaded diagnostics, simulated resume check and exact
source/test/documentation snapshot at commit
`47f70c2ae19b64d5cc065846a3ce4698347bacb2`. All 343 payload files were
independently rehashed from the ZIP against the included manifest, and ZIP CRC
verification passes. Size: 7,887,923 bytes. Archive SHA-256:
`10c55aab7b0f963d9713db798f57fc80bfa22ecf40924510c9c5a9abb39beaf6`.
The ZIP and raw fields are not in Git; their identity and regeneration
instructions are. This is verified local packaging, not confirmation of
an off-machine backup or OneDrive synchronization.

## 24. Resolved pressure carriers, active fronts and finite reservoirs (2026-10-28)

The recovered circular 200 N case remains unchanged: 15 mm radii,
113 GPa contact modulus, 2 m/s entrainment, 0.04 Pa s reference viscosity,
and G=2497.3. The three accepted roots are 33x49, 65x97 and 97x145
arrays in (ny,nx) order, with entrainment along +x. No production residual,
Jacobian, constitutive law, field gate, force gate or solver budget changed.

`examples.audit_transport_wavelengths` checks windowed sine/cosine pressure
directions at fixed physical wavelengths a, a/2 and a/4. Carriers with fewer
than four cells per wavelength are explicitly omitted. Separate two-cell
Nyquist directions retain the earlier alternating-mode reference. Pressure
directions have zero discrete force and zero boundary values. Their support
depends on the accepted pressure field; they are not operator eigenvectors.
The independently decomposed seven-term mass derivative agrees with the
complete JVP to 8.94e-15 relative error across all 43 probes. The largest
best feasible finite-difference error is 9.66e-9. Centered differences cover
carriers; second-order one-sided differences cover active-front directions.

Each physical root has a transverse-center +x rupture pair. Force-free
pressure transfers and content loss/gain probes preserve field bounds and
test the exact one-sided derivative of min(P,D). Near ties are not rounded
into exact ties. No reformation or exact tie occurs in these roots; reflected
and exact-tie unit tests provide separate coverage. Front-only best
finite-difference error is at most 1.09e-10. Trial field feasibility is not
complementarity convergence: these perturbed fields are not new roots.

The x-Nyquist scaled mass response is respectively .169, .225 and .371
times the corresponding windowed a-wavelength cosine response. These static
directional gains depend on state coordinates and the chosen window. They
do not establish temporal instability, a Jacobian spectrum, a convergence
order or a cause of the pressure pattern. Raw directions, components,
differences and figures are retained in
`results/wavelength_front_20261028_final`.

`examples.run_material_reservoir` separately extends the 65x97 source at
fixed dx=a/16 and dy=3a/32. Every original coordinate is preserved exactly.
Added pressure is initially zero; edge content is extrapolated as an
explicitly unaccepted guess. Full elasticity is recomputed at the original
signed offset. Tests independently verify force and shared-film preservation
and repaired-root agreement with a cold solve. The existing force Newton
engine repairs each extension under its original limits and acceptance gates.

| Added cells | Extended boundary | Target (ny,nx) | Newton corrections | Minimum film change | Central film change | Peak pressure change |
|---|---|---|---:|---:|---:|---:|
| 4 | inlet | 65x101 | 3 | -0.032445% | +0.095703% | +0.037536% |
| 4 | outlet | 65x101 | 1 | roundoff | roundoff | roundoff |
| 4 per side | transverse | 73x97 | 2 | +0.005928% | -0.007962% | +0.003725% |
| 8 | inlet | 65x105 | 4 | -0.058302% | +0.156815% | +0.062652% |
| 8 | outlet | 65x105 | 1 | roundoff | roundoff | roundoff |
| 8 per side | transverse | 81x97 | 4 | +0.005500% | -0.013514% | +0.005856% |

All six independently reopened roots close 200 N to printed float64
precision. Maximum scaled mass defect is 4.40e-11 (gate 1e-8), maximum
relative local mass defect 2.94e-11 (gate 1e-10), maximum complementarity
defect 1.55e-20, and maximum independently verified true linear residual
4.19e-10. Saved histories pass both line-search checks and the original
directional Armijo test. Direct source-cell elasticity checks at the center
and peak pass. Source file hashes remain unchanged. The largest shared-node
pressure L2 change is 0.4750%; maximum film difference is 2.335 nm, both for
the eight-cell inlet extension. Raw shared-node differences are plotted
without filtering or resampling.

Individual outward mass flows still depend substantially on reservoir extent.
The baseline west/east/each transverse flows are approximately
(-3.2753e-5, 1.3674e-5, 9.5396e-6) kg/s. Eight added inlet cells give
(-4.1612e-5, 1.3681e-5, 1.3965e-5); eight added transverse cells per side
give (-5.2997e-5, 3.0720e-5, 1.1138e-5). Small inner-film sensitivity over
this tested range does not prove infinite-domain validity. The earlier
15.08% minimum-film mesh change remains much larger; physical film-height
accuracy and the mesh-dependent pressure pattern remain unresolved.

Evidence directories are `results/material_reservoir_20261028` and
`results/material_reservoir_20261028_cells8`, with independent
`_audit_final` directories. README gives reproducible commands. Generated
data stay outside Git. GPU work is secondary; this material/JFO path remains
CPU-only and retains the existing 20,000-node cap.

Verification: Ruff clean; complete Windows CPU/CUDA suite **453 passed,
one skipped in 237.98 s**. The existing wider-longdouble test skips because
Windows longdouble provides no additional precision. Fourteen new CPU
regressions pass, including independent discrete Fourier symbols, exact
one-sided ties, reflected reformation, integer-input dtype handling and
same-spacing reservoir repair. The two new test modules also pass with
torch imports deliberately blocked. No full physical EHL validation is claimed.

## 25. Isolated donor transport and existing solver resource audit (2026-10-28)

Starting from 6ff85dc, `transport_candidate` provides an independently tested
conservative donor-capacity flux and its analytic smooth direction. It is
not a production solver option. The original centered-capacity/upstream-content
flux, material laws, boundaries, Newton/globalization, physical gates and
20,000-node cap remain unchanged. README and PHYSICS give equations and
reproduction instructions.

Eleven new tests cover independently evaluated scalar x/y faces, nonuniform
control-volume telescoping, material/constant-law JVPs, constant-capacity
reference parity, stationary p=0 and h*theta=constant, invalid inputs, and
manufactured-source orders. Donor density-advection symbols are independently
checked at 2/4/8/16 cells per wavelength, including their real numerical-
diffusion term and nonzero Nyquist response. These are not full EHL stability
results. The source-driven manufactured fields need not satisfy complementarity.

On 33/65/129/257 square manufactured grids, the finest full-film divergence
relative L2 errors are 1.00405e-4 for the reference and 1.27170e-2 for the
donor. Last observed orders are 1.99986 and 1.00559. With varying content,
errors are .00491706 and .01265527, orders 1.00070 and 1.00569. Thus a
first-order donor cannot be presented as an accuracy improvement over the
second-order full-film reference merely because it adds alternating-mode
response. It also does not establish why the loaded pressure pattern occurs.

At the three accepted 200 N roots, 43 full-elastic carrier/front flux
directions pass independent residual differences: maximum best error
8.01e-9. Front probes use positive, second-order forward samples and preserve
original field bounds; perturbed states are not roots. Exact candidate-minus-
reference face-flux identities agree to 5.87e-14 relative. The unchanged
accepted fields have candidate scaled mass maxima .00372048, .00466458 and
.00575769, far above the original 1e-8 gate. No candidate nonlinear solution
or physical improvement has been demonstrated.

The same benchmark measures the current CPU local sparse preconditioner and
existing residual/JVP paths on these saved roots. Windows Python 3.12.9,
NumPy 2.5.4, SciPy 1.18.1, one BLAS/OpenMP thread; five timed evaluations
after one excluded warm-up. Assembly/factorization are single measurements.
Timings vary between runs and do not include a nonlinear solve.

| (ny,nx) | Reference mass residual ms | Candidate mass residual ms | Existing full JVP ms | Existing film direction ms | Existing cached fast elasticity ms |
|---|---:|---:|---:|---:|---:|
| 33x49 | .322 | .350 | 6.315 | 2.971 | .664 |
| 65x97 | .460 | .794 | 12.654 | 6.092 | 1.367 |
| 97x145 | .800 | 1.212 | 38.852 | 21.213 | 4.184 |

Candidate residual evaluation was slower on all three meshes in this run.
The last two columns are different isolated calls: the film direction also
constructs the supplied pressure direction, while the cached operator takes
a ready pressure field. Their ratio is not a solver speedup. Cached ordinary-
padding and the already existing FFT-friendly ElasticityOperator were both
checked against the accepted deformation. The latter agrees to 4.89e-16
relative and uses FFT shapes 98x147, 196x294 and 294x440; this benchmark does
not activate it in the material solver.

| (ny,nx) | Block LU payload MiB | Bordered LU payload MiB | 81 state vectors MiB | Block/bordered factorization ms |
|---|---:|---:|---:|---:|
| 33x49 | .816 | 1.513 | 1.801 | 5.32 / 9.71 |
| 65x97 | 4.931 | 12.231 | 7.398 | 23.45 / 69.58 |
| 97x145 | 13.981 | 60.161 | 16.791 | 80.51 / 528.20 |

The block is the primary pressure/content preconditioner; the bordered factor
is the fallback. Independently recomputed factor-solve relative residuals
are below 2.75e-16. Sparse data/index/pointer payloads and generic restart-80
basis estimates exclude temporary workspaces, additional vectors, Python,
FFT buffers and allocator peaks. They are neither peak process memory nor
CUDA VRAM measurements. Finest dense full-Jacobian storage would be 5.50 GiB
and is not allocated. These measurements support further profiling, not an
unverified cap increase or extrapolated larger-grid convergence.
The material Newton driver explicitly uses restart 60, rather than the
generic correction helper's default 80; its 61-vector finest basis estimate
is 12.65 MiB. The table's 81-vector estimate is a conservative larger budget.

Evidence: `results/transport_candidate_20261028_final_verified`, including
source hashes, analytical-source errors, individual direction differences,
all timing samples, storage estimates and raw divergence arrays. Exploratory
and incomplete runs remain separately preserved. Verification: Ruff clean;
complete CPU/CUDA pytest **464 passed, one skipped in 207.86 s**. The existing
Windows wider-longdouble test skips; no tolerance was weakened. Next: verify
cached CPU elasticity inside the bordered workflow and measure complete
solve/peak memory before expanding grids; benchmark a higher-order conservative
transport alternative separately. Material/JFO GPU support remains open.

## 26. Cached CPU elasticity in bordered material Newton (2026-10-28)

Starting at 69b6a54, an explicit `elasticity_mode="cached"` option reuses the
existing NumPy ElasticityOperator in material force Newton, force continuation
and unaccepted-guess repair. Reference execution remains the default. The
same analytic integrated kernel is cached once per system with FFT-friendly
linear-convolution padding. No constitutive, transport, boundary, sparse
preconditioner, Newton, globalization, tolerance or node-cap change is made.

Six new regressions independently check complete bordered JVPs against finite
differences, a dense reference Jacobian/correction, Schur fallback, strict
film rejection, original final mass/force gates and directional Armijo checks,
active recovery with rejected trials, selected-model failure with safeguarded
backup, failed Krylov diagnostics and continuation retry seed/mode preservation.
Cached correction views reuse the same operator. Default reference code also
passes the full existing suite. Both modes remain CPU NumPy/SciPy execution.

The reproducible `examples.benchmark_cached_material` uses the same three
accepted 200 N source meshes. Each initialization preserves source pressure
and content but opens the signed offset by 0.01 href (48.98 nm), explicitly
creating an unaccepted guess. Both modes repair this identical guess at
the same 200 N force, using the existing local preconditioner and original
30-correction/eight-model/10-Krylov-cycle budgets. Each mode runs in its own
fresh process, warms once, then performs three measured repairs. Median
times include system/operator construction and the full nonlinear repair;
independent final audits run outside timing. This is a warm local repair
benchmark, not cold recovery or proof over a wider load/coupling range.

Every warm and measured root passes independent reference convolution/film,
sampled direct source-cell deformation, trapezoidal force, original SI mass/
complementarity/field gates, complete true-correction residual and saved
directional Armijo history checks. No audit is relaxed for cached mode.
Source hashes remain unchanged. Operating-system resident and pagefile
high-water marks cover the entire worker lifetime, including startup,
source loading, warm-up, all repeated solves and audits. They are not
incremental solve-only peaks, GPU allocations or a bound for larger meshes.

Windows results (Python 3.12.9, NumPy 2.5.4, SciPy 1.18.1, one BLAS/OpenMP
thread), three repetitions per mode after one warm-up:

| (ny,nx) | Reference median s | Cached median s | Reference/cached | Newton corrections ref/cached | Total Krylov iterations ref/cached | Peak resident MiB ref/cached |
|---|---:|---:|---:|---:|---:|---:|
| 33x49 | 1.3672 | .4664 | 2.93x | 4 / 4 | 181 / 181 | 129.18 / 129.32 |
| 65x97 | 4.3196 | 1.7702 | 2.44x | 4 / 4 | 231 / 230 | 140.81 / 141.15 |
| 97x145 | 17.4404 | 7.6672 | 2.27x | 5 / 5 | 332 / 332 | 162.32 / 164.27 |

Cached mode is faster on all tested meshes for this specific complete repair
workload. Finest worker-lifetime peak pagefile (commit) counter is 269.58/271.54 MiB
for reference/cached. The slight memory increase is consistent with the
retained spectrum/padding; no scaling law is inferred from three meshes.
Final relative pressure L2 differences are 2.91e-15, 6.04e-15 and 9.93e-15.
Maximum total-film difference is 6.78e-21 m; maximum h0 difference is
8.48e-22 m. Newton counts agree, while one Krylov iteration differs on the
middle mesh. These differences are not hidden by forcing identical histories.
Maximum final original scaled mass defect is below 1.85e-13; maximum audited
true correction residual is below 8.71e-10, within the unchanged 5e-9 gate.
Both paths retain the source's unresolved mesh-dependent pressure pattern.

Evidence: `results/cached_material_20261028_repeated`, six isolated worker
directories with final fields/histories/source hashes and all 18 measured
repair audits and timing samples. Each worker also passes an excluded warm-up audit. The
earlier single-measurement exploratory run remains separately retained.

Verification: Ruff clean; complete CPU/CUDA pytest **470 passed, one skipped
in 208.07 s**. The skip is the existing Windows wider-longdouble diagnostic.
All 17 new candidate/cached regressions also pass with torch imports blocked
(4.57 s). Material/JFO CUDA solving, mesh convergence, larger reservoirs and
physical film accuracy remain open.

The local `results/transport_cached_20261028_verified.zip` retains the new
carrier/front and reservoir evidence, isolated transport/resource benchmark,
repeated cached repairs, prior verified recovery ZIP and exact source snapshot
at `6336876c19a567d6b983f41c5b82a50028c902ae`. All 199 payload files pass
SHA-256 rehashing from the archive and ZIP CRC verification. Size: 19,502,445
bytes; archive SHA-256:
`fb4f12ec66659b30e25ba8e8b3f5d8aae9151506339a7c570876cd3aa5c6a035`.
Generated data remain Git-ignored. This is verified local packaging;
off-machine backup or OneDrive synchronization has not been confirmed.

## 27. Fourth-mesh resource trial and rejected nonlinear refinements (2026-10-28)

The default 20,000-node budget is preserved. Explicit
`maximum_physical_nodes=30000` permits the 24,897-node 129x193 case and is
propagated through registered geometry, transfer, force systems and repair.
There is no unbounded setting. Independent regression tests check default
rejection, explicit acceptance and rejection beyond 30,000 nodes. Before
the larger repair, the runner constructs the original local sparse block,
checks its factor true residual and requires OS peak resident memory below
1 GiB. The 61-vector Krylov basis estimate is 22.58 MiB. This is a resource
allowance, not a change in physical or nonlinear acceptance.

Two independent nested initializations retain the same 200 N force,
registered material case and +/-3a domain. Both pass preflight; factor
relative residuals are 3.33e-13 and 3.38e-13. Neither passes the unchanged
30-correction budget. All failed fields, histories and diagnostics remain
rejected. Independent reopened-field audits recompute original SI mass,
reference convolution, force and complementarity:

| Source mesh | Newton corrections | Models | Krylov iterations | Observed wall time s | Peak resident MiB | Final scaled mass defect | Final complementarity defect |
|---|---:|---:|---:|---:|---:|---:|---:|
| 65x97, factor 2 | 30 | 118 | 8,233 | 441.86 | 198.31 | .00284080 | 9.56e-14 |
| 33x49, factor 4 | 30 | 211 | 14,830 | 791.03 | 199.52 | .07918680 | .111184 |

Force errors are <=1.12e-16 relative, but the original scaled mass gate is
1e-8 and local/global gates also fail. Reference-film reconstruction errors
are <=6.78e-21 m. Force closure and small FFT differences cannot promote
these partials into solutions. Wall times are observed bounded-run times,
not isolated speedup benchmarks; verification work also ran during these trials.
The attempts establish neither nonexistence of a fine root nor mesh convergence.

The independent accepted-grid comparison uses raw fields and exact shared
physical nodes. From 33x49 to 65x97, sampled shared-node pressure L2 change
is 31.44%; from 65x97 to 97x145, it is 29.90% on their common 33x49 nodes.
The latter peak-pressure change is only -1.40%, while minimum/central film
changes are +15.08%/-10.58%. Minimum locations move; y-reflection can switch
which equal minimum is reported. The raw centerlines retain the alternating
pattern. No filtering or new transport stencil is applied.

An unequal-spacing three-point minimum-film fit gives apparent p=2.128,
but is explicitly unqualified: one very coarse mesh and moving extrema do
not establish an asymptotic range. Central film and peak pressure are
nonmonotone over the three accepted grids and receive no fit. A fourth
converged result is unavailable, so no reliable continuum error or new
film-height prediction is claimed. Mesh effects are expected; the missing
evidence is convergence of the full fields and quantities of interest.

Evidence: `results/refinement129_20261028` and
`results/refinement129_from33_20261028`, their separate `_rejected_audit`
directories, and `results/mesh_comparison_20261028_final`. Source roots are
preserved. The earlier section 26 timing experiment used 0.01 href; its
previous parenthetical conversion of 5 nm was a documentation error, now
corrected to 48.98 nm (href=4.897541492 um). Its code and measured results
are unchanged.

## 28. CPU geometric multigrid and mathematical bordered integration (2026-10-28)

Implemented fixed linear symmetric geometric V-cycles for the original
prescribed/frozen-film SPD Reynolds diffusion matrix. Bilinear P, R=P.T,
Galerkin coarse matrices, bounded damped Jacobi and small exact Cholesky
are independently verified. Default CG is unchanged; optional
`preconditioner="multigrid"` independently checks the true residual and
does not fall back silently. Tests also verify independent analytic-source
second-order discretization accuracy, rather than only algebraic inversion.

Windows Python 3.12.9, NumPy 2.5.4, SciPy 1.18.1, one BLAS/OpenMP thread.
Smooth positive film H=1+0.4X+0.2 sin(pi Y), unit square, random fixed-seed
RHS. Three solves per mode, one excluded V-cycle initialization, rtol=1e-10.
Setup includes the fine matrix and hierarchy and is reported separately:

| Grid | CG iterations | MG-CG iterations | CG median s | MG-CG median s | Solve-only speedup | MG setup s |
|---|---:|---:|---:|---:|---:|---:|
| 33x33 | 155 | 8 | .003702 | .001868 | 1.98x | .01040 |
| 65x65 | 323 | 8 | .012186 | .003554 | 3.43x | .02137 |
| 129x129 | 660 | 8 | .111485 | .013512 | 8.25x | .09475 |
| 257x257 | 1,335 | 8 | .823727 | .048066 | 17.14x | .66701 |
| 513x513 | 2,660 | 8 | 15.10135 | .374907 | 40.28x | 2.80949 |

Every listed solve passes the independently recomputed original relative
residual <=1e-10. Operator complexity is 1.47..1.59. These are linear solve
speedups, not coupled EHL speedups. Setup dominates small one-off solves;
including it reverses the apparent small-grid benefit. No universal
crossover grid is established.

Frozen material-mobility surrogates from the accepted 65x97 and 97x145 roots
span approximately 5.43e-14..547 and 7.38e-14..548 in dimensionless mobility.
Unpreconditioned CG reaches the fixed 500-iteration cap (relative residuals
19.83/18.36); MG-CG passes in 22/21 iterations. A speedup ratio is not valid
when the reference fails, and the benchmark marks it accordingly. Separate
known-pressure tests use symmetric diagonal equilibration at rtol=1e-12;
forward pressure errors are 2.13e-9/1.31e-9 and original true residuals
2.40e-13/7.12e-13. No material/elastic derivatives or JFO constraints enter
these SPD surrogate solves.

After these tests, optional MG was integrated into the signed full-film
mathematical bordered Newton pressure preconditioner. The inverse sign and
residual scaling match the original Jacobian; the original nonsymmetric
GMRES, Schur recovery, inexact forcing, dual line searches, continuation and
stopping rules are retained. Ten added MG configurations check complete
bordered corrections, Schur fallback, failed Krylov diagnostics, zero/weak/
moderate compliance, strict film feasibility, rejected steps, Armijo retry,
load-step retry and explicit MG setup failure without the legacy ILU fallback.

The moderate signed mathematical benchmark uses the existing reference
parameters (Rx=.01 m, Ry=.02 m, E*=1e7 Pa, href=20 um), not the 200 N material
case. Three full Newton solves per mode follow an excluded warm-up; times
include construction and all corrections. Independently recomputed final
Reynolds/load checks use 1e-9, with no cavitation or pressure clipping:

| (ny,nx) | ILU median s | MG median s | ILU/MG | Newton corrections both | Relative pressure difference |
|---|---:|---:|---:|---:|---:|
| 13x17 | .04583 | .05516 | .83x | 4 | 1.28e-10 |
| 25x33 | .42519 | .60581 | .70x | 4 | 1.46e-10 |
| 49x65 | 2.51847 | 2.28394 | 1.10x | 4 | 2.44e-10 |

MG is slower on the first two grids and modestly faster on the third.
Default ILU is retained. The signed solution can contain tensile pressure;
this remains a mathematical reference, not physical EHL validation.

Evidence: `results/multigrid_20261028_final` and
`results/multigrid_bordered_20261028_verified`. Full material/JFO multigrid,
nonlinear FAS, GPU multigrid and a converged 129x193 material result remain
unfinished. The next integration needs a separately verified pressure/
content block preconditioner, preserving active constraints and the load
border; the new SPD operator cannot simply replace that entire system.

Final verification: `ruff check .` clean; complete CPU/CUDA `pytest -q`
**490 passed, one skipped in 243.33 s**. The skip is the existing Windows
wider-longdouble diagnostic. All 65 selected refinement-resource, multigrid
and monolithic CPU tests also pass with torch imports deliberately blocked
(9.97 s). No tolerances were weakened. Pre-existing packaging modifications
are preserved and generated evidence remains outside Git.

Local evidence archive: `results/refinement_multigrid_20261028_verified.zip`,
23,354,856 bytes, 171 payload files independently rehashed from the ZIP,
with ZIP CRC verification passed. It includes both rejected mesh trials
explicitly as rejected evidence, independent audits, mesh comparisons, MG
benchmarks, the prior verified archive and exact source snapshot at
`077367498c4d4af7864fd26c2b1e72b562c70b42`. Archive SHA-256:
`85f52199152f76c69a95ed87a33d7d31c5953080caa23767973be819774e89a9`.
Raw data remain Git-ignored; no off-machine backup or synchronization is asserted.

## 29. Diagnostic material/JFO block MG foundation (2026-10-28)

Based on `1ab27ba`, adds a frozen-active-set pressure/content inverse as an
isolated CPU diagnostic, not a production Newton option. Physical definitions,
normalizations and the omitted mass coupling are specified in PHYSICS.md.

`tests/test_property_block_multigrid.py` independently verifies:

- Positive masked diffusion and exact Galerkin identities on 17x25 grids
  with full, cavitated, front and disconnected active sets; known-vector
  recovery with independently measured residuals.
- Fixed inverse linearity and exact complementarity rows for arbitrary RHS;
  exact cavitated mass rows and a dense lower-block surrogate comparison.
  A 7x9 case makes the pressure coarse solve exact and verifies the complete
  surrogate inverse against a dense solve.
- Full elastic/material JVP differences at a smooth state and complete
  load-border correction on a 9x13 pressure-dependent case, using both
  true-operator Schur reconstruction and full bordered GMRES. True linear
  residuals must be below 5e-9; corrections are compared with a separately
  assembled dense Jacobian solution.
- Explicit failures for invalid masks/RHS, unsupported material/local tangent
  modes, zero cavitated transport diagonal and nontriangular transport.

Linux CPU verification command uses the existing virtual environment:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/pytest -q`.
The complete run reports 440 passed, 62 optional backend/GPU skips in 332.47 s;
its collection preceded the last two transport-rejection tests. The final
prototype run reports all 13 passed in 19.42 s. The combined MG/prototype
run before those final two cases reports 19 passed in 22.20 s.
Repository-wide `.venv/bin/ruff check .` and `git diff --check` pass.

No production line search, retry, Krylov/Schur fallback, stencil, acceptance
criterion, nonlinear budget or node cap changes. Existing production
regressions pass, but they do not validate nonlinear use of the new prototype.
Next evidence must compare setup/action cost, memory and true Krylov work
against sparse preconditioners on loaded accepted/rejected states. Nonlinear
material/JFO MG, the unresolved 129x193 mesh and physical film accuracy remain
open; no performance or convergence improvement is claimed here.


## 30. Bounded loaded-state block-preconditioner comparison (2026-10-28)

Starting at `1432462`, implemented `examples.benchmark_property_block` and
independent regression tests, leaving all production numerical paths unchanged.
The three accepted 200 N roots and one rejected 97x145 partial come from
preserved commit `a4bee767fe01fc79793c20aec70ae359ea5d13c3`; exact paths and
reproduction command are in ROADMAP.md. The rejected partial's independently
recomputed scaled mass maximum is 0.0495271, so it remains rejected regardless
of linear-solver results.

Sixteen workers each perform three inverse-setup/action measurements and three
GMRES solves for each of two RHS vectors. Each mode uses cached CPU elasticity,
checked against the original complete reference JVP and residual. Maximum
relative JVP discrepancy is 1.58e-16 and maximum residual-entry discrepancy
4.19e-18. Source summary/state hashes and packed states agree across modes;
all inputs remain unchanged. No physical state, offset, front or active set is
rounded, clipped, filtered or updated by this experiment.

The diagnostic budget is restart 60, four cycles (240 inner iterations),
rtol 1e-9 and independent true relative residual <=5e-9. Both material and
local full-border preconditioners converge on both RHS vectors in all repeats.
Frozen and block MG exhaust all 240 iterations in every repeat; their true
relative residuals range from about 0.998 to 1.000004. This rejects production
adoption of the current prototype for these states under the tested budget;
it does not establish failure at every possible Krylov budget or prove why
these preconditioners stagnate.

Iteration counts are identical across the three repeats. N/K denote the saved
state Newton RHS and known-direction J*v RHS. All wall times below are medians
of three observations after inverse warm-up. Timings are machine-dependent;
workers are sequential, without exclusive machine reservation. Environment:
Linux x86_64, Intel i7-960 3.20 GHz, Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1,
BLAS/OpenMP/MKL threads one. Setup includes a new cached system and inverse on
an existing physical problem, excluding source audit. Application is one full
bordered surrogate inverse action. GMRES time excludes setup. RSS is OS
worker-lifetime peak, including imports, audits, warm-up and all measurements.

| Source | Inverse | N/K iterations | Setup ms | Apply ms | Newton GMRES ms | RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 33x49 accepted | frozen | 240/240 (exhausted) | 38.661 | 0.508 | 916.251 | 132.58 |
| 33x49 accepted | material | 108/93 | 41.833 | 0.517 | 383.910 | 132.89 |
| 33x49 accepted | local | 22/21 | 43.885 | 0.544 | 76.147 | 131.73 |
| 33x49 accepted | multigrid | 240/240 (exhausted) | 48.291 | 1.802 | 1260.712 | 129.73 |
| 65x97 accepted | frozen | 240/240 (exhausted) | 236.707 | 3.959 | 2950.179 | 169.28 |
| 65x97 accepted | material | 95/76 | 204.054 | 3.531 | 1056.434 | 164.62 |
| 65x97 accepted | local | 29/28 | 202.026 | 3.377 | 315.874 | 163.54 |
| 65x97 accepted | multigrid | 240/240 (exhausted) | 163.649 | 2.833 | 2595.726 | 140.07 |
| 97x145 accepted | frozen | 240/240 (exhausted) | 1142.063 | 13.906 | 8175.946 | 267.36 |
| 97x145 accepted | material | 85/60 | 880.413 | 12.859 | 2443.633 | 247.94 |
| 97x145 accepted | local | 33/32 | 1128.426 | 14.133 | 1007.665 | 271.74 |
| 97x145 accepted | multigrid | 240/240 (exhausted) | 372.936 | 4.621 | 5150.599 | 154.31 |
| 97x145 rejected | frozen | 240/240 (exhausted) | 1037.825 | 14.374 | 7412.480 | 266.44 |
| 97x145 rejected | material | 85/83 | 792.604 | 11.809 | 2537.610 | 247.82 |
| 97x145 rejected | local | 31/32 | 733.833 | 12.701 | 841.973 | 222.50 |
| 97x145 rejected | multigrid | 240/240 (exhausted) | 366.440 | 4.449 | 5496.641 | 154.57 |

Across verified results, the maximum true residual is 8.26e-10. Known-direction
relative forward errors are <=6.67e-8 for material and <=1.05e-8 for local;
small true residual alone is not a forward-error/conditioning guarantee.
Material/local Newton directions differ by <=2.49e-9 relative L2. No directions
from exhausted solves are compared as accepted corrections.

At the accepted finest grid, MG stores about 3.46 MiB of reported array payload
versus 60.31 MiB of local LU factor payload, and has lower measured lifetime
RSS, but does not solve the system within budget. Array/factor payloads exclude
workspaces and MG sums can count shared storage more than once. These memory
observations cannot justify a larger node cap or substitute for convergence.

The experiment factors full sparse borders and compares against an approximate
MG force-Schur inverse. It does not replay production block-Schur/recovery,
line searches, active-set exploration or continuation, and does not establish
nonlinear speedup or new load/mesh convergence. The selected local sparse
preconditioner remains the useful reference here. Next investigate how to
retain material and local elastic tangent information in a bounded MG/block
surrogate; verify the complete operator before any production selection change.
Physical film accuracy and the 129x193 convergence problem remain open.

Raw output: `results/block_mg_comparison_20261028_final`, with all vectors,
three-run traces, setup/action samples, source hashes, source snapshot and
worker reports. `plots/krylov_comparison.png` and `plots/resource_comparison.png`
show raw measurements and observed ranges without pressure-field filtering;
both were visually inspected. Input snapshots and original archive provenance
are in `results/block_mg_inputs_20261028`. These are local/Git-ignored artifacts.
The measured source snapshot precedes an exception-chaining-only lint cleanup;
the production numerical modules and measured algorithm are unchanged.

A separate post-run audit reloads all raw vectors, recomputes both RHS vectors
and true residuals using original reference FFT execution, and checks all
32 saved directions against the independent seven-term SI mass-flux derivative
and trapezoidal force direction. It confirms 16 verified and 16 exhausted
results; maximum relative SI mass-action discrepancy is 5.22e-15 and maximum
absolute force-row discrepancy is 2.47e-17. Rejected input classification is
preserved. Audit source, source/report/vector hashes and all individual checks
are retained in `results/block_mg_comparison_20261028_audit`.

Final verification: `ruff check .` and `git diff --check` pass. Complete CPU
suite with BLAS/OpenMP/MKL threads one: 459 passed, 62 optional backend/GPU
skips in 312.52 s. All 30 benchmark/prototype regressions pass in the targeted
run (19.95 s), including 17 new regressions for dense correction parity,
independent gates, false-success/exhaustion reporting, source/film/force audit,
setup rejection and worker-error retention. No GPU ran for this increment.

Local evidence package: `results/block_mg_evidence_20261028_verified.zip`,
12,078,003 bytes and 151 files, with all SHA-256 manifest entries and ZIP CRCs
verified. SHA-256:
`a73459777143a34dd204f2210ff0cd946054ee8310593d32be0495a0279a8b44`.
It retains input provenance, measured/published sources, raw vectors and traces,
inspected plots and independent audit. The package is Git-ignored; only code,
tests and documented evidence/identity are committed, with no asserted
off-machine archive synchronization.


## 31. Exact tangent-block reduction and nonsymmetric MG controls (2026-10-28)

Starting from `9150695`, implemented exact active-row elimination of the
existing sparse material/local mass matrix, plus explicit reduced-LU controls
and a separate nonsymmetric Galerkin/ILU prototype. Physics, nondimensionalization
and the complete mass/content/force algebra are specified in PHYSICS.md. These
components remain diagnostics; no production nonlinear code, acceptance gate,
node cap, stencil or material law changes.

Tests compare the reduced mass matrix with independently selected original
columns and solve the full sparse-surrogate dense system for arbitrary RHS.
Full, cavitated, front and disconnected active sets exercise nonzero active
constraint rows. Tests include signed/heterogeneous equilibration, exact
Galerkin identities, fixed-cycle linearity, multilevel known-vector recovery,
constant-property limits, complete elastic/material/load products and dense
bordered corrections. Invalid/singular/nonfinite cases and injected factor
failures propagate without fine-LU/reduced-LU fallback.

Twenty-four isolated workers use the same four input snapshots as section 30,
three observations per successful setup and the same restart-60/four-cycle
240-iteration GMRES budget. All modes use cached CPU elasticity, independently
checked against the original reference. Exact source snapshots and hashes are
retained; state vectors and input files remain unchanged. No trial correction
is applied to a physical state, including the previously rejected partial.

Both reduced-LU modes converge for both RHS vectors on all four states, with
identical iteration counts to their respective full sparse-LU references:
local N/K counts 22/21, 29/28, 33/32, 31/32; material 108/93, 95/76, 85/60,
85/83. All three observations reproduce those counts. Complete true relative
residuals are below 8.26e-10. Compared with the full local result, reduced-local
Newton directions differ by <=4.44e-14 relative L2; the known-vector directions
can differ by 1.05e-8. These are linear checks, not nonlinear root accuracy.

Measured medians below are in milliseconds, with worker-lifetime RSS in MiB.
RSS includes imports, source audit, warm-up, setup and solves; factor payload
excludes workspaces. Setup includes a fresh cached system/inverse on the
existing physical problem. GMRES excludes setup and the independent post-check.
One worker runs at a time; no exclusive machine reservation is asserted.
Environment: Linux x86_64/i7-960 3.20 GHz, Python 3.12.3, NumPy 2.5.4,
SciPy 1.18.1, BLAS/OpenMP/MKL threads one.

| Source | Inverse | Setup ms | Apply ms | Newton GMRES ms | Factor MiB | RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 33x49 accepted | local | 43.206 | 0.550 | 75.608 | 1.552 | 132.05 |
| 33x49 accepted | local-reduced | 38.492 | 0.362 | 70.225 | 0.332 | 130.17 |
| 33x49 accepted | material | 43.955 | 0.512 | 426.680 | 1.669 | 133.30 |
| 33x49 accepted | material-reduced | 35.840 | 0.374 | 396.780 | 0.333 | 130.96 |
| 65x97 accepted | local | 212.795 | 3.571 | 342.358 | 12.324 | 159.71 |
| 65x97 accepted | local-reduced | 126.766 | 1.211 | 252.029 | 2.077 | 143.32 |
| 65x97 accepted | material | 215.538 | 3.882 | 1127.141 | 12.363 | 163.75 |
| 65x97 accepted | material-reduced | 128.135 | 1.086 | 859.752 | 2.076 | 145.09 |
| 97x145 accepted | local | 1187.648 | 14.775 | 1096.602 | 60.311 | 271.97 |
| 97x145 accepted | local-reduced | 301.827 | 3.206 | 671.791 | 5.792 | 164.32 |
| 97x145 accepted | material | 824.084 | 13.384 | 2480.278 | 47.285 | 249.47 |
| 97x145 accepted | material-reduced | 292.822 | 3.675 | 1653.878 | 5.806 | 170.32 |
| 97x145 rejected | local | 670.715 | 10.339 | 814.803 | 36.383 | 223.90 |
| 97x145 rejected | local-reduced | 283.186 | 2.453 | 560.833 | 5.559 | 163.66 |
| 97x145 rejected | material | 840.598 | 12.155 | 2483.237 | 46.997 | 246.41 |
| 97x145 rejected | material-reduced | 282.394 | 2.838 | 1607.132 | 5.547 | 169.54 |

At the finest accepted grid, local reduced setup/application are 3.93x/4.61x
faster in this isolated comparison, Newton GMRES wall is about 1.63x faster,
and lifetime RSS is 271.97 -> 164.32 MiB. Local factor payload is
60.31 -> 5.79 MiB. This n=13,585 reduced field factor plus a scalar force
Schur action replaces the diagnostic 27,171-unknown full bordered factor;
the actual outer state/operator still has all 27,171 unknowns. It is not an
integrated nonlinear/continuation speedup or a larger-grid resource allowance.

The MG prototype retains all tangent/coupling terms, but the geometric mixed
hierarchy remains unreliable. Outcomes, identical across repeated solves:

| Source | Local tangent MG | Material tangent MG |
| --- | --- | --- |
| 33x49 accepted | Verified 47/41 N/K iterations | Both exhaust 240; true residuals 89.6/1.00 |
| 65x97 accepted | ILU fails at level 1 (1,457 unknowns) | Same level/size failure |
| 97x145 accepted | Both exhaust 240; true residuals 1.015/1.00 | ILU fails at level 1 (3,337 unknowns) |
| 97x145 rejected | ILU fails at level 1 (3,337 unknowns) | ILU fails at level 2 (805 unknowns) |

These are explicit rejected setup/linear results. Initial COLAMD incomplete
factors failed on the coarsest loaded input even though exact reduced LU
succeeded. The tested fixed NATURAL ordering preserves lexicographic upstream
structure and improves that case, but does not remove the later failures.
No data-dependent ordering/fill switch, shift, clipping, regularization or
silent sparse fallback is introduced. Bilinear coarse corrections mix pressure
and deficit roles, and no spectral/causal diagnosis is claimed from these
outcomes. Exact fine-block factorization success does not imply valid coarse
factorization or MG convergence.

A separate post-run audit reloads all 38 saved directions and the five setup
rejections. Original reference FFT and the independent seven-term SI mass
flux derivative confirm all 34 passing and four exhausted direction flags;
maximum relative SI mass-action discrepancy is 7.05e-15 and force-row
discrepancy 2.47e-17. Eight additional loaded-state inverse-action comparisons
check the original full sparse border against explicit reduced LU. Maximum
relative action difference is 9.83e-9 and maximum independently recomputed
sparse-surrogate residual is 1.85e-16. Small residual is not a conditioning or
forward-error guarantee; no condition number or physical accuracy is inferred.

Raw evidence: `results/tangent_block_comparison_20261028_final`, with source
snapshot, all worker reports, setup failures, vectors and repeated histories.
`results/tangent_block_comparison_20261028_audit` retains the separate audit
script and all checks. `plots/reduced_lu_costs.png` and
`plots/tangent_mg_status.png` show observed timing ranges and every MG outcome,
without physical-field filtering. Inputs remain the unchanged section-30
snapshots. Historical outputs are preserved.

Next bounded production work: integrate exact reduced LU only as an explicit
optional field preconditioner, preserving existing Schur/full-border recovery,
strict film rejection, both globalization checks, active backups, Krylov failure
and continuation retries. Independently audit original final mass/force/film
and matched-force repairs before selecting it. MG needs separate investigation
of coarse spaces/smoothing that respect the mixed active roles; it is not
selected by production. The 129x193 and physical film-validation gaps remain.

Final verification: repository-wide Ruff and whitespace checks pass. Complete
Linux CPU suite: 499 passed, 62 optional backend/GPU skips in 333.63 s,
BLAS/OpenMP/MKL threads one. Targeted tangent/benchmark run: 57 passed in
15.86 s, including 36 new tangent regressions and extended benchmark cases.
Both cost/status figures were visually inspected. No GPU ran for this change.

Local package: `results/tangent_block_evidence_20261028_verified.zip`,
13,233,407 bytes and 162 files, with all manifest hashes and ZIP CRCs verified.
Archive SHA-256:
`939061f061cf1679bf65a1e1c450ea1dce13cdd0d0b4e46972e39db4da48a514`.
It includes inputs, measured source snapshot, raw results, plots, independent
audit and validation source; it is Git-ignored, with no asserted off-machine
archive synchronization. Only code, tests and documented evidence are committed.


## 32. Optional reduced field LU in actual material Newton (2026-10-29)

Starting from `5726ff8`, added explicit `field_factorization="reduced"` to the
material force Newton, load continuation and unaccepted-guess repair APIs.
The default remains `"full"`. The original already assembled mass/active
matrix supplies the n-unknown factor; its solve reconstructs the original 2n
field inverse. It supports frozen/material/local pressure tangents without
changing their coefficients. Experimental MG remains unavailable to production.

Complete elastic/material JVPs, force row/offset column, both block GMRES
solves, true Schur and reconstructed correction checks remain unchanged.
Reduced-field RuntimeError failures, block Krylov failures and unsafe Schur
reconstruction retain the original full 2n+1 sparse-bordered GMRES recovery.
There is no hidden unreduced-field fallback. `used_bordered_fallback`, field
selection/setup failure reasons, individual true linear residuals and Krylov
traces are retained in corrections/errors and material model histories.
Default unreduced arithmetic is unchanged apart from additional diagnostics.

New regression coverage includes all three sparse pressure tangents with both
reference/cached elasticity, full/selected active rows, complete JVP finite
differences and independently dense bordered corrections. Normal Schur and
forced full-border paths pass. Tests verify one shared matrix assembly, strict
film rejection, actual nonlinear root parity, original SI mass/force/film
gates, full and fraction-to-boundary Armijo stages, active lookahead backup
after injected selected-model failure, setup/Krylov failures and retry seed/
selection preservation. Zero-budget guesses remain unaccepted. Parent benchmark
checks reject mismatched snapshots/modes, corrupted output hashes/nonfinite
fields, disagreeing roots and failed workers. Legacy globalization tests pass.

Six isolated workers compare actual 200 N nonlinear repairs on the same three
accepted inputs used in sections 30-31. Each starts with original pressure/
content and an explicitly unaccepted +0.01 Href offset perturbation. Both use
cached CPU elasticity and local sparse tangents, unchanged 30-correction/eight-
model budgets and physical node caps. After one excluded warm-up, three complete
repair calls are timed. Every returned state/history is independently audited
with original SI mass/force/film gates and direct-source deformation. All
measured histories and output hashes are retained. This is not cold load
continuation, a new converged mesh, or physical film-height validation.

| Grid (ny,nx) | Field LU | Repair median s | Observed range s | Newton corrections | Total Krylov iterations | Worker peak RSS MiB |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 33x49 | full | 0.773315 | 0.760800..0.780501 | 4 | 179 | 130.77 |
| 33x49 | reduced | 0.834165 | 0.816715..0.839077 | 4 | 179 | 130.17 |
| 65x97 | full | 2.982716 | 2.949588..3.086190 | 4 | 230 | 145.92 |
| 65x97 | reduced | 2.616079 | 2.504617..2.627960 | 4 | 230 | 146.41 |
| 97x145 | full | 8.898884 | 8.805906..8.930447 | 5 | 332 | 169.80 |
| 97x145 | reduced | 8.925125 | 8.591203..13.123601 | 5 | 331 | 179.53 |

The actual full/reduced median runtime ratios are 0.927/1.140/0.997.
Reduced is slower on the coarsest case, modestly faster on the middle case,
and effectively unchanged on the finest case. The reduced finest measurement
includes a 13.124 s sample; no sample is removed or hidden. Worker-lifetime RSS
includes imports, source audits, warm-up and post-solve auditing/hashing, not
incremental field-factor memory. No integrated memory reduction is established.
Timings are machine-dependent observed samples, not confidence intervals or
exclusive-machine reservations. Linux/i7-960 3.20 GHz, Python 3.12.3,
NumPy 2.5.4, SciPy 1.18.1; BLAS/OpenMP/MKL threads one.

Section 31 compared isolated full bordered inverse actions. The actual normal
Newton path instead uses two field-block solves plus a true load Schur
reconstruction. Its measured costs must not be replaced by those earlier
full-border timings. The modest/mixed repair results do not support changing
the default. All measured models use the requested field factorization;
no field setup failures or bordered recovery occur in these six normal
workers. Injected small-grid tests separately cover those recovery paths.

Matched-force full/reduced pressure differences are <=6.12e-15 relative L2,
film differences <=6.78e-21 m, content differences <=2.11e-15, and offset
differences <=8.48e-22 m. Both modes take 4/4/5 Newton corrections. The finest
Krylov totals differ by one iteration (332 versus 331); bitwise equivalence
is not claimed. Every returned root passes original field and 200 N force
gates. Existing mesh/domain accuracy and pressure-pattern limitations persist.

A separate post-run audit reloads all six saved states, verifies exact axes,
recomputes original force/mass/film/source diagnostics and checks every stored
history. All three measured field hashes and offsets per worker match its
saved final state, allowing 18 measured histories to be checked against those
independently recomputed fields. Maximum scaled mass defect is 1.85e-13;
maximum direct-source displacement discrepancy is 4.24e-21 m. Source state
and summary hashes remain unchanged, and parent output-identity checks pass.

Evidence: `results/reduced_field_repairs_20261029_final` retains exact measured
sources, all histories/hashes/audits and raw timings; `_audit` retains the
independent script/report. `plots/whole_repair_costs.png` shows all observed
ranges, including the outlier; `plots/mass_convergence.png` shows original
stored mass diagnostics without flooring/smoothing. Both figures are visually
inspected. The source states remain the unchanged section-30 inputs; no
historical output is replaced.

Next return to bounded nonlinear conditioning/globalization and unresolved
mesh diagnostics using the original gates and existing resource controls.
Keep unreduced LU as default, reduced LU as an explicit verified comparison,
and MG as separate research work. No new physical constitutive extension,
node-cap increase or film-height accuracy claim is justified by these repairs.

Final verification: full Linux CPU suite 525 passed, 62 optional backend/GPU
skips in 321.72 s, BLAS/OpenMP/MKL threads one. Targeted integration/benchmark/
legacy-globalization run: 37 passed in 8.80 s, including 26 new regression cases.
Repository-wide Ruff and whitespace checks pass; both figures were visually
inspected. No GPU ran for this increment.

Local evidence package: `results/reduced_field_evidence_20261029_verified.zip`,
3,041,195 bytes, 133 files, all member manifest hashes and ZIP CRCs verified.
Archive SHA-256:
`acf02d4af362a397bc666a11e3d90ce6962d8228435a93bd5b4aeb42cce7e872`.
It retains unchanged inputs, measured source snapshot, all histories/hashes,
raw timings, inspected plots, independent audit and validation source.
The package is Git-ignored, with no asserted off-machine archive synchronization;
only code, tests and documented evidence are committed.

## 33. Exact-state fourth-mesh Newton diagnosis (2026-10-29)

Based on `79a2b74`, added an internal observational hook and separate diagnostic
helpers/runners. Normal solver selection and every physical/nonlinear rule are
unchanged. The hook receives copied exact dimensionless state and deeply copied
SI fields/history, and records selected-row indices only when enabled. Default
runs do not copy or annotate observer data. Callback failure is distinct from
Newton failure. Raw state avoids reconstructing tiny D=1-theta deficits or
active ties from rounded saved physical fields.

Independent regressions verify observer mutation isolation (including forced
writes to the copied raw state), identical numerical outputs with/without
observation, initial/terminal state capture, zero-budget nonpromotion and exact
failed-linear selected-row metadata. SI bound tests cover pressure/content/
whole-grid film, non-square axes, uncapped limits, exact ties, strict film
boundary, SI rescaling, invalid/infeasible input and zero directions. Replay
regressions verify dense/true-correction consistency through the existing
linear solver, original merit/next-state reproduction, selected-row metadata
and corrupted history/state rejection. No numerical step choice is made by
the diagnostic bound helper.

One factor-two interpolation of the accepted 65x97 registered 200 N source
starts the explicit 129x193 trial on the unchanged +/-3a reservoir. The
original 20,000 default cap remains; the existing explicit 30,000 allowance is
used. Local sparse tangent, cached CPU elasticity and full field LU are used,
with 30 corrections, at most eight models, restart 60 and at most ten Krylov
cycles per solve. Sparse factor preflight residual is 3.51e-13 and resident
peak about 195.88 MiB, below the existing 1 GiB check; 61 basis vectors require
23,675,320 bytes (22.58 MiB). This is not a new resource allowance.

The diagnostic run ends at the original budget with 31 raw state snapshots.
It remains rejected by independently reopened original SI gates:

| Diagnostic | Observed value | Original gate |
| --- | ---: | ---: |
| Force | 200 N | relative error <=1e-8 |
| Scaled mass maximum | .0028407992 | <=1e-8 |
| Relative local balance | .05743805 | <=1e-10 |
| Relative global balance | 8.0948e-7 | <=1e-10 |
| Complementarity maximum | 9.13e-14 | <=1e-10 |
| Reference film reconstruction difference | 6.78e-21 m | <=1e-10 Href |

No fourth root, mesh convergence rate or physical film-height prediction is
accepted. The endpoint agrees with the earlier rounded .00284080 result in
section 27. Observed run time is 400.82 s, including snapshot I/O; terminal
worker-lifetime RSS is about 215.39 MiB. Verification work also ran during the
trial, so these are diagnostic wall/resource observations, not speedup numbers
or a cross-platform timing comparison with section 27.

All 118 returned linear models pass their complete true residual checks,
maximum 8.71e-10 (gate 5e-9); 8,236 Krylov iterations are recorded. There are no
field-factor setup failures, failed block true-residual checks, full-border
recoveries or active-recovery linear failures in this trial. Of these models,
79 have nonnegative original directional-merit slopes and are not line-searched.
The 39 descending models yield 30 accepted steps and nine deferred alternatives.
The nonaccepted trial log contains 38 pressure/content infeasibilities and nine
deferred feasible Armijo steps, with no Armijo-only rejection. Deferred steps
are not misclassified as physical failures.

Replaying exact raw state and recorded selected rows reproduces each accepted
next state and original slope with zero difference on this environment:

| Zero-based iteration | First internal constraint | Raw alpha limit | Accepted alpha | Limiting cell (iy,ix) |
| --- | --- | ---: | ---: | --- |
| 0 | Content upper | .0033667892 | .0033331213 | (5,109) |
| 15 | Content upper | 8.7898286e-7 | 8.7019303e-7 | (124,110) |
| 17 | Pressure lower | .80576967 | .79771197 | (83,108) |
| 29 | Content upper, beyond full step | 1.0000000018 | 1 | (83,105) |

At state 15 the limiter is at x=.11858036 mm, y=.76230229 mm (y=2.8125a),
outside the high-pressure contact. Saved pressure is about -6.71e-13 Pa,
within the original pressure tolerance, theta=.9999999987897328 and exact
D=1.210267180483277e-9. No pressure/content value is clipped or rounded into a
tie. SI and exact internal-variable bounds are both retained to expose
cancellation differences in theta=1-D. This is a recorded affine step limit,
not proof of a physical front instability or the cause of the pressure pattern.

All alternatives at iteration 15 were replayed independently:

| Model | Changed active rows | Original slope | Limiting raw alpha | Recorded outcome |
| --- | ---: | ---: | ---: | --- |
| 0 | 0 | -2.82504e-7 | 8.78983e-7 (content upper) | safeguarded backup selected |
| 1 | 147 | -9.43826e-9 | 1.19945e-7 (pressure lower) | feasible smaller-progress step deferred |
| 2 | 141 | +6.72028e-9 | 1.57620e-7 (content upper) | non-descending, no trial |
| 3 | 143 | +6.87376e-9 | 1.0000000010 (pressure lower) | non-descending, no trial |

For models 0/1, full trials independently fail the original pressure/content
bounds; the recorded bounded trial merits are reproduced exactly and satisfy
original Armijo/progress checks. Model 0 has the lower verified merit, so the
existing selection is reproduced. A larger feasible direction is not enough
when its original merit slope is non-descending. The final full step makes
substantial progress; no assertion of nonexistence or static total stagnation
follows from the 30-correction rejection.

A separate audit recomputes seven distinct selected directional actions using
the independent seven-term SI mass-flux decomposition and trapezoidal force.
Maximum relative mass-action discrepancy is 1.89e-16; maximum force-row
absolute discrepancy is 2.22e-18. Selected complementarity rows agree exactly.
This supports the recorded linear actions, not a temporal stability claim.

Evidence: `results/mesh_newton129_20261029` contains all exact/SI states,
history, source hashes and inspected `convergence_and_models.png` and
`raw_fields_and_limiter.png`; `_analysis` retains four accepted-model replays,
`_allmodels` retains every state-15 alternative, and `_flux_audit` retains raw
directions and the independent script/checks. Historical outputs are preserved.

Next testable hypothesis: a separately verified targeted content-upper-bound
active pivot may avoid coupling one weak-cell restriction to broad simultaneous
whole-step flips. Retain the current reference, original Jacobian/merit and
all bounds/budgets. Verify manufactured cases and physical benchmark behavior
before selection. Increasing the budget or clipping this partial would not
supply the missing convergence evidence. Physical film accuracy stays open.

Final checks: full Linux CPU suite 542 passed, 62 optional backend/GPU skips
in 337.33 s, BLAS/OpenMP/MKL threads one. Seventeen new CPU regressions pass;
repository-wide Ruff and whitespace checks are clean. Both figures were
visually inspected. No GPU ran for this change.

Local package: `results/mesh_newton_evidence_20261029_verified.zip`,
32,913,819 bytes, 477 files, all manifest hashes and ZIP CRCs verified.
Archive SHA-256:
`035f719ae23a2477362a8d2d81793f68cab1eea6e892e7fe412b034f9a4424ad`.
Original inputs, exact measured/replay sources, every state/history, all
alternative checks, independent flux audit/directions and plots are retained.
Generated data/archive remain Git-ignored, with no asserted off-machine
synchronization. Only code, tests and documented evidence are committed.

## 34. Controlled 50 N 223x223 limited-MC mesh refinement (2026-11-06)

The registered 50 N circular case (`cases/circular_50n_223.json`) was run on
49,729 physical nodes, preserving its SI inputs, +/-4a domain, equations and
acceptance tolerances. The input specifies Q=50 N, R=15 mm, contact modulus
113 GPa, eta0=0.04 Pa s, rho0=850 kg/m^3 and positive-x entrainment 2 m/s.
The exact MC continuation used the independently audited 199x199 root as a
bilinearly interpolated, unaccepted initial guess. Eight stages reached
transport fraction 1.0; none were rejected.

The final root passes the solver's direct-elasticity and physical audit:

| Diagnostic | 223x223 result |
| --- | ---: |
| Integrated force | 49.99999999999999 N |
| Relative local / global mass balance | 6.093e-11 / 1.802e-14 |
| Maximum complementarity | 1.481e-28 |
| Direct-elasticity film reconstruction difference | 1.736e-20 m |
| Minimum / central film | 287.867 / 470.732 nm |

`compare_limited_paths.py` independently reloads both complete paths,
recomputes their SI and direct-elasticity audits, and compares fine fields at
coarse nodes. For 199x199 -> 223x223, minimum/central film changes are
+0.411%/+0.217%; pressure relative L2/maximum changes are 1.772%/10.397%;
exit-region L2/maximum changes are 4.530%/11.790%. The minimum moves 0.85/0.58
fine cells in x/|y|. Thus both roots are accepted, but pressure differences
fail the provisional 1% sensitivity gate. The small film changes and
sub-cell minimum movement do not imply pressure convergence, continuum
convergence or physical EHL accuracy. The HD reference values remain
correlations, not acceptance thresholds.

The resource preflight used the requested coarse approach, not measured-peak
profiling. Its adjusted estimate was 1,259,911,478 bytes, with
3,249,704,960 bytes available at preflight and a 3.446e-16 sparse-factor
known-vector residual. The nonlinear run completed, but no peak RSS was
measured and the estimate is not a peak-memory guarantee. Improved peak and
factor-fill estimation is explicitly deferred to an optimization round; no
larger mesh budget is authorized by this result.

Reproduction and retained local archives are documented in
[LIMITED_ENDPOINT.md](LIMITED_ENDPOINT.md). The root is
`results/limited_mesh_50n_223_20261106/`; its independently audited
comparison is `results/limited_comparison_50n_199_to_223_20261106/`. Both
preserve source-summary hashes and detailed SI audits. This is numerical
verification of a finite-grid root and mesh change, not physical validation.

Verification: repository-wide `ruff check .` passed; the complete CPU suite
passed **701 tests with 62 optional skips** in 442.57 s. This run included the
preserved local driver edit. No GPU validation was run or claimed.

## 35. Fixed-domain directional 50 N refinement (2026-11-06)

The 50 N case was independently solved on anisotropic meshes with the same
physical inputs and fixed +/-4a domain: `(ny,nx)=199x223` refines the
entrainment/x direction, and `223x199` refines the transverse/y direction.
Each grid has 44,377 nodes. The accepted 199x199 exact MC root was used only as
an interpolated initial guess; it was not promoted as a candidate solution.
Both continuation paths pass the unchanged exact endpoint/direct-elasticity
gates. The x-refined path accepts fraction one in ten attempts (one rejection);
the y-refined path does likewise.

| Grid | Force [N] | Relative local/global mass balance | Complementarity | Minimum / central film [nm] |
| --- | ---: | ---: | ---: | ---: |
| 199x223 | 50.00000000000000 | 1.561e-11 / 1.063e-15 | 1.235e-27 | 287.077 / 470.678 |
| 223x199 | 49.99999999999999 | 4.492e-11 / 1.259e-14 | 1.886e-30 | 287.478 / 469.764 |

Each final direct-source elastic reconstruction agrees with the coupled film
to <=1.53e-20 m. Independent spatial comparisons against the 199x199 root
find:

| Refinement | Full pressure L2 / max | Exit pressure L2 / max | Minimum / central film change |
| --- | ---: | ---: | ---: |
| x: 199x199 -> 199x223 | 1.357% / 9.929% | 3.588% / 10.884% | +0.135% / +0.206% |
| y: 199x199 -> 223x199 | 1.096% / 9.572% | 2.427% / 10.575% | +0.275% / +0.011% |

The x-direction refinement causes the larger measured pressure change in this
pair, though the y refinement also changes pressure beyond the provisional 1%
gate. Comparing each anisotropic result against the square 223x223 endpoint
adds one refinement in the other direction and reproduces the ordering: y
refinement gives 1.140% full / 2.563% exit L2 changes; x refinement gives
1.350% full / 3.577% exit L2 changes. These are non-nested bilinear
fine-at-coarse comparisons; they include interpolation effects and are not a
formal directional truncation-error estimate. Mesh/continuum convergence,
physical accuracy and experimental validity remain unestablished.

The accepted cases and complete independent comparisons are recorded in
[LIMITED_ENDPOINT.md](LIMITED_ENDPOINT.md). Coarse adjusted resource estimates
were 1,042,314,243 bytes (x-refined) and 1,097,789,678 bytes (y-refined);
measured peak RSS remains deliberately deferred to a separate optimization
round. Verification: full CPU suite **702 passed, 62 skipped** (373.03 s),
`ruff check .` passed and `git diff --check` passed. The added configuration
regression checks equal domains, unchanged physical signatures, axis order,
per-axis spacing and node caps. No GPU validation was run or claimed.

## 36. NASA Table I inspired circular-contact probe (2026-11-07)

This is a numerical comparison, **not an experimentally validated or fully
matched NASA benchmark**. Table I of [NASA TN D-8317](https://ntrs.nasa.gov/api/citations/19770004457/downloads/19770004457.pdf),
printed p. 17, gives the point-contact numerical result `Hmin=hmin/Rx =
3.367e-6` at `k=1`, `U=0.1683e-11`, `W=0.1106e-6` and `G=4522`.
For the independently chosen `Rx=0.015 m`, this corresponds to a target
minimum film of 50.505 nm. This target is a tabled numerical result, not a
measurement or the fitted correlation. No central film value is asserted
from this table row.

The three new case files explicitly choose `E*=113 GPa`, `E'=2E*=226 GPa`,
`eta0=0.04 Pa s`, `Ue=0.14263425 m/s`, `Q=5.62401 N` and
`alpha=4522/E'=2.0008849557522124e-8 Pa^-1`. They reproduce the rounded
table groups using `U=eta0 Ue/(E' Rx)`, `W=Q/(E' Rx^2)` and
`G=alpha E'`; these SI values are one selected parameterization, not
reported NASA test conditions. Here `Ue=(u1+u2)/2`, x is entrainment, and
arrays are `(ny,nx)`. Grids are 33x33, 65x65 and 129x129 on the same
square `[-4a,4a]` domain, where `a=82.42141277 um`; maximum node budget is
20,000. Refinement seeds are interpolated initial guesses and are never
accepted without a new solve and audit.

Matching these four groups does **not** make this the same physical problem.
NASA defines G via the asymptotic isoviscous pressure, approximated by
`1/alpha`; this probe instead uses alpha as the *ambient* logarithmic slope
of a chosen Roelands curve. Ambient viscosity, density 850 kg/m^3, Roelands
reference viscosity 6.31e-5 Pa s, pressure 196 MPa and the present
Dowson-Higginson coefficients are explicit choices; they are not established
as the table's exact fluid. The older NASA TN D-8049 reference-viscosity SI
and cP labels disagree by a factor of 100; see
[CONSTITUTIVE.md](CONSTITUTIVE.md). NASA applies a Reynolds rupture condition,
whereas this solver uses finite-domain conservative pressure/content
complementarity and donor transport. A flooded west node on a finite domain
does not prove the fully flooded far-field limit. The older tabulated solution
also has its own discretization uncertainty. None of these differences is
absorbed into a tolerance or treated as solved by matching a film value.

The donor CLI's initial Hertz state is unaccepted; its correlation-based
offset is only a starting guess, not a constraint on the final film. Each
root was independently reopened using `load_run` and re-audited with direct
source-cell elasticity, SI fluxes and trapezoidal load:

| Grid | Minimum / central film [nm] | Minimum vs NASA table | Relative local / global mass balance |
| --- | ---: | ---: | ---: |
| 33x33 | 47.17520 / 92.82430 | -6.593% | 3.190e-15 / 3.774e-17 |
| 65x65 | 53.95931 / 94.47625 | +6.840% | 6.400e-15 / 9.202e-17 |
| 129x129 | 54.40787 / 91.71068 | +7.728% | 1.379e-14 / 0.000e+00 |

Each integrated force is 5.62401 N; the maximum complementarity is
1.567e-26 and the greatest direct film-reconstruction discrepancy is
1.748e-21 m. Minimum film changes +14.381% and +0.831% over successive
refinements; central film changes +1.780% and -2.927%. A small final
minimum-film change or approximate agreement with the NASA value does not
prove domain independence, continuum convergence or matched physics.

As a separate solver-robustness probe, the 33x33 limited-MC continuation
exhausted its 24-attempt budget at fraction `0.9999703125`. Its last blended
stage passed its own audit but is **not** an exact limited-MC endpoint; its
film must not be reported as a successful MC result. The driver snapshot
includes the pre-existing local changes to
`examples/run_limited_transport_path.py`; those changes were not part of this
milestone. No altered stopping gates, clipped fields or bigger budget were
used.

Reproduce the donor calculations into fresh output directories:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src
.venv/bin/python -m ehl_solver.prototype_cli --case cases/nasa_d8317_table1_k1_probe.json --output results/nasa_probe_33_new --no-plots
.venv/bin/python -m ehl_solver.prototype_cli --case cases/nasa_d8317_table1_k1_probe_65.json --seed results/nasa_probe_33_new --output results/nasa_probe_65_new --no-plots
.venv/bin/python -m ehl_solver.prototype_cli --case cases/nasa_d8317_table1_k1_probe_129.json --seed results/nasa_probe_65_new --output results/nasa_probe_129_new --no-plots
.venv/bin/python -m examples.run_limited_transport_path --case cases/nasa_d8317_table1_k1_probe.json --output results/nasa_probe_mc_new --max-attempts 24
```

The runs are retained locally under `results/nasa_d8317_table1_k1_donor_*`
and `results/nasa_d8317_table1_k1_probe_20261107/`, including inputs, source
snapshots, checksums and audit histories; they are not committed datasets.
Next, establish the exact historical fluid parameters and boundary
assumptions or construct a matched experimental case with independently
measured lubricant properties. Do not equate this preliminary comparison
with physical validation.

Verification: `ruff check .` passed; the complete CPU suite passed **718
tests, 62 skipped** with BLAS/OpenMP/MKL limited to one thread. The four
new input regressions check the three sets of dimensionless groups, ambient
viscosity slope, domain and nested grid consistency. No GPU validation was
run or claimed.

## 37. Source audit and conditional PAO4 experiment probe (2026-11-07)

**Outcome: reproducible numerical roots, not experimental validation.**
The published experiment supplies useful operating data, but the
pressure-viscosity coefficient is fitted from film measurements and the
computed film is not mesh converged. The further NASA source information
below also does not make the section-36 case a fully matched benchmark.

### Historical NASA inputs

[NASA TN D-8166](https://ntrs.nasa.gov/api/citations/19760010312/downloads/19760010312.pdf),
printed p. 13, Table I, specifies equal solid Young's moduli 200 GPa and
Poisson ratios 0.3, a ball radius of 11.113 mm in both directions, ambient
viscosity 0.0411 Pa s, Roelands index 0.67, density-law coefficients
5.83e-10 and 1.68e-9 Pa^-1, and asymptotic isoviscous pressure 45.22 MPa.
It gives force 10 N and x surface velocity 0.1 m/s; its Table II identifies
the k=1 opposing surface as flat and gives U=0.1683e-11, W=0.3686e-6,
G=4522. These are historical source inputs, not the freely chosen SI
parameterization of section 36.

There is a material-group inconsistency needing resolution: the printed
moduli yield E*=109.8901 GPa and E'=219.7802 GPa; E'/45.22 MPa is 4860.24,
not the printed 4522, using the later D-8317 definition G=E'/p_iv,as.
No cause or correction has been established. D-8166 Table III's k=1 ratio
is 0.1366 relative to its printed Hmin,L=6.955e-6; **D-8317 explicitly
reports corrected ellipticity results**. Moreover, the section-36 D-8317
k=1 row has W=0.1106e-6, not D-8166's W=0.3686e-6. Neither the old ratio
nor the different-load row is an interchangeable validated target, and we
have not established that all fluid inputs carry over unchanged. Thus the
earlier statement that exact inputs were not established remains true for
the selected D-8317 row, but should not be read as saying no historical
fluid table exists. The public case adapter currently uses its default
density coefficients; the historical coefficients have not been silently
substituted. Differences in rupture and finite boundaries also remain.

### PAO4 conditions and limitations

[MacLaren and Kadiric (2024)](https://doi.org/10.1007/s11249-024-01894-w),
Tables 1-4 and Fig. 7, motivate a steel/BK7-glass, 50 N, 9.525 mm ball,
20 C, pure-rolling 1.2 m/s PAO4 probe. The Fig. 7 caption does not itself
identify the disc; the steel/glass assignment to that figure is a conditional
interpretation. E*=58.749804 GPa follows from 210 GPa, nu=0.3 steel and
75 GPa, nu=0.22 glass, giving dry Hertz radius a=182.514 um. The VFT and
density fits give eta0=0.033872735 Pa s and rho0=816.75 kg/m^3.

Table 3 reports alpha=17.6 GPa^-1, while the Fig. 7 legend rounds it to
17.5. We conditionally interpret the Table 3 value as the ambient
logarithmic Roelands slope using reference viscosity 6.31e-5 Pa s and
pressure scale 196 MPa. This does not establish a measured high-pressure
viscosity law; the default Dowson-Higginson pressure density and isothermal,
fully flooded Newtonian model are additional assumptions. Crucially, the
paper fits alpha to Chittenden's correlation using low-speed film
measurements, including the range containing 1.2 m/s. Comparing the same
measurements with this input is therefore circular, not independent
validation. Figure 7 contains plotted means of five and +/-3 sigma error
bars, not a tabulated 1.2 m/s target. The supplementary PDF contains graphs,
not the underlying raw table; the authors state that data are available on
reasonable request. No value or experimental tolerance has been fabricated.

### Results and reproduction

The donor cases are fixed-temperature SI inputs, with x entrainment and
arrays ordered (ny,nx). Each root was separately reloaded with a checksum
and re-audited with SI mass flux and direct source-cell film reconstruction:

| Grid and half-width | Minimum / central film [nm] | Relative local / global mass balance |
| --- | ---: | ---: |
| 33x33, 4a | 12.41300 / 174.26474 | 2.671e-12 / 3.623e-17 |
| 65x65, 4a | 70.65507 / 237.34708 | 2.435e-14 / 1.233e-16 |
| 129x129, 4a | 106.20606 / 245.12753 | 5.189e-14 / 2.087e-16 |
| 97x97, 6a | 70.62335 / 237.56843 | 4.179e-14 / 0 |

All four have force 50 N to roundoff, maximum scaled mass defect 5.31e-15,
maximum complementarity 9.98e-29, and direct film reconstruction discrepancy
at most 1.49e-20 m. Min/central changes relative to the preceding grid are
+469.20%/+36.20% and +50.32%/+3.28%. The 97x97 wider domain holds the
65x65 spacing fixed and changes min/central film by approximately
-0.045%/+0.093%, respectively. This is only one domain comparison on the
coarser spacing, not a finer-mesh domain-convergence bound. Agreement of
the 65x65 central film with the solver's Hamrock-Dowson correlation
(238.233 nm) is not evidence of experimental accuracy or mesh convergence.

The following commands create fresh output directories. Seeds are initial
guesses only; each target is independently solved and accepted.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src
.venv/bin/python -m ehl_solver.prototype_cli --case cases/maclaren_pao4_20c_1p2_probe.json --output results/pao4_33_new --no-plots
.venv/bin/python -m ehl_solver.prototype_cli --case cases/maclaren_pao4_20c_1p2_probe_65.json --seed results/pao4_33_new --output results/pao4_65_new --no-plots
.venv/bin/python -m ehl_solver.prototype_cli --case cases/maclaren_pao4_20c_1p2_probe_129.json --seed results/pao4_65_new --output results/pao4_129_new --no-plots
.venv/bin/python -m ehl_solver.prototype_cli --case cases/maclaren_pao4_20c_1p2_probe_wide_97.json --seed results/pao4_65_new --output results/pao4_wide_97_new --no-plots
```

Archived local outputs are in `results/maclaren_pao4_20c_1p2_donor_*_20261107/`
and are not committed. Independently reload with `load_run` and call
`audit_root(case, root)` to recompute the stated checks. Future work requires
independently measured pressure-dependent fluid properties and numerical
experimental data, matched configuration and uncertainty, plus controlled
mesh and domain error. No GPU or new constitutive physics was added.

Verification: `ruff check .` passed; the complete CPU suite passed **723
tests, 62 skipped** with BLAS/OpenMP/MKL limited to one thread. Five new
tests check the three input conversions, nested refinement and fixed-spacing
domain expansion. Optional device tests were skipped; no GPU result is
claimed.

### Ciulli Figure 13: initial donor-discretization closure and convergence status

**Historical status note:** the field-availability and “comparison deferred”
statements in this section and its retained-pressure-field subsection were
accurate when written, but are superseded by the completed reanalysis at the
end of this report. The failed 193x193 iterate remains rejected.

The initial CPU studies use the Figure 13 steady pure-rolling operating point:
15 N, `u_e = 0.2 m/s`, `S = 0`, `T = 12 deg C`, `eta_0 = 0.497 Pa s`,
`rho_0 = 892 kg/m^3`, `alpha_p = 2.87e-8 Pa^-1`, and the reported 10.319 mm
ball diameter. The effective modulus is inferred from the paper's rounded
0.1 mm Hertz radius. The configured fluid uses the Roelands index matched to
the reported zero-pressure viscosity slope and the assumed
Dowson-Higginson density coefficients described in
[EXPERIMENTAL_VALIDATION.md](EXPERIMENTAL_VALIDATION.md). These are declared
model assumptions, not measured full-pressure SN500 properties.
The CPU runs used Python 3.13 with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`.

The accepted roots below solve the donor-flux discretization
(`transport_fraction = 0`) on the fixed +/-0.4 mm square domain. They passed
the independent force, local and global SI mass, JFO complementarity, film
reconstruction, and direct-elasticity audits:

| Grid | Spacing [um] | Minimum / central film [nm] | Relative local / global mass balance | Maximum complementarity |
| --- | ---: | ---: | ---: | ---: |
| 33x33 | 25.000 | 216.706 / 456.136 | 2.93e-15 / 5.91e-17 | 2.13e-29 |
| 65x65 | 12.500 | 282.747 / 492.246 | 6.25e-15 / 3.83e-17 | 3.36e-28 |
| 129x129 | 6.250 | 285.231 / 485.267 | 1.79e-14 / 7.58e-17 | 4.14e-28 |

Load is 15 N to the reported precision. The maximum film-reconstruction
discrepancy on the 33x33 root is `3.39e-21 m`; the direct elasticity audit
passed on each accepted root. Refinement is not yet a convergence result:
minimum/central film change by about +30.5%/+7.9% from 33x33 to 65x65 and
+0.88%/-1.42% from 65x65 to 129x129. No accepted finer-grid pressure field is
available for common-coordinate main-contact comparison or the separate
outlet-spike analysis.

A separate fixed-spacing sequence holds the grid spacing at approximately
12.5 um while expanding both domain half-widths:

| Grid and half-width | Minimum / central film [nm] | Relative local / global mass balance |
| --- | ---: | ---: |
| 65x65, +/-0.4 mm | 282.747 / 492.246 | 6.25e-15 / 3.83e-17 |
| 81x81, +/-0.5 mm | 283.483 / 493.981 | 9.65e-15 / 3.79e-17 |
| 97x97, +/-0.6 mm | 283.809 / 494.698 | 6.41e-15 / 8.59e-17 |

The scalar film changes are small in this sequence, but the accepted 65x65,
81x81, and 97x97 pressure arrays were not retained. The 33x33 pressure field
is the only Ciulli donor field currently available in the local result
archives, so no fixed-spacing pressure profiles can be compared on common
physical coordinates. This is preliminary scalar boundary-sensitivity
evidence only and does not establish domain independence.

The 193x193 fixed-domain attempt (`+/-0.4 mm`, spacing `4.167 um`) exhausted
its 30-correction Newton budget. Its final iterate was **not accepted**:
relative local/global mass balance was `8.589e-1`/`3.273e-4`,
maximum complementarity was `1.047e-3`, and maximum SI mass divergence was
`97.08 kg/(m^2 s)`. The audit reported 15 N load, but direct elasticity was
not checked for this failed iterate. Its reported 339.455 nm minimum and
462.396 nm central film are not solution values and must not be used in a
refinement comparison.

These results establish Level 1 closure only for the listed donor-flux roots.
They do not close the distinct limited-capacity endpoint
(`transport_fraction = 1`): the bounded transport continuation previously
stopped at fraction 0.9995491, and its blended state is not an exact endpoint
root. Level 2 remains incomplete because a finer accepted root and
common-coordinate pressure/domain comparisons are missing. Therefore no
solver-vs-Ciulli film-profile comparison has been made; Level 3 is deferred.
The digitized Figure 13 profiles remain the initial experimental reference,
with their stated digitization and uncorrected-optics uncertainty. They
contain film thickness only and cannot validate pressure or the outlet spike.

#### Retained pressure-field analysis and missing archives

The local Ciulli archives contain complete stage fields for the 33x33
continuation only. `attempt_000/state.npz` is an independently accepted
donor root (`transport_fraction = 0`) with pressure, coordinates, content,
film and displacement. The remaining saved states in those continuation
archives have nonzero transport fraction; they are blended equations and are
not substituted for donor or exact limited-capacity roots. No pressure arrays
for the accepted 65x65, 129x129, 81x81, or 97x97 donor roots were found in
the Ciulli result archives.

The reproducible analysis command is:

```text
PYTHONPATH=src python examples/analyze_pressure_field.py results/ciulli_20261109_33_fixed_domain/attempt_000/state.npz
```

It verifies the adjacent accepted-stage metadata, samples the centerline
(`y=0`) on the saved physical x coordinates, and reports global/local maxima,
pressure support and the first centerline zero-pressure node with deficient
film content. Its shared-coordinate profile comparator and explicit-window
outlet-spike diagnostic are in `src/ehl_solver/pressure_analysis.py`; spike
shoulder bounds must be specified from the finest field and are never guessed
by the tool.

For the retained 33x33 donor field (`+/-0.4 mm`, `dx = dy = 25 um`), the
sampled two-dimensional maximum is `760.606 MPa` at
`(x,y) = (50,-25) um`. The centerline maximum is `751.145 MPa` at `x=0`,
with immediate neighboring samples `721.167` and `738.187 MPa`; it is the
only interior centerline local maximum. Centerline pressure is positive from
`x=-375` through `+100 um`; at `x=+125 um` pressure is zero and
`theta=0.78219`. Thus this coarse profile has no separately identifiable
downstream local peak. It does not show that a physical outlet spike is
absent: the spike window, shoulders, half-prominence width, and resolving
cell count are undefined for this field and must not be reported as zero.
No main-contact cross-grid pressure difference is possible from one grid.

The separate 193x193 donor attempt reached its 30-correction limit with load
exactly `15 N`, but relative local/global mass balance remained
`8.589e-1`/`3.273e-4`, maximum complementarity `1.047e-3`, and maximum SI
mass divergence `97.08 kg/(m^2 s)`. Direct elasticity was not checked on
that iterate, and it is not a solution. The available record establishes
nonlinear equation nonconvergence despite force closure; its iteration,
active-set, line-search and Krylov histories were not retained, so it cannot
identify which correction mechanism stalled. Do not infer a 193x193 pressure
profile or diagnose a specific linear-solver cause from the terminal audit.

To complete pressure sensitivity, rerun and archive the accepted donor fields
for 65x65 and 129x129 on the fixed `+/-0.4 mm` domain, and 81x81 and 97x97
on their respective fixed-spacing `+/-0.5 mm` and `+/-0.6 mm` domains. The
65x65 fixed-domain root can serve both sequences. Each archive must retain
`pressure_pa`, `x_m`, `y_m`, the exact case, transport fraction and passing
independent audit. Then compare the main-contact centerline profiles on an
explicit common physical interval and analyze any outlet peak separately
with shoulder bounds frozen from the finest accepted field. The 193x193
failure needs a diagnostic rerun that preserves nonlinear/Krylov histories
before choosing a numerical remedy; do not extend iteration budgets blindly.

The field-analysis helpers have regression coverage for y=0 interpolation,
physical-coordinate comparison across unlike grids, regional spike-width
measurement, unresolved-peak reporting, and rejection of nonaccepted field
archives. `ruff check src tests examples` passed. The Windows-compatible CPU
suite passed **797 tests, 1 skipped**; `tests/test_joint_refinement.py` remains
uncollectable on Windows because it imports the Unix-only `resource` module.
`ruff check .` still reports the pre-existing 101-character line in
`.kilo/skills/jupyter-notebook/scripts/new_notebook.py`.

#### Exact limited-endpoint linearization diagnosis (2026-11-09)

A bounded diagnostic was run from the archived rejected `attempt_047` state
at `transport_fraction = 0.9995532653808594`, using the exact
`transport_fraction = 1` equations. No tolerance, solver budget, or production
equation was changed, and this did not produce an accepted endpoint root.

The natural active-set frozen bordered preconditioner has structural rank
1907 of 1923 and sparse LU reports an exactly singular factor. This is a
preconditioner-matrix failure; it is not proof that the full matrix-free
Jacobian is singular. Continuing the existing one-way free-content pivot rule
as a diagnostic beyond its production eight recovery models raised structural
rank only to 1914 of 1923 after 29 pivots, then stopped finding eligible
content pivots. Unpreconditioned GMRES reached its existing 10-cycle,
restart-60 budget (600 iterations) with independently recomputed relative
residual `0.519`.

Two preconditioner-only alternatives were screened, without promotion:

- Replacing the limited-capacity reconstruction by donor reconstruction in
  the sparse preconditioner made that matrix structurally full rank, but the
  true limited matrix-free solve stalled at relative residual `1.216e-2`
  after 600 iterations. Extending this diagnostic to 50 cycles (3000
  iterations) did not improve the residual materially.
- An all-full active-row matrix factorized, but using it to precondition the
  natural active-set Jacobian did not converge (true residual `0.283` after
  1200 iterations). Solving its own alternate active-row system produced a
  direction with relative linear residual `6.78e-13`, but the original
  nonlinear feasibility bound reduced the trial fraction to `1.83e-9`;
  this is not meaningful progress.

A nonzero guide formed from the preceding archived continuation-stage
difference also left the natural exact-endpoint preconditioner singular.
These results do not justify increasing iteration limits, relaxing mass
acceptance, or changing the nonlinear equations. The remaining numerical
obstacle is the rank-deficient/ineffective active-set linear correction near
the limited-capacity endpoint. A future remedy must retain the exact
matrix-free operator and independent true-residual checks, demonstrate a
feasible Armijo step, and pass all unchanged endpoint audits before it can be
considered. No pressure-spike or experimental-validation conclusion follows
from this diagnostic.

## Ciulli Figure 13 donor-flux reanalysis (2026-11-11)

This reanalysis uses only independently accepted donor-flux
(`transport_fraction = 0`) roots. It supersedes the older statements above that higher-grid fields
were unavailable and that no Figure 13 comparison had been made. It does not
change the equations, physical model, or any acceptance criterion. Donor-flux
results are not results for the limited-capacity (`transport_fraction = 1`)
endpoint.

### Reproduction and retained fields

The SI case is
[`cases/ciulli_15n_donor.json`](../cases/ciulli_15n_donor.json). The runner
[`examples/run_ciulli_donor_grid.py`](../examples/run_ciulli_donor_grid.py)
solves and independently audits a requested grid, then archives the full
dimensionless state, pressure, film, content, coordinates, exact input,
constitutive coefficients, solver budgets, package/runtime metadata, source
and field checksums, and convergence history. It refuses to overwrite an
existing output directory. Set BLAS/OpenMP to one thread for reproducibility:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:PYTHONPATH = "src"
python examples/run_ciulli_donor_grid.py --grid 33 --half-width-mm 0.4 --output results\ciulli_33
```

The five distinct root configurations are `(33, 0.4 mm)`, `(65, 0.4 mm)`,
`(129, 0.4 mm)`, `(81, 0.5 mm)`, and `(97, 0.6 mm)`; the 65x65 root is shared
between mesh and domain studies. Analyze retained roots without rerunning them:

```powershell
python examples/analyze_ciulli_donor_grid.py `
  --mesh results\ciulli_33 results\ciulli_65 results\ciulli_129 `
  --domain results\ciulli_65 results\ciulli_81 results\ciulli_97 `
  --output results\ciulli_analysis.json
```

The result archives and analysis JSON remain in ignored `results/`; large
simulation output is not committed. All five accepted archives are available
locally and can be recreated from the tracked case, runner and solver commit.
The exact run summaries record inputs and hashes; the verified 33x33 runner
execution took about 1.5 seconds on this machine.

### Level 1: equation closure

| Grid and half-width | Spacing [um] | h_min / h_c [nm] | p_max [MPa] and (x,y) [um] | Relative local / global mass residual |
|---|---:|---:|---|---:|
| 33x33, +/-0.4 mm | 25.000 | 216.706 / 456.136 | 760.606 at (50,-25) | 2.93e-15 / 5.91e-17 |
| 65x65, +/-0.4 mm | 12.500 | 282.747 / 492.246 | 746.301 at (0,0) | 6.25e-15 / 3.83e-17 |
| 129x129, +/-0.4 mm | 6.250 | 285.231 / 485.267 | 741.270 at (0,0) | 1.79e-14 / 7.58e-17 |
| 81x81, +/-0.5 mm | 12.500 | 283.483 / 493.981 | 746.063 at (0,0) | 9.65e-15 / 3.79e-17 |
| 97x97, +/-0.6 mm | 12.500 | 283.809 / 494.698 | 745.813 at (0,0) | 6.41e-15 / 8.59e-17 |

Every listed root passed the independent force, local and global SI mass,
JFO complementarity, film-reconstruction, and direct-elasticity audits.
Integrated force is 15 N to reported precision. These checks establish
equation closure for the declared donor discretization and case; they do not
establish mesh convergence or physical accuracy.

### Level 2: mesh and domain studies

The fixed-domain mesh study and fixed-spacing domain study are independent.
Fields were interpolated to common physical coordinates, with no comparison
by grid index. The declared two-dimensional main-contact comparison window
is `x=[-100,60] um`, `y=[-75,75] um`; the separate midplane comparison uses
`[-100,100] um`.

| Fixed-domain comparison | Main-region pressure relative L2 / RMS / Linf | Main-region film relative L2 / RMS / Linf |
|---|---|---|
| 33 -> 65 | 7.74% / 42.69 / 236.02 MPa | 10.15% / 44.58 / 109.64 nm |
| 65 -> 129 | 5.71% / 31.71 / 233.23 MPa | 2.01% / 9.59 / 40.92 nm |

On the `y=0` longitudinal and `x=0` transverse midplanes, 65 -> 129 gives
pressure relative L2/RMS of 5.51%/33.79 MPa longitudinally and
4.98%/28.95 MPa transversely; film relative L2/RMS is 2.21%/10.84 nm and
1.92%/8.24 nm, respectively. Reprocessing found that the earlier midplane
analysis applied a micrometre-to-metre conversion twice to coordinates
already stored in SI, so the earlier midplane values in this report were
incorrect and are superseded here. This was an analysis-only correction;
the accepted solver fields are unchanged. Corrected 33 -> 65 midplane film
RMS is 49.76 nm longitudinally and 46.38 nm transversely. The main-contact
2D Linf pressure changes remain large, and the minimum/central-film sequence
is nonmonotone (`h_min`: +30.5%, then +0.88%; `h_c`: +7.92%, then -1.42%).
The two finer meshes show reduced film sensitivity but do not prove
full-field convergence.

| Fixed-spacing domain expansion | Main-region pressure relative L2 / RMS / Linf | Main-region film relative L2 / RMS / Linf |
|---|---|---|
| 65 +/-0.4 -> 81 +/-0.5 mm | 0.299% / 1.656 / 9.504 MPa | 0.329% / 1.572 / 3.202 nm |
| 81 +/-0.5 -> 97 +/-0.6 mm | 0.177% / 0.980 / 5.339 MPa | 0.143% / 0.688 / 1.679 nm |

The domain sequence holds spacing at about 12.5 um. Its contact-region
scalar changes are also small: 65 -> 97 changes minimum/central film by
0.38%/0.50% and maximum pressure by -0.07%. The centerline pressure-release
start is at `x=125 um` for these three domain roots. This supports low
artificial-boundary sensitivity in the stated contact region at this
resolution; it is not a claim that the full pressure-release region or
outlet feature is domain-independent.

### Outlet spike and pressure release

The pressure regions are treated separately: main contact as above, outlet
spike in `x=[75,175] um`, and downstream pressure release/cavitation thereafter.
At 33x33 and 65x65 no interior local peak is identified in the outlet window.
At 129x129 the diagnostic identifies a peak at `x=81.25 um`,
`617.834 MPa`; the adjacent sampled pressures are `599.135 MPa` before and
`205.708 MPa` after the peak. The linearly interpolated half-prominence width
is `3.593 um`, only `0.575` of a 6.25 um grid cell. This peak is
under-resolved; neither its height nor width is a converged physical
quantity. Its changing shape and coarse-grid absence are not pass/fail
comparisons to a smooth Hertz pressure curve. The centerline first
cavitated node (`p <= 1 kPa`, `theta < 1-1e-10`) is 125 um on 33/65 and
118.75 um on 129.

The failed 193x193 attempt remains a failed iterate, not a fourth mesh root.
The recorded 30-correction limit, 15 N integrated force, relative local/global
mass residuals `8.589e-1`/`3.273e-4`, maximum complementarity `1.047e-3`,
and maximum SI divergence `97.08 kg/(m^2 s)` establish nonlinear
nonconvergence. Its iteration, line-search, active-set and Krylov histories
were not retained. The available evidence cannot identify a specific
linear-solve or correction mechanism, and the distinct limited-capacity
endpoint diagnosis must not be used to attribute this donor-run failure.
Do not use the final iterate as a solution or increase iteration limits
without a diagnostic plan.

### 257x257 fixed-domain refinement and three-grid assessment

The accepted 257x257 donor-flux root used the same `+/-0.4 mm` domain,
equations, physical inputs, boundary conditions and acceptance tolerances as
the 129x129 root. The accepted 129x129 pressure/content fields were
bilinearly prolonged as a feasible but unaccepted initial guess. The only
resource-setting change was an explicit 70,000-node budget for the 66,049
physical nodes. Seven Newton corrections completed in 365.8 s. The run
recorded 4-thread OpenBLAS configuration, but GMRES/SuperLU, matrix-vector
products and FFT convolution were not configured as parallel solver paths.
Its full fields, per-correction diagnostics and checkpoints remain in the
local ignored result archive; the reproducible runner and analysis scripts
are committed separately.
The numerical run used source revision `097637d74adb4c62fe523d0de8f9b705c1139548`.
Afterward, the persisted per-iteration records were reconciled against the
compact solver history to restore accepted-trial markers; this repaired
diagnostic metadata only and did not alter the accepted state. The runner
now performs that reconciliation on completion, and the Newton diagnostic
callback writes after accepted line-search selection for future runs.

The independent final audit reports 15 N integrated load (zero relative
force error), relative local/global mass balances `6.45e-14`/`5.66e-17`,
maximum SI mass divergence `8.75e-12 kg/(m^2 s)`, maximum JFO
complementarity `1.18e-25`, film reconstruction error `1.82e-20 m`, and a
passing direct-elasticity check. The root is accepted as equation closure
for this discrete donor-flux problem, not as evidence of physical validation.

The nested fixed-domain grids have 64, 128 and 256 intervals (refinement
ratio 2). The common-coordinate pairwise field comparisons are:

| Region and grid pair | Pressure relative L2 / RMS / Linf | Film relative L2 / RMS / Linf |
|---|---|---|
| Full contact, 65 -> 129 | 7.43% / 38.52 / 237.48 MPa | 2.40% / 11.24 / 47.49 nm |
| Full contact, 129 -> 257 | 6.18% / 32.23 / 299.17 MPa | 2.12% / 9.94 / 23.91 nm |
| Main contact excluding outlet, 65 -> 129 | 5.71% / 31.71 / 233.23 MPa | 2.01% / 9.59 / 40.92 nm |
| Main contact excluding outlet, 129 -> 257 | 4.48% / 24.83 / 275.09 MPa | 2.21% / 10.47 / 22.63 nm |
| Outlet/spike region, 65 -> 129 | 18.50% / 59.06 / 237.48 MPa | 3.77% / 16.23 / 47.49 nm |
| Outlet/spike region, 129 -> 257 | 15.41% / 53.39 / 299.17 MPa | 1.69% / 7.48 / 23.91 nm |

Pressure RMS norms contract only modestly in the main region, from 31.71 to
24.83 MPa (factor 1.28); the corresponding difference-norm order is about
0.35 and is descriptive, not an accepted field convergence order. Main
pressure Linf increases from 233.23 to 275.09 MPa. Main film RMS increases
from 9.59 to 10.47 nm, while film Linf decreases. These mixed behaviors do
not demonstrate asymptotic full-field convergence. The outlet pressure RMS
also contracts only slightly (59.06 to 53.39 MPa) and its Linf increases.
In the full-contact `x=[-100,100] um` pressure-error energy, the main/outlet
shares are 54.70%/45.30% for 65 -> 129 and 48.12%/51.88% for 129 -> 257.
The separate release-region pressure-error energy is only 0.044% and 0.011%
of the respective full-contact energy, so it is not the source of the
discrepancy.

The dry-Hertz footprint with `a_H=100 um` is used as the core/transition
partition; it covers 93.65% of the declared main-region area. The core
contains 99.787% of main-region squared pressure-error energy for 65 -> 129
and 99.977% for 129 -> 257. Thus the remaining difference is in the
physically relevant core, not primarily in the outer transition.

| Region / grid pair | Pressure relative L2 / RMS / Linf | Film relative L2 / RMS / Linf |
|---|---|---|
| Hertz core, 65 -> 129 | 5.71% / 32.73 / 233.23 MPa | 2.04% / 9.57 / 40.92 nm |
| Hertz core, 129 -> 257 | 4.49% / 25.65 / 275.09 MPa | 2.25% / 10.46 / 22.63 nm |
| Transition, 65 -> 129 | 5.57% / 5.80 / 13.84 MPa | 1.62% / 9.82 / 15.46 nm |
| Transition, 129 -> 257 | 1.53% / 1.51 / 5.00 MPa | 1.77% / 10.58 / 13.07 nm |

As expected from the spatial error-energy shares, the core contains nearly
all the pressure Linf for both refinements; transition differences are
small in absolute pressure even where their relative L2 is nonzero.

The largest 65 -> 129 main-region difference is 233.23 MPa near
`(50,-56.5) um`; a `10 um` half-width neighborhood there contributes 17.07%
of the main-region error energy. For 129 -> 257 the largest difference is
275.09 MPa near `(59.5,53) um`. Both comparisons have zero cavitated area
and zero active-set mismatch in the main region, with theta differing only
by roundoff. This is a broader full-film pressure-ridge shape/location
change, not an isolated hotspot or cavitation-front displacement.

Corrected common-coordinate midplane differences are:

| Grid pair / profile | Pressure relative L2 / RMS / Linf | Film relative L2 / RMS / Linf |
|---|---|---|
| 65 -> 129, longitudinal (`y=0`) | 5.51% / 33.79 / 204.36 MPa | 2.21% / 10.84 / 41.27 nm |
| 65 -> 129, transverse (`x=0`) | 4.98% / 28.95 / 112.43 MPa | 1.92% / 8.24 / 22.41 nm |
| 129 -> 257, longitudinal (`y=0`) | 4.22% / 25.76 / 216.38 MPa | 2.32% / 11.28 / 19.16 nm |
| 129 -> 257, transverse (`x=0`) | 3.19% / 18.50 / 90.64 MPa | 2.15% / 9.20 / 11.64 nm |

The midplane pressure L2/RMS differences decrease, but longitudinal
pressure Linf increases and both film RMS differences increase. Midplanes
therefore also do not establish convergence.

| Scalar | 65x65 | 129x129 | 257x257 | Assessment |
|---|---:|---:|---:|---|
| `h_c` [nm] | 492.246 | 485.267 | 473.629 | Changes grow (6.98 then 11.64 nm); no convergence |
| `h_min` [nm] | 282.747 | 285.231 | 281.997 | Nonmonotone; no convergence estimate |
| `p_max` [MPa] | 746.301 | 741.270 | 739.669 | Monotone scalar decrease |
| Integrated load [N] | 15.000 | 15.000 | 15.000 | Prescribed load; not a discretization indicator |

For `p_max` alone, the three values give a formal observed order of 1.65,
Richardson estimate 738.922 MPa and fine-grid GCI 0.126% (safety factor
1.25), with asymptotic ratio 1.002. This is a conditional scalar estimate,
not a field GCI. The maximizing location moves from `(0,0) um` on 65x65 and
129x129 to `(53.125,56.25) um` on 257x257, so the scalar fit is not strong
evidence that the peak location or surrounding field has converged. Do not
use this GCI to support physical film validation.

The 257x257 outlet peak is at `x=84.375 um`, `635.175 MPa`, with an
interpolated half-prominence width of `3.960 um`, only `1.267` grid cells.
At 129x129 it is at `81.25 um`, `617.834 MPa`, width `3.593 um` or `0.575`
cells. The adjacent sampled pressures are `599.135`/`205.708 MPa` at 129
and `594.469`/`253.466 MPa` at 257 (before/after the peak). Integrated
absolute pressure over each grid's detected half-prominence x-window and
`|y|<=100 um` is `0.11479 N` and `0.10245 N`, respectively. Because the
window follows the grid-dependent detected width, this is a diagnostic
local integral, not a fixed-window convergence norm. The spike remains
under-resolved: neither its height nor width is converged.

The 257 result shows some contraction in pressure L2/RMS profiles, but the
main pressure field, film field, `h_c`, and `h_min` do not yet support a
mesh-converged claim. A uniform 513x513 grid would be a useful next
refinement for the main field, but entails 263,169 nodes (four times the
257 grid) and exceeds the current explicit node budget; it is unlikely to
resolve a roughly 4 um spike adequately (about 2.5 cells across the prior
width estimate). Do not launch it without a bounded memory/runtime
preflight and diagnostic capture. The solver currently requires uniform
axes, so nonuniform/local refinement is a future capability needed to
study the spike efficiently. The 257 data do not establish convergence
sufficient for a defensible physical-validation claim.

### Level 3: comparison with Figure 13

The finest accepted field was sampled on the two digitized midplanes. For 21
points in each profile:

| Profile | Bias [nm] | RMSE [nm] | Maximum absolute error [nm] | NRMSE / mean; range | Within pointwise digitization estimate |
|---|---:|---:|---:|---:|---:|
| Longitudinal | +19.22 | 40.75 | 129.60 | 8.71%; 21.45% | 5/21 |
| Transverse | +32.98 | 37.37 | 70.73 | 9.59%; 17.79% | 4/21 |

Digitization RMS estimates are 24.20 nm longitudinally and 24.69 nm
transversely. The paper's method uncertainty is about 10 nm and the authors
estimate an additional optical effect of about the same order; treating all
three as independent RMS errors would give only an illustrative combined
scale near 28-29 nm, not a formal confidence interval. The estimated
coordinate read-off uncertainty is +/-2 um and has not been propagated at
steep gradients. The measured profiles were not corrected for
pressure-dependent refractive index. The maximum discrepancies and the
transverse bias exceed the quoted plot-digitization scale. Thus these results
are a quantitative comparison, not agreement within the known experimental
uncertainty and not a physical-validation pass.

### Model/input uncertainty and evidence classification

The case uses the paper's `eta_0=0.497 Pa s`, `rho_0=892 kg/m^3`,
`alpha_p=2.87e-8 Pa^-1`, 12 deg C, 15 N, and 0.2 m/s. The effective modulus
is inferred from the paper's rounded Hertz radius. Roelands `z=0.626999` is
matched to the zero-pressure viscosity slope only. The solver's assumed
Dowson-Higginson coefficients are `a=0.6e-9/Pa`, `b=1.7e-9/Pa`; they are not
reported by Ciulli. At the computed 129x129 peak pressure of 741.27 MPa this
law gives density about 1067.5 kg/m^3 (19.7% above reference). At the same
pressure, slope-matched Barus and Roelands formulas evaluate to roughly
`8.62e8` and `1.56e6 Pa s`, respectively. This illustrates potentially
material constitutive-model difference, not an uncertainty bound or evidence
that Barus is the correct high-pressure law. Controlled solution sensitivities
are reported below. No fluid parameters were tuned.

The paper reports the nominal load, entrainment speed, temperature and ball
diameter used here but does not provide uncertainty ranges for these inputs or
for the rounded Hertz-radius-based modulus inference. Those uncertainties
were not perturbed; assigning arbitrary ranges would not produce a defensible
input-uncertainty estimate.

A controlled fixed-domain sensitivity was solved on both 65x65 and 129x129
grids. It compared four combinations of the reported zero-pressure viscosity
slope and two density assumptions. Each alternative root used the same
bounded Newton/Krylov budgets and passed the unchanged independent force,
mass, JFO, film and elasticity audits:

| 65x65 model | h_min / h_c [nm] | p_max [MPa] | Figure 13 longitudinal / transverse RMSE [nm] |
|---|---:|---:|---:|
| Roelands + Dowson-Higginson (baseline) | 282.75 / 492.25 | 746.30 | 42.70 / 39.82 |
| Roelands + constant density | 294.59 / 574.78 | 848.79 | 100.53 / 94.47 |
| Barus + Dowson-Higginson | 292.49 / 507.07 | 747.56 | 52.11 / 51.33 |
| Barus + constant density | 305.86 / 594.44 | 867.72 | 117.54 / 110.41 |

| 129x129 model | h_min / h_c [nm] | p_max [MPa] | Figure 13 longitudinal / transverse RMSE [nm] |
|---|---:|---:|---:|
| Roelands + Dowson-Higginson (baseline) | 285.23 / 485.27 | 741.27 | 40.75 / 37.37 |
| Roelands + constant density | 299.58 / 567.39 | 969.64 | 96.29 / 91.57 |
| Barus + Dowson-Higginson | 294.90 / 500.36 | 742.81 | 50.02 / 49.23 |
| Barus + constant density | 311.13 / 586.95 | 1015.31 | 113.41 / 107.95 |

Relative to each grid's baseline, at 65x65 constant density with Roelands
changes central film by 16.77%, p_max by 13.73%, pressure relative L2 by
11.88%, and film-field RMS by 37.61 nm. Barus with Dowson-Higginson changes
central film by 3.01%, p_max by 0.17%, pressure relative L2 by 1.16%, and
film RMS by 11.49 nm. Barus with constant density changes central film by
20.76%, p_max by 16.27%, pressure relative L2 by 13.29%, and film RMS by
51.30 nm.

At 129x129, constant density with Roelands changes central film by 16.92%,
p_max by 30.81%, pressure relative L2 by 17.85%, and film-field RMS by
37.96 nm. Barus with Dowson-Higginson changes central film by 3.11%,
p_max by 0.21%, pressure relative L2 by 1.24%, and film RMS by 11.80 nm.
Barus with constant density changes central film by 20.95%, p_max by
36.97%, pressure relative L2 by 19.77%, and film RMS by 52.05 nm. The
corresponding film-field Linf differences are 82.13, 17.93, and 101.68 nm.
Profile RMSE increases for each alternative at both resolutions.

These are controlled model-form sensitivities, not a probability
distribution or error bar. The alternatives are not fitted to SN500 and do
not identify which law is physically correct. They show that the unresolved
pressure-density and viscosity-pressure laws materially affect the
comparison; a defensible physical-validation claim needs a justified
constitutive choice or uncertainty range. Similar percentage changes at the
two grids are sensitivity evidence, not proof of convergence for each model.

The study is reproducible with
[`examples/run_ciulli_constitutive_sensitivity.py`](../examples/run_ciulli_constitutive_sensitivity.py),
using the accepted 65x65 or 129x129 root as `--baseline`. It stores every
alternative full field, independent audits, source and input hashes, and
runtime/solver controls under ignored `results/`. Each four-case run took
about 6 seconds at 65x65 and 181 seconds at 129x129 on this machine.

```powershell
python examples\run_ciulli_constitutive_sensitivity.py `
  --baseline results\ciulli_20261111_donor_129_fixed `
  --output results\ciulli_sensitivity_129
```

| Review area | Status | Evidence and minimum remaining work |
|---|---|---|
| Reproducibility | PARTIAL | Tracked input, run/analyze scripts, checksums and regression exist; archives are local/ignored. Recreate them from the committed inputs/scripts when needed. |
| Equation/implementation verification | DONE | Five donor roots pass independent load, mass, JFO, film and elasticity checks; component tests cover Hertz/Reynolds/cavitation. This does not substitute for mesh convergence. |
| Formulation target | DONE | Donor-flux is the declared initial target. Exact `tau=1` limited-capacity remains a separate unfinished formulation and is not a donor blocker. |
| Mesh convergence | PARTIAL | Fixed-domain common-coordinate metrics are measured, but fine-grid changes and spike resolution prevent an asymptotic/full-field convergence claim. |
| Domain independence | DONE | Contact-region sensitivity is small over +/-0.4 to +/-0.6 mm at about 12.5 um spacing; release/spike and other domains are outside this conclusion. |
| Outlet pressure spike | PARTIAL | Detected only at 129x129 and resolved by less than one cell. A finer accepted field is needed before reporting stable peak/width. |
| Frozen numerical reference | PARTIAL | Case, formulation, solver controls and grids are specified; no numerically converged mesh or uncertainty bound is frozen yet. |
| Ciulli physical validation | PARTIAL | Quantitative comparison is complete, but mismatch exceeds estimated reference uncertainty and numerical/model uncertainty is not closed. |
| Model/input uncertainty | PARTIAL | Controlled 65x65 and 129x129 alternative-law roots show material effects; a justified high-pressure SN500 law or credible model/input range remains missing. |
| Regression testing | DONE | A deterministic 33x33 donor solve checks independent audits and h_min, h_c, p_max regression values; analysis metrics have unit tests. |

### Shortest defensible next steps

**Blocking for a physical-validation claim:** (1) quantify the measured-profile
mesh error with an additional accepted refinement or other independently
verified convergence evidence; assess the steep constriction/outlet samples
separately rather than requiring 193x193 by default; (2) use applicable SN500
high-pressure properties or establish a defensible range around the already
measured constitutive-model sensitivity; (3) propagate the stated +/-2 um
coordinate uncertainty at steep profile gradients and keep it separate from
model/numerical errors. Do not tune inputs to improve the fit.

**Recommended:** resolve the outlet spike with a finer, well-conditioned
accepted grid and assess peak location, shape and integral behavior; add a
matched published numerical EHL reference (the NASA candidates remain
unmatched). These strengthen numerical verification and any pressure-field
claim but do not change what Figure 13 measures.

**Optional/future:** original interferograms, calibration files and
pressure-dependent refractive-index data may improve the optical reference,
but are not prerequisites for this initial comparison. The exact limited-
capacity endpoint, thermal behavior, nonzero slide-to-roll cases and other
papers remain separate milestones.

The 13 pre-existing untracked files were reviewed and left untouched: `.kilo/`
agent/skill assets are local customization, `ehl-run.spec` is a generated
machine-specific PyInstaller specification with absolute paths, and
`cases/circular_50n_316.json` duplicates the tracked `cases/circular_50n_100k.json`.
They are not inputs to these results and should not be committed. New
validation deliverables are limited to the Ciulli case, runner, analysis,
tests, and the directly related documentation; result arrays remain ignored.

### Regional localization of the 65x65 -> 129x129 difference

The accepted complete pressure and film fields were reprocessed without
rerunning either solve. Bilinear field interpolation onto a common 0.5 um
coordinate grid covers `x=[-100,175] um`, `y=[-75,75] um`. The comparison
reports area-weighted trapezoidal L2/RMS values; pressure L2 and RMS use the
union of samples with either field above 1 kPa. The Linf location is reported
in physical `(x,y)` coordinates. The windows are partitioned into main
contact `[-100,60] um`, outlet/spike `[60,100] um`, and release `[100,175] um`;
the full contact window is `[-100,100] um`.

| Region | Pressure relative L2 / RMS / Linf | Film relative L2 / RMS / Linf |
|---|---|---|
| Full contact, x=[-100,100] um | 7.43% / 38.52 / 237.48 MPa | 2.40% / 11.24 / 47.49 nm |
| Main, x=[-100,60] um | 5.71% / 31.71 / 233.23 MPa | 2.01% / 9.59 / 40.92 nm |
| Outlet/spike, x=[60,100] um | 18.50% / 59.06 / 237.48 MPa | 3.77% / 16.23 / 47.49 nm |
| Pressure release, x=[100,175] um | 43.35% / 3.26 / 8.38 MPa | 0.16% / 2.13 / 9.16 nm |

About 54.70% of the full-contact squared pressure difference is in the main
region and 45.30% in the outlet/spike region. The release-region pressure
difference is only 0.044% of the full-contact squared difference despite its
large relative L2, because pressure there is small. Thus the 5.71% main
pressure difference is real and cannot be attributed to the outlet spike
alone. The main-region pressure Linf is at approximately `(50,-56.5) um`;
the outlet-region Linf is at `(75,31) um`. This indicates substantial
trailing/lateral pressure-support structure changes as well as a localized
outlet difference. Main-region film RMS is 9.59 nm, with its Linf at about
`(50,56.5) um`.

On the centerline outlet profile the 65x65 field has no strict local peak in
`[75,100] um`; its maximum in the window is the left boundary value
645.43 MPa at 75 um. The 129x129 field has a strict sampled peak of
617.83 MPa at 81.25 um. As the coarse result does not resolve a local peak,
this is not a defensible estimate of peak displacement. A profile-translation
diagnostic reduces centerline outlet RMS from 74.31 to 48.42 MPa with a
2.75 um shift, but the remaining shape error is substantial and this fitted
shift is not a physical peak-location estimate. The pressure-release region
is therefore not the source of the large L2 difference; the main and
outlet/spike regions both matter, and spike location versus shape cannot yet
be separated reliably.

Reproducible regional JSON metrics and common-coordinate difference arrays
are generated by the analysis script's optional `--regional-fields` output.
The compressed arrays are intentionally local under ignored `results/`, not
committed as a large derived dataset. The 65x65 -> 129x129 common-coordinate
maps are sufficient to locate current differences, but not to establish
convergence or to treat either field as a reference solution.

Uniform global refinement is currently the only available mesh strategy:
the coupled solver enforces uniform axes and its elasticity/Reynolds
discretizations assume uniform spacing. The regional result does not support
refining only the spike; roughly half of pressure error energy is already in
the main contact, and the main-region Linf is large. A spatially interpolated
129x129 donor solution is the starting field for the next consistent
refinement `65 -> 129 -> 257`, with 64, 128 and 256 intervals (`r=2`).
The prior failed 193x193 attempt is not reused as a solution or repeated.
The 257x257 node count (66,049) requires the explicitly supported
70,000-node budget instead of the case's default 20,000; all physical,
nonlinear, Krylov and independent acceptance gates remain unchanged.

### Main-contact pressure structure and 257x257 run diagnostics

The main-contact split uses the case's Hertz radius, `a_H=100 um`, calculated
from the reported load, equivalent radius and modulus. This is a geometry-
based partition independent of the pressure-difference map: the core is the
portion of `x=[-100,60] um`, `y=[-75,75] um` inside `x^2+y^2<=a_H^2`;
the transition region is the remainder of that rectangle.

| Region | Area share of main | Pressure relative L2 / RMS / Linf | Share of main pressure-error energy |
|---|---:|---|---:|
| Hertz-footprint core | 93.65% | 5.71% / 32.73 MPa / 233.23 MPa | 99.787% |
| Outside-footprint transition | 6.35% | 5.57% / 5.80 MPa / 13.84 MPa | 0.213% |

This partition indicates the discrepancy lies overwhelmingly inside the
nominal dry-contact footprint, not in the small outside-footprint corner
transition. It is not a Reynolds-region or cavitation-boundary definition.
The maximum difference at `(50,-56.5) um` is not isolated: the common-grid
10x10 um half-width window around it contains 17.07% of the main-region
squared pressure error. The largest-error area is mirrored at positive and
negative lateral coordinates, and pressure-ridge maxima move along `x`:
at `y=-56.5 um`, the maximum over `x=[0,60] um` moves from 12.5 um on 65x65
to 44 um on 129x129; at `y=-60 um` it moves from 12.5 to 31 um. The maximum
pressures on those slices change less than the large pointwise differences,
consistent with a broad ridge displacement/reshaping rather than one
isolated pressure defect.

The main region is full-film on both grids: the `p<=1 kPa, theta<1-1e-10`
cavitation-mask mismatch is zero, both cavitated-area fractions are zero,
and the maximum interpolated theta difference is roundoff
(`4.44e-16`). The hotspot therefore is not an active-set/cavitation-boundary
effect. This evidence identifies a mesh-dependent full-film pressure ridge
and field shape within the Hertz footprint.

The 257x257 seed was preflighted by bilinear interpolation of the accepted
129x129 pressure and content fields while preserving the physical central
offset. The target state is strictly feasible but is only an unaccepted
initial guess. Its scaled initial residual has L2 `0.17383`, Linf
`3.887e-3`, mass-block Linf `2.341e-3`, complementarity Linf `3.887e-3`,
and relative force residual `1.18e-16`. The identical `+/-0.4 mm` domain,
donor flux (`transport_fraction=0`), material inputs, boundary conditions
and solver tolerances are retained; only the explicitly supported node
budget changes to 70,000.

The Newton driver now persists per-iteration scaled residual components,
independent force/mass/JFO/film/elasticity audit, all recovery-model
statuses, Krylov histories and true residuals, all line-search trials,
correction norms, and packed active-set masks. It writes field checkpoints
at each audited nonlinear iterate and preserves the rejected final candidate
plus termination reason. A checkpoint can be supplied again as `--seed`;
it remains a guess, never an accepted root. Runtime metadata records
environment thread limits and independently queried NumPy and SciPy
OpenBLAS thread counts. Checkpoints retain pressure, fractional content,
film, deformation, density, viscosity, axes, and the dimensionless restart
state.

Backend inspection found SciPy sparse GMRES and SuperLU in the Newton linear
path, with NumPy's FFT convolution and Python matrix-free products; those
parts are not configured here as multithreaded algorithms. SciPy reports an
OpenBLAS build. The run is launched with OpenBLAS/OpenMP/MKL/BLIS/NumExpr
limits set to four; the loaded SciPy OpenBLAS runtime reports four threads.
No competing solver jobs are used. OpenBLAS is the only positively
verified threaded backend setting; this does not imply the complete solve
uses four cores.

The accepted 257x257 root has now been compared with the accepted 65x65 and
129x129 roots on common physical coordinates; the results are detailed in
the preceding "257x257 fixed-domain refinement and three-grid assessment"
section. Pressure RMS differences contract slowly, but the main pressure
and film fields and `h_c`/`h_min` do not demonstrate mesh convergence. Only
`p_max` admits a conditional scalar Richardson/GCI estimate, with the
maximizing location changing on the finest grid. The outlet peak is still
under-resolved at 1.267 cells across its estimated width.

### Resource preflight for 513x513 donor-flux refinement

A 513x513 preflight was run against the actual accepted 257x257 seed and
target-grid donor linearization, including assembly and SuperLU
factorization of the sparse preconditioner. The measured host has 63.2 GiB
physical RAM; 26.18 GiB was available at the start of the instrumented
preflight and 24.74 GiB remained after the target factorization. The measured
peak working set was 1.447 GiB and peak private usage 2.510 GiB.

| Component | Measured/estimated size |
|---|---:|
| 513x513 grid / interior unknowns | 263,169 / 261,121 |
| Newton vector `(2*n+1)` | 4.18 MB |
| GMRES restart basis (60+2 vectors) | 259 MB |
| Elasticity operator workspace estimate (`128 * product(fft_shape)`) | 304 MB |
| Sparse bordered preconditioner, assembled | 2,608,168 nonzeros / 33.4 MB |
| SuperLU L/U factors | 52,903,113 nonzeros / 639 MB |
| Preflight process peak working set / private bytes | 1.447 / 2.510 GiB |
| Available RAM before / after exact matrix and factorization | 26.18 / 24.74 GiB |
| Free results-volume storage | 103 GiB |

An 8 GiB conservative process working-set budget is set for the run, retaining
at least 16 GiB available to the OS and other applications. Uncompressed
field checkpoints (restart state, pressure, film, deformation, density,
viscosity, cavitation content and axes) are estimated below 0.6 GiB for 31
correction checkpoints plus final/latest snapshots; active-set and JSON
diagnostics are budgeted below 0.1 GiB. A 1 GiB storage reserve is therefore
well below available disk.

The run environment sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `BLIS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 4.
Independent runtime queries report 4 threads for both NumPy and SciPy
OpenBLAS. The iterative NumPy FFT, Python matrix-free products, and SuperLU
factorization are not configured as parallel solvers; no nested solver
parallelism or competing solver jobs are used. The existing runner's node
budget accepted up to 100,000 nodes, so this specific run uses the explicit
300,000-node limit for 263,169 nodes; equations, model, line-search,
nonlinear, Krylov, and independent acceptance gates are unchanged.

The resource preflight passed. The 513x513 diagnostic run and four-grid
comparison are now complete; the evidence and decision are recorded below.

### 513x513 donor-flux run and four-grid mesh assessment

The accepted 513x513 donor-flux run was seeded by bilinear prolongation from
the accepted 257x257 state. It retained the same +/-0.4 mm domain, governing
equations, donor-flux formulation (`transport_fraction=0`), material model,
boundary conditions and physical/nonlinear/audit acceptance gates. The only
required setting change was the supported node budget, raised to 300,000 for
263,169 physical nodes. It ran with Python 3.12.9, NumPy 2.5.4 and SciPy
1.18.1. NumPy and SciPy OpenBLAS independently reported four threads under
four-thread BLAS/OpenMP/MKL/BLIS/NumExpr environment limits. This does not
mean the full solve ran in parallel: the sparse GMRES/SuperLU, FFT and
Python matrix-free paths were not configured as multithreaded solvers.

The run completed in 1,383.96 s with seven Newton corrections and was
accepted by the independent endpoint audit. The 15 N force error was
2.22e-16 relative; relative local and global mass balances were 8.29e-14 and
7.52e-17; maximum SI mass divergence was 3.69e-11 kg/(m^2 s);
complementarity was 3.98e-27; film reconstruction error was 5.38e-20 m; and
direct elasticity was checked. The local ignored archive
`results/ciulli_20261111_donor_513_fixed` contains the complete final fields,
restart state, eight checkpoints, seven nonlinear diagnostic records and
active-set snapshots. Its state checksum is
`4d6376f51c0056f0414a8bed3cab8916c88f27db8f738d8ab77f060ffcf0ce06` as
recorded in `summary.json`. Compact committed run and seed provenance is
provided in [cases/ciulli_513_donor_run_metadata.json](../cases/ciulli_513_donor_run_metadata.json).
The 4-grid reanalysis is
`results/ciulli_20261111_mesh_convergence_65_129_257_513.json`; these large
result artifacts remain local and are not committed.

The resource-monitor wrapper captured the Python launcher rather than its
spawned solver interpreter. Consequently it did not measure solver peak
working-set/private memory; its launcher values are not solver-memory
measurements. It did observe available system RAM fall no lower than
23.81 GiB from 26.15 GiB initially, with no configured RAM threshold
crossed. The instrumented preflight's direct solver-process measurements,
not the wrapper's launcher sample, remain the valid process-memory evidence.

The 65x65, 129x129, 257x257 and 513x513 roots were compared on the common
physical region `x=[-100,60] um`, `y=[-75,75] um` for the main contact and
the documented full-contact/outlet/release regions. Pressure norms use the
union of samples with pressure above 1 kPa; film norms use the full sampled
region. The pairwise maps/norms use a 0.5 um common grid, and the line
profiles use common physical coordinates from -100 to 100 um. The analysis
is reproducible with `examples/analyze_ciulli_donor_grid.py --mesh
results/ciulli_20261111_donor_65_fixed
results/ciulli_20261111_donor_129_fixed
results/ciulli_20261111_donor_257_fixed
results/ciulli_20261111_donor_513_fixed --mesh-only --output
results/ciulli_20261111_mesh_convergence_65_129_257_513.json
--regional-fields
results/ciulli_20261111_mesh_differences_65_129_257_513.npz`. The compressed
pairwise coordinate and difference maps are local in that ignored NPZ archive.

#### Main pressure field and spatial error

| Grid pair | Main pressure relative L2 | RMS (MPa) | Linf (MPa) | Linf location (um) | Hertz-core error share |
|---|---:|---:|---:|---|---:|
| 65 -> 129 | 5.711% | 31.712 | 233.225 | (50,-56.5) | 99.787% |
| 129 -> 257 | 4.481% | 24.825 | 275.093 | (59.5,53) | 99.977% |
| 257 -> 513 | 3.451% | 19.075 | 296.020 | (48.5,-61) | 99.998% |

Across the full contact rectangle, main/outlet shares of squared pressure
difference are respectively 54.70/45.30%, 48.12/51.88%, and 45.78/54.22%.
The pressure RMS-difference norms contract by factors 1.277 and 1.301,
corresponding to descriptive orders of the *difference norm* of 0.353 and
0.380 for ratio-two refinements. These are not observed orders of the
unknown solution error. The main-contact RMS trend is encouraging but slow;
the increasing Linf and migrating maximum-error location show that the
localized field has not stabilized. Almost all main-region error energy is
inside the dry-Hertz footprint, not in the outer transition or exclusively
at the outlet.

The sampled pressure ridge maximum over `x=[0,60] um` moves at `y=-60 um`
from 12.5 to 31 to 46.5 to 48.5 um, and at `y=-56.5 um` from 12.5 to 44 to
53 to 56 um across the four grids. At `y=-50 um`, the sampled ridge maximum
moves 50 -> 56 -> 60 -> 5 um, indicating a switch in the maximizing ridge
branch rather than a stable ridge profile. The 513-grid global pressure
maximum jumps from 739.669 MPa at `(53.125,56.25) um` on 257x257 to
812.881 MPa at `(51.5625,59.375) um`. Its local 5x5 pressure patch is sharply
peaked and varies strongly across neighboring cells; pressure remains
positive and `theta=1` throughout that patch. This maximum is an interior,
non-cavitating core feature, not the downstream outlet spike. Existing
evidence identifies its spatial/grid sensitivity but does not yet identify
the responsible discretization mechanism.

#### Film, profiles and scalar quantities

| Grid pair | Main-film relative L2 | RMS (nm) | Linf (nm) | Longitudinal pressure RMS / Linf (MPa) | Transverse pressure RMS / Linf (MPa) | Longitudinal film RMS / Linf (nm) | Transverse film RMS / Linf (nm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 65 -> 129 | 2.005% | 9.588 | 40.918 | 33.787 / 204.361 | 28.947 / 112.428 | 10.842 / 41.269 | 8.243 / 22.411 |
| 129 -> 257 | 2.207% | 10.469 | 22.630 | 25.763 / 216.378 | 18.496 / 90.642 | 11.285 / 19.163 | 9.202 / 11.638 |
| 257 -> 513 | 1.532% | 7.124 | 13.330 | 20.321 / 237.867 | 10.825 / 71.699 | 7.703 / 10.997 | 6.415 / 7.981 |

Main-film RMS first increases and then decreases, so the three differences
are not monotone; the last-pair reduction does not establish an asymptotic
rate. The film Linf and the longitudinal/transverse film-profile errors
decrease, but they remain direct intergrid differences rather than estimated
errors to an exact solution.

| Quantity | 65x65 | 129x129 | 257x257 | 513x513 |
|---|---:|---:|---:|---:|
| Central film `h_c` (nm) | 492.246 | 485.267 | 473.629 | 465.649 |
| Minimum film `h_min` (nm) | 282.747 | 285.231 | 281.997 | 278.512 |
| Maximum pressure `p_max` (MPa) | 746.301 | 741.270 | 739.669 | 812.881 |
| Integrated load (N) | 15.000 | 15.000 | 15.000 | 15.000 |
| `p_max` location (um) | (0,0) | (0,0) | (53.125,56.25) | (51.5625,59.375) |

`h_c` changes by -6.979, -11.638 and -7.981 nm; the signed changes are not
monotone contracting. `h_min` is nonmonotonic and its location shifts along
the contact edge. The 513x513 `h_c` and `h_min` differ from the 257x257
values by 1.69% and 1.24%, respectively, but these last-grid changes are not
convergence uncertainty estimates. The `p_max` increase and location change
rule out a scalar convergence claim. Load closure is excellent but is an
equation audit, not a discretization-accuracy measure.

No Richardson extrapolation or GCI is justified for the full fields or
important scalars: pressure Linf grows, the pressure maximizing ridge changes
branches, `h_c` changes do not contract monotonically over all intervals,
`h_min` is nonmonotone, and `p_max` jumps. The descriptive pressure
difference-norm order near 0.37 must not be used as solution-error order.
The available numerical uncertainty is therefore bounded only by direct
successive-grid changes; no defensible asymptotic discretization uncertainty
has been quantified.

#### Outlet spike and next decision

| Grid | Outlet peak (MPa) | Peak x (um) | Half-prominence width (um) | Cells across width | Variable-window pressure integral (N) | Fixed `x=[75,100] um`, `|y|<=a_H` pressure integral (N) |
|---|---:|---:|---:|---:|---:|---:|
| 65x65 | Not identified | -- | -- | -- | -- | 0.39334 |
| 129x129 | 617.834 | 81.250 | 3.593 | 0.575 | 0.11479 | 0.44166 |
| 257x257 | 635.175 | 84.375 | 3.960 | 1.267 | 0.10245 | 0.48035 |
| 513x513 | 689.502 | 85.938 | 2.704 | 1.730 | 0.06080 | 0.49746 |

The fixed-corridor integral includes background pressure and is supplied as a
common-region comparison, not as the spike's isolated load. The local
half-prominence integral uses a different, grid-dependent width on each
mesh and is not a convergence quantity. At 513x513 the peak has 624.89 MPa
at the left adjacent sample and 315.78 MPa at the right. The spike height,
width and shape remain unresolved; the finer grid does not provide enough
cells to claim spike convergence.

**Decision:** the 513x513 root is an accepted donor-flux equation-closure
solution and the latest main-contact pressure/film RMS differences contract.
However, full main-field convergence, scalar/profile stabilization and
outlet-spike resolution are not established. Do not launch 1025x1025
automatically. First inspect the core ridge pressure-gradient/flux
discretization, elastic-convolution grid treatment and quadrature, and
symmetry/grid alignment; then design a targeted or local/nonuniform
refinement that resolves both the ridge and outlet feature without loosening
acceptance gates. The 513-grid result does not redo or supersede the existing
Figure 13 comparison and does not establish physical validation. The
strongest supported claim remains independently audited equation closure
plus a pressure/film L2 contraction trend, not numerical mesh convergence
or physical accuracy.

## Core pressure-ridge mechanism diagnosis (post-processing only)

The accepted 65x65, 129x129, 257x257 and 513x513 donor roots were reanalyzed
without running or modifying a solver. The analysis is reproducible with
[`examples/diagnose_ciulli_core_ridge.py`](../examples/diagnose_ciulli_core_ridge.py);
focused regression tests are in
[`tests/test_ciulli_core_ridge_diagnostics.py`](../tests/test_ciulli_core_ridge_diagnostics.py).
The full post-processing output and field/checkpoint archives remain local
under `results/`. Compact values and source-state hashes are committed in
[`cases/ciulli_core_ridge_diagnostic_summary.json`](../cases/ciulli_core_ridge_diagnostic_summary.json).
The command used was:

```powershell
$env:PYTHONPATH='src'
$env:OPENBLAS_NUM_THREADS='4'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:BLIS_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
$env:VECLIB_MAXIMUM_THREADS='4'
python examples\diagnose_ciulli_core_ridge.py --roots `
  results\ciulli_20261111_donor_65_fixed `
  results\ciulli_20261111_donor_129_fixed `
  results\ciulli_20261111_donor_257_fixed `
  results\ciulli_20261111_donor_513_fixed `
  --output results\ciulli_20261111_core_ridge_diagnostics.json
```

The diagnostic used Python 3.13, NumPy 2.5.4 and SciPy 1.18.1. NumPy
reported the `scipy-openblas` 0.3.34.106.0 backend (build maximum 24);
process environment limits were set to four threads for BLAS/OpenMP-style
backends. The SciPy FFT path used its default single worker. These are
post-processing resource limits, not evidence that the earlier solver used
four compute threads; no EHL solve was run here.

### Symmetry and ridge geometry

The expected reflection is `y -> -y`. For every accepted grid, pressure,
film, deformation, theta, density and viscosity are symmetric to floating
point roundoff. The pressure relative-L2 errors are `1.39e-15`, `2.64e-15`,
`3.72e-15` and `4.86e-15` from 65x65 through 513x513. Across all four grids,
the largest relative-L2 symmetry errors for film, deformation, theta,
density and viscosity are respectively `2.00e-16`, `2.02e-15`, `1.32e-16`,
`8.38e-17` and `5.97e-14`; their largest RMS/Linf errors are respectively
`2.17e-21/1.02e-20 m`, `9.28e-22/3.39e-21 m`,
`1.10e-16/1.78e-15`, `7.54e-14/1.59e-12 kg/m^3`, and
`6.06e-9/5.27e-7 Pa s`. Per-grid relative-L2, RMS and Linf values for each
field are in the compact diagnostic metadata. Reflection in x is not a
physical symmetry because entrainment selects +x; its pressure relative-L2
asymmetry is about 0.24-0.25.

The equal upper/lower ridge maxima differ by at most about `4e-12 MPa`.
Consequently, a global `argmax` changing from one lateral sign to the other
is a tie-selection/roundoff effect, not symmetry breaking. This does not
explain the amplitude and location changes: the two mirrored local maxima
move from `(50,50) um`, through `(56.25,50) um` and
`(53.125,56.25) um`, to `(51.5625,59.375) um`; their nodal pressure rises
from `687.30` to `705.39`, `739.67` and `812.88 MPa`. The global maxima on
the first two grids are instead the central values `746.30` and `741.27 MPa`.
Thus the coarse-grid fixed-ROI ridge maxima are boundary-limited in y and
must not be treated as resolved ridge vertices.

Ridge traces were extracted as the maximum along fixed `x=[25,75] um` cuts
at fixed y coordinates, not by defining a window around each global maximum.
For example, the cut maxima at `y=50/55/60 um` have x positions (um)
`50/25/25`, `56.25/43.75/31.25`, `62.5/53.125/46.875`, and
`64.05/57.8/48.45` on the four grids. Some coarse cut maxima touch the
analysis-window edge. A quadratic fit to these cuts produces trace lengths
of `70.6`, `147.7`, `56.4` and `50.1 um`, with large changes in orientation
and curvature; these are fit descriptors of jagged, sometimes boundary-
limited maxima, not physical ridge length/orientation estimates. The trace
does not settle to a unique geometrical ridge.

### Sampling, fixed neighborhoods and pressure-error energy

For comparison, all fields were also evaluated by piecewise-bilinear
interpolation on the nested 513x513 nodal coordinates. The maximum in the
fixed ridge region on this aligned grid equals the native nodal maximum for
each root (the grids are nested). A separate 0.5 um sampling lattice does
not include the 513-grid peak coordinate: its maximum is only `795.16 MPa`
at `(45,62.5) um`, versus the aligned-grid/nodal `812.88 MPa` at
`(51.5625,59.375) um`. This quantifies a real peak-sampling sensitivity, but
the aligned-grid comparison confirms that sampling alone does not remove the
257-to-513 increase. A local 3x3 quadratic fit is rejected as not a strict
maximum on the 129-, 257- and 513-grids (the 65-grid fit leaves the fixed
ridge band). Independent one-dimensional quadratic fits at 513 estimate
`821.9 MPa` in x and `835.0 MPa` in y at inconsistent coordinates; neither is
a defensible two-dimensional subcell peak. No extrapolated peak is accepted.

At 513x513 only one x interval (`1.5625 um`) is contiguous above 90% of the
local peak on the ridge-peak row, and no adjacent y interval is above that
threshold on the peak column. The 257-grid has no adjacent interval in
either direction; the 65-grid threshold trace is clipped by the fixed
analysis band. At 513, only 3 of the 5x5 neighboring nodes exceed 95% of the
peak and 8 of the 9x9 nodes exceed 90%. The core ridge width is therefore
not resolved.

The same fixed upper-lobe neighborhood `x=[45,60] um`, `y=[50,65] um`
(and its mirror) has mean pressure `323.5`, `398.1`, `448.1`, `468.2 MPa`
and integrated pressure `0.07269`, `0.08985`, `0.10167`, `0.10642 N` from
65x65 through 513x513. Both neighborhood measures change even though the
integration window is fixed. Their last increments are smaller, but do not
establish convergence. The peak of grid-specific cell averages in the larger
fixed ridge region rises from `487.6` to `646.9`, `667.1` and `724.8 MPa`;
because those averages have different cell sizes, they are not a common-
resolution peak estimate.

The pressure-error energy was split using the fixed core-ridge mask
`x=[35,60] um`, `45<=|y|<=75 um`, with the complement retained inside the
main contact region:

| Grid pair | Main RMS (MPa) | Fixed-ridge RMS (MPa) | Main error energy in ridge | Main RMS outside ridge (MPa) |
|---|---:|---:|---:|---:|
| 65 -> 129 | 31.64 | 86.35 | 46.65% | 23.87 |
| 129 -> 257 | 24.78 | 71.68 | 52.38% | 17.66 |
| 257 -> 513 | 19.06 | 58.06 | 58.12% | 12.74 |

Both the ridge-region and complementary RMS differences contract, so bulk
pressure behavior can be assessed separately from the ridge. The contraction
is slow, and the ridge accounts for an increasing fraction of the remaining
main-region squared error. The main RMS difference-norm orders remain only
about 0.35 and 0.38; these are descriptive orders of intergrid differences,
not solution-error orders. No asymptotic range, Richardson estimate or GCI is
established.

### Pressure/flux and elasticity operator checks

At the 513-grid ridge maximum, `p=812.88 MPa`, `h=405.87 nm`,
`rho=1074.65 kg/m^3`, `eta=4.839e6 Pa s`, mobility is
`1.237e-24 m s`, and `theta=1`. The 3x3 surrounding stencil is entirely
full-film. With positive +x entrainment, the donor is always the west/left
node; no cavitation branch, limiter or coefficient clipping is active there.
Face mobility is arithmetically averaged. At the peak, west/east x pressure-
gradient fluxes are `-1.42e-10` and `3.32e-9 kg/(m s)`, compared with
Couette fluxes `8.76e-5` and `8.72e-5 kg/(m s)`. The pressure-gradient
contribution is tiny relative to Couette transport in x; y flux is pressure
diffusion only. The center discrete divergence is
`5.93e-14 kg/(m^2 s)` and the surrounding 3x3 residuals remain of order
`1e-13 kg/(m^2 s)` or smaller. The diagnostic output retains the neighboring
3x3 pressure, film, density, viscosity, mobility, theta, directional face
fluxes, donor nodes and residuals for all four meshes.

At the 257-grid ridge maximum, viscosity is `1.523e6 Pa s`, density is
`1067.36 kg/m^3` and film is `423.07 nm`. From 257 to 513, peak pressure
increases by `73.21 MPa`, peak viscosity by a factor of about 3.18, while
film decreases by `17.20 nm`. This is consistent with strong constitutive
feedback in the local discrete solution, but is correlation rather than
proof that the viscosity law causes the ridge.

A separate smooth, fully flooded manufactured-field check of the production
flux gives first-order total residual convergence, first-order donor/Couette
advection (donor selection follows velocity sign) and second-order
pressure-diffusion convergence (orders about `1.00` and `1.99`). This matches
the implemented first-order upwind donor stencil and second-order centered
pressure gradient; it identifies no flux implementation defect. It does mean
the scheme has an expected low-order contribution to the slow coupled-field
convergence. The
test is a smooth full-film operator check, not a local truncation-error
estimate for the narrow ridge.

The elastic operator analytically integrates the rectangular source-cell
kernel `1/(pi E* r)`, including the self-cell, and uses zero-padded linear
FFT convolution with a centered crop. A positive smooth Gaussian pressure
field compared with its analytic infinite-domain center displacement gives
relative errors `0.1172%`, `0.03094%`, `0.00794%` and `0.00201%` on 65, 129,
257 and 513 grids, with observed orders `1.92`, `1.96` and `1.98`. The
response is nonnegative, symmetric to roundoff, translates away from the
boundaries with relative-L2 discrepancy `1.1e-7`, and doubles linearly when
pressure is doubled. This argues against an obvious standalone elastic-kernel
defect; it does not prove that elasticity has no coupled effect on the ridge.

### Seed history, interpretation and next decision

The 513-grid seed already contained the 257-grid precursor ridge at
`739.67 MPa`. After the first Newton correction, the global maximum was
`811.02 MPa` at `(60.94,-53.125) um`, while the fixed upper-branch ridge was
`809.35 MPa` at `(51.5625,59.375) um`. The maximum subsequently moved between
the exactly symmetric lateral branches and settled at `812.88 MPa`.
The first correction changed 309 cavitation-state nodes; 6 changed at the
third correction, then the active state remained fixed. The first Newton step
used four recovery models across seven line-search trial checks; the second
accepted a damped step `alpha=0.1356`; subsequent accepted corrections used
full steps.
Saved Krylov counts were 63-69 iterations with true residuals about
`1.3e-10` to `3.8e-10`; the nonlinear residual reached `2.26e-11`.
Thus the final ridge is amplified/moved during nonlinear correction rather
than simply being copied unchanged from the prolonged seed. This history
alone does not establish seed independence.

### Controlled same-grid seed sensitivity

A resource-preflighted 513x513 solve was then run on the same `+/-0.4 mm`
domain with the same donor-flux JFO equations, material model, boundary
conditions, nonlinear limits, tolerances and independent acceptance audits.
It was initialized by direct bilinear pressure/content prolongation of the
accepted 129x129 root, rather than the accepted 257x257 root. No solver
physics or acceptance gate was changed. The new state is independently
accepted with seven Newton corrections. It is reproducible from the archived
run at `results/ciulli_20261111_donor_513_from129`; the complete pressure,
film, deformation, density, viscosity and theta arrays, all eight periodic
checkpoints, seven diagnostic records and generated standardized summary
figure remain local with the run.

The paired accepted 513x513 roots have identical coordinates and case inputs.
Direct nodal comparison gives pressure relative-L2 `6.36e-15` (RMS
`7.17e-7 Pa`, Linf `7.00e-5 Pa`); the largest pressure difference is at
`(29.6875,71.875) um`. Film relative-L2 is `9.47e-17` (Linf
`3.39e-21 m`), deformation relative-L2 `1.82e-15`, theta relative-L2
`1.00e-16`, density relative-L2 `1.29e-16`, and viscosity relative-L2
`8.39e-14`. On the same fixed upper neighborhood
`x=[45,60] um`, `y=[50,65] um`, both roots have mean pressure `485.099 MPa`
and nodal-sum pressure integral `0.118432 N`; their mirrored lower
neighborhoods agree to roundoff as well. Both yield `p_max=812.880981 MPa`
at the symmetric pair `(51.5625,+/-59.375) um`, `h_c=465.648691 nm`,
`h_min=278.512053 nm` at `(23.4375,+/-82.8125) um`, and integrated load
`15 N`. The two roots therefore agree to numerical roundoff, not just in a
single peak metric.

The prolonged 129-grid seed starts with its global maximum at
`(0,0) um`, `741.270 MPa`, and already has a fixed-window off-axis ridge
maximum of `705.395 MPa` at `(56.25,+/-50) um`. After the first Newton
correction, the off-axis ridge becomes the global maximum at
`(37.5,+/-65.625) um`, `820.849 MPa`; over the remaining six corrections it
moves to `(51.5625,+/-59.375) um` and settles at `812.881 MPa`. By contrast,
the original 257-grid seed began at the core ridge maximum and its first
correction moved the peak to a neighboring location before reaching the same
endpoint. This indicates that the final ridge is not simply inherited
unchanged from one particular seed; in both runs it develops through the
nonlinear correction path.

The 129-seeded trajectory changed the checkpointed `theta < 1-1e-10`
cavitation classification by 905 nodes on the first correction, then by
20, 0, 6, 0, 0 and 0 nodes. The accepted full-film active mask changes
recorded in subsequent correction models were 20, 0, 6, 0, 0 and 0 nodes.
The first correction required six Newton recovery models and rejected several
infeasible or non-Armijo trials before accepting a full step; correction 2
accepted `alpha=0.21497`; other accepted corrections used full steps.
GMRES used 60-67 iterations per model with independently recomputed true
residuals from `9.0e-11` to `6.24e-10`. The final independently audited
local/global relative mass balances are `1.50e-13`/`1.50e-16`; force error
is zero, film reconstruction error is `5.51e-20 m`, and maximum JFO
complementarity is `5.12e-25`.

The preflight budget was 8 GiB peak working set and a 16 GiB minimum free-RAM
guard; no competing solver was present. During the run the external monitor
measured a maximum process working set of `1.32 GB` and minimum available RAM
of `40.0 GB` (about `37.3 GiB`). The complete local run directory, including
eight approximately 11.4 MB checkpoints and all diagnostics, occupies
`108.96 MiB`. The reported solver runtime is `1021.4 s`. All configured
OpenBLAS/OpenMP/MKL/BLIS/NumExpr thread variables were set to four, and
NumPy and SciPy independently reported four active OpenBLAS threads. The
linear backend is SciPy sparse GMRES/SuperLU; NumPy FFT and Python
matrix-free operations are not parallel solver paths, so the four-thread
setting is an upper bound on supported backend parallelism, not evidence that
the whole solve used four CPU cores. The run generated and visually checked
`ciulli_15n_donor_15N_513x513_donor-flux_ACCEPTED_summary.png` and its
machine-readable centerline profile CSV.

The compact comparison, state hashes, fixed-region metrics and checkpoint /
Newton histories are stored in
[`cases/ciulli_513_seed_sensitivity_summary.json`](../cases/ciulli_513_seed_sensitivity_summary.json).
The reproducible post-processing command is:

```powershell
$env:PYTHONPATH='src'
$env:OPENBLAS_NUM_THREADS='4'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:BLIS_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
python examples\diagnose_ciulli_core_ridge.py --roots `
  results\ciulli_20261111_donor_65_fixed `
  results\ciulli_20261111_donor_129_fixed `
  results\ciulli_20261111_donor_257_fixed `
  results\ciulli_20261111_donor_513_fixed `
  --seed-comparison `
  results\ciulli_20261111_donor_513_fixed `
  results\ciulli_20261111_donor_513_from129 `
  --seed-comparison-output results\ciulli_20261111_seed_sensitivity.json `
  --output results\ciulli_20261111_core_ridge_seed_comparison.json
```

**Mechanism assessment:** common-grid sampling contributes to the reported
point maximum, but cannot explain the rising aligned-grid peak and fixed-
neighborhood means/integrals (A is not the whole cause). The evidence supports
a narrow, grid-sensitive feature of the discrete coupled solution, but does
not show that the continuous physical solution has a singular or equally
narrow ridge (B remains unproven). First-order donor advection is verified as
an expected discretization property and may contribute to slow convergence;
no donor, cavitation, coefficient-switching or clipping defect was found (C
is not established as a bug). The standalone elasticity operator passes the
smooth reference tests (D is less likely, not excluded in the coupled
system). The controlled 129-to-513 seed study reaches the same accepted nodal
root as the 257-to-513 path, making seed/branch dependence less likely for
these two prolongated starts (E is not demonstrated, though other branches
cannot be excluded).

**Decision:** do not launch 1025x1025 and do not change solver physics based
on this evidence. The seed-path discriminator is complete and does not
justify another full solve. Prioritize a controlled pressure/flux
discretization or local-resolution study of the unresolved full-film ridge,
keeping the fixed-region metrics, operator benchmarks and outlet-spike
analysis separate. A 1025x1025 run remains gated on a specific diagnostic
question that uniform refinement can answer, and on resource/backend checks.
Keep all existing acceptance gates.

The strongest supported statement after this phase is: five accepted donor
roots (65/129/257/513 from 257 and 513 from 129) pass their independent
equation-closure audits, preserve transverse symmetry and show no measurable
seed sensitivity between the two tested 513-grid initializations. Smooth
standalone operators converge at their expected orders, but the coupled
ridge remains under-resolved and grid-sensitive. Neither mesh convergence,
outlet-spike resolution nor physical EHL validation is established.

### Standardized run summary and outlet-feature relationship

The shared
[`plot_run_summary`](../src/ehl_solver/visualization.py) output now retains the
unfiltered pressure range, identifies tied global pressure maxima and the
secondary centerline outlet peak, and marks global, centerline-minimum and
central-film locations. Its film map uses a display-only contact-sensitive
colour range when far-field film values dominate; the exact range is printed
in the subplot title. The centerline film panel uses the same range as a
viewport and labels any clipping; underlying field/profile arrays remain
unchanged. Separate raw centerline pressure/film CSVs and compact metadata
accompany the compatibility combined-profile CSV. Available full-field
archives are linked from the metadata. Failure-state figures remain
explicitly labelled as not accepted solutions.

The accepted 513x513 `513_from129` result was re-plotted from its saved
`state.npz`; no solver was run and no constitutive or acceptance setting was
changed. The high-resolution summary is saved under the local
`results/ciulli_20261111_donor_513_from129/summary_v2/` directory. It reports
`h_c=465.649 nm`, global `h_min=278.512 nm` at the symmetric pair
`(23.4375,+/-82.8125) um`, centerline minimum `398.828 nm` at `x=93.75 um`,
and `p_max=812.881 MPa` at `(51.5625,+/-59.375) um`. The centerline outlet
peak is `689.502 MPa` at `x=85.938 um`. The film colour cap is `1000 nm`;
the full far-field values and CSV profiles remain uncapped.

The existing cut-wise outlet analysis identifies a connected sequence of
local pressure maxima over 77 transverse rows from `y=-59.375` to `+59.375
um`, with maximum adjacent x displacement `3.125 um`. This describes one
sampled curved downstream ridge linking the centerline outlet peak to the two
off-axis global maxima. It does not establish a continuous physical ridge:
the measured half-prominence width is only `1.34` cells at `y=0` and `2.06`
cells at `y=+/-59.375 um`. The transverse film-minimum locus lies downstream
of the pressure-max locus by `7.8125 um` at the centerline and `14.0625 um`
at `y=+/-59.375 um` (the row-wise separation is `7.8125-14.0625 um` over
these 77 rows). The two are related features in the same outlet region, but
are not coincident sampled loci and neither spike width nor physical
pressure/film feature alignment is resolved. Preserve separate pressure-peak
and film-constriction diagnostics.

The exact figure and numerical profile paths are recorded in the accompanying
`*_summary_metadata.json`. The plot is a visualization correction, not new
evidence for mesh convergence or physical validation.

### Standard run summaries and CPU thread scaling (2026-11-12)

The centerline-film viewport was refined after visual inspection showed that
the full far-field parabola (about 14,000 nm at the domain edge) obscured the
contact minimum on a linear y-axis. The lower-right panel now uses the same
contact-sensitive display range as the film map, states the displayed range
in its title when values lie outside it, and leaves both plotted source arrays
and exported profile CSVs unmodified. The accepted 513x513 `513_from129`
figure and the thread-benchmark figures were regenerated from their saved
accepted fields with this version of the writer; no solver was rerun for
figure generation. The 513x513 figure now visibly shows the centerline
constriction and marks the 398.828 nm centerline minimum separately from the
278.512 nm off-axis global minimum.

A controlled 257x257 CPU thread series used the same 15 N donor-flux case,
0.4 mm half-width, 129x129 accepted seed, solver source, and seven accepted
Newton corrections. All three endpoints passed the numerical-root
acceptance gate and their saved full fields agree to roundoff:

| OpenBLAS threads (NumPy/SciPy) | Solver elapsed [s] | Monitored wall [s] | Mean process-tree CPU [cores] | Peak process-tree working set [GiB] |
| ---: | ---: | ---: | ---: | ---: |
| 1/1 | 99.490 | 105.3 | 0.93 | 0.37 |
| 2/2 | 97.505 | 105.2 | 0.94 | 0.36 |
| 4/4 | 96.707 | 104.9 | 0.94 | 0.37 |

Relative field L2 differences versus the one-thread endpoint were
`3.626e-15` / `4.025e-15` for pressure, `2.145e-17` / `1.062e-16` for film,
`5.783e-16` / `2.522e-15` for deformation, `6.286e-17` / `6.864e-17` for
density, `4.438e-14` / `4.720e-14` for viscosity, and `6.674e-17` /
`8.149e-17` for theta (2-thread / 4-thread). Thus the endpoints are
numerically equivalent, but the measured speedups are only 1.020x and 1.029x.
NumPy and SciPy both reported the requested 1/2/4 OpenBLAS limits; OMP, MKL,
BLIS, NumExpr and VECLIB were each capped at one to avoid nested
oversubscription. Process-tree sampling found about 0.94 effective CPU cores
for every setting, so the four-thread configuration was available to the
backend but this workload did not use four cores concurrently. Raising the
thread cap is not a useful performance recommendation for this solver path.

A one-thread cProfile rerun from the same seed accepted in seven corrections;
its solver-reported elapsed time was 94.396 s. The local profile and reports
are in
`results/ciulli_20261112_cpu_thread_benchmark_v3/` (including
`profiled_1thread.prof`, the full cumulative report, a Matplotlib-filtered
solver-function report, and compact manifest). The profiler's instrumented
total was 194.045 s, dominated by excessive Matplotlib text-layout
instrumentation, and must not be interpreted as runtime or scaling evidence.
In the filtered table, the Newton/GMRES matrix-free correction and its
Jacobian-vector products are the largest solver-side costs; the independent
audit also includes two direct `convolve2d` calls. This is diagnostic evidence
for a later targeted performance study, not a solver-physics change.

The generated figures, field archives, resource monitors and cProfile files
remain local under `results/`. Donor/JFO CUDA parity is still unavailable:
the repository's optional CUDA signed-full-film path is a distinct
mathematical verification solver, not a donor/JFO implementation. No
1025x1025 calculation was run or authorized. These visualization and
throughput results do not establish mesh convergence, outlet-spike
resolution, or physical EHL validation.

Verification: the focused run-summary tests pass (**6 passed**) and linting
the changed Python files passes. The full Windows `pytest -q` collection
remains blocked by the existing POSIX-only `resource` import in
`tests/test_joint_refinement.py`; excluding that one module, **824 passed,
1 skipped**. `git diff --check` passes. Repository-wide `ruff check .` is
blocked by an E501 in the pre-existing untracked
`.kilo/skills/jupyter-notebook/scripts/new_notebook.py`; the changed Python
files pass Ruff. No unrelated file was changed to hide either limitation.

## Outlet geometry, profiling and device operators (2026-11-12)

The checksum-verified saved 513x513 donor state (SHA-256
`dc1faecef8ec8a5c5fe29929b1c384fcc1654c52d5aa8217dccfb3d73579b30f`)
was analyzed without rerunning the solver. Its summary figure now includes a
contact-detail inset and a film heatmap capped only for display to
275--1000 nm; saved arrays and profile CSV remain unmodified. Output figures,
the full transverse-locus CSV, and JSON are local under
`results/ciulli_20261111_outlet_tracked_v2/`.

The tracker starts at the rightmost centerline downstream local pressure
maximum with prominence at least 1 MPa in `20 <= x <= 120` micrometres and
follows the nearest candidate in each direction, stopping at a missing peak
or an adjacent x jump above 6.25 micrometres. It similarly tracks the first
film minimum downstream of the centerline peak. The pressure maximum locus
connects the centerline to both global maxima across 77 transverse rows; its
largest adjacent jump is 3.125 micrometres. The centerline peak is 689.502 MPa
at x=85.9375 micrometres; the two off-axis maxima are 812.881 MPa at
(51.5625, +/-59.375) micrometres. Their half-prominence widths are 2.0943 and
3.2246 micrometres (1.34 and 2.06 cells), respectively. Integrating pressure
over +/-5 micrometres around each tracked x gives 4410.309 N/m at the
centerline and 5764.355 N/m at the off-axis maxima. The film constriction
is 7.8125 and 14.0625 micrometres downstream, respectively; global film
minima are 278.512 nm at (23.4375, +/-82.8125) micrometres.

These are connected **sampled cut maxima**, not proof of a smooth or singular
continuous feature. The tiny sampled widths preclude a resolution claim.
The nominal Hertz circle in the overlay is geometric context, not a
discretization-error reference. The earlier bulk RMS contraction, symmetry,
operator, and seed checks remain as reported above; separation from the
localized structure does not prove full-field convergence. Pointwise p_max is
not a reliable convergence observable at this resolution.

### Measured 257x257 profile

The following are selected cProfile costs from the archived accepted
one-thread run, in seconds and call counts. Percentages use the instrumented
194.045 s **whole profile**, including ~94.9 s distorted Matplotlib text
layout; they are not ordinary wall fractions or acceleration predictions.
Cumulative rows overlap and must not be added. Solver-reported elapsed was
94.396 s; the separate unprofiled run was 99.490 s.

| Component (calls) | Self s / % | Cumulative s / % | Suitability / opportunity |
| --- | ---: | ---: | --- |
| Newton coordination (1) | 0.165 / 0.09 | 80.067 / 41.26 | orchestration; optimize inner operations |
| Cavitating corrections (12) | 0.062 / 0.03 | 56.618 / 29.18 | active-set and linear solves |
| GMRES (24) | 3.018 / 1.56 | 42.848 / 22.08 | device Krylov requires true-residual checks |
| Complete JVP (803) | 0.790 / 0.41 | 30.377 / 15.65 | elasticity and donor kernels |
| Independent audits (10) | 0.009 / 0.00 | 28.141 / 14.50 | preserves separate acceptance evidence |
| Direct audit convolution (2) | 27.752 / 14.30 | 27.752 / 14.30 | independent audit, not iterative GPU physics |
| Elastic operator (1032) | 1.621 / 0.84 | 19.934 / 10.27 | GPU FFT is promising |
| Raw FFT (4156) | 17.908 / 9.23 | 17.912 / 9.23 | GPU cuFFT through PyTorch |
| SuperLU solves (755) | 11.745 / 6.05 | 11.745 / 6.05 | device preconditioner still needed |
| Sparse LU factorization (12) | 5.758 / 2.97 | 5.758 / 2.97 | CPU direct sparse factorization |
| Donor JVP flux (1630) | 4.760 / 2.45 | 9.478 / 4.88 | tested resident GPU kernel |
| Fluid ratios (1834) | 3.811 / 1.96 | 4.108 / 2.12 | vectorized GPU candidate |
| Field reconstruction (169) | 0.043 / 0.02 | 3.428 / 1.77 | overlap with elasticity |
| Line searches (12) | 0.018 / 0.01 | 1.298 / 0.67 | nonlinear control and residual |
| Donor face flux (86) | 0.124 / 0.06 | 0.312 / 0.16 | tested resident GPU kernel |
| JFO residual (24) | 0.023 / 0.01 | 0.865 / 0.45 | device operator subsequently verified below |

State arrays contain 66,049 nodes and use float64; the 513 grid has 263,169.
The archived profile records 695 total GMRES iterations in 24 Krylov histories
across 12 attempted correction models. Each 257-grid scalar field is 528,392
bytes; each 513-grid scalar field is 2,105,352 bytes, excluding workspaces.
The 257-grid elastic FFT padding is 770x770, versus 1540x1540 at 513.
Save-state calls occurred 17 times with 2.736 s cumulative; profile timing of
plotting is distorted and is not a useful performance estimate. No new 513
nonlinear profiling run was launched: the accepted root was instead used
for bounded component measurements below. Their changed cost ratios show
that a whole-solver 513 distribution cannot safely be extrapolated.
The donor-zero path also constructs some limited-flux/limiter machinery before
multiplying its correction by zero. This is a plausible CPU optimization,
not a demonstrated equation defect; it needs full residual, JVP, correction,
active-set, line-search and audited-root equivalence before adoption.

### Device capability and component measurements

The CPU path uses NumPy/SciPy, NumPy pocketfft, sparse GMRES and SuperLU;
OpenBLAS threads do not parallelize the whole solve. MKL, CuPy and Numba CUDA
were not installed; no HIP/ROCm runtime was available in this environment.
PyTorch CUDA uses its device FFT facilities and provides CUDA linear-algebra
libraries, but no cuBLAS/cuSPARSE-specific donor/JFO implementation or
performance was independently established. The existing GPU GMRES, Jacobi
and DST preconditioners and nonlinear infrastructure target the separate
signed full-film formulation. PyTorch's CUDA interface may target ROCm in a
ROCm build, which was not tested here.

On an RTX A2000 Laptop GPU (4 GiB, PyTorch 2.11.0+cu128), ten warmed,
synchronized applications of the saved 257/513 fields gave:

| Resident operator | 257 CPU / GPU ms | 257 speedup | 513 CPU / GPU ms | 513 speedup | Max relative L2 difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elasticity | 18.076 / 2.047 | 8.83x | 91.569 / 8.436 | 10.85x | 3.33e-16 |
| Donor face flux | 3.941 / 2.203 | 1.79x | 41.255 / 2.955 | 13.96x | 4.34e-17 |
| Donor smooth JVP | 4.995 / 2.671 | 1.87x | 62.559 / 4.335 | 14.43x | 9.38e-17 |

These component-only timings exclude setup and field transfers; scalar
validation in the donor kernel synchronizes, but fields remain on device.
They do not measure solver acceleration. The elasticity operator peak GPU
allocation was 33.8 MB / 135.0 MB (257 / 513); full solver and Krylov memory
would be additional. The donor operator uses the actual prototype's Roelands
viscosity and pressure-dependent density, not a new physical model. Its JVP
uses deterministic synthetic directions around the accepted field. It does
not yet implement complementarity, load closure, boundary residuals, Krylov,
preconditioning or acceptance checks. No CUDA JFO root, CPU/GPU full-solver
equivalence, or 1025x1025 run is claimed.

Reproduction commands and next gates are in [USER_GUIDE.md](USER_GUIDE.md) and
[ROADMAP.md](ROADMAP.md); compact checksums and results are in
`cases/ciulli_outlet_device_diagnostic_summary.json`.

Verification: **54 focused tests passed**, including CPU and CUDA donor
operator parity; **828 passed, 1 skipped** with the Windows-incompatible
joint-refinement module excluded. Complete `pytest -q` remains blocked during
collection by its existing POSIX-only `resource` import. Changed-file Ruff and
`git diff --check` pass; `ruff check .` reports the existing E501 in the
untracked `.kilo/` notebook helper.

## Device donor/JFO residual and bordered products (2026-11-12)

The full-strength **donor endpoint only** now has a resident PyTorch
residual and selected generalized Jacobian-vector product. These reproduce
the CPU interior state `(P, D, h0/h_ref)`, where `P=p/p_ref` and
`D=1-theta`, the copied content boundaries, zero boundary pressure,
elastic deformation, donor face fluxes, `min(P,D)` complementarity and
normalized load row. The CPU endpoint's nondimensional mass residual is
divided by 100.
At exact complementarity ties the guide selects the generalized derivative;
the separate true one-sided derivative selects the direction-dependent
minimum. Active-mask overrides affect the selected model, not the nonlinear
residual. Input and film feasibility checks synchronize scalar decisions,
but do not transfer field arrays to the host.

Four saved 257x257 CPU Newton states, checkpoints 0, 3, 6 and 7, were
compared without another nonlinear solve. The accepted state SHA-256 is
`ecac5724a26374879a9e028e43631d043f6b16ebf5e00f8f001fc8dea038010b`.
Across smooth, localized pressure, localized content, offset, combined and
near-tie-masked smooth directions, the largest CPU/CUDA selected and
directional JVP relative L2 discrepancy was `5.68e-15`. Residual absolute
Linf differences were at most `1.22e-14` in the scaled residual. At the final
root the relative L2 residual difference is `0.403`, and relative Linf is
`10.47`: these ratios are dominated by a reference residual close to
roundoff and are **not** a relative-residual parity pass.

Second-order one-sided CPU residual finite differences used four feasible
steps and a deterministic seed; in this set no complementarity branch changed.
Best relative L2 discrepancies were about `2.5e-12` for smooth directions,
`6.4e-8` for localized pressure, `1.1e-13` to `5.4e-12` for localized content,
and `2.7e-9` to `5.2e-9` for offset directions. These samples do not establish
accuracy across every active-set switch; separate focused tests exercise exact
ties and overridden active rows. An earlier unmasked perturbation did cross
two complementarity branches at finite steps, so such differences must not
be interpreted as unconditional derivative errors or proofs.

For five warmed applications at the accepted 257x257 checkpoint, the CPU
residual averaged 47.15 ms and synchronized CUDA 31.45 ms (1.50x); the frozen
JVP averaged 49.37 ms and 27.27 ms (1.81x). These short resident-component
measurements omit setup, transfers, preconditioner, Krylov and audits;
observed timings varied substantially between checkpoints and shared-device
activity is not controlled. They are not end-to-end solver speedups.

This is **not a device nonlinear solver**. In the tested PyTorch CUDA build,
`torch.sparse.spsolve` reports that CUDA cuDSS support was not compiled in,
and CUDA CSR triangular solve is unavailable. The CPU preconditioner uses
SuperLU; moving full Krylov fields to the host for each application violates
the residency requirement. A robust device-native JFO pressure/content
active-set preconditioner and its verified bordered Krylov recovery remain
open work. Existing GPU signed-full-film solvers do not fill that gap.
No accepted CUDA root, full CPU/GPU solver comparison, GPU-specific run
figure, end-to-end speedup, or 513/1025 GPU solve is claimed. The 1025 gate
remains closed, and neither mesh convergence nor physical validation follows
from these operator checks. See [USER_GUIDE.md](USER_GUIDE.md) for reproduction.

Verification: the focused operator/directional suite passed **64 tests**;
the Windows-compatible suite passed **832, with 1 skipped**. Full `pytest -q`
remains blocked during collection by the existing POSIX-only `resource`
import in `tests/test_joint_refinement.py`. Changed-file Ruff passes;
repository-wide `ruff check .` reports the existing E501 in the untracked
`.kilo/` notebook helper. No unrelated files were changed to mask these
limitations.

## Saved-state device JFO preconditioner screening (2026-11-12)

The local frozen donor/JFO matrix matches the CPU model on focused small
systems, but the experimental multigrid/content-sweep preconditioner is not
production-ready. It is explicitly named
`ExperimentalDeviceDonorJFOPreconditioner`; no nonlinear driver uses it.
`examples/benchmark_ciulli_jfo_preconditioner.py` repeats its bounded linear
screen on saved checkpoints 0, 3 and 6 from the checksum-verified accepted
257x257, one-thread CPU root. It runs no nonlinear solve and does not change
the equations, tolerances or saved fields. Compact results, checkpoint
hashes, transfer estimates and hardware/software identity are committed in
`cases/ciulli_20261112_device_preconditioner_diagnostics.json`; reproduction
is documented in `USER_GUIDE.md`.

The CPU local sparse matrix contains 130,050 field unknowns. Across the three
checkpoints, matrix assembly took 0.325--0.536 s, SuperLU factorization
0.468--0.771 s, and a repeated factor solve averaged 0.0145--0.0224 s.
Factor L/U contained 9.30--9.39 million nonzeros. Full CPU correction solves
took 4.50--5.53 s, converged in 56--60 Krylov iterations, and had
independently checked relative linear residuals of `1.03e-10` to `5.55e-10`.
These factor/setup/apply timings are microbenchmarks; they are not additive to
the full correction time.

On the NVIDIA RTX A2000 Laptop GPU (PyTorch 2.11.0+cu128, CUDA 12.8), every
candidate failed the required `5e-9` true unpreconditioned residual gate:

| Checkpoint | Device total s | Setup s | GMRES s | Steps | Operator / preconditioner calls | True relative residual | Peak allocated MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 13.923 | 0.796 | 12.936 | 600 | 612 / 610 | 0.8694 | 134.0 |
| 3 | 11.216 | 0.464 | 10.686 | 600 | 612 / 610 | 0.7849 | 156.1 |
| 6 | 14.683 | 0.739 | 13.877 | 600 | 612 / 610 | 0.7945 | 156.1 |

The CPU converged in about one tenth as many iterations and in less wall time.
The experimental preconditioner therefore provides no acceleration and cannot
be accepted as a linear solver. Its repeated failure across smooth, middle,
and late Newton states closes this candidate, not the general search for a
device-native preconditioner. Focused small CPU/CUDA tests still pass; they
verify matrix/residual mechanics, not production-scale convergence.

The candidate keeps full Krylov vectors and sparse hierarchy operations on
the GPU: there are no full-field transfers per Krylov iteration. It does copy
the mobility field and active mask to the host once when building each
preconditioner, and uploads the sparse multigrid hierarchy. The estimated
total preconditioner storage is about 22.6 MB at 257x257, including both the
hierarchy and local device arrays; it is not a measured transfer-volume
counter. The input state and mask are uploaded once per correction. Scalar
validation still synchronizes the device, so “no full-field transfer” does
not mean an asynchronous loop. As a comparison, a host SuperLU application
would round-trip roughly 2.08 MB of full field vectors; at roughly 610
preconditioner applications this is about 1.27 GB of aggregate traffic per
correction, before synchronization overhead. The measured candidate avoids
that traffic but still loses because the preconditioner is ineffective.

This bounded linear benchmark does not include Newton line search, full
nonlinear iterations, final audits, plotting or run-output I/O. The archived
instrumented 257x257 CPU profile above remains the only component breakdown:
its 194.045 s total is distorted by roughly 94.9 s of Matplotlib layout and
must not be treated as ordinary runtime shares. At the time of this
preconditioner screening, no accepted CUDA root or 513x513 CUDA solve existed.
The subsequent hybrid CUDA-JVP results are documented separately below and do
not alter the failed candidate's conclusion. Do not extend iteration limits
or loosen tolerances to turn its stagnation into apparent success; it remains
unintegrated and no physical-validation claim follows from that candidate.

## Hybrid CPU-preconditioned/CUDA-JVP donor/JFO verification (2026-11-12)

This is a distinct, bounded acceleration path after the device-resident
multigrid candidate failed. The nonlinear limited-load solve, its residual,
feasibility checks, line searches and independent endpoint audits remain CPU
float64. The hybrid correction uses the existing CPU frozen local bordered
matrix and SuperLU Schur factorization as its preconditioner, while the
donor/JFO matrix-free JVPs and restarted GMRES vector operations run on the
selected PyTorch CUDA device. A full bordered Krylov vector is round-tripped
for each CPU preconditioner application; the pressure/content/material fields
are not transferred during those applications. This is not a GPU-resident
preconditioner or a fully device-resident nonlinear solver.

### Why the CPU residual is the Krylov right-hand side

The CUDA residual is computed and independently compared with the CPU
residual at every correction. The linear right-hand side is then formed from
the CPU residual and uploaded once to the device. This avoids a false late-
Newton failure caused by normalizing the tiny cross-device residual difference
by an already tiny residual norm. On saved 257x257 checkpoint 6, the CPU
residual norm is `9.6578e-10` and the CUDA/CPU residual absolute Linf
difference is only `1.208e-14`; nevertheless a CUDA-residual-based
correction has a CPU-recomputed relative true defect of `1.257e-5`. Using
the CPU residual as the right-hand side gives a CPU-recomputed relative true
defect of `1.178e-10` and relative correction difference `6.945e-11`.
The CUDA residual's relative defect is retained as a diagnostic, not used as
the correction's residual gate; its absolute parity remains checked against
`1e-10`. The `1e-9` Krylov tolerance and `5e-9` independently recomputed
linear-residual gate are unchanged.

The failed device-native preconditioner is separately diagnosed in
[`cases/ciulli_20261112_device_preconditioner_root_cause_v2.json`](../cases/ciulli_20261112_device_preconditioner_root_cause_v2.json).
On the three production checkpoints, CUDA residual/JVP and the local frozen
field-matrix action match CPU at roundoff scale, while the sampled
experimental bordered-preconditioner defect amplifies mass residual by
approximately 295--353 times. The failed corrections differ from the CPU
reference by relative L2 `0.55--0.59` and have true relative residual
`0.785--0.869`. This implicates that preconditioner approximation, not an
observed CUDA operator mismatch; the candidate remains isolated and rejected.

### Saved 257x257 correction comparison

Use the checksum-verified accepted one-thread CPU root
`results/ciulli_20261112_cpu_thread_benchmark_v3/threads_1`, with identical
saved checkpoints 0, 3 and 6. No nonlinear solve is performed by this
correction benchmark. All three hybrid corrections meet the independently
recomputed `5e-9` residual gate, `5e-7` CPU correction-parity gate, and the
absolute CPU/CUDA residual-parity `1e-10` gate:

| Checkpoint | CPU s / iterations / true residual | Hybrid s / iterations / true residual | Hybrid relative correction L2 | Correction Linf |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.345 / 56 / 5.547e-10 | 2.457 / 26 / 5.940e-10 | 3.676e-10 | 1.709e-10 |
| 3 | 5.409 / 59 / 1.587e-10 | 2.261 / 28 / 1.346e-10 | 1.416e-10 | 2.382e-12 |
| 6 | 5.307 / 60 / 1.029e-10 | 1.951 / 28 / 1.178e-10 | 6.945e-11 | 3.576e-18 |

The hybrid independently recomputed normalized true residuals therefore
reduce a unit initial residual by factors of approximately `1.68e9`,
`7.43e9` and `8.49e9`. The listed hybrid wall times include the complete
correction call; the internally timed totals are `2.444`, `2.249` and
`1.941 s`. Per-correction listed transfers were about `37.7--40.5 ms`
(`1.6--2.0%` of internal total): roughly `28.1--30.2 MB` each way for the
CPU preconditioner vectors, `1.04 MB` for the CPU residual RHS upload, and
`0.32--0.58 ms` for the correction download. The state/guide upload
(`2.08 MB`, `2.6--2.8 ms`) is a one-time initialization cost, not repeated
each correction. Transfer and full-call wall timings include synchronization
and should not be summed as independent shares.

The benchmark records per-correction residual/property work, CUDA
linearization and JVP timings, CPU matrix assembly and SuperLU setup/apply
timings, Krylov counts, explicit synchronization, transfer volumes, and peak
device allocation. It is reproducible with
`examples.benchmark_ciulli_hybrid_jfo`; output JSON remains under ignored
`results/`.

### Accepted 257x257 hybrid root

The hybrid nonlinear run starts from the exact state in CPU checkpoint 0;
it uses the same case, domain, 70,000-node budget, transport endpoint,
nonlinear/linear tolerances and seven accepted Newton corrections as the
CPU reference. Both roots pass the unchanged independent force, local/global
mass, JFO complementarity, reconstructed-film and direct-elasticity audits.
CPU elapsed time is `99.490 s`; hybrid elapsed time is `86.955 s` (12.6%
less for this run). The final fields agree as follows:

| Field | Relative L2 | RMS | Linf |
| --- | ---: | ---: | ---: |
| Pressure [Pa] | 3.784e-15 | 4.258e-7 Pa | 1.192e-5 Pa |
| Film [m] | 2.080e-17 | 2.348e-22 m | 3.388e-21 m |
| Theta | 6.974e-17 | 5.854e-17 | 9.992e-16 |
| Deformation [m] | 5.611e-16 | 2.580e-22 m | 3.388e-21 m |
| Density [kg/m^3] | 6.526e-17 | 5.869e-14 kg/m^3 | 1.137e-12 kg/m^3 |
| Viscosity [Pa s] | 4.587e-14 | 4.452e-9 Pa s | 2.512e-7 Pa s |

These are CPU/GPU implementation differences at one grid, not estimates of
spatial discretization error. The standard accepted 2x2 run-summary figure
and raw centerline profile CSVs were produced from this hybrid run's own
final fields.

### 513x513 checkpoint probe and accepted hybrid root

The 513x513 gate used the existing accepted CPU checkpoint/root rather than
repeating it for the diagnostic. At saved checkpoint 0, CPU SuperLU correction
took `41.588 s` and 69 iterations; hybrid took `15.022 s` and 29 iterations.
Relative correction L2 difference is `3.468e-10`, and the hybrid true
relative residual is `3.712e-10`. This correction's measured CUDA peak was
427.5 MiB allocated / 468 MiB reserved on an RTX A2000 Laptop GPU with
PyTorch `2.11.0+cu128` and CUDA `12.8`. This is a one-correction probe, not
the full nonlinear run's peak-memory measurement.

The subsequent hybrid solve uses the same accepted CPU 257x257 root as the
accepted CPU 513x513 reference seed (seed SHA-256
`9c3e870e9d2aa7ec8950556ff522559f5a41ecda739b3bc5c0df1337c1b3137a`).
Both 513x513 roots take seven Newton corrections and pass the independent
endpoint audit including direct elasticity. CPU elapsed time is
`1,383.955 s`; hybrid elapsed time is `921.324 s` (33.4% less). Final-field
comparisons are:

| Field | Relative L2 | RMS | Linf |
| --- | ---: | ---: | ---: |
| Pressure [Pa] | 4.545e-15 | 5.121e-7 Pa | 1.240e-5 Pa |
| Film [m] | 7.978e-17 | 8.967e-22 m | 3.388e-21 m |
| Theta | 8.151e-17 | 6.841e-17 | 1.110e-15 |
| Deformation [m] | 2.138e-15 | 9.845e-22 m | 3.388e-21 m |
| Density [kg/m^3] | 7.704e-17 | 6.928e-14 kg/m^3 | 1.364e-12 kg/m^3 |
| Viscosity [Pa s] | 5.309e-14 | 5.397e-9 Pa s | 3.772e-7 Pa s |

The accepted hybrid result has `h_min=278.512 nm`,
`h_c=465.649 nm`, `p_max=812.881 MPa`, force relative error
`2.22e-16`, local/global relative mass defects `8.58e-14` /
`1.13e-16`, maximum dimensionless mass defect `4.49e-13`, maximum
complementarity defect `2.28e-27`, and direct film-reconstruction error
`5.46e-20 m`.
Its standard per-run summary figure and machine-readable profiles are in
the local ignored run directory
`results/ciulli_20261112_hybrid_513/`.

The instrumented 513 run records 11 accepted model corrections. Across them,
the measured CPU SuperLU bordered-preconditioner setup totals `126.69 s`,
GMRES wall time `58.48 s`, CPU preconditioner applications `34.84 s`, and
CUDA JVP execution `11.13 s`; these timings are nested and must not be added
as disjoint shares. Repeated CPU factorization is the largest measured
correction cost, so this is not evidence that a fully device-resident
preconditioner is unnecessary at larger grids. The device RHS/field transfer,
active-mask bytes, operator counts, histories and source hashes remain in the
local per-iteration diagnostics.

### Scope and next decision

The hybrid path preserves the CPU solver as the reference and gives
roundoff-level CPU/GPU root parity at 257x257 and 513x513. It verifies
equation closure and backend equivalence for these cases; it does not
establish mesh convergence, physical validity, or strong-compliance
robustness. The end-to-end speedup is modest at 257x257 and 33.4% at 513x513;
the largest profiled correction cost is CPU SuperLU setup. Detailed
exclusive timings for every flux/property/FFT subkernel, Newton coordination,
line search, independent final audit, plotting and disk I/O are not available
from this hybrid run and remain a profiling limitation.

### Core pressure/flux disposition and 1025x1025 preflight (2026-11-12)

#### Targeted core pressure/flux review

The existing saved core-stencil diagnostics for the accepted 65x65, 129x129,
257x257 and 513x513 roots were reviewed; no operator or mesh study was rerun.
At the fixed nominal coordinate `(52, 59.375) um`, each nearest 3x3 stencil
has `theta=1` throughout and selects the west node for +x donor transport.
The cell mass divergences are `5.88e-15`, `0`, `1.29e-14` and
`5.93e-14 kg/(m^2 s)` respectively; the maximum absolute neighboring 3x3
residual at 513x513 is `8.38e-14 kg/(m^2 s)`. The sample falls on different
nearest nodes at the coarse grids, so these residuals are local closure checks,
not pointwise convergence measures.

At the 513x513 peak (`p=812.88 MPa`, `h=405.87 nm`, `rho=1074.65 kg/m^3`,
`eta=4.839e6 Pa s`, mobility `1.237e-24 m s`), the west/east x pressure
fluxes are `-1.42e-10` / `3.32e-9 kg/(m s)` versus Couette fluxes
`8.76e-5` / `8.72e-5 kg/(m s)`. The x and y divergence contributions at
the cell are approximately `-0.23132` and `+0.23132 kg/(m^2 s)`, cancelling
to the reported residual. The full 3x3 is flooded; there is no cavitation
branch transition, donor switch, coefficient clipping or limiter at this
sample. The production donor flux uses the documented arithmetic face
mobility and west-node Couette value for positive entrainment. Its independent
smooth manufactured-field check gives the expected first-order donor
advection and second-order pressure diffusion. No implementation defect was
identified to block further refinement. This does not remove the known
first-order donor error or establish convergence/resolution of the narrow
pressure ridge.

#### Bounded resource estimate for 1025x1025

The host/GPU snapshot was 63.2 GiB host RAM total with 29.1 GiB free, and an
RTX A2000 Laptop GPU with 4096 MiB total and 3955 MiB free. Resource estimates
use the existing hybrid defaults and do not assume any solver or physical
parameter changes.

| Resource | Measured 257x257 | Measured 513x513 | 1025x1025 estimate |
| --- | ---: | ---: | ---: |
| Full run directory | 26.91 MiB | 109.23 MiB | 436 MiB by field-area scaling |
| Hybrid end-to-end runtime | 86.96 s | 921.32 s | 9,794 s (2.72 h) from the two-point power-law exponent 3.415 |
| Sparse field matrix | 130,050 square; 518,925 nnz | 522,242 square; 2,086,413 nnz | 2,093,058 square; about 8.35M nnz by area scaling |
| SuperLU `L+U` factor nnz | 9.30M | 52.90M | about 301M using the measured 5.69x factor-fill growth per grid doubling |
| SuperLU factor storage | 113.7 MB | 643.2 MB | about 3.64 GB using the measured 257/513 factor-storage ratio |

One isolated SuperLU setup took `0.443 s` at 257 and `11.314 s` at 513.
Post-factorization process-private usage increased by approximately `536 MiB`
at 257 and `2.18 GiB` at 513 from each probe's pre-factor baseline. The measured
513 peak working set rose `1.21 GiB` above its probe baseline; scaling that
increment by the measured LU-fill ratio projects about `7.2 GiB` peak resident
memory for the 1025 factorization. Scaling the 513 private-usage increment the
same way projects about `12.4 GiB` additional committed memory, or `13.5 GiB`
including the 513 probe process baseline. These are two-grid, single-state
extrapolations, not measured 1025 peaks; compared with 29.1 GiB currently free,
host RAM appears likely to fit. The estimated run directory is also modest.
The 2.72 h runtime is an extrapolation from only two complete hybrid runs, not
a service-time guarantee.

The CUDA workspace admission fails with the current verified settings:

* The limited-load field has `n=(1025-2)^2=1,046,529` interior cells and the
  bordered GMRES vector has `2n+1=2,093,059` entries. The production DC-FFT
  estimate is `128 * 3080^2 = 1,214,259,200 bytes` (`1.131 GiB`), exceeding
  the configured `1 GiB` workspace cap.
* With the production restart of 60, the device-GMRES estimate
  `16*N*(restart+5)+32*(restart+1)^2` is `2,176,900,432 bytes`
  (`2.027 GiB`). It exceeds the configured `1 GiB` cap and half of the
  currently free GPU memory (`1.931 GiB`); actual free memory would be lower
  after operator initialization. Thus the current hybrid runner would reject
  this grid before a nonlinear solve. The observed 513 checkpoint reserved
  peak of 468 MiB, area-scaled to about 1.82 GiB, is not a full-run peak and
  does not override these explicit admission checks.

#### Decision

The core numerical review found no flux-discretization defect to block a
uniform-refinement experiment. The initial 1025x1025 preflight above records
the default restart-60/1-GiB configuration and its failure; that is a
superseded configuration, not permission to bypass admission checks. The
following saved-correction measurements select a lower restart with unchanged
linear tolerances and acceptance gates.

### Measured hybrid workspace/restart settings and 1025x1025 gate

The hybrid runner now exposes `--gmres-restart` (integer 1–60) and
`--fft-workspace-limit-mib` (integer 1–4096). The workspace limit is propagated
to the CPU field-construction FFT, the CUDA elasticity/JFO operator and GMRES
admission. The default remains restart 60 and 1024 MiB. The explicitly
supported physical-node budget now includes 1,100,000 nodes; the 1025x1025
grid has 1,050,625 physical nodes. These changes expose resource settings and
do not change the equations, constitutive laws, tolerances or nonlinear gates.

Saved 513x513 checkpoints 0/3/6 were tested with restart 20, 30 and 40, a
1536-MiB workspace limit, ten maximum Krylov cycles and `rtol=1e-9`. Each CPU
reference correction was computed once. All nine hybrid corrections passed
the `5e-9` independently recomputed true-residual gate, `5e-7` correction
relative-L2 parity gate and `1e-10` CPU/CUDA residual-parity Linf gate.
Measured correction outcomes:

| Restart | Krylov iterations (0/3/6) | Hybrid correction seconds (0/3/6) | Maximum true residual | Maximum correction relative L2 | Peak allocated / reserved |
| ---: | --- | --- | ---: | ---: | ---: |
| 20 | 40 / 40 / 40 | 16.47 / 16.75 / 17.06 | `3.75e-12` | `1.09e-9` | 280.5 / 323.0 MB |
| 30 | 29 / 30 / 30 | 15.98 / 16.00 / 15.10 | `5.20e-10` | `1.14e-9` | 322.4 / 343.9 MB |
| 40 | 29 / 31 / 32 | 16.29 / 15.02 / 15.97 | `3.72e-10` | `3.47e-10` | 364.4 / 385.9 MB |

CPU corrections took 40.00–42.05 s each and required 67–69 Krylov
iterations. Restart 20's hybrid true residual was 45.7–448 times smaller than
the matching CPU residual; for restarts 30/40 it was not uniformly smaller,
but remained within the unchanged absolute acceptance gate. Hybrid transfer
overhead was 0.061–0.092 s per correction. Restart 30 was fastest on average
at 513x513, but it does not pass the conservative 1025 projection below.
Measured peak reserved increments were about 323 MB for restart 20/30 and
365 MB for restart 40. The benchmark JSON and checkpoint hashes are preserved
under ignored `results/ciulli_20261112_hybrid_restart_memory_preflight.json`.

The measured host/GPU snapshot immediately before the 1025 launch was 28.83
GiB free host RAM, 113.13 GiB free disk, and 3,455,791,924 bytes free on the
4,294,443,008-byte RTX A2000 Laptop GPU. The prior 1025 host estimate remains
about 13.5 GiB committed memory, 436 MiB result storage and 2.72 hours; these
are extrapolations, not measured 1025 usage. At 1025 the FFT workspace estimate
is 1,214,259,200 bytes (1.131 GiB), so a 1536-MiB cap leaves 396,353,536 bytes
(378 MiB) above that estimate. This cap is 1.72 GiB below the measured free
GPU memory at the start of the selected 513 correction configuration.

For the GMRES admission check, the measured 513 reserved increments were
scaled by the 1025-to-513 grid-area ratio, then conservatively subtracted from
measured free VRAM before applying the production half-free-memory rule:

| Restart | Projected 1025 reserved increment | 1025 GMRES workspace estimate | Half projected free VRAM | Admission margin |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 1,289,328,641 B | 837,237,712 B | 1,083,231,641 B | +245,993,929 B |
| 30 | 1,289,328,641 B | 1,172,143,792 B | 1,059,114,393 B | -113,029,399 B |
| 40 | 1,456,773,919 B | 1,507,056,272 B | 975,391,754 B | -531,664,518 B |

Restart 20 is therefore the only tested configuration with positive
conservative admission headroom. It was selected despite being about 6.8%
slower per saved 513 correction than restart 30. The unchanged `1e-9`
Krylov tolerance and all true-residual/parity gates are retained.

The first 1025x1025 hybrid setup used the accepted 513x513 seed, restart 20,
a 1536-MiB FFT/workspace cap and the explicit 1,100,000-node budget. Its output
directory, including its initial checkpoint and failure summary, is
`results/ciulli_20261112_hybrid_1025_r20_1536mib_workspacecap/`; it has been
preserved. The setup stopped before the first Newton correction because the
nested `LimitedLoadLinearization` probe constructed a CPU DC-FFT operator
without receiving the configured workspace limit, and therefore used the
1-GiB default. This was an internal resource-plumbing defect rather than an
admission or equation failure. The limit is now propagated through that
constructor. A retry in
`results/ciulli_20261112_hybrid_1025_r20_1536mib_workspacecap_retry1/` was
then stopped cleanly during the correction following checkpoint 6. The
checkpoint is the latest completed Newton state, not an accepted solution;
the in-progress correction was discarded. Checkpoint 6 was preserved with
SHA-256
`42F4EAFE1EBFAAE38EC22CB07B445A9A399D0D1679E9E2F4FC45E0CCDFA2BE89`, as
were prior logs and a pre-interruption summary. The run summary now marks the
attempt `interrupted`. Its last complete diagnostic is iteration 5; no final
audit, outlet-feature analysis or 513-to-1025 field comparison exists.

The runner and hybrid correction now support scalar-only live heartbeats in
stdout and `heartbeat.jsonl`. During a correction they report Newton
iteration, correction/stage elapsed time, stage, GMRES iteration and restart
counts, estimated and most recently recomputed true residuals, cumulative
JVP, preconditioner/SuperLU, GPU and host-device transfer times, and the
complete correction timing once available. Heartbeats flush immediately and
are emitted at least every 30 seconds; GMRES also reports at five-iteration
intervals and restart boundaries. Diagnostics flag three weak consecutive
restarts, a long correction/setup/application/audit, and approach to existing
Newton/GMRES iteration limits. A `cpu_stage_without_recent_cuda_jvp` warning
is an inference from instrumented calls, not a sample of actual GPU
utilization. The solver has no global wall-time limit; no new one was added.
These changes do not alter equations, Krylov convergence checks, tolerances,
preconditioner mathematics or acceptance gates.

After the final active-SuperLU timing refinement, the focused heartbeat,
device JFO and limited-Newton tests pass (49 passed, three existing PyTorch
warnings). The Windows-compatible suite, excluding
`tests/test_joint_refinement.py`, passed 859 tests with one skip immediately
before that timing-only refinement; that excluded module imports the Unix-only
`resource` package. A full `pytest -q` collection on Windows therefore stops
at that module. Scoped Ruff (`ruff check docs examples src tests`) passes;
repository-wide `ruff check .` reports one pre-existing 101-character line in
the unrelated, untracked
`.kilo/skills/jupyter-notebook/scripts/new_notebook.py`. Pylance reports no
syntax errors and confirms all 39 `solve_limited_load` call sites remain
signature-compatible. The follow-up must use a fresh output directory and
seed from checkpoint 6; it is a new bounded Newton solve and may repeat work.
Resource projections remain estimates, not measured 1025 usage, and one
513-to-1025 pair cannot establish Richardson/GCI mesh convergence or physical
EHL validation.

### Retry-2 audit bottleneck and revised independent elasticity check

The heartbeat-enabled retry was started from checkpoint 6 in the fresh
directory
`results/ciulli_20261112_hybrid_1025_r20_1536mib_heartbeat_retry2/`. Its
checkpoint 0 is byte-identical to the source checkpoint 6 (SHA-256
`42F4EAFE1EBFAAE38EC22CB07B445A9A399D0D1679E9E2F4FC45E0CCDFA2BE89`). The
initial SI audit passed, after which the final independent elasticity audit
entered `scipy.signal.convolve2d`. That direct spatial convolution scales
quadratically with the number of grid cells. Process CPU time increased by
more than five minutes while the heartbeat file remained at sequence 9,
stage `final_audit`; the long native call prevented the Python heartbeat
thread from running. The process was interrupted at the user's request. Its
summary was preserved before being marked `interrupted`, and no root was
written as accepted.

To keep the same film-reconstruction equation and acceptance tolerance while
avoiding this audit bottleneck, `audit_limited_state(..., direct=True)` now
uses direct `convolve2d` at up to 20,000 nodes and independent, zero-padded
`scipy.signal.fftconvolve` above that size. The reported method and
`independent_elasticity_checked` flag distinguish an independent FFT
reconstruction from a direct spatial sum; `direct_elasticity_checked` is
false when the FFT path is used. Small-grid regression tests compare both
methods, including boundary samples, and the portable limited-path loader
accepts either recorded independent method while re-auditing the fields.
No solver equation, preconditioner, tolerance, or acceptance gate changed.

Applied to the saved 1025x1025 checkpoint state, the independent FFT audit
completed in 2.16 s and reported `numerically_accepted=true`,
`relative_local_balance=6.26e-12`, force relative error `2.22e-16`, and film
reconstruction error `3.39e-21 m` versus the unchanged `1.94e-16 m` limit.
At this point in the timeline, this post-interruption audit was diagnostic
evidence only and the interrupted run summary remained unaccepted. The later
fresh runner restart and its accepted status are documented below. The
targeted audit/loader/runner regression set passes
(77 passed); the full suite was not rerun after this audit implementation
change.

### Retry 3: accepted 1025x1025 checkpoint re-audit and 513-to-1025 analysis

The heartbeat-enabled retry 3 used a fresh output directory,
`results/ciulli_20261112_hybrid_1025_r20_1536mib_heartbeat_retry3/`, and
preserved checkpoint 6. The checkpoint SHA-256 remained
`42f4eafe1ebfaae38ec22cb07b445a9a399d0d1679e9e2f4fc45e0ccddfa2be89`.
The runner completed in 30.49 s with `status=accepted_numerical_root` and
zero new Newton corrections. It re-audited and accepted the saved checkpoint;
it did not resume or accept the interrupted correction, and it was not a new
1025 nonlinear solve. The run used hybrid CUDA JFO Jacobian-vector products
and CPU residual/SuperLU work, GMRES restart 20, and a 1536 MiB FFT workspace
cap.

| Independent audit [SI unless noted] | 513x513 | 1025x1025 |
| --- | ---: | ---: |
| Numerically accepted | yes | yes |
| Integrated force [N] | 15.000000000000002 | 15.000000000000002 |
| Relative force error | 2.22e-16 | 2.22e-16 |
| Relative local mass balance | 8.58e-14 | 6.26e-12 |
| Relative global mass balance | 1.13e-16 | 9.39e-17 |
| Maximum complementarity | 2.28e-27 | 6.07e-29 |
| Film reconstruction error [m] | 5.46e-20 | 3.39e-21 |
| Independent elasticity check | direct convolution | zero-padded `fftconvolve` |

The 1025 audit reports `independent_elasticity_checked=true` and
`direct_elasticity_checked=false`: its independent film reconstruction used
the scalable zero-padded SciPy FFT method, not direct spatial convolution.
The unchanged film reconstruction tolerance is `1.94e-16 m`.

The accepted saved 513 and 1025 fields were compared on common physical
coordinates using the existing pairwise mesh analysis; no solver was run for
this post-processing. Scalar changes are:

| Quantity | 513x513 | 1025x1025 | Change |
| --- | ---: | ---: | ---: |
| `h_min` [nm] | 278.512 | 276.288 | -0.80% |
| `h_c` [nm] | 465.649 | 461.126 | -0.97% |
| `p_max` [MPa] | 812.881 | 896.066 | +10.23% |
| Integrated load [N] | 15.000 | 15.000 | 0% |

The minima are at `(23.4375, +/-82.8125) um` and
`(25.0000, +/-82.03125) um`, respectively. The maximum pressure moves from
`(51.5625, +/-59.375) um` to `(53.125, +/-59.375) um`. Point `p_max` is only a
diagnostic and is not used as a convergence pass criterion.

| Common-coordinate region | Pressure relative L2 | Pressure RMS [MPa] | Film relative L2 | Film RMS [nm] |
| --- | ---: | ---: | ---: | ---: |
| Main contact, excluding outlet | 2.678% | 14.787 | 0.883% | 4.045 |
| Outlet spike | 9.473% | 34.848 | 0.684% | 2.983 |
| Full contact | 3.830% | 20.039 | 0.850% | 3.857 |

The outlet region accounts for 55.55% of the full-contact squared pressure
difference. The maximum common-coordinate pressure difference is 345.394 MPa
at `(57,57) um`; this localized discrepancy is not a pressure-error estimate
against Hertz. Both grids have zero cavitated area in the analyzed main-contact
window, with zero cavitation-state mismatch. These are pairwise field changes,
not an observed order, Richardson/GCI estimate or mesh-convergence result.

The existing cut-wise outlet tracker was applied to the accepted 1025 field.
It follows pressure maxima from the centerline to both symmetric global
maxima over 153 transverse rows from `y=-59.375` to `+59.375 um`; the
positive-branch maximum adjacent x step is 1.5625 um. The classification is
one **sampled connected curved pressure-maximum locus**, rather than a
separate centerline spike and off-axis core ridge. It does not prove a
continuous physical feature. The tracked film-constriction locus remains
downstream of the pressure locus: at `y=0`, the pressure maximum is at
`x=86.71875 um` and the film minimum at `x=94.53125 um`; at
`y=+/-59.375 um`, they are at `x=53.125 um` and `x=66.40625 um`.

Using the same fixed-band half-prominence definition as the existing 513
outlet-locus study:

| Feature | 513x513 | 1025x1025 |
| --- | ---: | ---: |
| Centerline peak x [um] | 85.938 | 86.719 |
| Centerline peak pressure [MPa] | 689.502 | 765.733 |
| Centerline width [um] / cells | 2.094 / 1.34 | 1.366 / 1.75 |
| Centerline +/-5 um pressure integral [N/m] | 4410.309 | 4164.425 |
| Off-axis global peak pressure [MPa] | 812.881 | 896.066 |
| Off-axis half-prominence width [um] / cells | 3.225 / 2.06 | 2.278 / 2.92 |
| Off-axis +/-5 um pressure integral [N/m] | 5764.355 | 5357.296 |

The widths use the 20-to-120 um fixed x-band and local-valley baseline from
the outlet-locus tracker. They differ from the separate centerline
75-to-175 um shoulder-window diagnostic elsewhere in this report. The
narrower sampled widths and still-small cell counts mean the pressure feature
remains under-resolved; its peak amplitude and continuum shape are not
established. The global `h_min` locations are farther out in y than the
pressure maxima and are not the outlet-constriction minima. This 1025
comparison does not justify physical-parameter tuning or another uniform
refinement by itself.

The re-generated 2x2 summary and top-down pressure/film-locus overlay are in
`results/ciulli_20261112_hybrid_1025_r20_1536mib_heartbeat_retry3_outlet_locus/`.
The film map is display-clipped from the global `h_min` (276.288 nm) to
996 nm; raw numerical fields and exported profiles remain unchanged. The
overlay marks the symmetric global pressure maxima, global film minimum,
theta boundary, nominal Hertz boundary and +x entrainment. Raw locus outputs
are `outlet_feature_locus.csv` and `outlet_feature_locus.json`; the common-
coordinate comparison is
`results/ciulli_20261112_hybrid_513_to_1025_analysis.json`.

The pairwise analyzer previously hard-coded cavitated-area labels `65` and
`129`; its output now keys those percentages by the actual `(ny,nx)` grids.
The focused regression verifies nonstandard grid labels as well as the
percentages.

**Conclusion:** the 1025 state passes discrete load, local/global mass,
complementarity, film and independent elasticity audits. This is an accepted
equation-closure root by re-audit, not evidence that a fresh Newton run
completed. The adjacent 513-to-1025 field changes and under-resolved outlet
width preclude a mesh-convergence claim. Experimental/physical validation
remains open; next work should target bounded local resolution or a
discretization-error study, not another automatic uniform refinement.

Verification for this analysis/documentation increment: the focused
`tests/test_ciulli_grid_analysis.py` suite passes (12 passed), and the
Windows-compatible full suite passes (862 passed, 1 skipped, 3 existing
PyTorch warnings) when excluding `tests/test_joint_refinement.py`. Plain
`pytest -q` still fails collection because that test imports the POSIX-only
`resource` module on Windows. `ruff check docs examples src tests` passes;
repository-wide `ruff check .` reports the existing E501 in the unrelated,
untracked `.kilo/skills/jupyter-notebook/scripts/new_notebook.py`.
