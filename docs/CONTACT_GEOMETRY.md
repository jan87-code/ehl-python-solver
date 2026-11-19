# Shared contact geometry and dry-contact foundation

## Scope and frozen circular baseline

The circular EHL regression baseline is commit
`c8b1577320c5473b61e4d1b500dcb1c0c959e2b2`, tagged
`circular-point-contact-frozen-2026-11-19`. The production circular case now
obtains its undeformed gap from `CircularPointContactGeometry`; its formula,
elasticity operator, solver equations and acceptance gates are unchanged.
`tests/test_geometry.py` compares the geometry and pressure-to-film field
bit-for-bit with the former circular paraboloid expression.

The geometry contract is `h_geom(x_m, y_m) -> array(ny,nx)`, with SI inputs
and output. The currently available implementations are:

- `CircularPointContactGeometry`: `x**2/(2R) + y**2/(2R)`.
- `EllipticalPointContactGeometry`: `x**2/(2Rx) + y**2/(2Ry)`.
- `LineContactGeometry`: `x**2/(2Rx)`, invariant in `y`.
- `ProfiledRollerGeometry`: `x**2/(2Rx) + g(y)` for a caller-supplied axial
  profile function.

`contact_geometry_from_config` is the shared case-input factory for
`circular_point`, `elliptical_point`, `classical_line`, and `profiled_roller`
geometry types. It builds geometry objects; it does not dispatch a solver or
turn the finite 2-D point-contact elasticity into an infinite-line model.
Contact formulations select their mathematically appropriate solver after
the geometry is built.

The profiled-roller `g(y)` contract is the `AxialProfile` callable protocol.
No profile or a `cylindrical` profile means `g(y)=0`; `TabulatedAxialProfile`
linearly interpolates user-supplied or measured `(y_m, g_m)` values and rejects
queries outside their tabulated range. Arbitrary callable profiles can also
be supplied. Built-in circular/parabolic crown and logarithmic profile laws
are intentionally not added yet; they can implement the same protocol without
changing `ProfiledRollerGeometry`.

`AxialTiltGeometry` is a distinct additive component, `h_tilt(y)=y*tan(alpha)`,
and is never folded into `g(y)`. The case-input factory currently permits
this tilt with profiled-roller geometry, not with the infinite-line selector.
`CompositeContactGeometry` sums base,
profile, tilt and other compatible height components. Its optional `time_s`
evaluation argument allows later time-dependent surface-defect or roughness
components to participate in the same sum; no transient surface model or
time-stepping behavior is implemented here. The geometry sum excludes the
central film offset `h0` and elastic deformation, which remain separate solver
terms.

`CavitatingElasticityProblem` accepts this geometry interface and retains its
legacy radii inputs for compatibility. `LineContactGeometry` is used by the
separate dry plane-strain line-contact mode below. `ProfiledRollerGeometry`
remains a geometry interface only; no finite crowned-roller EHL solver is
implemented. Tilt is available as geometry composition but has not been
independently validated in a coupled solver.

`integrate_contact_pressure` provides the shared force and contact-moment
hooks for cell-centered pressure fields. For positive compressive pressure
acting in `-z`, it returns `F=integral(p dA)`,
`Mx=-integral(y p dA)`, and `My=integral(x p dA)` about the coordinate
origin. The current dry solver uses rectangular cell-area quadrature, not
nodal trapezoidal integration.

## Dry elastic contact

`solve_dry_contact` is separate from Reynolds/JFO and uses the existing
cell-integrated, zero-padded FFT elasticity operator in float64. Its
load-controlled, frictionless, non-adhesive conditions are:

```text
p >= 0
F = sum(p) * dx * dy
gap = h_geom + w(p) - approach >= 0
p * gap = 0
```

The solver minimizes the associated convex quadratic over nonnegative
cell-loads with a fixed total load, using accelerated projected-gradient
iterations. It raises `DryContactConvergenceError` if the scaled KKT residual
does not meet the requested tolerance and retains the latest diagnostics on
the exception. The `numpy` backend is the reference; PyTorch CPU/CUDA runs the
same operator without silently transferring iterative fields to the CPU.

### Hertz verification evidence

The references use `E*` with
`1/E* = sum((1-nu_i**2)/E_i)`. The elliptical axes use the elliptic-integral
relations in Hokada et al., [Proposal for fast computational method for
Hertzian contact theory](https://arxiv.org/abs/2508.06967), Eqs. 5, 8 and 11.
The near-circular limit uses the leading curvature-ratio series below `1e-5`
to avoid cancellation in the elliptic-integral ratio.
The tested 5 N circular case uses `R=15 mm`, `E*=113 GPa`, and a `+/-3a`
square domain. The elliptical case uses principal radii `Rx=15 mm`,
`Ry=60 mm` and a domain of three analytical semi-axes in each direction.

| Case / grid | Load relative error | Approach relative error | Peak-pressure relative error | Scaled KKT residual |
|---|---:|---:|---:|---:|
| Circular, 33x33 | `4.4e-16` | `-0.1874%` | `+0.1136%` | `1.08e-6` |
| Circular, 65x65 | `-5.6e-16` | `-0.0277%` | `+0.0413%` | `1.76e-6` |
| Elliptical, 65x65 | `-3.3e-16` | `-0.0315%` | `+0.0197%` | `1.95e-6` |

For the elliptical case, the analytical values are
`a_x=61.560306 um`, `a_y=153.944249 um`, `p0=251.911010 MPa`, and
`approach=323.812642 nm`. The circular and elliptical comparisons are
finite-grid verifications, not a mesh-independent validation of all contact
geometries. At 33x33, CUDA versus NumPy maximum pressure-field difference is
`7.60e-15` relative to the NumPy peak. The CUDA test skips when no compatible
GPU is available.

## Dry plane-strain line contact

`solve_dry_line_contact` is a separate one-dimensional, load-controlled dry
contact mode. Its API accepts `line_load_n_m` in N/m; it must not be confused
with the finite two-dimensional point-contact `load_n` in N. The line load is
`sum(p_i) * dx`, where `p_i` is cell-centered pressure in Pa. The line moment
hook reports `sum(x_i p_i) * dx` in N, i.e. moment per unit axial length.

The plane-strain displacement uses the explicit datum
`w(x)=2/(pi E*) integral(p(s) log(L_ref/abs(x-s)) ds)`. The operator integrates
this logarithmic kernel over source cells and applies a zero-padded linear
convolution in one dimension. `L_ref` must be positive and is recorded with
the approach because it sets the additive displacement datum. This operator
is mathematically distinct from the existing 2-D half-space point-contact
operator; making the 2-D domain wider is not equivalent.

The Hertz reference uses
`b=sqrt(4 W R/(pi E*))`, `p0=2W/(pi b)`, and
`p(x)=p0 sqrt(max(0,1-(x/b)**2))`. Its approach for the stated logarithmic
datum is
`2W/(pi E*) * (log(2 L_ref/b)+1/2)`. For `W=1000 N/m`, `R=10 mm`,
`E*=113 GPa`, and `L_ref=45.1133 um`, the reference is
`b=10.61490 um`, `p0=59.97418 MPa`, and `approach=14.87361 nm`.

### Refinement and independent checks

The uniform-grid study used the fixed domain `+/-4.25b`. Profile errors are
relative discrete L2 differences against the nodal analytical Hertz profile;
contact width is the farthest active pressure node and is therefore
grid-quantized.

| Nodes | Line-load relative error | Pressure-profile relative L2 | Half-width relative error | Peak-pressure relative error | Approach relative error | Scaled KKT residual |
|---:|---:|---:|---:|---:|---:|---:|
| 33 | `2.22e-16` | `1.619%` | `-20.31%` | `+0.5832%` | `+0.2172%` | `1.81e-6` |
| 65 | `0` | `1.278%` | `-7.031%` | `-0.00776%` | `-0.0530%` | `1.68e-6` |
| 129 | `2.22e-16` | `0.3259%` | `-0.3906%` | `+0.00380%` | `-0.00868%` | `1.66e-6` |

A direct quadrature of the logarithmic influence integral, performed
cell-by-cell on a 9-node pressure field, agrees with the FFT displacement to
`2.44e-15` relative maximum error. The available Torch CPU and CUDA paths agree
with NumPy to `3.39e-16` and `2.82e-16` relative to peak displacement. The
NumPy implementation remains the independent reference, and all operators
retain float64. The grid trend verifies this particular dry Hertz case; it is
not a general error bound or proof for coupled lubrication.

## Stationary isothermal line-contact EHL/JFO

`LineContactEHLCase` and `solve_line_ehl_at_load` implement a distinct
stationary, infinitely long line-contact formulation. The public load field
is `line_load_n_m` [N/m]; the `classical_line_ehl` config factory rejects
finite-contact `load_n` and `force_n` fields and requires
`LineContactGeometry`. It is not routed through the point-contact total-force
API. The dimensional gap is

```text
h(x) = h0 + h_geom(x) + w(x)
h_geom(x) = x**2/(2 Rx)
w(x) = 2/(pi E*) integral(p(s)*log(L_ref/abs(x-s)) ds)
```

`h0` [m] is the rigid film offset solved by a bracketed scalar root. Positive
pressure produces positive, gap-opening deformation. The target constraint is
`integral(p dx) = line_load_n_m` [N/m], evaluated on the zero-pressure-edge
uniform grid as `sum(p_i)*dx`. The positive logarithmic displacement datum
`L_ref` remains an explicit input; report it with every `h0` result.

The line JFO flow reuses `IsothermalFluid` and its pressure-dependent
Roelands viscosity and Dowson-Higginson density. With
`X=x/L`, `H=h/h_ref`, `P=p/p_ref`, `R=rho/rho0`, `V=eta/eta0`, its mass
flux is

```text
q_hat = -avg(R H**3/V)*dP/dX + Lambda*avg(R H)*theta_west
Lambda = 12*eta0*Ue*L/(h_ref**2*p_ref)
```

This is mass flux per unit axial width; the dimensional SI face flux is
`-avg(rho*h**3/(12 eta))*dp/dx + Ue*avg(rho*h)*theta` [kg/(m s)].
Pressure/content are solved by a sparse 1-D active set with bounded
pressure-property Picard updates. The elastic coupling uses compliance
continuation and relaxed film fixed-point iterations, followed by a monotone,
bracketed `h0` load root. Failed property, coupling, or load-root iterations
retain diagnostics and are not returned as accepted roots. There is no
pressure clipping, film clamping, or change of the constitutive law.

The CPU sparse pressure/content solve remains the reference path. Selecting a
Torch CUDA backend accelerates the plane-strain zero-padded FFT elasticity
operator; the 1-D pressure and film fields cross the device boundary for that
operator, while the independent CPU path and direct-convolution final audit
remain available. This is not GPU acceleration of the sparse active-set
solver.

The final root gates include the N/m load residual, strict positive film,
zero pressure at both x boundaries, pressure/content admissibility,
complementarity, unrelaxed SI mass-flux residual, local/global flux balance,
film reconstruction and a direct (non-FFT) elasticity sum. A symmetric
pressure input must produce symmetric elastic deformation; the rolling EHL
pressure itself is not expected to be symmetric.

The minimum-film comparison uses the approximate fully flooded, isothermal
line-contact proximity correlation
`H_min=1.6*G**0.6*U**0.7*W**(-0.13)` summarized by Marian et al. (2020),
[DOI 10.3390/lubricants8100095](https://doi.org/10.3390/lubricants8100095).
Here `E'=2 E*`, `U=eta0*Ue/(E'*Rx)`, `G=alpha*E'`, and
`W=line_load_n_m/(E'*Rx)`. The correlation is a scale comparison, not an
acceptance gate or an exact matched reference for the solver's Roelands and
Dowson-Higginson property laws. A dimensionless M=20, L=10 published case is
configured in `examples/run_line_ehl_physical_diagnostics.py` and has now
been reached with the fixed-input, CUDA/CPU-SuperLU Newton continuation in
`examples/run_line_ehl_m20_newton_continuation.py`. Accepted 257-, 513- and
1025-node roots, CPU audits and digitized Figure 5/6 comparisons are recorded
in `PHYSICS.md`. This closes the earlier high-pressure continuation blocker,
not the physical-validation phase: the outlet-film discrepancy and sharp
spike resolution remain open, and the publication's cavitation treatment is
not the solver's JFO closure.

### Limits

This is a numerical stationary line-EHL/JFO mode, not design-grade physical
validation. Its uniform inlet supply, zero pressure at both x ends, isothermal
Roelands/Dowson-Higginson fluid, plane-strain elastic datum and fully
developed infinite line assumptions limit physical claims. The fixed-input
Venner-ten Napel `M=20,L=10` case is now accepted and compared with the
digitized pressure/film profiles; see `PHYSICS.md` and `ROADMAP.md` for the
continuation evidence, regional comparison and remaining limitations.
Pressure-spike width and other
sharp local features remain unresolved despite bulk load closure. The former
outlet-spike detector searched only at `x>b`; this excluded the published
Venner-ten Napel spike near `x/b=0.9035`. The corrected diagnostic searches
for the most prominent local maximum at `x>=0.5b` with prominence at least
0.1% of the global pressure maximum. Width is measured at half prominence and
the pressure integral is over that interpolated interval. It is a grid
diagnostic, not a physical acceptance gate. The accepted 4097-node field was
re-analyzed with the corrected search and still has no qualifying outlet
maximum; prior coarser-grid “not detected” records used the incomplete search
region and must not be interpreted as evidence of spike convergence.

The reported `outlet_constriction_film_m` is the minimum film value over
`x>=0.5b`, not a local-minimum detector. It can therefore exceed the global
minimum when the film is increasing throughout that outlet window. Finite
crowned rollers, tabulated measured profiles, tilt-coupled EHL, transient
EHL and thermal behavior remain out of scope.
