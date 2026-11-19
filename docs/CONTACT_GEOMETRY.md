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

`CavitatingElasticityProblem` accepts this geometry interface and retains its
legacy radii inputs for compatibility. `LineContactGeometry` is used by the
separate dry plane-strain line-contact mode below. `ProfiledRollerGeometry`
remains a geometry interface only; crowned-roller contact and tilt are not
implemented.

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

## Limits and next step

The point-contact EHL and 2-D dry solvers remain finite-domain models using
their respective 3-D half-space formulations. The new dry line mode verifies
only plane-strain elastic Hertz contact; it does not yet include line-contact
Reynolds/JFO equations or coupled load closure. Before using it for steady
line-contact EHL/JFO, add and verify the corresponding 1-D/2-D flow
normalization, cavitation/mass conservation, boundary conditions and
independent load audits. Only after those pass should the axial-profile
interface be exercised as finite crowned-roller contact. Tilt, transient EHL
and thermal EHL remain unimplemented.
