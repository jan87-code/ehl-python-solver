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
legacy radii inputs for compatibility. The line and profile geometries are
geometry models only: this does not implement a plane-strain line-contact
elasticity model, crowned roller, or tilt.

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

## Limits and next step

The present solver is for finite two-dimensional point-contact domains with
the existing 3-D half-space kernel. `LineContactGeometry` does not make that
kernel a classical line-contact model: a line-contact verification mode
needs a suitable plane-strain elasticity formulation, explicit load-per-unit
length conventions, and comparison against the analytical Hertz line
pressure, width and approach. Establish that mode before using the profile
interface for finite crowned rollers. Do not infer that crowned profiles,
tilt, transient EHL or thermal EHL are implemented.
