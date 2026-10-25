# Physics model and numerical conventions

## Ultimate target
Steady, isothermal 2D point-contact EHL, extended stage-wise to pressure-dependent viscosity and density, cavitation, shear thinning and thermal effects. The initial foundation is *not yet an EHL solver*.

## Coordinates
Array shape `(ny, nx)`. `x` follows lubricant entrainment; `y` is transverse. Use SI units at interfaces and explicit grid spacing `dx`, `dy`. Center odd-sized grids about x=y=0 for analytical comparisons.

## Film thickness
The prescribed-offset model uses
`h(x,y) = h0 + x²/(2 Rx) + y²/(2 Ry) + w(x,y)`. Here `h0` is the
dimensional central rigid-gap offset [m], not a prescribed load or an unknown
solved by force balance. `w` is the positive gap-increasing displacement [m]
from the existing cell-integrated half-space kernel. Thus positive pressure
produces positive displacement and increases the film gap in this model.
The actual minimum sampled film need not equal `h0` if the grid does not
include the coordinate origin or if the elastic deformation varies. Rigid
approach and reference indentation are not load-controlled.

## Prescribed-film Reynolds solver (implemented foundation)
For steady, incompressible, isoviscous flow, the dimensional flux is
`q = -h**3 / (12 eta0) * grad(p) + Ue * h * e_x`, where
`Ue = (u1 + u2) / 2` is the mean of the two surface velocities in the
entrainment direction. Thus
`div(h**3 grad(p)) - 12 eta0 Ue * dh/dx = 0`.

The implemented residual uses `X = x/L`, `Y = y/L`, `H = h/h_ref`, and
`P = p/p_ref`. Its dimensionless form is
`div(H**3 grad(P)) - Lambda * dH/dX = 0`, with
`Lambda = 12 eta0 Ue L / (h_ref**2 p_ref)`. Here `eta0` is in Pa s,
`Ue` in m/s, `L` and `h_ref` in m, and `p_ref` in Pa. `Lambda` is
nonnegative for the documented positive entrainment direction. If the
entrainment speed is instead defined as `u1 + u2`, the coefficient is
`6 eta0 (u1 + u2) L / (h_ref**2 p_ref)`; do not mix these conventions.

`reynolds_residual` in `src/ehl_solver/reynolds.py` computes a conservative
finite-volume divergence on uniform grid arrays of shape `(ny, nx)`, with x
along entrainment and y transverse. The edge entries represent boundary
nodes, and the residual is returned only for the `(ny - 2, nx - 2)` interior
control volumes. `dx` and `dy` are dimensionless spacings. Face values of
`H**3` use arithmetic interpolation.

`assemble_reynolds_system` assembles the same interior stencil as a float64
SciPy sparse system. The edge nodes are fixed at homogeneous Dirichlet
pressure `P = 0`, and only interior pressures are unknown; no other boundary
condition is implied. The operator is `-div(H**3 grad(P))`, positive definite
for positive prescribed `H` and these Dirichlet edges. The source-free right
hand side is `-Lambda*dH/dX`. `solve_reynolds_pressure` uses conjugate
gradients with configurable relative and absolute linear-system tolerances.
It reports the algebraic residual of the assembled system separately from a
global discrete flux-balance error computed from the net boundary flux and
integrated source. Nonconvergence raises an error.

The source-free regression wedge uses `H(X) = 1.4 - 0.4 X` over a unit
rectangular domain with `Lambda = 2`. It checks positive interior pressure,
transverse symmetry, zero edge pressure, small algebraic residual, and global
boundary-flux balance. This is a numerical verification case, not a
calibrated experimental benchmark.

`solve_reynolds_pressure` is a full-film solve: pressure values, including
negative values, are returned without clipping. Negative pressures indicate
that the unconstrained full-film assumption may not be physically valid in
that region; they are not corrected by this solver. Optional dimensionless
source `S` gives `div(H**3 grad(P)) - Lambda*dH/dX - S = 0` for manufactured
solution verification and is not an additional physical model. The solver
assumes a prescribed, strictly positive film, uniform grid spacings, steady
flow, constant viscosity and density, and homogeneous pressure on all domain
edges. It does not implement cavitation, elasticity, film coupling, load
control, or pressure-dependent properties; it is not a complete EHL solver.

## Prescribed-film mass-conserving cavitation prototype
`src/ehl_solver/cavitation.py` adds a separate stationary, prescribed-film
JFO/Elrod-Adams formulation. It retains the full-film pressure diffusion
stencil and solves the conservative dimensionless flux balance
`div(q)=0`, with
`q = -H**3 grad(P) + Lambda*H*theta*e_x`. Here `P=p/p_ref` is gauge
pressure relative to the cavitation pressure, and `theta` is fractional
film content. The complementarity conditions are `P >= 0`,
`0 <= theta <= 1`, and `P*(1-theta)=0`: full-film cells have `theta=1`,
while cavitated cells have `P=0` and may carry reduced lubricant content.
This is not pressure clipping; pressure and saturation are obtained together
from a sparse active-set system.

The positive-x saturation flux uses first-order upwinding of `theta` and the
same arithmetic face average of `H` as the validated Reynolds operator.
Consequently, setting `theta=1` exactly recovers the existing full-film
convection stencil. Diffusion retains the arithmetic face conductance for
`H**3`. Pressure is zero on all domain edges. The west boundary has uniform
prescribed `inlet_fractional_film_content` in `[0,1]`; the default `1` is
fully flooded, and lower values prescribe a starved supply. The east edge is
outflow, and no transverse advective saturation flux exists. This assumes
constant lubricant density, zero-gauge cavitation pressure, steady flow, a
uniform inlet, and no transient squeeze term. The starvation boundary is
simplified; it is not a feed-groove or inlet-meniscus model. The formulation
is a fixed-film numerical prototype, not yet integrated with elasticity,
load control, or the coupled outer iteration.

The active-set solve rejects negative saturation, active-set cycling and
linear-system failures rather than clipping state variables. It reports
local control-volume mass residuals, the sum of outward boundary fluxes and
the complementarity violation separately. A reproducible nondimensional
converging-diverging wedge (`H=1.4-0.4 X+0.8 X**2`, `Lambda=2`, `X in
[0,1]`, `Y in [-0.5,0.5]`) gives positive pressure in the converging region
and reduced saturation downstream. For grids `(ny,nx)` of 9x21, 17x41 and
33x81, the trapezoidal dimensionless pressure integrals are respectively
`2.4104e-4`, `2.5522e-4` and `2.5883e-4`; successive changes decrease.
The finest-grid maximum local mass residual is `1.42e-11`, complementarity
violation is zero, and net boundary flux is approximately `-2.9e-15`.

The script `examples/run_cavitation_verification.py` additionally tests a
repeated-wedge film `H=1.5+0.05*cos(4*pi*X)` at `Lambda=2`, with two
pressurized zones, two rupture fronts and a reformation front. On
`(ny,nx)=21x41`, `31x61`, and `51x101`, the maximum centerline pressure in
the full-film cells directly adjacent to any front, normalized by the
centerline peak, was `2.40e-1`, `6.96e-2`, and `3.73e-2`. Detected fronts
were at `X=(0.35, 0.55, 0.85)`, `(0.35, 0.533, 0.85)`, and
`(0.35, 0.53, 0.84)`; maximum local mass residuals stayed below `6.8e-11`.
This supports front-pressure reduction with refinement for this smooth,
mild geometry, but the mesh-aligned location and first-order discretization
do not establish general front accuracy or higher-order convergence.

Inlet starvation was checked on the converging-diverging wedge above at
`(ny,nx)=33x81`: reducing inlet saturation from `1` to `0.99` reduced the
dimensionless pressure integral from `2.5883e-4` to `2.4068e-4` and peak
pressure from `1.5691e-3` to `1.4850e-3`. At inlet saturation `0.9`, this
geometry had no pressurized cells and a zero pressure integral; saturation
remained in `[0.7017, 0.9317]`. Net boundary flux magnitude remained below
`3.5e-15`. These are prescribed-film numerical checks, not experimental
validation or a general starvation law.

The model still assumes a uniform scalar inlet supply and fixed film
geometry. A stronger repeated-wedge exploratory case can drive the active
set to negative saturation and is rejected explicitly; saturation is not
clipped to make that case appear successful. General reformation-front
continuity, feed geometry, transient starvation, experimental comparison,
elastic coupling and load control remain unvalidated.

## Prescribed-offset Reynolds-elasticity coupling
`solve_coupled_reynolds_elasticity` in `src/ehl_solver/coupled.py` couples
the prescribed rigid geometry to the independently validated Reynolds solver
and cell-integrated elastic kernel. Its public geometry and material inputs
are SI: coordinate vectors `x_m`, `y_m` [m], central rigid-gap offset `h0`
[m], radii [m], effective modulus `E*` [Pa], and dimensional reference
scales `L`, `h_ref` [m] and `p_ref` [Pa]. Coordinates must be uniformly
spaced, strictly ascending 1-D arrays. Internal pressure, film, and grid
spacing passed to the Reynolds solver are `P=p/p_ref`, `H=h/h_ref`,
`dx=delta_x/L`, and `dy=delta_y/L`. The dimensionless speed parameter
`Lambda=12*eta0*Ue*L/(h_ref**2*p_ref)` is passed by the caller, with
`Ue=(u1+u2)/2`.

The coupled gap is the reference rigid paraboloid plus the elastic
displacement: `h=h0+x**2/(2*Rx)+y**2/(2*Ry)+gamma*w`. The elasticity routine
receives pressure `P*p_ref` [Pa], dimensional cell widths [m], and `E*`
[Pa], returning `w` [m]. Its kernel represents the integral of `1/r`
divided by `pi*E*`; linear FFT convolution is zero-padded rather than
circular. The displacement is interpreted as positive gap increase; it is
not subtracted from the film. The dimensionless continuation factor gamma
multiplies compliance only as a numerical continuation device and reaches
one by default. `continuation_factors=(0.0,)` isolates the exact zero
compliance limit for comparison to the prescribed-film solver.

At each outer iteration, conjugate gradients solve the Reynolds equation on
the current film, pressure is relaxed toward that solution, and the elastic
gap is recomputed from the relaxed dimensional pressure. The film update is
also explicitly relaxed. Convergence requires all three independent
criteria: relative RMS pressure update, relative RMS film update, and the
Reynolds residual normalized by the RMS diffusion and entrainment terms.
The Reynolds residual is recomputed on the updated pressure and film. The
history records these metrics, minimum film, pressure extrema, compliance
stage, and inner linear iteration count. The standard logger emits progress
at the configured `log_every` interval. A non-finite or nonpositive proposed
film is rejected; it is never clamped.

### Weak-coupling verification and limits
The reproducible CPU regression case uses the parabolic geometry and
low-pressure scaling above (`p_ref=21.25 kPa`, `Lambda=12`) on a
17 x 13 grid with `x=[-0.5, 0.75] mm`, `y=[-0.5, 0.5] mm`. `E*=100 MPa`
is an intentionally compliant numerical benchmark material chosen to make
the small elastic feedback measurable at low pressure; it is not a
representative metal contact modulus. With pressure and film relaxation
0.6, default continuation factors `(0, 0.25, 0.5, 0.75, 1)`, and tolerances
`1e-6`, `1e-6`, `1e-8` for pressure update, film update, and relative Reynolds
residual respectively, the measured stage histories were:

| Compliance factor | Outer iterations | Final relative pressure update | Final relative film update | Final relative Reynolds residual |
|---:|---:|---:|---:|---:|
| 0.00 | 20 | 1.65e-8 | 0 | 5.50e-9 |
| 0.25 | 16 | 2.55e-8 | 1.58e-10 | 5.27e-9 |
| 0.50 | 16 | 2.57e-8 | 1.52e-10 | 5.26e-9 |
| 0.75 | 16 | 2.59e-8 | 1.52e-10 | 5.26e-9 |
| 1.00 | 16 | 2.62e-8 | 1.59e-10 | 5.29e-9 |

The final minimum gap was `20.0297 um`; peak absolute elastic displacement
was `70.5 nm`. Pressure ranged from approximately `-13.99 kPa` to
`7.94 kPa`. These observations show convergence of this weak-compliance
fixed-point test, not validation at realistic EHL loads or modulus. The
zero-compliance test compares the pressure field to the existing
prescribed-film solve. Relaxation-setting tests verify convergence to the
same pressure within their nonlinear tolerances.

This remains an unconstrained, stationary, isoviscous, incompressible
full-film calculation with homogeneous zero-pressure boundary conditions.
Negative pressures are retained and indicate that the full-film assumptions
are not physically valid there; no cavitation or mass-conserving rupture
model is included. There is no prescribed-load controller or load balance:
`h0` is fixed, and integrated pressure is not forced to match an applied
load. Grid/domain boundary sensitivity, realistic strong-coupling
continuation, and physical load closure remain unvalidated. A test with excessive
compliance confirms nonpositive film iterates raise an error rather than
being modified.

An exploratory stronger-compliance run on the same small grid/scales with
`E*=100 kPa` did **not** converge reliably using the existing fixed-relaxation
Picard iteration. At relaxation 0.5 and 100 iterations per stage it failed
at `gamma=0.25` with relative pressure update `2.78e-6`, film update
`2.38e-7`, and Reynolds residual `1.09e-6`. Reducing both relaxations to
0.2 and allowing 1200 iterations moved the failure to `gamma=0.75`, where
the final metrics had grown to `0.301`, `0.0338`, and `0.198`, respectively.
This is evidence of instability/divergence at stronger compliance, not a
converged strong-coupling result. Adaptive globalization and separately
tracked rejected trials are still required before extending this method.

### Mathematical signed-full-film load control
`solve_full_film_at_load` in `src/ehl_solver/load_control.py` searches for
the prescribed central offset `h0` such that the signed pressure integral
matches a requested signed target load. The interface uses dimensional
coordinates and offset [m], pressure [Pa], target/integrated load [N] and
returns the converged coupled state. Each load-function evaluation calls
`solve_coupled_reynolds_elasticity`; nonconverged states propagate as errors
and are never accepted by the scalar root finder. Before Brent's bracketed
method runs, equally spaced samples must show a strictly monotone load
response across the supplied `h0` interval and bracket the target. This
finite sampling is a local plausibility check, not proof of global
monotonicity. The returned evaluation history stores every converged h0,
calculated signed load, load residual, relative error and inner iteration
count. The present implementation does not warm-start across h0 samples or
continue in applied load.

This is a mathematical verification tool only. The full-film model retains
negative pressure; the signed load can be negative and includes tensile
contributions. On the weak-coupling verification grid with zero compliance,
the sampled signed load increased monotonically from `-7.0291e-3 N` at
`h0=10 um` to `-7.2887e-4 N` at `h0=40 um`. A 20 um reference-offset case
was recovered by a bracketed root over 15--25 um with relative signed-load
error below `1e-6`. This tests root closure against a numerical reference
generated by the same discretization; it is not independent physical
validation and does not establish positive-load monotonicity, cavitation, or
load balance for realistic EHL contacts.

### Monolithic bordered Newton signed-load verification
`BorderedReynoldsElasticityProblem` and
`solve_monolithic_full_film_load` in `src/ehl_solver/monolithic.py` solve
simultaneously for the interior dimensionless pressure and the dimensional
central offset `h0` [m]. Pressure boundary nodes are fixed to zero. The
coupled film is `h=h0+x**2/(2 Rx)+y**2/(2 Ry)+gamma*w`, where the existing
cell-integrated zero-padded elastic operator maps `p=p_ref*P` [Pa] to
gap-increasing displacement `w` [m]. `compliance_factor=gamma` is a
dimensionless numerical continuation parameter in `[0,1]`; gamma=0 is the
exact rigid-film limit and gamma=1 applies the full elastic operator. The
unknown offset is scaled as `h0/h_ref`. The Reynolds residual is divided by
the configured positive scale; the load residual is
`(integral(p dx dy)-Q_target)/Q_scale`, with dimensional integrated signed
load and target in N. Its default scale is
`max(abs(Q_target), p_ref*L**2*1e-6)` [N], and a positive explicit scale may
be supplied. Coordinates and material quantities use SI, arrays use
`(ny,nx)`, and x is the entrainment direction. Pressure remains signed;
there is no pressure clipping.

The complete directional residual, including pressure-to-elasticity-to-film
feedback, is applied matrix-free with an analytic derivative of the discrete
conservative Reynolds face-flux stencil. The pressure direction passes
through the same zero-padded DC-FFT operator used by the film equation; face
conductance, pressure gradients, entrainment source, and the exact `h0`
direction are all differentiated. No dense elasticity Jacobian is formed.
Centered finite-difference checks of the complete JVP are regression-tested.

The frozen-film pressure preconditioner uses `-A/R_scale`, where `A` is the
positive sparse matrix assembled for `-div(H^3 grad(P))` and `R_scale` is the
configured Reynolds residual scale. This sign and scale match the pressure
block of the residual Jacobian. Configurable ILU, Jacobi, and identity
strategies are available; a failed ILU factorization falls back to Jacobi,
then identity if the diagonal is invalid. A single sparse approximate inverse
is reused for both Schur pressure solves and any full bordered fallback in a
Newton correction; the matrix is rebuilt when the current film changes.
`compare_bordered_preconditioners` times and verifies the same correction for
each strategy. The `schur_condition_indicator` is the normalized magnitude
of the Schur product relative to the norms of its factors; it is a
relative-angle diagnostic, not a matrix condition-number estimate.

Inexact Newton forcing starts at `1e-2`, adapts to consecutive nonlinear
residual reduction within `[1e-4, 1e-2]`, and is tightened if the computed
direction is not a descent direction. A requested looser tolerance is capped
at `1e-2`; both each GMRES true residual and the complete bordered correction
are independently checked against hard accuracy limits. The preconditioned
GMRES callback history and the independently recomputed true residual are
stored separately. If Schur/Krylov correction fails, the full bordered
GMRES fallback is attempted and its termination and true residual are kept
on `MonolithicNewtonError`.

Each Newton step first computes a common fraction-to-boundary step that
preserves `h>film_floor_m` including both pressure-induced deformation and
the `h0` correction. Armijo backtracking then reduces the combined,
dimensionlessly scaled merit `M=0.5*(||R_p||_2^2/N + R_Q^2)` using the true
directional Armijo condition `M_trial <= M + c*alpha*(dM/dalpha)`. The
direction must have a negative derivative; up to two tighter-Krylov retries
are attempted before a non-descent failure is raised. A rejected Armijo
search retries the bordered correction up to twice with tighter Krylov
tolerances only when the independently checked linear residual exceeds
`max(1e-7, 2*krylov_rtol)`. This avoids oversolving an already accurate
correction when the failure is nonlinear rather than linear. Each retry
recomputes strict film feasibility and Armijo from the unchanged nonlinear
state; accepted steps still use one shared pressure/offset step length. The
iteration record aggregates true Krylov diagnostics and rejected trials
across retries and reports the termination reason. A small step alone is
never accepted as convergence. Accepted-iteration history records both residuals, pressure
extrema, film minimum, feasible and accepted step lengths, rejected trials,
Krylov iterations and residual histories, Schur diagnostics, and the bordered
correction accuracy. `MonolithicNewtonError` retains accepted history,
current-state extrema, deformation-to-rigid-gap ratio, line-search merit data
and failed Krylov records. Configure `log_every` for periodic standard Python
logger output; the history and return values remain the authoritative
machine-readable diagnostics.

`continue_monolithic_load` verifies its supplied converged starting solution
against the same problem, then warm-starts from the last accepted solution.
Failed trials are recorded and retried with a reduced load increment; a
rejected or unconverged state is never used as the next initial condition.
The controller has minimum step and retry limits, and raises if the target
cannot be reached.

### Bordered Newton benchmark and present limits
Run `python examples/run_monolithic_load_control_demo.py` to compare the
existing Picard reference, nested signed-load controller, and bordered
Newton solver, compare the sparse preconditioners, execute two adaptive load
steps, and save field and convergence plots below
`results/monolithic_signed_load_demo/`. The example uses the established
low-pressure numerical case (`E*=100 MPa`, grid
`(ny,nx)=13x17`, `p_ref=21.25 kPa`, `h_ref=20 um`) and sets the target to
the signed load of the converged prescribed-offset reference at `h0=20 um`.
On the development CPU, bordered Newton converged in 3 corrections with
independent load residual `4.34e-19 N`, 13 counted residual evaluations,
and relative pressure L2 difference `7.23e-8` from the Picard reference.
The nested root controller returned relative pressure L2 difference
`1.12e-11`, load residual `-2.44e-14 N`, and used 11 converged load
evaluations; those evaluations required 917 aggregate Picard outer
iterations. These timings and method-specific work counts are single-run
observations, not performance guarantees.

The weak reference state has peak `|w|` approximately `70.5 nm`, about
`0.36%` of its minimum rigid gap, and retains negative full-film pressure.
In the same small-grid demonstration, ILU, Jacobi and no preconditioner
required 8, 77 and 122 aggregate Krylov iterations for one bordered
correction. Their independently recomputed maximum true residuals were
`9.0e-10`, `2.66e-9` and `8.85e-9`; single-run setup-plus-solve times were
approximately `0.023`, `0.123` and `0.221 s` on the development CPU. The
moderate `E*=10 MPa` case converged in four corrections with a deformation
ratio about `4.18%`. These are numerical verification comparisons, not
material validation or guaranteed performance.

The `E*=1 MPa` full-compliance stress case is still unresolved. It reaches a
positive-film iterate with deformation about `32.9%` of the minimum rigid
gap, but the nonlinear Reynolds residual remains about `4.55e-2`; load error
is `7.48e-8 N`. Pressure spans approximately `-14.6 kPa` to `5.81 kPa`.
GMRES terminates with `info=0` and a true relative residual of `7.96e-3`;
the Newton correction is descending, but the Armijo search fails after 16
accepted corrections as the trial merit ceases to decrease. Thus the
reproduced failure is nonlinear line-search stagnation, not an immediate
Krylov iteration-limit failure. On the related rigid-reference load case,
tighter forcing caps of `1e-3` and `1e-4` also stagnated near Reynolds RMS
`4.55e-2`, despite bordered-correction residuals of `9.93e-4` and
`2.06e-5`, respectively. The exact JVP agrees with centered finite
differences along the difficult Newton direction. At
`E*=100 kPa`, the `h0=18 um` initial pressure gives a negative film minimum
under full compliance. `make_feasible_initial_state` can explicitly raise
`h0` by about `64.2 um` to establish a `20 nm` film margin; this only creates
an admissible starting state and does not demonstrate convergence or a
physical solution. Neither stress case establishes whether another solution
branch exists.

One numerical correction during this robustness work was replacing the prior
residual-relative Armijo surrogate with the directional-derivative Armijo
criterion above. The finite-difference JVP checks and merit-history
regressions independently verify the derivative and acceptance condition.
Despite this correction, the strong-coupling benchmark remains unverified.
The monolithic solver is a signed mathematical verification model, not a
physically validated load-controlled EHL solver. The prescribed-h0 Picard
and nested load solvers remain independent references.

`examples/run_monolithic_load_control_demo.py` prints the residuals and
globalization/Krylov/Schur diagnostics for each accepted Newton correction,
final independently recomputed Reynolds RMS and load closure, preconditioner
comparisons, normalized deformation strengths, and adaptive continuation
attempt history. The Newton visualization module plots Reynolds and load
residuals, feasible/accepted step lengths, Krylov iteration counts and
inexact-Newton forcing tolerances, preconditioned residual histories (with
true-residual markers), Schur indicators, minimum film, both pressure
extrema, and coupling-strength comparisons. Continuation plots distinguish
rejected load attempts from the accepted `h0` and minimum-film branches.
Exceptions preserve accepted history and failed-state diagnostics for
post-run inspection.

## Coupled visualization
`src/ehl_solver/visualization.py` is independent of the numerical iteration
and plots dimensional pressure [MPa], film thickness [um], central
entrainment-direction sections, transverse sections, and nonlinear history.
Section requests use linear interpolation between nodes and reject
coordinates outside the domain. Film maps mark the sampled minimum and its
coordinate. Pressure color scales include both signs; negative full-film
pressure is displayed rather than clipped.

The coupled solver collects no snapshots by default. `plot_every=N` requests
a copied pressure [Pa] and film [m] snapshot at every Nth accepted global
outer iteration. `snapshot_iterations=(...)` requests selected one-based
global accepted iterations; both selectors may be used together. Invalid
film proposals fail before snapshot collection. The report writer creates
`results/<case_name>/` and saves pressure/film maps, x- and y-directed
sections, convergence history, and paired pressure/film snapshots as PNGs.
The monolithic solver has separate Newton and load-continuation plotting
functions in the same module and a separate signed-load demonstration.
These generated results are excluded from version control. The weak Picard
demo is `examples/run_coupled_ehl_demo.py`; the Newton comparison demo is
`examples/run_monolithic_load_control_demo.py`.

### Parabolic point-contact verification
`parabolic_point_contact_film` provides the prescribed rigid geometry
`h = h_min + x**2/(2 Rx) + y**2/(2 Ry)` in SI metres; it adds no elastic
deformation. The verification case uses `h_min = 20 um`, `Rx = 10 mm`,
`Ry = 20 mm`, `eta0 = 0.04 Pa s`, surface speeds `u1 = 0.85 m/s` and
`u2 = 0 m/s`, so `Ue = 0.425 m/s`. With `L = 0.5 mm`,
`h_ref = 20 um`, and `p_ref = eta0*Ue*L/h_ref**2 = 21.25 kPa`, the
`dimensionless_reynolds_parameter` utility returns `Lambda = 12`. This also
checks equivalence of `12*eta0*((u1+u2)/2)*L/(h_ref**2*p_ref)` and
`6*eta0*(u1+u2)*L/(h_ref**2*p_ref)`; using the sum speed in the first formula
would double the correct coefficient.

The default grid-convergence case covers `x = [-2, 1.5] mm` and
`y = [-1.5, 1.5] mm`, with homogeneous Dirichlet pressure on the rectangular
edges. It has asymmetric x extents so the signed integral does not cancel by
symmetry. On the development CPU, the following values were measured:

| Grid `(ny, nx)` | Maximum pressure [Pa] | Signed pressure integral [N] | RMS `P` change on shared nodes |
|---|---:|---:|---:|
| 33 x 33 | 24,405 | 0.0052909 | — |
| 65 x 65 | 24,671 | 0.0053056 | 0.00378 |
| 97 x 97 | 24,720 | 0.0053083 | 0.000710 |

The decreasing successive changes demonstrate spatial convergence for this
specific prescribed geometry and domain; they are not an EHL reference
solution. Pressure is negative in parts of the full-film solution. The signed
integral retains those negative values and is not a physically corrected
load.

The domain study uses symmetric x extents from `[-1, 1] mm` through
`[-4, 4] mm`, with a fixed `0.05 mm` x spacing and fixed `y = [-1.5, 1.5] mm`.
The maximum pressure changes from approximately `19.68 kPa` at a `2 mm`
streamwise domain to `25.13 kPa` at an `8 mm` domain. The change from the
`6 mm` to `8 mm` domain is below 1% of the latter maximum, but the results
show that pressure remains sensitive to nearby zero-pressure boundaries.
For this symmetric geometry/domain, the unconstrained pressure is odd in x
and the signed integrated load is approximately zero; that cancellation is
not evidence of a physical zero contact load.

`reynolds_diagnostics` reports dimensional pressure extrema and trapezoidal
signed load, the relative algebraic L2 residual `||A p - b||2/||b||2`, the
maximum absolute and locally normalized control-volume flux imbalance, and
each integrated outward boundary flux. Boundary and local fluxes are
dimensionless integrals of `q = -H**3 grad(P) + Lambda*H*e_x`. For the
source-free solved cases, their net boundary flux should balance to solver
tolerance. Local balances are independent measurements from the global sum.
For the 33 x 33 asymmetric parabolic case, a representative run reported
relative algebraic residual `7.3e-11`, maximum relative local imbalance
`4.0e-11`, maximum absolute local balance `4.1e-10`, and net boundary flux
`2.4e-9` (dimensionless integrated-flux units).

`compare_reynolds_linear_solvers` times SciPy `spsolve` and unpreconditioned
conjugate gradients on the same assembled float64 system; it reports each
elapsed solve time and relative algebraic residual. One representative
development run gave:

| Grid `(ny, nx)` | Direct time [s] | Direct relative residual | CG time [s] | CG relative residual |
|---|---:|---:|---:|---:|
| 33 x 33 | 0.036 | 6.9e-15 | 0.380 | 9.7e-10 |
| 65 x 65 | 0.051 | 2.8e-14 | 0.987 | 8.3e-10 |

These single-run timings are machine- and load-dependent, not performance
guarantees. The benchmark timer excludes matrix assembly and includes the CG
iteration callback. The direct solver is a useful reference and can be faster
on these small systems; its memory and factorization costs grow quickly.
Unpreconditioned CG uses less factorization memory but converges slowly as the
grid is refined. Use `rtol=1e-10` (or tighter when verifying) and an explicit
`atol` for near-zero right-hand sides; check reported residuals rather than
relying on SciPy's status alone. For larger grids, a tested preconditioner or
iterative method study is still needed before recommending a solver.

## Elastic deformation
For combined elastic half-spaces, the normal displacement is
`w(x,y) = 1/(pi E*) integral[p(x',y') / r] dA'`, where
`r = sqrt((x-x')**2 + (y-y')**2)` and
`1/E* = sum((1 - nu_i**2) / E_i)`. This is the same `E_star` convention
used by the Hertz reference, not the alternative modulus that is twice as
large.

The implemented cell-integrated influence coefficient integrates `1/r` over
each rectangular pressure cell and evaluates displacement at cell centers.
The singular central cell is integrated analytically, not sampled at its
singular center. The resulting odd-sized lag kernel has axes `(ny_lag,
nx_lag)` with x along entrainment and y transverse; its center is the
zero-offset cell. Convolution uses zero-padded linear FFTs (no wraparound),
and its coefficients have units m/Pa, so a pressure field in Pa produces
displacement in m. Pressure values are piecewise constant within their
cells. The current deformation benchmark discretizes a dry Hertz pressure
field and checks convergence of its center displacement to the Hertz
approach `a**2/R` under grid refinement. It does not yet compare the full
displacement profile. This elastic component is validated independently and
does not constitute a coupled EHL solver.

## Validation
1. Hertz contact radius and peak pressure with known analytical values.
2. FFT convolution against direct loops, including corners and non-square arrays.
3. Cell-integrated kernel tests, direct-convolution parity, and Hertz center-displacement grid refinement.
4. Reynolds manufactured residual and prescribed-film pressure convergence,
   homogeneous Dirichlet boundary conditions, sparse-system consistency,
   parabolic point-contact spatial/domain studies, local/global flux balance,
   positive-film SPD checks, and direct-versus-iterative comparisons
   (validated). The separate prescribed-film cavitation prototype has
   conservative wedge, front-refinement, and inlet-starvation regressions;
   independent experimental validation and elastic/load-controlled
   cavitation remain planned.
5. Prescribed-offset weak Reynolds-elasticity fixed-point convergence and
   zero-compliance limit (validated only for the documented low-pressure
   numerical benchmark); the weak signed mathematical load-control cases
   also pass. Strong coupling, coupled grid/domain studies, and physically
   validated load closure remain planned.
6. Reference film-thickness cases and GPU double precision parity (planned).

## Computational strategy
GPU: use PyTorch tensors for both NVIDIA CUDA and AMD ROCm. On ROCm, PyTorch also reports GPUs via `torch.cuda`. A single environment generally uses one GPU-enabled build (CUDA **or** ROCm), not both at once.

### Fixed-load adaptive compliance continuation
`continue_monolithic_compliance` follows the signed-load branch by varying
only the dimensionless compliance multiplier in `[0,1]`. It independently
checks the supplied starting state and each accepted candidate, retains the
same feasibility/Armijo searches, halves rejected increments, and grows
accepted increments by 1.5 up to the configured maximum. Warm starts always
come from the last converged state; infeasible trials are rejected without
an offset repair. Exhaustion restores the last accepted multiplier and
raises with all attempts, failed diagnostics and the last converged solution.
The problem's load and residual scales stay fixed along this path.

Run `python -m examples.run_compliance_continuation` for a reproducible CPU
probe on `(ny,nx)=13x17`, with the rigid `h0=20 um` reference signed load.
The zero-compliance starting residual is `3.44e-14` scaled RMS/combined
norm. The full-compliance `E*=100 MPa` weak reference reaches gamma=1 in six
accepted stages (12 Newton, 57 Krylov iterations, zero line-search rejects),
with independently recomputed scaled Reynolds RMS `4.31e-10`, minimum film
`19.934 um`, and maximum displacement `71.18 nm`; measured continuation time
was `0.688 s`. The moderate `E*=10 MPa` case also reaches gamma=1 in six
accepted stages, with 16 total Newton iterations, 87 Krylov iterations, no
rejected line-search trials, independently recomputed scaled Reynolds RMS
`3.04e-10`, zero load error at printed precision, minimum film `18.506 um`,
and maximum displacement `0.786 um`; measured time was `0.665 s`. Timings
are hardware- and load-dependent.

For `E*=1 MPa`, the baseline recorded in the earlier audit already stalled
at gamma `0.179174805` after 17 attempts, 13 rejected stages. The refined
solver still reaches the same last accepted gamma: 90 aggregate Newton
iterations over accepted parts of attempted stages, 23,629 Krylov iterations
and 1,249 rejected Armijo trial evaluations including the final failed
corrections, and 78.319 s for the compliance probe. One failed bordered GMRES
fallback reached its 300 restart-cycle limit with true relative residual
`4.35e-8`. Its accepted state has independently
recomputed scaled Reynolds RMS `6.13e-9`, signed load error
`1.08e-10 N`, minimum film `15.013 um`, and maximum deformation `10.562 um`.
Thus tighter-Krylov recovery is now attempted only when the true linear
residual warrants it, and its work is reported, but it does **not** advance
the strong-coupling branch. Most rejected compliance stages terminate on
Armijo failure; one fails the bordered GMRES residual check.

For `E*=100 kPa`, a bounded continuation probe targeting gamma `0.2` rejects
gamma `0.1` and `0.05` from the rigid reference with Armijo failures (5 and 4
accepted Newton corrections, 155 and 105 Krylov iterations respectively);
the last accepted state remains gamma=0. The probe takes `1.043 s`. The separate
initialization-only test requires raising `h0` by `46.588 um` to move the
minimum film from `-46.568 um` to `20.001 nm`.
That adjustment demonstrates strict feasibility only, not convergence or a
physical branch. A branch/turning-point investigation, potentially with
pseudo-arclength continuation, remains necessary before interpreting
stagnation as absence of a solution. These results remain signed full-film
mathematical checks, with no physical-load or cavitation validation.
