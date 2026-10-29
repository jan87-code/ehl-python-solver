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
simplified; it is not a feed-groove or inlet-meniscus model. This fixed-film function remains a numerical prototype. The separate
prescribed-offset JFO/elasticity reference is described below; prescribed
load control and broad physical validation remain unfinished.

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

### Jacobian conditioning and experimental pseudo-arclength
Run `python examples/run_pseudo_arclength_diagnostic.py` for the small-grid
`E*=1 MPa` comparison. It writes fixed-compliance and pseudo-arclength
conditioning, continuation, branch, and accepted-state pressure/film plots to
`results/monolithic_pseudo_arclength_audit/` and refuses to overwrite an existing
output directory. On the `(ny,nx)=13x17` grid, the independently assembled
finite-difference Jacobian tests use the actual nonlinear residual, including
its pressure, elastic-film, mobility, entrainment, `h0`, and load terms; at
compliance 0 and 0.175 the explicit matrix relative error is below `2e-8`.
Directional checks cover three perturbation sizes. These checks verify the
discrete derivatives, not the physical model.

At accepted 1 MPa fixed-compliance states, the scaled Jacobian condition
estimate rises from `1.10e3` at gamma `0.1`, to `5.54e3` at `0.175`,
`1.40e4` at `0.178516`, and `5.33e4` at the last accepted gamma
`0.179174805`; its smallest singular value falls from `2.22e-2` to
`3.48e-4`. Over those same states the directly assembled Schur denominator
decreases from `1.36` to `5.63e-2`. Exact small-grid pressure-block Schur
reconstruction and full dense bordered solves agree to relative correction
differences below `4e-15`; their directly evaluated relative linear residuals
are below `3.1e-13`. This confirms the Schur algebra on these states, but
does not imply that iterative solves remain well-conditioned.

At gamma `0.175`, frozen-film ILU, Jacobi, and identity preconditioners
required 14, 82, and 134 Krylov iterations, with independently recomputed
true residuals `4.97e-9`, `3.68e-9`, and `6.33e-9`. The existing ILU is the
best of these tested choices at this state; no new preconditioner is claimed.
For the ILU Schur solve at gamma `0.175`, the iterative Schur denominator
differed from the exact small-grid value by `1.45e-9` relative; its pressure
and complete bordered true residuals were `4.97e-9` and `7.72e-9`. This
single-state agreement does not establish robustness close to the stall.
The fixed-compliance run still stalls at gamma `0.179174805` after 23,629
Krylov iterations and 1,249 rejected Armijo trials (132.1 s in this diagnostic
run). Its last accepted state has independently recomputed scaled Reynolds
RMS `6.13e-9`, load error `1.08e-10 N`, and minimum film `15.013 um`.

The experimental pseudo-arclength corrector uses compliance gamma as a
continuation parameter, analytic state-JVP columns, an analytic gamma
derivative (since the compliance-derivative verification increment), RMS-scaled pressure variables, strict positive-film feasibility,
and dense bordered Newton solves. It is limited to small state dimensions.
On the same grid it crossed a tangent sign reversal near a shallow maximum at
gamma `0.179191378` (condition estimate `6.59e4`, smallest singular value
`2.77e-4`, Schur denominator `-4.75e-2`), then continued back down the branch.
With a 20-step budget it accepted 20 correctors (50 dense Newton corrections,
zero rejected line searches) and reached a maximum gamma of only `0.179191378`,
ending at gamma `0.168947605`; it did not reach full compliance. The final
state independently has scaled Reynolds RMS `4.29e-8`, load error
`8.67e-19 N`, and minimum film `13.278 um`, while pressure remains signed.
This is evidence of a turning point in the computed mathematical branch, not
proof of a physical EHL fold or a solution beyond it. The small increase in
maximum gamma does not resolve the strong-coupling failure.

For `E*=100 kPa`, a bounded continuation probe targeting gamma `0.2` rejects
gamma `0.1` and `0.05` from the rigid reference with Armijo failures (5 and 4
accepted Newton corrections, 155 and 105 Krylov iterations respectively);
the last accepted state remains gamma=0. The probe takes `1.043 s`. The separate
initialization-only test requires raising `h0` by `46.588 um` to move the
minimum film from `-46.568 um` to `20.001 nm`.
That adjustment demonstrates strict feasibility only, not convergence or a
physical branch. Although the small-grid 1 MPa pseudo-arclength experiment
now detects and follows a local turning point, it does not establish
continuation to full compliance, nor explain the 100 kPa failure. These
results remain signed full-film mathematical checks, with no physical-load
or cavitation validation.

### Analytic compliance derivative verification

At fixed dimensionless pressure and offset, `dH/dgamma = w/h_ref`.
The residual derivative differentiates the arithmetic face averages of `H**3`
using `3*H**2*dH/dgamma`, and the centered entrainment term using the same
film direction. Its load row is zero because load depends only on pressure.
`compliance_derivative(state)` and the optional `compliance_direction` in
`jacobian_vector` apply this derivative without perturbing the parameter or
changing the problem. Pseudo-arclength now uses this analytic column.

CPU regressions compare the column against independent second-order residual
differences at gamma 0, 0.175 and 1 on a rectangular grid at E*=10 MPa,
with relative errors below 1e-7 over steps 1e-3 through 1e-5 and below 1e-8
at the smallest step. Joint state/parameter perturbations independently check
the complete augmented derivative. An E*=1 MPa positive-film test places the
minimum just above the film floor: the analytic column remains finite while
the previous centered parameter difference would sample an infeasible film.
This verifies discrete derivatives and removes a parameter-differencing
limitation; it does not demonstrate improved strong-branch convergence.

### Exact pseudo-arclength target correction

A predictor that crosses the requested parameter is shortened to that
parameter. Its corrector replaces the tangent-plane constraint by
`gamma - gamma_target = 0`; the original nonlinear residual, feasibility
checks, and directional Armijo search remain active. If curvature causes an
otherwise converged tangent-plane corrector to cross the target, it is
re-corrected with the same endpoint constraint before acceptance. The
recorded `arclength_residual` for an endpoint step is consequently the
parameter-constraint residual; `arclength_step` retains the nominal attempted
step budget, which can exceed the shortened endpoint predictor length. An initial root already at the target returns
without taking a step; targets behind the initial requested direction raise.

Exact scalar branches test increasing/decreasing bounded parameters and
corrector-only target crossings. The existing E*=10 MPa rectangular coupled
regression now covers targets 0.05 and exactly 1, independently recomputes
Reynolds/load closure and positive film, and compares pressure and offset
with a separate fixed-compliance matrix-free Newton solve. This establishes
endpoint correctness for these cases, not strong-compliance robustness or
physical EHL validation. A fixed-parameter endpoint at a singular fold may
still fail and must retain failed-corrector diagnostics.

### Bounded grid/domain audit of the 1 MPa signed branch

`examples/run_branch_sensitivity.py` compares three rectangular meshes and an
expanded domain. Its target is held at the baseline 13x17 rigid `h0=20 um`
signed load, **-0.0026894389511675915 N**, for every case. Each other rigid
starting root solves for its own offset at that same load; retargeting each
mesh at the same rigid offset would mix load sensitivity into this study.
The modulus is 1 MPa; radii are 10 and 20 mm; `L=0.5 mm`, `h_ref=20 um`,
`p_ref=21250 Pa`, `Lambda=12`, and Reynolds residual scale is 100.

The base domain is `x=[-0.5,0.75] mm`, `y=[-0.5,0.5] mm` with zero boundary
pressure. The expanded domain is `x=[-0.65625,0.90625] mm`,
`y=[-2/3,2/3] mm`; it adds two nodes at every edge and holds the baseline
spacings fixed. The dense continuation diagnostic caps the state size at
512, takes at most 12 continuation steps, allows 3 retries per step, and uses
initial/max arclength steps 0.03/0.05 and residual tolerance 1e-8. Its
corrector retains the existing 12-iteration and 16-backtrack bounds.

Measured with Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1 and one BLAS/OpenMP
thread, all four runs accepted 12 states, found a tangent sign reversal and
terminated at the step budget, without reaching gamma=1:

| Case `(ny,nx)` | Initial h0 [um] | Largest sampled gamma | Minimum film there [um] | Scaled Jacobian condition there | Continuation + audit [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coarse (7,9) | 19.50891 | 0.182129572 | 14.62403 | 8.30e3 | 1.03 |
| Baseline (13,17) | 20.00000 | 0.179202822 | 14.84055 | 8.04e4 | 6.54 |
| Fine (19,25) | 20.08891 | 0.178327919 | 14.81936 | 1.32e5 | 28.29 |
| Expanded (17,21) | 25.27835 | 0.176400509 | 18.83067 | 2.34e5 | 15.71 |

These are sampled maxima, not localized folds or bounds on the exact maximum.
The same-domain successive changes decrease (about 0.00293 then 0.000875),
but three small meshes and a fixed arclength sampling do not establish grid
convergence. Expanding the domain shifts the sampled maximum by about
0.00280 at fixed spacing and load. Therefore neither a mesh-independent
turning point nor domain independence is established. Condition estimates
use the existing scaled Jacobian and change with discretization size; they
are diagnostics, not invariant measures of physical stiffness.

`audit_full_film_solution` independently evaluates direct spatial convolution
using the existing separately verified cell-integrated coefficients, film
coupling, dimensional pressure scaling, edge conditions, signed trapezoidal
load and face-flux conservation. It never calls the FFT state reconstruction
or the nonlinear residual, and is capped at 1024 grid points by default.
Using the supplied film, it forms `qx=-avg(H**3)*diff(P)/dX + Lambda*avg(H)`
and `qy=-avg(H**3)*diff(P)/dY`, then differences these fluxes. This also
checks the Reynolds stencil by a distinct flux implementation. Summed
interior-volume balances telescope to the outward flux through the outer
faces of those volumes, halfway between interior and boundary nodes.

Across all 48 accepted states, the largest independently computed scaled
Reynolds RMS was 9.73e-12; absolute load error was below 8.68e-19 N;
maximum local integrated flux imbalance was below 1.43e-10 and absolute
net boundary flux below 7.07e-10 (dimensionless integrated flux). Direct/FFT
coupling differences were at roundoff. The benchmark separately enforces
Reynolds RMS <=1.01e-8, scaled load error <=1e-6, strict film feasibility,
zero pressure edges, direct displacement/film errors <=1e-12*h_ref, and
local/net flux errors <=1e-6. Telescoping mismatch must be <=1e-10.

Reproduce with a new output directory (existing directories are refused):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python examples/run_branch_sensitivity.py --plots \
  --output-directory results/branch_sensitivity_new
```

The JSON report records settings, runtime versions, all attempt diagnostics,
independent accepted-state audits and tangent-reversal state intervals. NPZ
files preserve every attempted pressure/offset state, parameter and acceptance
flag. Optional figures show pressure, film, both centerline directions,
unscaled deformation, branch, corrector and conditioning diagnostics. Dense
correctors have no Krylov history. Output data and plots are not tracked.
Negative pressures and the negative signed target remain mathematical model
features, not cavitation-corrected physical loads. This audit does not close
strong-compliance or physical EHL validation.

### Modulus/compliance equivalence and continuation scaling

For fixed geometry, load, discretization, lubricant parameters and reference
scales, the present linear elastic operator satisfies `w_E[p] = K[p]/E*`.
The coupled film is consequently `h=h_rigid+(gamma/E*)*K[p]`.
The actual compliance coefficient is `C=gamma/E*` [Pa^-1], or equivalently
`E_effective=E*/gamma` [Pa] for gamma>0. Gamma zero is the rigid limit.
Moduli with equal gamma/E* therefore define identical residual roots, state
Jacobians and bordered corrections in this model. This equivalence does not
apply if other physical inputs or the reference scales are changed along
with the modulus. Unscaled displacement w differs with modulus; gamma*w is
the film contribution that must be compared.

New regressions check this equivalence for `(E*,gamma)` of `(1 MPa,0.1)`,
`(100 kPa,0.01)`, `(5 MPa,0.5)` and `(10 MPa,1)`, including the analytic
parameter derivative's inverse-modulus factor. Separate fixed-compliance
Newton solutions and Schur/GMRES corrections agree for `(1 MPa,0.175)` and
`(100 kPa,0.0175)`. Thus the earlier failed 100 kPa gamma=0.05 trial probes
`C=5e-7 Pa^-1`, well beyond the observed baseline turning region near
`1.79e-7 Pa^-1`; it is not evidence that the rigid branch cannot be followed
at smaller compliance. Neither this observation nor a local turning point
rules out other branches.

`pseudo_arclength_continue(parameter_weight=...)` now supports the metric
`sum(state_weights*dz**2)+parameter_weight*dgamma**2`. With the optional
monolithic `compliance_reference_modulus_pa=E_ref` [Pa], the weight is
`(E_ref/E*)**2`, giving arclength in the dimensionless normalized compliance
`c=gamma*E_ref/E*`. Tangent normalization/orientation, the corrector plane,
and its Armijo merit use that metric consistently. A fixed endpoint uses
`sqrt(parameter_weight)*(gamma-gamma_target)` as its constraint residual.
Returned parameters and parameter-tangent components still refer to gamma;
recorded arclength step sizes refer to the chosen weighted metric. Omitting
the option preserves the existing unit-gamma metric. Invalid, underflowed
and overflowed weights raise rather than being clamped.

Exact scalar-fold and coupled small-grid regressions verify invariance of
accepted states and tangent reversals under a linear change of parameter,
including exact target endpoints. This improves parameter scaling, not the
state-Jacobian conditioning or the existence of roots beyond a turning point.

The modulus study uses the baseline 13x17 grid, the same negative signed
target load, and the previous 12-step resource limits. With the default
metric the 100 kPa run detects a reversal and samples gamma=0.017900999,
with maximum independently audited Reynolds RMS 1.22e-11. With
`E_ref=1 MPa`, the 1 MPa and 100 kPa runs instead take matching steps: their
largest sampled gammas are 0.179202821579 and 0.017920282158. Their full
accepted dimensionless `(P, h0/h_ref)` vectors differ by at most 6.67e-16, and normalized
gammas differ by at most 1.95e-16. Both peak states have minimum film
14.84055 um, effective modulus about 5.58027 MPa, and scaled state-Jacobian
condition about 8.04e4. The 100 kPa branch audit has maximum scaled Reynolds
RMS 8.87e-12, absolute signed load error below 8.68e-19 N, and absolute net
flux below 6.43e-10. Both runs stop at the 12-step budget after a reversal,
without reaching their requested full compliance. Observed continuation plus
audit wall times were 6.78 and 6.61 s, excluding plotting.

Reproduce the normalized comparison with a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python examples/run_branch_sensitivity.py --study modulus --plots \
  --compliance-reference-modulus-pa 1000000 \
  --output-directory results/modulus_scaling_new
```

Omit the reference-modulus option to compare the original parameter metric.
The report records the selected metric, per-case modulus, normalized
compliance and effective modulus at each sampled maximum. These are signed
mathematical branch checks, not physical bearing-material validation, a
localized fold, or full-compliance success at 100 kPa.


### Bracketed parameter-turn localization

`turning_point.localize_parameter_turn` is a bounded dense diagnostic for
callbacks using the same explicitly scaled state/parameter conventions as
`pseudo_arclength_continue`. It accepts two nearby converged roots whose
parameter tangents, oriented along their weighted chord, have opposite
signs. EHL benchmark callbacks use interior dimensionless `P`, `h0/h_ref`
and gamma; dimensional solution fields remain Pa and m. The metric is the
existing RMS pressure/unit offset metric with parameter weight
`(E_ref/E*)**2` when normalized compliance is selected.

Each refinement takes one tangent-plane predictor/corrector from the left
root, with step equal to half the weighted bracket chord length. The
existing residual, feasibility, Armijo, and tangent calculations are reused.
The candidate must lie within 0.9 old bracket widths of both endpoints;
otherwise refinement stops with the old bracket intact. Opposing oriented
tangent signs select the next bracket. Localization requires both weighted
bracket width <=1e-6 and `abs(sqrt(parameter_weight)*t_gamma)<=1e-6`.
The residual-root tolerance is 1e-10 RMS, with at most 24 refinements, one
12-iteration/16-backtrack corrector per refinement, no retries, and at most
512 state variables. Failed predictors/correctors, noncontracting probes,
and budget exhaustion retain the bracket and all attempted probe records.
Initial invalid/non-root/non-bracketing inputs raise. No pressure or film
clamps are introduced.

The contraction check is a safeguard for a *local* branch bracket, not a
proof of branch uniqueness or an interval bound for a continuum solution.
A tangent reversal locates a stationary parameter in the computed branch;
it alone does not prove the nondegeneracy conditions for a simple fold, a
physical instability, or the absence of other branches. Generic analytical
regressions cover a quadratic fold, a cubic turn with a transverse state,
parameter-rescaling invariance, failed feasibility, a distant noncontracting
root, invalid controls, and budget exhaustion. An independently audited
small coupled case verifies localization and restoration of mutable problem
compliance after the benchmark callback adapter returns.

The branch study accepts `--localize-turn` and
`--turn-bracket-tolerance` to enable/refine localization. Per-case JSON files
preserve both endpoint roots/tangents and every attempted corrector before
independent state audits run. The main report records the best root,
contraction history, normalized parameter tangent, and direct elasticity,
Reynolds, load and conservation audits for each accepted probe. Optional
plots show bracket contraction and parameter-tangent reduction alongside
the existing sampled-branch/field plots. Existing directories are refused.

#### Localized 1 MPa grid/domain results

Run the previous four-case study with localization and normalized compliance:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python examples/run_branch_sensitivity.py --localize-turn --plots \
  --compliance-reference-modulus-pa 1000000 \
  --output-directory results/turn_localization_new
```

At the same fixed negative signed target and physical inputs documented
above, every case localized after 16 refinements:

| Case `(ny,nx)` | Localized gamma | Change from sampled maximum | Final bracket width | Normalized parameter tangent | Continuation + audits + localization [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coarse (7,9) | 0.182176253504 | 4.67e-5 | 7.63e-7 | -1.82e-7 | 2.94 |
| Baseline (13,17) | 0.179225816365 | 2.30e-5 | 7.64e-7 | 4.38e-7 | 17.05 |
| Fine (19,25) | 0.178395632035 | 6.77e-5 | 7.64e-7 | 3.77e-7 | 66.71 |
| Expanded (17,21) | 0.176425633616 | 2.51e-5 | 7.64e-7 | 2.32e-7 | 38.65 |

All 64 accepted localization probes pass independently evaluated scaled
Reynolds RMS <=1.01e-10, signed load, positive film, direct/FFT coupling,
zero pressure boundary and local/global flux checks. The largest measured
scaled Reynolds RMS is below 8.87e-11, absolute load error below 1.31e-18 N,
and absolute net integrated flux below 3.73e-9 (dimensionless flux units).
The direct/FFT quadrature coefficients are the same separately verified
cell integrals; this checks summation and coupling, not an independent
elasticity quadrature. The wall timings above exclude plotting and vary with
machine/load. All runs use one BLAS/OpenMP thread and the earlier grid caps.

A baseline restart from its retained bracket at tolerance 1e-7 took three
more refinements: gamma=0.179225816365158, bracket width 9.55e-8, normalized
tangent -1.72e-8. Its gamma change was 1.67e-13. This is an empirical
localization sensitivity check under the fixed 1e-10 residual tolerance,
not a rigorous error bound or an independently proved fold condition.

The localized coarse-to-baseline shift is about 0.00295044 and the
baseline-to-fine shift about 0.00083018. Both exceed the observed sampling
corrections. At fixed baseline spacing and target load, the expanded-domain
shift is about -0.00280018. Thus resolving the parameter turn does not
remove discretization/domain dependence. These small grids and one domain
expansion do not support a continuum limiting gamma, an asymptotic order
claim, or physical instability. Gamma=1 remains unreached by these branches.
The negative target cannot serve as a nonnegative-pressure physical EHL
load benchmark; future coupled cavitation verification needs an admissible
positive-load case.

#### Discrete nondegeneracy checks at saved turns

`audit_turning_point_conditions` checks additional evidence at an approximate
root of `F(z,c)=0`. For an exact turn with a one-dimensional Jacobian kernel,
let phi/psi be right/left null directions. Nonzero
`a=psi.T*F_c` and `b=psi.T*F_zz[phi,phi]` provide transversality and quadratic
nondegeneracy tests. These are standard folded-manifold conditions; see
[Zhang et al., equation (6)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3520151/).
For SVD-based continuation conditioning, see
[Dickson et al. (2007)](https://doi.org/10.1137/060654384).
These mathematical tests do not validate EHL physics.

The diagnostic normalizes phi with the supplied state metric, psi with the
Euclidean residual norm, and uses `c=sqrt(parameter_weight)*gamma`.
It orients psi so that a is nonnegative. At an exact turn, differentiating
the residual twice along unit weighted arclength gives the local parameter
curvature `c''=-b/a`. This curvature does not determine dynamic stability.
A centered full residual-difference Jacobian independently checks the
analytic matrix; centered second residual differences at fixed gamma check
b for steps 1e-2, 3e-3 and 1e-3. Every perturbed state must be feasible.
The state dimension remains capped at 512. No residual, pressure or film
clipping is used. Zero transversality gives undefined curvature (NaN),
multiple null directions give no isolated singular gap, and zero quadratic
curvature is preserved rather than labeled nondegenerate.

`python -m examples.audit_localized_turns` replays retained roots, validates
their model settings and independent SI state audits, and writes a new JSON
report without rerunning continuation. The numerical evidence flags require:
analytic/FD Jacobian relative error <2e-8; smallest/next singular ratio <1e-4;
next/largest ratio >1e-8; normalized transversality and curvature projections
>1e-4; relative variation of b against the smallest-step estimate <1e-3;
and negative predicted parameter curvature for every step. These explicit
thresholds classify numerical evidence, not exact rank or a rigorous fold.
The CLI returns a failure status if any case does not satisfy those checks.

For the four saved localized 1 MPa cases (`E_ref=1 MPa`), the measured checks
all pass:

| Case | sigma_min / sigma_next | Normalized transversality | Predicted c'' at step 1e-3 | Relative b step variation | Analytic / FD Jacobian relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coarse | 6.06e-8 | 0.3469 | -1.078094 | 1.31e-5 | 1.39e-11 |
| Baseline | 1.30e-7 | 0.3152 | -1.191400 | 1.53e-5 | 9.37e-12 |
| Fine | 1.09e-7 | 0.3075 | -1.226672 | 1.60e-5 | 8.99e-12 |
| Expanded | 5.03e-8 | 0.3073 | -1.285089 | 1.72e-5 | 6.01e-12 |

The next-smallest singular values range from 0.0937 to 0.1454, versus
smallest values from 5.68e-9 to 1.36e-8. Normalized curvature projections
are between 0.103 and 0.145, separated from zero. These measurements support
a nondegenerate local maximum in each *discrete signed branch*, while the
localization and residual errors preclude an exact singularity proof. They
also do not supply a continuum limit or an exclusion of other branches.

Reproduce the diagnostics directly from an existing localized study:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m examples.audit_localized_turns \
  --source-directory results/turn_localization_new \
  --output-directory results/turn_conditions_new
```

The source roots remain unchanged. Regression tests include exact analytic
coefficients, parameter scaling, a transverse state with nonuniform metric,
zero-curvature and zero-transversality counterexamples, nullity two,
incorrect supplied Jacobians, infeasible perturbations, and a coupled saved
root replay. Nondegeneracy checks do not close the physical load/cavitation
or full-compliance roadmap milestones.

### Prescribed-offset JFO/elasticity CPU reference

`cavitating_coupled.py` now couples the existing fixed-film active-set JFO
solver with the cell-integrated elastic operator, as a reference algorithm
at **prescribed h0**. `CavitatingElasticityProblem` accepts SI coordinates,
radii, offset, modulus, reference lengths and pressure. Array order is
`(ny,nx)` and x follows positive entrainment. Internal quantities remain
`P=p/p_ref`, `H=h/h_ref`, `dx=dx_m/L`, `dy=dy_m/L` and
`Lambda=12*eta*Ue*L/(h_ref**2*p_ref)`. It preserves zero boundary gauge
pressure, uniform west inlet saturation, and the fixed-film upwind
saturation/arithmetic-H face flux. No density/viscosity variation is added.

The reference iterates on film, solves pressure and saturation together,
and constructs `h_exact=h0+geometry+gamma*w(p)`. It relaxes only the film
iterate, avoiding pressure/saturation mixing that can break complementarity.
Both update/film defects and mass/admissibility checks are required. Mass
RMS, maximum divergence and maximum relative local integrated imbalance
must be <=1e-8 on **h_exact**, not the previous inner-solve film. Pressure
and saturation feasibility and `max|P*(1-theta)|` use tolerance 1e-10.
Pressure and saturation are never clipped. Returned film coupling is exact
to FFT rounding. Nonconvergence carries full history and the last candidate
in an exception, without reporting that candidate as a converged root.
This reference does not replace the experimental monolithic architecture.

`audit_cavitating_solution` independently checks SI fields using direct
spatial convolution and the existing opposing-face JFO balance stencil,
while the coupled implementation uses whole-face arrays and `np.diff`.
It verifies elasticity/film coupling, boundary/inlet conditions, mass,
complementarity, bounds, and trapezoidal positive pressure force. Numerical
checks cover the zero-compliance fixed-film identity, full-film stencil
identity, 100/10/1 MPa small cases, relaxation independence, inlet starvation,
invalid film rejection, active-set failure and false convergence from tiny
film steps. Prescribed load is not yet solved.

The reproducible benchmark uses `h0=h_ref=20 um`, radii 10/20 mm,
`E*=10 MPa`, `L=0.5 mm`, `p_ref=21250 Pa`, `Lambda=12`,
`x=[-0.5,0.75] mm`, `y=[-0.5,0.5] mm`, compliance stages
0/0.25/0.5/0.75/1, film relaxation 0.5 and at most 100 outer iterations per
stage. Measured results at full compliance are:

| `(ny,nx)` | Inlet theta | Force [N] | Minimum film [um] | Minimum theta | Independent max mass residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| (9,13) | 1 | 0.002210200631 | 20.3253 | 0.492227 | 9.48e-9 |
| (17,25) | 1 | 0.002342149561 | 20.3441 | 0.470920 | 5.38e-9 |
| (25,37) | 1 | 0.002357024095 | 20.3450 | 0.464569 | 5.44e-9 |
| (17,25) | 0.99 | 0.002320457963 | 20.3417 | 0.470752 | 5.35e-9 |

Same-domain force changes decrease under refinement; this is a numerical
convergence observation, not experimental validation or domain independence.
Reducing inlet supply reduces force and conserves mass. Each field has
nonnegative pressure within the stated tolerance and complementary reduced
saturation downstream. These small positive-load prescribed-offset cases
are substantially different from the earlier negative signed-load branch.
They do not establish prescribed-load equilibrium, a bearing-steel EHL
film-height reference, strong-coupling robustness, or a full EHL solution.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python examples/run_cavitating_elasticity.py --plots \
  --output-directory results/cavitating_elasticity_new
```

The new output directory holds SI fields, saturation, full convergence
history, independent audits and pressure/film sections, deformation,
saturation and convergence plots. Existing output directories are refused.

### Monolithic positive-load JFO/elasticity Newton verification

`cavitating_newton.py` adds a separate experimental monolithic system for
interior `P=p/p_ref` and deficit `D=1-theta`, with optional simultaneous
`h0/h_ref`. All geometry, material, solution fields and target force inputs
are SI; arrays remain `(ny,nx)`, x entrainment. Residual rows are
`div(q)/100`, `min(P,D)`, and (when a positive target is supplied)
`(integral(p)-Q_target)/Q_target`. The film is reconstructed exactly as
`h0+geometry+gamma*w(p)` on every residual call. Zero boundary gauge pressure,
uniform inlet saturation and the existing conservative upwind JFO flux
are preserved. This is a numerical isoviscous/incompressible model, not a
validated full EHL solver or replacement for the retained signed references.

The analytic matrix-free generalized JVP differentiates pressure gradients,
`H**3` face mobility, pressure-driven elastic film, convective film/content,
the optional offset, and the exact force row. Complementarity uses dD in
full-film cells (`P>=D`) and dP otherwise. Away from switches, complete
Jacobian columns and directional products pass centered residual-difference
checks at zero/full compliance, with/without the load border. At exact ties,
the merit slope uses the true one-sided `min(dP,dD)` directional derivative.

A sparse frozen-film active-set LU is the pressure/content preconditioner.
For prescribed force, two pressure/content GMRES solves and an explicit h0
Schur correction reconstruct the bordered Newton step. Its complete true
linear residual is checked. Failed block solves, an unsafe Schur indicator,
or an inaccurate reconstruction recover through full bordered GMRES; tests
force both Schur and fallback paths and compare to a complete dense
linearization. Every GMRES result is independently checked using the actual
JVP rather than trusting its status or preconditioned norm. Default rtol is
1e-9, with true-residual ceiling 5*rtol, restart 80 and at most 20 cycles.
The state-dependent film Jacobian is not included in the frozen-film block.

A fraction-to-boundary safeguard for P/D bounds precedes a directional
Armijo search with strict film feasibility. The stated dimensionless bound
tolerance is 1e-10; tiny negative roundoff is reported, never clipped.
Acceptance requires max mass divergence and relative local imbalance <=1e-8,
min-residual and complementarity <=1e-10, admissibility, and scaled force
error <=1e-8 where applicable. Small updates do not imply convergence.
Defaults cap Newton corrections at 30 and backtracks at 20. Exceptions
retain the last candidate/history and failed true-Krylov diagnostics.
Accepted history records pre/post merits, slope, step, active cells, post-step
film/offset/primal extrema, Schur/fallback indicators and complete
preconditioned/true Krylov histories.

At prescribed h0 on the 9x13 reference grid, cold-start Newton converges in
2/10/20 corrections for E*=100/10/1 MPa and agrees with the independent
fixed-point reference to below 1e-9 relative pressure difference. Known
positive-force roots from those reference states are also recovered by
simultaneously solving offset after starting from 22 um instead of 20 um.
These are discrete numerical parity tests, not physical material validation.

A separate prescribed-force benchmark holds **Q=0.002 N** and E*=10 MPa
fixed, using the previous parabolic domain, scales and inlet. Cold-start
results at full compliance are:

| `(ny,nx)` | Solved h0 [um] | Minimum film [um] | Peak pressure [Pa] | Minimum theta | Max independent mass residual | Newton / Krylov iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (9,13) | 20.935777690 | 21.231098 | 8263.721 | 0.502535 | 1.33e-11 | 11 / 110 |
| (17,25) | 21.489860487 | 21.785007 | 7912.916 | 0.487266 | 3.83e-10 | 7 / 70 |
| (25,37) | 21.550575804 | 21.844698 | 7929.420 | 0.481504 | 2.45e-13 | 19 / 190 |

Force error is zero at printed float64 precision, complementarity is below
3.57e-23, and net boundary flux below 4.29e-11 in dimensionless integrated
units. Minimum Schur indicators range from 0.549 to 0.574; these runs need no
bordered fallback. Maximum complete true linear residual is below 5.38e-10.
An independent nested scalar h0 root on the coarse grid uses nine converged
fixed-point force evaluations; its h0 differs by 4.13e-17 m and pressure by
8.25e-13 relative L2. Each nested solve tightens mass tolerance to 1e-10.
Decreasing offset changes under same-domain refinement do not establish
domain independence, asymptotic order, or independent EHL film-height
validation. The low-pressure cases and artificial low moduli must not be
presented as validated bearing-steel results.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_cavitating_newton --plots \
  --output-directory results/cavitating_newton_new
```

New directories preserve raw SI fields, content, convergence/globalization,
Schur/Krylov diagnostics, nested force evaluations, independent audits and
field/section/saturation/deformation/convergence plots. No existing baseline
is overwritten. Broad load/continuation robustness and pressure-dependent
properties remain unverified.

### Adaptive positive-force continuation

`cavitating_continuation.continue_cavitating_load` advances from a supplied
converged SI state toward a positive SI force target, holding compliance and
all other physical inputs fixed. Only the last independently recomputed
root supplies a warm start. Acceptance checks force, exact elastic film,
displacement, boundary/inlet data, mass balance, min(P,D), product
complementarity and primal bounds. A solver success flag is insufficient.

Step fractions refer to the original force span. Defaults begin at 0.1,
grow accepted steps by 1.5 up to 0.25, halve rejected *actual* force steps,
and stop below 0.005 or after 30 attempts/6 retries per stage. Shortened
final steps are reduced on rejection rather than repeatedly attempting the
same target. A target already within the configured load tolerance returns
the verified initial root without roundoff-sized steps. Configuration does
not permit overriding the physical compliance or warm-start/target data via
solver options. Exhaustion retains every trial, rejected candidate, Newton/
Krylov failure diagnostics and the last independently accepted root. The
problem and starting state are not modified.

Regression tests cover increasing/decreasing loads, independent direct-root
parity, failure/retry warm starts, exhausted budgets, a corrupted returned
candidate, invalid starts, near-identical targets and shortened-target retry
behavior. The benchmark additionally uses direct spatial elasticity and the
separate opposing-face mass stencil for every accepted trial.

On `(ny,nx)=13x17`, E*=10 MPa, the previous geometry/scales and an initial
0.002 N root, each target below is reached in six accepted load stages with
no rejected load stages:

| Target force [N] | Solved h0 [um] | Minimum film [um] | Maximum displacement [um] | Minimum theta | Total Newton / Krylov iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 28.801894 | 29.007154 | 0.201756 | 0.565454 | 25 / 248 |
| 0.005 | 13.621900 | 14.390149 | 1.043520 | 0.400688 | 23 / 265 |
| 0.020 | 4.620289 | 7.818798 | 4.102689 | 0.285608 | 29 / 436 |

Across the 18 accepted load stages, maximum independently measured mass
divergence is below 3.35e-9, product complementarity below 9.51e-23, and
absolute integrated net flux below 1.15e-10. Final force errors are below
8.68e-19 N. Tiny Newton step lengths near content switches remain visible
in history; acceptance still requires final equation residuals. At 0.02 N,
peak pressure is about 80.775 kPa and displacement is appreciable relative
to the minimum film. An independent prescribed-offset fixed-point solve at
that final h0, with mass tolerance 1e-10, converges in 132 aggregate outer
iterations and agrees in pressure to 1.46e-12 relative L2. Its force is
0.01999999999999384 N, independently computed from the field.

These tests expand numerical coupling/load coverage; they do not establish
bearing-steel material validity, domain independence at these loads, an
experimental film-height comparison, or a pressure-dependent fluid model.
No broad physical-EHL validation box is closed by continuation alone.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_cavitating_load_continuation --plots \
  --output-directory results/cavitating_continuation_new
```

The fresh directory retains accepted/rejected trial metadata, SI candidate
arrays, all Newton/Krylov diagnostics, independent audits and field/branch
plots. `strong_reference.json` records the independent final reference.

### Fixed-positive-force domain sensitivity and warm-start recovery

`examples.run_cavitating_domain` holds Q=0.002 N, E*=10 MPa, all physical
scales/radii and inlet theta=1 fixed while adding boundary nodes at constant
spacing `dx=78.125 um`, `dy=83.333333 um`. Each domain first obtains a
verified prescribed-h0 coupled reference at h0=20 um, then uses adaptive
monolithic force continuation to the common target. This prevents changing
force between domains or interpreting an unconverged cold start as a root.

A bounded cold-start probe reached the base domain but exhausted the existing
30-correction Newton cap on 21x25 and 29x33, with remaining maximum mass
residuals about 1.95 and 7.97. The warm-start study does not raise that cap.
All three domains subsequently reach the target:

| `(ny,nx)` | x interval [mm] | y interval [mm] | Final h0 [um] | Peak pressure [Pa] | Minimum film [um] | Accepted / rejected load trials |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| (13,17) | [-0.5,0.75] | [-0.5,0.5] | 21.300524 | 8029.09 | 21.661017 | 6 / 0 |
| (21,25) | [-0.8125,1.0625] | [-0.833333,0.833333] | 46.368401 | 3065.96 | 46.606636 | 6 / 0 |
| (29,33) | [-1.125,1.375] | [-1.166667,1.166667] | 76.902079 | 1617.33 | 77.086215 | 8 / 3 |

All 20 accepted load roots pass direct elasticity, film, force, mass,
complementarity, boundary and inlet audits. Maximum independent mass
residual across those roots is below 8.76e-9 and maximum absolute integrated
net flux below 1.12e-8. Final force errors are below 8.68e-19 N. Three actual
29x33 trial failures exhaust the same 30-correction cap with maximum mass
residuals 0.3694, 0.2862 and 3.84e-7. They are rejected; reduced steps from
the preceding converged roots subsequently reach the requested force.
Accepted/rejected trial states and full Newton/Krylov histories are retained.

The domain effect is substantial and not converged. Prior same-domain mesh
refinement and load closure do **not** establish a physical film prediction.
Moving a fully supplied inlet also changes local inlet geometry and the
resulting supply/backflow budget. Individual outward dimensionless fluxes
are now recorded by `audit_cavitating_solution`, rather than reporting only
their near-zero sum. At the three final roots, west/east/south/north fluxes
are approximately `(-30.653,27.095,1.779,1.779)`,
`(-121.247,104.326,8.461,8.461)`, and
`(-298.724,251.779,23.473,23.473)`. Thus flow budgets vary with the domain
and boundary model even when net balance and force closure pass. These data
do not isolate one cause or prove a boundary implementation defect; inlet/
far-field consistency and larger-domain studies remain necessary.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_cavitating_domain --plots \
  --output-directory results/cavitating_domain_new
```

The largest grid has 957 nodes, within the 1024-point direct-audit cap.
Each output directory must be new. Reference roots, all trial candidates,
SI grid/fields, failure reasons, independent audits and field/path plots are
preserved. This study supports verified numerical recovery and exposes a
material limitation; it does not close domain independence, broad physical
EHL validation, or film-height comparison milestones.

### Desktop native HIP audit (2026-10-25)

The desktop exposes an AMD Radeon RX 6600 XT (`gfx1032`) to host processes;
sandboxed processes do not see `/dev/kfd` or `/dev/dri`. GPU checks therefore
ran outside the sandbox. The existing staged HIP feature was copied with
`git checkout-index` into `/tmp/ehl-gpu-validation`, preserving the original
index/worktree. Its extension was built there using `scripts/build_hip.py`
with explicit `--arch gfx1032` and the existing GPU-Testing SDK, version
10.3.0a20261023. No drivers, system packages or original staged sources were
changed. The tested C++ source SHA256 is
`8c56da57e3b69da3a23bfaeaa899a7b376a48203ad17984d0d11739640af8662`.
The tested runtime reported HIP version integer 71726423 and
hipFFT version integer 10041. This optional HIP feature remains separate
from the current develop backend API; it was not silently merged/enabled.

Host regression selection returned 40 passed, 5 optional PyTorch skips;
all seven native HIP regressions executed. Additional checkpointed checks
cover 12 random/asymmetric/even/edge/zero/strided-input cases against direct
spatial convolution, four invalid native-buffer cases, and 100 repeated
calls. Largest direct error normalized by reference peak was 8.44e-16.
The fixed-shape repeated-call free-memory readings were identical after
warmup; this is an observation, not a leak proof under desktop activity.
A broader 48-shape sweep exhausted its 180 s budget before producing timings;
plan/compilation cost is suspected but was not profiled. The reduced sweep checkpoints each case and completes.

Three-repetition medians, one BLAS/OpenMP thread, float64 inputs and
complex128 HIP FFTs:

| Grid | HIP event compute [ms] | HIP convolution wall [ms] | NumPy convolution [ms] | HIP full elasticity [ms] | NumPy full elasticity [ms] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 129x129 | 0.442 | 21.168 | 7.519 | 45.475 | 31.700 |
| 257x257 | 2.944 | 31.263 | 119.581 | 133.223 | 220.970 |
| 513x513 | 16.139 | 48.387 | 239.674 | 470.378 | 696.261 |
| 1025x1025 | 153.718 | 239.056 | 2821.556 | 2084.967 | 4653.391 |

Convolution wall includes allocation/planning and host/device transfers;
full elasticity also includes CPU influence-coefficient generation. Maximum
displacement differences were below 4.88e-21 m and errors normalized by
reference peak below 2.59e-15. Sizes are capped at 1025; the large case uses
about 353 MB in explicit device buffers and is gated against four times that
amount of free device memory for workspace room. This gate is conservative
experimental protection, not a proven peak-workspace estimate. Timings vary
with planning/cache/desktop activity and are not performance guarantees.

Fresh staged-selector checks choose NumPy at 129x129 (34.808 ms versus HIP
48.300 ms full elasticity) and HIP at 513x513 (494.919 ms versus NumPy
648.566 ms). The selector checks float64/FFT capability and memory before
using synchronized workload timings. These observations concern the staged
selector; develop's production numerical solvers still default to CPU.

`examples.validate_native_hip` also compares the current positive-load solver
with temporary, explicit GPU-convolution hooks, leaving all Reynolds,
complementarity, sparse preconditioning and Krylov operations on CPU. At
9x13, Q=0.002 N, the 10 MPa and 1 MPa cases retain 11 and 20 Newton
corrections. Pressure differences from NumPy are 1.67e-16 and 2.44e-16
relative L2; independent direct elasticity, mass and force checks pass.
The native path never imports PyTorch.

Those small solver runs are **slower**: CPU walls are 0.346/0.793 s, versus
13.586/27.746 s using 467/1065 transferred GPU convolution calls. Summed GPU
event compute is only 87.8/231.8 ms, while transferred/planned convolution
walls sum to 13.263/27.073 s. Per-call setup and transfers dominate. This is
component parity, not a device-resident iterative GPU EHL solver or a reason
to enable GPU automatically for small Newton problems.

The project CPU venv has no PyTorch. In the separate installed GPU-Testing
venv, PyTorch 2.15.0+rocm10.3.0a20261023 terminates an isolated import/probe
with SIGILL (return code -4). Core dumps were disabled. Its cause was not
inferred from that signal alone; no package/driver replacement was attempted.
The native HIP result remains independently operational.

To repeat the extended checks using an explicitly built extension:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  timeout 180 python -m examples.validate_native_hip \
  --extension-path /tmp/ehl-gpu-validation/src/ehl_solver/_hip_native.cpython-312-x86_64-linux-gnu.so \
  --output-directory results/desktop_hip_validation_new
```

Run with host GPU access. The new directory records device/binary hash,
checks, separated timings, memory observations and SI solver fields.
The original staged sources and develop's CPU reference remain preserved.

### Analytical rigid parabolic inlet/rupture reference

`parabolic_reference.parabolic_jfo_reference` supplies an independent
infinite-width, rigid, steady incompressible/isoviscous reference, with a
finite fully flooded zero-pressure inlet and one smooth rupture. It is not
an elastic or two-dimensional point-contact solution. Public inputs are
SI x, minimum gap, radius, viscosity and **mean** entrainment speed Ue;
outputs are p [Pa], h [m], content, dp/dx [Pa/m], volume flux per transverse
width [m²/s] and force per width [N/m]. The latter is not a point-contact
force in N. No negative-pressure clipping is used.

For `h=h_min+x²/(2R)`, choose `L=sqrt(2R*h_min)`, `X=x/L`, `H=1+X²`,
and `p_ref=12*eta*Ue*L/h_min²`; the dimensionless speed parameter is one.
With primitives
`I2(X)=(atan(X)+X/(1+X²))/2` and
`I3(X)=X/(4*(1+X²)²)+3*I2(X)/4`, determine the positive rupture coordinate
Xc from
`I2(Xc)-I2(Xin)=(1+Xc²)*(I3(Xc)-I3(Xin))`.
This is the zero-inlet/zero-rupture pressure condition. Full-film pressure is
`P=I2(X)-I2(Xin)-(1+Xc²)*(I3(X)-I3(Xin))`; downstream p=0 and
`theta=(1+Xc²)/H`. Both sides carry `q=Ue*h_min*(1+Xc²)` and the smooth
rupture has p=dp/dx=0. Integration by parts gives the analytical force per
width `p_ref*L*(Hin-Hc)²/(4*Hc*Hin²)` where Hin=1+Xin² and Hc=1+Xc².
The formulas are derived from the stated flux; pressure, rupture and force
are independently checked against adaptive quadrature in the regressions.

Boundary conditions and inlet flow are coupled in analytical lubrication
limits; see [On boundary conditions in lubrication with one dimensional
analytical solutions](https://doi.org/10.1016/j.triboint.2011.11.021) for
related boundary-condition analysis. The parabolic formulas and numerical
values here are independently derived/verified for this repository's stated
smooth rigid geometry, not attributed to that paper or transferred directly
to the elastic point-contact model.

The wide 2-D JFO comparison sets h independent of y, uses x=[-3L,3L] and
y=[-8L,8L], and compares centerline fields to the infinite-width reference.
Here h_min=20 um, R=10 mm, eta=0.04 Pa s, Ue=0.425 m/s,
L=0.632455532 mm and p_ref=322552.321 Pa. The transverse grid has 17 nodes;
it is a wide-domain approximation with zero side pressure, not an exact
one-dimensional discrete boundary condition.

| Grid `(ny,nx)` | Relative pressure L2 error | Relative force-per-width error | Max content error | Maximum local mass residual |
| --- | ---: | ---: | ---: | ---: |
| (17,33) | 0.0317444 | 0.0334057 | 0.0721956 | 5.20e-14 |
| (17,65) | 0.00641594 | 0.00645935 | 0.0370292 | 2.19e-13 |
| (17,129) | 0.00163204 | 0.00165019 | 0.0185179 | 8.59e-13 |

The analytical rupture is x=0.293551858 mm and force per width is
32.380268794 N/m. The last grid locates its first reduced-content node at
0.296463531 mm and integrates centerline pressure to 32.326835053 N/m.
Complementarity is zero at printed precision and absolute net integrated
flux is below 1.09e-12. Tests require decreasing pressure/force errors,
last-grid errors below 2%, bounded rupture-location error and conservation.
The content error decreases more slowly, consistent with its upwind flux.
These measurements do not prove an asymptotic order or full 2-D independence.

An analytical inlet-location study at the same physical parameters shows
that load can depend strongly on a nearby inlet even with exact mass balance:

| Xin | Xc | Force per width [N/m] |
| ---: | ---: | ---: |
| -0.5 | 0.225494 | 1.231921 |
| -1 | 0.357869 | 8.592849 |
| -2 | 0.444700 | 24.622984 |
| -4 | 0.470121 | 35.984054 |
| -8 | 0.474451 | 40.074694 |
| -16 | 0.475043 | 41.214090 |

This verifies a rigid finite-inlet/rupture limit and demonstrates a plausible
boundary-sensitivity mechanism. It does not identify the entire cause of the
elastic point-contact study's sensitivity or make its film heights validated.
The next study should vary inlet, outlet and transverse extents separately,
keeping material/load and inner resolution matched, and monitor flow budgets.
No physical boundary condition is changed solely to force domain convergence.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_parabolic_cavitation_reference --plots \
  --output-directory results/parabolic_reference_new
```

Fresh output directories preserve SI grid/fields, analytical reference,
centerline comparisons, rupture/conservation metrics and inlet-study data.

### Separating point-contact inlet, outlet and transverse extents

`examples.run_cavitating_boundaries` extends each boundary family separately
by four/eight original cells. The baseline is `(ny,nx)=(13,17)`,
x=[-0.5,0.75] mm, y=[-0.5,0.5] mm; dx=78.125 um, dy=83.333333 um.
Inlet extensions put xmin at -0.8125/-1.125 mm; outlet extensions put xmax
at 1.0625/1.375 mm; transverse extensions put both y edges at
+/-0.833333/1.166667 mm. All original nodes are retained exactly. Every case
uses E*=10 MPa, Rx=10 mm, Ry=20 mm, h_ref=20 um, L=0.5 mm,
p_ref=21250 Pa, Lambda=12 and flooded west content. This matches eta=0.04
Pa s and mean entrainment Ue=0.425 m/s. Pressure is zero on all four edges.

Each independent prescribed-h0=20 um compliance continuation supplies both
a rigid (gamma=0) force and an elastic (gamma=1) initial root; adaptive
positive-force continuation then solves h0 at Q=0.002 N. No field is
interpolated from another domain, no boundary condition is changed, and
iteration/tolerance limits are unchanged.

| Extended boundary | Grid `(ny,nx)` | Rigid force at h0=20 um [N] | Elastic force at h0=20 um [N] | h0 at Q=0.002 N [um] | Relative pressure change on baseline nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | (13,17) | 0.00231802 | 0.00229602 | 21.3005243 | 0 |
| Inlet +4 | (13,21) | 0.00545200 | 0.00530370 | 32.9530689 | 0.475451 |
| Inlet +8 | (13,25) | 0.00742901 | 0.00718947 | 40.7639950 | 0.599390 |
| Outlet +4 | (13,21) | 0.00231802 | 0.00229602 | 21.3005243 | 4.44e-16 |
| Outlet +8 | (13,25) | 0.00231802 | 0.00229602 | 21.3005243 | 4.90e-16 |
| Transverse +4 each | (21,17) | 0.00380367 | 0.00375144 | 27.0190576 | 0.377863 |
| Transverse +8 each | (29,17) | 0.00450645 | 0.00443703 | 29.7131725 | 0.473898 |

All seven targets complete in six accepted load stages each, without
rejections. Newton corrections total 17/22/29/17/17/21/21 in table order.
The 42 accepted states pass separate direct-spatial elasticity and
opposing-face flux audits: maximum dimensionless local mass residual
6.68e-9, absolute net flux 2.20e-9, complementarity 3.29e-22,
film-coupling error 2.04e-20 m. Final force errors are below 4.34e-19 N.
The largest grid here has 493 nodes, below the existing audit cap.

At the target force, outward west/east/south/north dimensionless fluxes are:

| Case | West | East | South | North |
| --- | ---: | ---: | ---: | ---: |
| Baseline | -30.652797 | 27.095336 | 1.778730 | 1.778730 |
| Inlet +4 | -54.673314 | 40.630072 | 7.021621 | 7.021621 |
| Inlet +8 | -83.179240 | 49.602691 | 16.788274 | 16.788274 |
| Outlet +4/+8 | -30.652797 | 27.095336 | 1.778730 | 1.778730 |
| Transverse +4 | -69.047158 | 64.903837 | 2.071661 | 2.071661 |
| Transverse +8 | -118.907177 | 114.396258 | 2.255459 | 2.255459 |

For these cases, the outlet is already beyond rupture: added downstream
nodes carry zero pressure and transport reduced content through the widening
gap without altering the pressure upstream. A regression independently solves
the baseline and extended load roots, checks direct audits, unchanged film
and pressure on shared nodes, and conserved downstream row flux. The old
east boundary content was copied from the interior; it becomes a genuine
flow unknown when the domain extends and is excluded from content overlap
comparisons. This observation is not a universal outlet-independence claim.

The inlet and transverse boundaries both materially affect these small-domain
roots. Their influence also exists in the rigid reference, so it cannot be
attributed solely to elastic coupling. Finite-inlet and lateral outflow
effects remain coupled in the point problem; the one-dimensional analytical
limit does not supply a complete point-contact far-field condition. The
offsets above remain domain-dependent numerical results, not independently
validated physical film heights. Next verification should establish a
controlled rigid point-contact far-field limit with explicit extent versus
resolution checks before extending material or constitutive comparisons.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_cavitating_boundaries --plots \
  --output-directory results/cavitating_boundaries_new
```

New directories contain unique names even for equal-shaped inlet/outlet
grids, all reference/Newton histories, SI states, trial audits, flux budgets,
pressure-overlap changes and a comparison plot. Prior results are preserved.

### Active-set linear equilibration on rigid far-field grids

Expanding a rigid parabolic point domain exposes a separate numerical issue:
H cubed gives very large pressure-mass matrix entries far from the minimum
gap, while pressure/content constraints retain unit coefficients. The
original sparse direct solve can have an unacceptable **unscaled** local
mass residual even though integrated pressure has stabilized numerically.

The fixed-film JFO solver now defaults to positive diagonal row/column
equilibration. For the original active-set system A z=b, define
Rii=1/max_j(abs(Aij)), then Cjj=1/max_i(abs((R A)ij)). Solve
`(R A C) u=R b` and recover `z=C u`. Zero/nonfinite rows/columns or
nonfinite scales raise. This changes linear coordinates only: the stencil,
active-set decisions, tolerances, pressure/content variables and boundary
conditions are unchanged. Nothing is clipped or regularized. Public option
`equilibrate_linear_system=False` retains the original direct reference.
All returned mass diagnostics use original face fluxes, not the scaled
linear-system residual. No device backend or monolithic Jacobian changes.

A normalized rigid probe uses H=1+X²+Y²/2, Lambda=1,
X=[-A,3], Y=[-sqrt(2) A,sqrt(2) A], with dY=sqrt(2) dX.
Dimensionless integrated pressure is W=integral(P dX dY).

| A | dX | Grid `(ny,nx)` | W, equilibrated | Original max mass residual | Equilibrated max mass residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.25 | (65,45) | 0.498109443073311 | 3.39e-9 | 9.24e-13 |
| 16 | 0.25 | (129,77) | 0.562361207496827 | 4.93e-8 | 9.56e-12 |
| 8 | 0.125 | (129,89) | 0.507565861780342 | 4.03e-8 | 4.95e-12 |
| 16 | 0.125 | (257,153) | 0.571843278633597 | 1.41e-6 | 4.70e-11 |

The final W changes by only 3.4e-14 relative between linear paths. These
are observed float64 results, not universal error bounds across sparse
libraries/platforms. An independent whole-face divergence regression checks
the original equation on the 129x89 case; a separate dense solve verifies
the complete recovered linear vector. Existing full-film, front, starvation,
elasticity and positive-load tests remain in force. Domain dependence is a
physical/boundary question distinct from this now-improved algebraic accuracy.

## Optional CUDA operator verification (2026-10-26)

`ElasticityOperator` retains the existing analytically cell-integrated kernel:
the rectangular source integral of 1/r divided by pi E*, including its
singular self-cell. NumPy constructs this kernel once; torch uploads it once
and caches its complex128 spectrum in the operator object. Float64 pressure
in Pa gives positive gap-increasing displacement in m; there is no extra
cell-area multiplier. Axis order is (ny,nx), x along entrainment.
For the (2ny-1,2nx-1) lag kernel, linear convolution needs padding at least
(3ny-2,3nx-2). The cached path rounds each length up with SciPy next_fast_len,
then extracts indices [ny-1:2ny-1,nx-1:2nx-1]. Default backward-normalized
rfft2/irfft2 retain the reference normalization. Larger padding changes no
discrete coefficient or wraparound convention. The original uncached NumPy
and torch functions remain available.

The default memory guard estimates 128 bytes per padded grid point for
real/complex arrays and workspace. It requires this estimate to fit both
1 GiB and half of current free CUDA memory, before kernel allocation.
At 513x513 the padded shape is 1540x1540 and estimate is 303,564,800 bytes
(289.50 MiB). It is a conservative admission guard, not a guarantee about
cuFFT plans, driver allocations, concurrent processes or multiple operators.
Spectrum lifetime is object-owned; release unused operators. PyTorch's FFT
plan and allocator caches are separate. No device error falls back to CPU.

The optional backend of reynolds_residual preserves exactly
div(H**3 grad(P))-Lambda*dH/dX-S on interior nodes: arithmetic H cubed face
conductance, adjacent pressure differences and centered entrainment gradient.
The scalings remain X=x/L, Y=y/L, H=h/h_ref, P=p/p_ref and
Lambda=12 eta Ue L/(h_ref**2 p_ref). All boundary values participate without
being overwritten. This is the existing full-film residual, not a port of
JFO's upwind content equation. Rigid parabolic film construction also supports
resident tensors, with SI coordinates, radii and gap. Validation checks
finite pressure and strictly positive film; scalar predicates synchronize
CUDA but do not export full fields.

DeviceFullFilmEvaluator snapshots a signed full-film bordered problem's
parameters, geometry and load gradient, uses zero boundary pressure, and
evaluates the complete scaled nonlinear residual plus analytic JVP. The JVP
includes elastic pressure-to-film changes, h0, optional compliance direction,
mobility derivative 3H**2 dH, entrainment and the force row. Film feasibility
is strictly h>film_floor_m. Snapshot parameters must be refreshed explicitly
after changing the CPU problem. This audit path does not change the CPU
solver. SciPy sparse assembly/factorization, ILU, GMRES, Schur recovery,
line searches and continuation remain CPU operations. There is no transferred
inner-Krylov bridge, GPU JFO implementation or complete GPU nonlinear solver.

Independent tests cover odd/even and anisotropic grids, signed/zero/edge
pressure, direct scipy.signal.convolve2d, repeated spectrum reuse, and dry
Hertz center-displacement refinement at 33/65/129. Complete face-flux arrays
independently reproduce Reynolds divergence with nonzero boundary pressure
and a manufactured source. All small-grid Jacobian columns (including h0 and
force), joint compliance directions and residual finite differences are
checked against the CPU equations. Invalid/nonfinite/infeasible states raise.
Resident-call tests prohibit Tensor.cpu/numpy exports; injected FFT errors
propagate. CPU tests also run with torch imports deliberately unavailable.

### Measured component performance

Windows 11, Python 3.12.9, NumPy 2.5.4, torch 2.11.0+cu128, CUDA runtime
12.8, driver 596.71, RTX A2000 Laptop GPU, 4096 MiB. OPENBLAS_NUM_THREADS
and OMP_NUM_THREADS are both one. Five warmups precede 15 repetitions;
reported times are synchronized wall medians in milliseconds. CPU baselines
below use the same cached, padded operator as CUDA. Standalone Reynolds
uses identical precomputed H on each device. Allocation, scalar validity
checks and Python launches are included; initial setup and field transfers
are separate. These are component evaluations, not solved EHL cases.

| Grid | DC-FFT CPU / CUDA [ms] | CPU/CUDA | Reynolds CPU / CUDA [ms] | CPU/CUDA |
| --- | ---: | ---: | ---: | ---: |
| 33x33 | 0.173 / 0.215 | 0.80 | 0.076 / 0.890 | 0.09 |
| 65x65 | 0.606 / 0.282 | 2.15 | 0.157 / 0.972 | 0.16 |
| 129x129 | 4.069 / 0.534 | 7.62 | 0.443 / 0.864 | 0.51 |
| 257x257 | 18.595 / 2.208 | 8.42 | 1.856 / 0.883 | 2.10 |
| 513x513 | 84.802 / 8.990 | 9.43 | 20.910 / 1.256 | 16.65 |

| Grid | Complete residual CPU / CUDA [ms] | CPU/CUDA | Complete JVP CPU / CUDA [ms] | CPU/CUDA |
| --- | ---: | ---: | ---: | ---: |
| 33x33 | 0.296 / 1.788 | 0.17 | 0.497 / 2.437 | 0.20 |
| 65x65 | 0.819 / 1.816 | 0.45 | 1.636 / 5.678 | 0.29 |
| 129x129 | 4.574 / 2.042 | 2.24 | 8.941 / 2.592 | 3.45 |
| 257x257 | 21.422 / 3.762 | 5.69 | 40.244 / 5.851 | 6.88 |
| 513x513 | 111.921 / 11.540 | 9.70 | 213.863 / 21.440 | 9.97 |

Ratios below one mean slower on CUDA. All components are slower at 33x33;
Reynolds alone is slower through 129x129. FFT gains begin at 65x65, while
complete residual/JVP gains against cached CPU begin at 129x129 here.
The original uncached CPU residual/JVP at 513x513 measure 241.098/467.405 ms;
using those as baselines would mix caching/padding gains with device gains.
Desktop scheduling, laptop clocks and thermal state cause substantial timing
variation (an earlier 513 FFT median was 33.049 ms). These observations are
not hardware-independent crossover rules or guaranteed speedups.

| Grid | One field H2D / D2H [ms] | GPU elasticity setup [ms] | Peak allocated / reserved [MiB] |
| --- | ---: | ---: | ---: |
| 33x33 | 0.031 / 0.026 | 102.733 | 0.69 / 2 |
| 65x65 | 0.039 / 0.046 | 196.586 | 2.72 / 4 |
| 129x129 | 0.047 / 0.045 | 74.203 | 10.46 / 24 |
| 257x257 | 0.096 / 0.104 | 82.953 | 41.79 / 72 |
| 513x513 | 0.253 / 0.293 | 342.570 | 166.94 / 202 |

Transfers use ordinary pageable NumPy memory, include allocation and are not
overlapped. Setup includes CPU kernel generation/upload and the initial
spectrum FFT; it is measured once and not a median. Memory columns are
maximum PyTorch allocated/reserved peaks over component timing runs with
resident benchmark fields and two GPU spectra. They exclude allocations
outside PyTorch's allocator; reserved memory can persist between operations.
The runner clears allocator cache between grids, never inside timed calls.

Across the five grids, max(abs(GPU-CPU))/max(abs(CPU)) is at most 8.46e-16
for elasticity, 5.95e-16 for Reynolds, 3.51e-15 for the complete nonlinear
residual and 4.40e-16 for complete JVP. The independent 33x33 direct spatial
elasticity error is 1.36e-15 by the same peak normalization. These are observed
float64 errors on the benchmark fields, not universal bounds or pointwise
relative tolerances near zeros. The signed operator audit remains mathematical
and adds no physical film-height or positive-load/cavitation validation.

Reproduce on Windows using the README PowerShell thread settings, or on Linux:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.benchmark_cuda \
  --output-directory results/cuda_new --warmup 5 --repeats 15
```

The new directory contains checkpointed summary.json with metadata, per-grid
accuracy, CPU/CUDA timings, uncached CPU timings, transfers and memory peaks.
The measured local run is results/cuda_20261026_a2000_final (untracked).
Windows execution is verified here; Linux GPU execution and ROCm remain open.
No full coupled GPU execution or end-to-end solver speedup has been established.

## Resident signed Newton/Krylov verification (2026-10-26)

This increment supersedes the preceding operator-only execution limitation.
It demonstrates complete resident *signed full-film mathematical* load solves,
not GPU JFO or physically validated positive-load EHL. The NumPy/SciPy solver,
pressure sign, SI scales, (ny,nx) axis order, conservative centered Reynolds
stencil, elastic kernel and nonlinear equations are unchanged.

### Transfer audit and baseline

At 97a35f1 there was no GPU Newton driver: the reference used NumPy fields,
SciPy ILU/GMRES, CPU Schur corrections, CPU dual line searches and scalar
convergence checks. It performed no CPU/GPU transfers because it did no GPU
calculation. Before changes, solve-only CPU walls were 232.117/4565.424 ms on
33/65 square grids, each with three corrections, default restart 80 and
1e-7 Reynolds/load tolerances. Those are retained baseline observations,
not the baselines for the tighter, matched-tolerance speedups below.

| Stage | Resident path | Transfers/synchronizations |
| --- | --- | --- |
| Setup | CPU integrated kernel; device spectrum, geometry, load weights and initial state | One-time uploads |
| Residual and frozen-state analytic JVP | torch fields, padded FFTs, conservative face operations | Scalar finite/film predicates |
| GMRES | Device basis, Hessenberg, rotations and triangular solves | Scalar norms/stopping decisions |
| Preconditioner | Device Jacobi or DST-I scaled Laplacian inverse | No full-vector/matrix transfer |
| Schur/full-bordered correction | Device vectors, dot products and reconstructed correction | Scalar denominator/true-residual checks |
| Feasibility and Armijo | Device film increments, trial state and residual | Scalar feasible step, slope and merit |
| Newton/continuation | Device accepted state; Python scalar control | Scalar residual/load/film/history data |
| Independent final audit | Explicit state/field export after timing | Final CPU comparison only |

Tests prohibit Tensor.cpu and Tensor.numpy during resident GMRES, bordered
correction and complete Newton execution. There is no bridge into SciPy
inner iterations and no CPU fallback after a device failure. CUDA availability
is selected explicitly; invalid/runtime device operations propagate.

### GMRES and frozen-film preconditioning

`device_gmres` implements real float64 nonsymmetric restarted left-preconditioned
GMRES. It uses two classical Gram-Schmidt passes (CGS2), accumulated device
Givens rotations and a device triangular solve. Only scalar norms leave the
device. Its max_iterations counts total Arnoldi steps; SciPy maxiter instead
counts restart cycles. Explicit termination distinguishes convergence,
iteration limits, Arnoldi/preconditioner breakdown and nonfinite calculations.
The returned solution is accepted only if a newly evaluated original residual
satisfies ||b-Ax|| <= max(atol,rtol*||b||). Hessenberg/preconditioned estimates
only trigger true checks. Restart boundaries also recompute true residuals.
Absolute estimated preconditioned histories are retained; these differ in
normalization from SciPy's relative pr_norm callback histories.

Basis admission estimates 16*N*(m+5)+32*(m+1)^2 bytes, where m is the effective
restart. This includes temporary headroom beyond the stored basis. It must
fit a default 1 GiB cap and half of free device memory. Elasticity retains
its separate workspace guard. Neither guard guarantees vendor workspace or
availability under concurrent device use; actual allocator peaks are below.

The Jacobi strategy uses the exact signed/scaled frozen-film Reynolds
diagonal, independently checked against SciPy assembly. The laplacian strategy
defines k=H^3, S=diag(sqrt(k_interior)) and the positive zero-Dirichlet
five-point negative Laplacian L with the original dimensionless dx,dy.
It applies the exact inverse of B=-S L S / reynolds_residual_scale:
`B^-1 b = -reynolds_residual_scale * S^-1 L^-1 S^-1 b`.
This is an explicit surrogate for the actual arithmetic-face-conductance
variable-coefficient Reynolds block. It omits elastic/entrainment derivatives
and is neither ILU nor a claimed exact inverse of the Reynolds block.

L is inverted with two-dimensional DST-I. The unnormalized transform is
implemented using odd extensions and rFFT imaginary parts; its square gives
4*(ny_interior+1)*(nx_interior+1) times the input in two dimensions.
Eigenvalues are
`4*sin(pi*i/(2*(nx_interior+1)))^2/dx^2`
plus the analogous y term. Independent SciPy DST, dense surrogate inversion,
and original sparse face-operator tests verify transform/sign/scaling/indexing.
No sparse matrix, ILU factor or Krylov basis is exported from CUDA.

### Bordered and nonlinear safeguards

The pressure solves retain A*x=-R_p and A*y=b, where b is the h0 column.
With force row g, the Schur denominator is -g.y,
delta_h0=(-R_load-g.x)/(-g.y), and delta_p=x-y*delta_h0. Unsafe scaled Schur
denominators, failed pressure solves or inaccurate reconstructed corrections
trigger recorded full-bordered device GMRES with block preconditioning.
All attempted Krylov diagnostics are retained. Reconstructed corrections are
checked using the complete analytic JVP plus original residual. Effective
linear tolerance is capped at 1e-2, and the bordered accuracy ceiling follows
the reference's min(1e-2,max(1e-7,2*rtol)) default. Device GMRES enforces the
requested original-equation tolerance directly; it does not relax tolerances
for parity. Tiny denominators used in relative-residual normalization are
machine-underflow guards, not pressure/film regularization.

The separate resident Newton implementation preserves the CPU formulas:
merit is half the sum of squared scaled Reynolds RMS and load residual;
forcing starts at 1e-2 then uses min(1e-2,max(1e-4,0.9*ratio^1.5)). The exact
film direction first limits the common pressure/h0 step using fraction-to-
boundary 0.99; directional-derivative Armijo (default 1e-4) then backtracks.
Non-descent recovery and inaccurate-linear-solve Armijo recovery each have
the reference's bounded tighter-Krylov retries. Film must remain strictly
above the original floor. There is no clipping or implicit h0 raise.
Final force is recomputed with nested device trapezoids, independently of
the load-gradient residual evaluation.

Adaptive device load continuation retains the reference growth/halving and
load-resolution rules. Only accepted roots warm-start later attempts;
rejected states, nonlinear/Krylov diagnostics and last accepted roots remain
available. Candidate residual/film/target metadata are checked before
acceptance. Unexpected device exceptions propagate and restore the last
accepted evaluator load. GPU compliance/pseudo-arclength continuation is
not implemented by this load driver.

Independent tests cover nonsymmetric dense/SciPy linear systems, restart,
zero RHS/exact starts, happy and failed breakdown, scaled preconditioners,
memory/iteration limits and propagated runtime errors. Complete corrections
are compared with dense Jacobians, residual differences and CPU Schur solves,
including forced full-bordered recovery. Newton tests cover zero compliance,
100 MPa weak and 10 MPa moderate cases, and gamma=0.15 at 1 MPa as a stronger
exploratory case. Direct spatial elasticity, whole-face flux divergence and
SI trapezoidal force independently check final small-grid fields. Tests
also cover both line-search stages, rejected/exhausted Armijo trials, tighter
linear recovery, continuation in both force directions, injected retry,
exhaustion and false-success rejection. These limited cases do not establish
robust full-compliance convergence on the known 1 MPa branch.

### Benchmark inputs and timing scope

Windows 11, Python 3.12.9, NumPy 2.5.4, torch 2.11.0+cu128, CUDA 12.8,
driver 596.71, RTX A2000 Laptop GPU (4096 MiB); BLAS/OpenMP threads are one.
The benchmark uses x=[-0.5,0.75] mm, y=[-0.5,0.5] mm; Rx=10 mm, Ry=20 mm,
E*=100 MPa, gamma=1, L=0.5 mm, h_ref=20 um, p_ref=21250 Pa and Lambda=12.
Each grid's signed target is the independently solved rigid-film force at
h0=20 um; initial pressure is the rigid solution at h0=18 um. The 513 target
is -0.002724790847933 N. This negative target and signed pressure make the
case mathematical. Initial pressure/target preparation is common and excluded.
The Reynolds residual scale is 100, load residual scale is abs(target_load_n),
and strict film floor is 1e-12 m. Reported scaled RMS values use these scales.

Standalone linear solves use rtol=1e-8, atol=0 and restart 40. CPU maxiter
cycles are mapped to the same maximum Arnoldi-step budget as the GPU.
All nonlinear/load tolerances in comparison runs are 1e-10. Linear timings
are synchronized wall medians of three calls after one warmup, except the
explicitly bounded large CPU attempts described below. Full Newton walls are
one measured solve in a warmed process, including its repeated preconditioner
builds, all Krylov work, scalar checks and line searches. Per-iteration time
is the full wall divided by four accepted Newton corrections. Setup-inclusive
walls add evaluator/kernel construction, initial upload and final state
download. CUDA context/library initialization and independent CPU audits
remain outside those walls; these are not cold process-launch measurements.

### Preserved NumPy/SciPy reference comparison

| Grid | CPU / CUDA pressure GMRES [ms] | CPU / GPU Arnoldi steps | CPU / CUDA Newton [ms] | Setup-inclusive GPU [ms] | Setup-inclusive speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 33x33 | 23.413 / 54.949 | 5 / 13 | 282.955 / 314.470 | 424.188 | 0.67x (slower) |
| 65x65 | 569.624 / 54.254 | 30 / 13 | 3440.043 / 309.947 | 469.907 | 7.32x |
| 129x129 | 2977.015 / 57.718 | 195 / 13 | 11939.597 / 498.240 | 572.027 | 20.87x |

All three CPU and GPU nonlinear runs take four corrections. Total CPU/GPU
Newton Krylov counts are 20/46, 140/46 and 684/50 respectively. GPU mean
Newton walls are 78.618, 77.487 and 124.560 ms. GPU solve-only speedups over
the preserved reference are 0.90x, 11.10x and 23.96x; standalone GMRES ratios
are 0.43x, 10.50x and 51.58x. These combine hardware, cached FFT/state fields,
and a different verified preconditioner. They are not hardware-only gains.
Nonlinear residual histories consequently differ within the same forcing
and convergence criteria; matching every inexact trajectory is not claimed.

At 129, an initial 160-step CPU budget stopped at true relative residual
2.69e-7; increasing the budget to 640 allowed convergence in 195 steps.
The final comparison uses 640 for both backends. At 257/513 the retained
160-step CPU attempts stop at residuals 2.61e-3/2.74e-1, with measured walls
36.758/91.019 s. Those unsuccessful attempts are not speedup baselines.
Full NumPy/SciPy Newton is bounded to grids <=129 in this benchmark; no
complete reference-runtime comparison exists here at 257/513.

At 33/65/129 the final CUDA pressure differs from the converged original
CPU reference by 7.99e-13/2.49e-12/1.90e-11 relative L2; h0 differences are
1.01e-18/1.70e-17/2.29e-16 m. The standalone pressure-solve differences are
at most 2.28e-9 relative L2 and the reconstructed bordered-correction
differences at most 1.48e-8, with the same 1e-8 linear stopping tolerance.

### Identical cached algorithm on torch CPU and CUDA

To isolate device effects, a separate sequential run uses the same cached
JVP, DST preconditioner, CGS2 GMRES and Newton policies on both devices.
Both pressure solves take 13 steps, and full solves take four corrections.

| Grid | CPU / CUDA GMRES [ms] | CPU / CUDA Newton [ms] | Warm solve speedup | Setup-inclusive CUDA [ms] |
| --- | ---: | ---: | ---: | ---: |
| 65x65 | 26.385 / 57.957 | 155.556 / 342.536 | 0.45x (slower) | 523.383 |
| 129x129 | 65.656 / 57.112 | 436.722 / 396.362 | 1.10x | 471.981 |
| 257x257 | 220.194 / 88.802 | 1448.538 / 755.151 | 1.92x | 845.364 |
| 513x513 | 1103.041 / 312.562 | 6947.224 / 2103.297 | 3.30x | 2496.381 |

Matched pressure agreement is within 3.11e-16 relative L2. The first tested
warm-solve win is 129, but its 10% margin is small and vanishes when device
setup/transfers are included. First tested setup-inclusive win is 257
(1.71x); 513 gives 2.78x including setup. Reference-solver gains start at 65.
These are workload/algorithm/session observations, not universal crossover
sizes. Laptop scheduling/clocks and allocator/FFT-plan state cause timing
variation; the benchmark makes no automatic backend selection.

### Preconditioner, transfer and memory observations

| Grid | CPU ILU build/apply [ms] | CUDA surrogate build/apply [ms] | H2D / D2H state [ms] | Peak allocated / reserved [MiB] |
| --- | ---: | ---: | ---: | ---: |
| 33x33 | 8.294 / 0.069 | 211.142 / 0.692 | 0.075 / 0.077 | 9.46 / 22 |
| 65x65 | 35.165 / 0.265 | 161.042 / 0.686 | 0.055 / 0.094 | 13.34 / 26 |
| 129x129 | 125.076 / 0.788 | 86.854 / 1.000 | 0.062 / 0.110 | 28.55 / 48 |
| 257x257 | not measured in final matched run | 190.562 / 0.664 | 0.108 / 0.161 | 88.24 / 108 |
| 513x513 | not measured in final matched run | 220.610 / 23.799 | 1.423 / 0.330 | 329.85 / 390 |

Build is a single first-use observation including field reconstruction;
apply is a warmed median. The first three rows are the preserved-reference
run; large rows are the final matched run. Large retained CPU attempts
separately measure ILU build/apply at 635.233/4.569 ms (257) and
2744.446/20.977 ms (513). Allocated/reserved peaks cover live benchmark
fields, preconditioners and Krylov/Newton work, not just the basis. Reserved
pool memory and vendor plans can persist; driver/non-PyTorch allocations
are excluded. The observed 4 GiB machine completes all cases without OOM.

At 513 the GPU h0 is 19.9013602695 um and minimum film 19.8745823867 um.
Independent original CPU equations give scaled Reynolds RMS 9.82e-14,
scaled load error 2.04e-14 and nested-trapezoid device force error 4.34e-19 N.
Final load errors, film minima and residual histories are recorded per grid.
These mathematical closure checks add no experimental film-height evidence.

Reproduce with the same thread settings (PowerShell equivalents in README):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m examples.benchmark_device_krylov --compare --sizes 33 65 129 \
  --repeats 3 --max-krylov-steps 640 --output-directory results/newton_small_new
python -m examples.benchmark_device_krylov --compare --sizes 257 513 \
  --repeats 1 --max-krylov-steps 160 --output-directory results/newton_budget_new
python -m examples.benchmark_device_krylov --compare --skip-cpu-linear \
  --matched-torch-cpu --sizes 65 129 257 513 --newton-max-size 0 \
  --repeats 3 --max-krylov-steps 160 --output-directory results/newton_matched_new
```

Output directories must be new. The runner checkpoints linear diagnostics,
records failed CPU attempts explicitly, and saves summary.json plus per-grid
JSON and final NPZ fields after complete solves. Current retained runs are
results/device_krylov_baseline, device_krylov_final_small,
device_krylov_compare_large and device_krylov_final_matched_cpu. Generated
files remain untracked. Extra exploratory runs are not performance baselines.

CPU without torch remains verified. GPU tests skip gracefully without torch
or CUDA; runtime calculation errors on available devices fail. Windows CUDA
is exercised; Linux CUDA/ROCm execution, other devices, arbitrary strong
coupling, broad continuation robustness and GPU JFO remain unverified.
The unresolved physical far-field/domain and film-height questions are
unchanged. Full physical GPU EHL support is not marked complete.

### Scientific audit and reproducible CPU validation hierarchy

`docs/VALIDATION_REPORT.md` consolidates the governing equations, units,
deformation/Hertz conventions, boundary/coordinate audit, diagnostic plot
interpretation, exact benchmark settings and numerical-versus-physical
validation limits. It identifies accessible NASA classical-EHL target
tables and their missing constitutive/boundary requirements without claiming
an unmatched isoviscous comparison. It also records the controlled rigid
point-contact extent/resolution probe through 257x257, with separate SI
flux audits and dimensional-similarity regressions. Domain independence
remains open despite small conservative residuals.

`python -m examples.run_scientific_validation --output-directory results/new_validation`
regenerates Hertz, finite-inlet analytical JFO, rigid point far-field,
loaded-grid and separately extended-boundary calculations. Figures include
physical units, orientation, operating conditions, sampled extrema and
status/residuals; SI NPZ states, matched nested reference fields, JSON
histories and source/runtime provenance are saved. Existing output
directories are refused. This benchmark runner needs no PyTorch and does
not modify the signed full-film CUDA implementation.

The generic pressure contour renderer now explicitly covers the entire
supplied pressure range, including tiny negative roundoff values. Automatic
levels previously excluded those values in near-zero cavitated regions,
leaving white polygons; the generic full-film title was also misleading
when reused for cavitation. Data are neither clipped nor smoothed by this
correction. Zero load errors remain exact in the saved data and are shown
using a symmetric-log axis rather than replaced by an arbitrary floor.

### Far-field error budget: partial reduction, no domain-independence claim

The follow-up in section 5 of `docs/VALIDATION_REPORT.md` separates fixed-
domain mesh refinement, nested independent boundary/domain extension,
free-space elasticity support and actual SI boundary-flow budgets.
`rectilinear_cavitation.py` adds a prescribed-film CPU diagnostic for graded
far fields, independently checked against the unchanged uniform solver and
the analytical parabolic limit. It preserves arithmetic face coefficients,
upwind content and pressure/content constraints on midpoint control volumes.
The coupled solver and uniform DC-FFT operator are unchanged; an FFT kernel
is never applied to unequal cells. `elasticity_audit.py` evaluates bounded
target-point displacements by direct physical source-cell integration.

The accepted refined domain sequence reduces the previous 5.51% load change
to roughly 1.17% (1.16% relative to the later state), still above the target.
Two larger fine-core candidates fail the declared original max-continuity
criterion and remain explicitly rejected. A coarser sequence has two
successive changes below 1% but fails the independent mesh error budget.
Source pressure beyond 16 natural lengths contributes appreciably more to
load than to center displacement; this is distinct from FFT wraparound and
from re-solving a smaller fluid domain. The tested single-boundary inlet
effect dominates the transverse effect; the already-cavitated outlet remains
insensitive here. Fixed flooded content must not be confused with a fixed
integrated inlet feed when the parabolic inlet gap/domain expands.

The 27-case runner produces SI fields, source/runtime provenance, explicit
accepted/rejected audits, front brackets, complete cold-start failure and
Picard recovery records, and 11 comparative figures with common-coordinate
overlays. The existing uniform coupled cases still have about 13.3% load
change from extent 8 to 16; their small film/pressure changes alone cannot
establish domain independence. Growing far-field slopes also limit physical
interpretation of the very large parabolic-domain calculation.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_farfield_budget --output-directory results/farfield_budget_new
```

The report records exact grids/spacings, units, acceptance thresholds,
quantitative errors and next actions. No constitutive physics or GPU port
is introduced, and the milestone remains open.

### Reservoir matching and physical-flux conditioning

Section 6 of `docs/VALIDATION_REPORT.md` accounts for the remaining 1.17%
rigid and 13.3% coupled force changes on unchanged nested cells. Added-area
load contributes about 38%/39%, and pressure readjustment over the shared
rectangle about 62%/61%. Most readjustment is outside the inner +/-2L,+/-2Ly
core. On identical uniform grids the rigid control already changes 12.9%,
so elasticity is not the main origin of the coupled-domain sensitivity.
This spatial identity does not make nonlinear boundary effects additive.

The inlet component condition `(|q_C|+|q_P|)/|q_C+q_P|` is about 3.46 in
the large 2-D point cases, with maximum face condition below 10. Catastrophic
component cancellation is not observed there. The h^3 range nevertheless
reaches 2.51e15. Wider reevaluation of a stored float64 state barely changes
the failed residual, while one pressure ULP can change that defect by about
1e-8. These are diagnostics of mobility/state sensitivity, not matrix
condition numbers or proofs of an unavoidable accuracy floor.

The rectilinear **prescribed-film** reference offers at most three opt-in
physical-flux defect corrections, accepted only when the original independent
float64 SI flux defect decreases and the same constraints hold. States and
linear corrections remain float64; only residual evaluation uses explicitly
wider arithmetic. Requests raise when wider precision is unavailable. The
label-192 original defect falls from 1.53e-8 to 4.76e-9; label 256 remains
rejected at 1.71e-8. Loads change only at roundoff. The default is zero
corrections, and the coupled solver and its tolerances remain unchanged.

Optional nonnegative SI pressure-edge data in that reference distinguish a
computational cut from a physical ambient reservoir. Known positive-pressure
edges are wet; west content must be one if its pressure is positive. Exact
nonuniform linear-pressure tests, an independent analytical infinite-width
matched-window limit and a finite point-reference restriction verify the
boundary assembly. The 1-D comparison restores omitted force per width
[N/m] by independent quadrature; the point comparison restores known outside
force [N] from its larger finite reference. Neither invents a point outer law
or omits exterior deformation from a coupled solution. Default pressure
edges stay ambient. Moving an ambient inlet inward, prescribing supply, or
changing content represents a different physical reservoir/feed problem.

In the 1-D full region, constant q=Ue hc gives q_C=Ue h and
q_P=Ue(hc-h); component cancellation grows like h/hc as the inlet retreats.
This independently derived limit has no transverse leakage, unlike the
point case, and cannot prescribe its outer pressure. A finite dimensioned
reservoir is a valid stated model; a physical infinite-domain point limit
remains unverified, especially outside small-slope lubrication assumptions.
Grading limits resource cost but does not supply missing outer pressure,
load or elastic sources. Nonuniform FFT coupling remains prohibited.

### Discrete finite-reservoir exterior response

`exterior.py` condenses exterior pressure **and** transported content for each
JFO active set. It computes the Schur response of the full constrained
prescribed-film system, reconstructs exterior fields and updates their wet/dry
sets. Only the physical outer boundary is ambient/flooded; the numerical cut
has neither a new pressure condition nor a prescribed feed. Its response is
affine only while the active set and film stay fixed. No reference pressure
trace is supplied to this solver. The independent original uniform solver
checks the fixed-reservoir benchmark in validation report section 7.

Exterior pressure contributes explicitly to SI load and to diagnostic
cell-integrated half-space displacement. Retained/exterior source masks
partition whole midpoint source cells, including interface cells; this differs
from a geometric integral up to the inner node rectangle. The total source
load equals full trapezoidal load when physical edge pressures are zero.
Target displacements use direct source-cell integration and agree with the
uniform zero-padded FFT in independent regressions. The rigid prescribed film
is not updated; no nonuniform coupled elasticity or infinite-domain boundary
law is claimed. Resource caps and original SI acceptance criteria remain
explicit. Partition independence on one mesh is distinct from mesh accuracy
or insensitivity to moving the physical reservoir.

### Graded exterior at fixed physical reservoir

The section-8 benchmark holds the same ambient reservoir at +/-8L,+/-8Ly,
retains an exactly uniform [-2,2] core and independently varies core spacing
and outer geometric growth. Final outer cells end at the prescribed physical
bounds; the reservoir is not snapped outward. A 139x139 graded prescribed-film
response agrees with an independent 257x257 uniform reference to 0.129% full
load, 0.056% core pressure and 0.065% total center displacement. Its exterior
source force and exterior center displacement differ by 0.177% and 0.139%
on matched core source-cell supports. Both retained and exterior pressure
are included; no nonuniform FFT or elastic feedback is introduced.

The native boundary flux audits surround midpoint cells whose outer faces
move with the edge cell width. They remain the conservative acceptance test.
A separate diagnostic at the same physical section x=-L,y=+/-Ly measures
supply changes consistently across meshes using nodal flux and centered
pressure differences. It never replaces the original face residual. Core
front comparisons match transition kinds/counts and brackets on shared y
rows without interpolation. Coarse candidates can conserve mass while failing
load/field accuracy, and remain explicitly rejected for approximation use.
Reference mesh changes are reported separately; finite-reservoir approximation
success is not continuum, infinite-domain or coupled physical validation.

### Conditional film derivative and all-inner source evaluation

The exterior result records its wet/dry set. A prescribed-film directional
response differentiates arithmetic H^3 mobility by 3 H^2 dH and face film
by dH, preserving upstream theta and fixed physical supply. The same
constrained exterior Schur solve reconstructs pressure/content directions
throughout the full finite reservoir. SI directions use h+epsilon v, v [m]
and dimensionless epsilon. Independent paired solves verify the derivative
on unchanged active sets on both uniform and graded grids. It is not a
generalized derivative at a moving front, nor the Jacobian of a new nonlinear
coupled solver. Mismatched/nonconverged root films are rejected explicitly.

Bounded direct cell integration now evaluates retained/exterior displacement
at every retained target, not just sampled center/peak nodes. Uniform
pressure-direction results match the full FFT; a frozen-film restriction of
an independently rerun uniform coupled root matches pressure/content/load
and all 289 inner displacements. Validation report section 9 distinguishes
this preparatory restriction from solving a new coupled problem. A graded
coupled solver still needs a consistent film/source map at exterior as well
as inner nodes, plus active-set/globalization verification. No pressure
trace, missing-source correction or nonlinear film update is inferred from
the frozen-root agreement.

### Separate uniform exterior/Picard coupling experiment

`exterior_coupled.py` independently cold-starts a prescribed-h0 iteration
with compliance factors 0,.25,.5,.75,1 and film relaxation .5. Each step
recomputes the exterior pressure/content response, then applies the unchanged
full-source uniform FFT to update film at both inner and exterior nodes.
Physical continuity is evaluated on the exact film from the reconstructed
pressure. Stage acceptance additionally requires original SI local/global
conservation, complementarity, supply and film feasibility. Failure retains
histories and the last candidate. Iteration/node/memory caps are unchanged.

This separate damped-Picard verification is neither globalized Newton nor
force control. It matches independent full monolithic roots while leaving
the established coupled solver unchanged. All exterior pressure sources
contribute to deformation; only physical outer pressure is zero, and elastic
displacement is not fixed at that boundary. The experiment requires uniform
physical cells. It cannot be ported to graded grids by applying this FFT to
unequal sources. Validation report section 10 separates reference parity,
cut independence and remaining mesh/physical-reservoir limitations.

### Pressure-dependent laws (2026-10-27)

Standalone float64 Roelands viscosity and Dowson-Higginson density are now
implemented in `constitutive.py`. SI coefficients, source-equation checks,
analytic derivatives, illustrative reference-temperature parameters and the
resolved 1975 viscosity-unit inconsistency are documented in
`docs/CONSTITUTIVE.md`. No existing isoviscous solve is changed by this module.
Pressure-dependent mass-flux integration and initial coupled runs are the next
two authorized milestones; the finite-reservoir/model limitations remain.

### Pressure-dependent conservative JFO mass flow

The separate prescribed-film CPU path now uses liquid rho(p),eta(p) in
both pressure mobility and entrainment capacity. It conserves mass in kg/s;
variable density prevents interpreting these as the earlier volume budgets.
`docs/CONSTITUTIVE.md` specifies the arithmetic face interpolation, original
SI audit, coefficient-only damping, bounds and benchmark references. An
independent Kirchhoff pressure-driven channel checks the continuum limit,
and complete smooth flux derivatives match finite differences. Existing
isoviscous and GPU paths remain independent; this milestone does not alter
coupled film or claim physical classical-EHL accuracy.

### Initial compressible/piezoviscous elastic coupling

`property_coupled.py` now couples the pressure-dependent conservative JFO
solve to full-source uniform FFT film, at prescribed h0 and fixed finite
reservoir dimensions. Only film iterates are relaxed; final acceptance uses
unrelaxed material laws and exact pressure-coupled film. Material/compliance
continuation, independent monolithic limit and small full nonlinear reference,
physical mass/load and direct elastic checks are in `docs/CONSTITUTIVE.md`.
No old solver or GPU path is replaced. This initial experiment has no new
load controller or graded elasticity and does not establish mesh-converged
classical EHL or experimental fluid accuracy.

### Pressure-dependent positive-load control

`property_load.py` now replaces prescribed offset by a bracketed scalar
force equation around independently audited coupled solves. SI target force
is the integral of gauge pressure [N]; density and saturation contribute
through mass transport, not a separate mechanical load. Every trial keeps
physical grid, reference scales and supply assumptions unchanged, updates
only h0, and must pass original true-property mass, complementarity, gap,
source-coupling and independent pressure-integration checks. Details and
failure semantics are in `docs/CONSTITUTIVE.md`.

Only the current positive-offset cold-start branch is supported. The gap
remains h0+geometry+w(p), so minimum film is independently measured. Exact
force closure by adjusting h0 can conceal mesh/domain sensitivity in h0,
film and pressure; those errors must be evaluated separately at fixed load.
No old signed/full-film controller or production bordered Newton method is
modified. This nested CPU reference does not add transient squeeze, signed-
offset continuation, GPU support or classical EHL physical validation.

## Smooth material-transport consistency probe (2026-10-27)

`examples.run_transport_consistency` compares the existing SI mass-flux
stencil with an independently differentiated continuum source on rectangular
(ny,nx) grids. It uses L=1 mm, h_ref=2 um, p_ref=10 MPa, Ue=.1 m/s,
and the existing IsothermalFluid(.04 Pa s, 850 kg/m^3) material laws.
Both density and viscosity vary with pressure. Analytic derivatives of smooth
pressure/film/content fields form div[-rho h^3 grad(p)/(12 eta)
+ Ue rho h theta e_x], in kg/(m^2 s); no manufactured source is inserted
into a solver and no production root is assessed.

On 9x17, 17x33, 33x65 and 65x129 grids, full-film relative L2 divergence
errors are .0317437, .00867311, .00222986 and .000563468 (last observed
order 1.98455). Smooth variable-content errors are .0366970, .0120777,
.00461058 and .00205672, approaching the expected first-order upwind
asymptote. These deliberately non-complementary manufactured fields isolate
smooth flux consistency; they do not verify cavitation-front accuracy.

A separate uniform frozen-film tangent applies alternating x pressure.
Centered density-advection divergence is zero, but diffusion remains nonzero
and agrees with the independent 4*mobility/dx^2 Fourier symbol to 4.08e-16
relative error. Its density-only cell Peclet number is .000270037.
This limited null-mode observation is not evidence of an unstable complete
elastic/material/JFO operator, nor an explanation of the archived pressure
pattern. The coupled nonlocal film tangent and active fronts remain untested
by this probe. No production stencil, tolerance, fluid law or floor changed.

Reproduce into a new directory (existing directories are refused):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m examples.run_transport_consistency \
  --output-directory results/transport_consistency_new
```

The JSON records errors, orders, units, limited diagnostic scope and source/
runtime provenance. Recover and independently audit the saved 200 N roots
before extending these observations to the coupled benchmark.

## Best-candidate active lookahead (2026-10-27)

Explicit finer-grid repair now compares the final accepted active-model
trial with all retained feasible Armijo backups. The lowest measured merit
wins; a worse final trial no longer overrides a better deferred candidate.
Histories mark a displaced acceptance as superseded and retain exactly one
accepted trial per correction. A best-mode trial must change the float64
state and strictly reduce computed original merit as well as satisfy the
original directional Armijo inequality. Equality caused by rounding is not
nonlinear progress. This adds no pressure/content/film clamp and relaxes no
physical, force, linear-residual or feasibility criterion.

`repair_property_load_guess(..., lookahead_selection="legacy")` reproduces
the preceding final-model preference and rounded Armijo behavior for
comparison. `--lookahead-selection legacy` exposes that reference in
`examples.run_load_refinement`; the policy is recorded in the output JSON.
`--no-active-lookahead` still disables lookahead. Existing warm continuation,
whose lookahead is disabled, retains its prior behavior. The default explicit
repair uses `best`; the eight-model and 30-correction caps are unchanged.
Regression tests isolate final-candidate selection, exact-state/merit
stagnation, legacy behavior and failure retention. Physical convergence
still requires the independently audited original field/force gates.

## Targeted near-empty-content pivots (2026-10-27)

In P,D=1-theta variables, the lower content bound is D<=1+1e-10.
Earlier recovery changed models only for negative proposed P or D; it did
not handle outward motion at theta~=0. Explicit repair can now identify the
earliest affine pressure/content blocker when D>=1-1e-10 and the full
proposal exceeds D<=1+1e-10. Only tied earliest upper-D constraints (within
one floating-point step in the ratio) are selected. The corresponding new
model imposes D_new=0 (full film), and is solved and checked with the same
complete discrete operator. It gets priority over unrelated full-step lower-
bound violations. No state, direction or bound is clipped or shifted.

This is a model-selection heuristic: the original nonsmooth merit slope,
strict progress, true linear error, exact positive total film and final
physical/force checks still govern acceptance. Film feasibility is checked
separately by the existing line search. Repeated active models stop recovery;
there are still at most eight models and 30 corrections. Interior flattened
(y,x) pivot indices are recorded in every enabled model's diagnostics.

`content_bound_pivots=False` or `--no-content-bound-pivots` retains the
preceding best-candidate reference. To reproduce the older legacy policy,
combine `--no-content-bound-pivots --lookahead-selection legacy`.
Warm-root continuation remains unchanged; the new option is enabled only
by default for explicit unaccepted-guess repair. Analytic bounded-complement-
arity tests cover root recovery, ties, competing bounds, roundoff and cycling;
they are algorithm verification, not a physical EHL benchmark.

## Loaded-root conservative transport decomposition (2026-10-28)

`transport_audit.mass_direction_components` is an independent diagnostic; no
production stencil is modified. For SI fields in (ny,nx), +x entrainment,
write M=rho*h^3/(12*eta), C=rho*h and
q=-M_face grad(p)+Ue*C_face*theta_up*e_x. Face coefficients use arithmetic
averages and content uses the existing west/upwind value. For prescribed
directions dp [Pa], dh [m], dtheta [dimensionless],

```
dM = rho_p*dp*h^3/(12*eta) - M*eta_p*dp/eta + 3*M*dh/h,
dC = h*rho_p*dp + rho*dh,
dq = -M_face grad(dp) - dM_face grad(p)
     + Ue*(dC_face*theta_up + C_face*dtheta_up)*e_x.
```

The seven returned opposing-face divergences separate direct pressure
diffusion, density diffusion, viscosity diffusion, film diffusion, density
advection, film advection and content advection. Units are kg/(m^2 s).
Summing them gives the complete directional mass equation. The uniform-state
density-advection tangent of an alternating pressure direction has the centered
null response, while direct diffusion reproduces its nonzero analytical
symbol. A separate finite-difference test changes pressure, film and content
together and checks the sum against independently evaluated physical fluxes.

The loaded-root runner requires an accepted 200 N archive with matching
benchmark metadata and freshly verifies original physical/film/boundary
criteria and independently integrated SI force. Its probes have 1 MPa peak
pressure direction, zero pressure edges and zero uniform-grid force direction.
Pressure support is p>0.1*pmax, with a fixed physical window
exp(-2*((x/a)^4+(y/a)^4)), where a is the registered dry Hertz radius.
Subtracting the window-weighted mean makes each probe force-free; this
changes only the artificial direction, never the production state. Modes
are alternating x, y, xy and windowed smooth x/y; dtheta=dh0=0. Nonlocal
elastic dh is computed on the whole physical domain, not just that support.

The complete analytic mass JVP is checked against the independently summed
SI components with the original mass nondimensionalization. Complementarity
and force rows are checked as well. Feasible two-sided residual differences
at factors .01/.001/.0001 retain all errors, rather than declaring the smallest
step best. These pressure-supported directions avoid changing active fronts;
front/tie perturbations remain a distinct next benchmark. The accepted
three-grid test gives decomposition errors below 1.62e-14 and best complete
residual-difference errors below 9.41e-9. It verifies derivative actions,
not a temporal stability criterion or continuum accuracy.

Raw alternating pressure projections use the same fixed physical window:
abs(sum(window*p*parity))/sum(abs(window*p)), for x, y and xy index parity.
Nearest-axis-zero centerline projections expose cancellation between opposite
row/column phases. No pressure is smoothed, detrended or filtered. The same
diagnostic on independently sampled dry Hertz pressure establishes a finite-
sampling reference, not an EHL solution. Nonzero projection alone is not an
instability test, and a small global projection need not mean a smooth field.
See validation-report section 23 for actual values, recovered fields and plots.

Optional CUDA checks evaluate only the unchanged linear elastic operator at
the accepted pressure fields and those five directions. They preserve Pa to m,
positive gap increase and the existing source-integrated kernel. They are
component parity checks; no GPU material/JFO solve or speedup is claimed.

## Resolved carriers and active-front diagnostics (2026-10-28)

`transport_probes.resolved_pressure_probes` uses the same pressure-supported
physical window as the preceding audit, with zero pressure edges and zero
uniform-grid force direction. Sine/cosine carriers have nominal physical
wavelength a, a/2 or a/4 in x or y, where a is the registered dry Hertz
radius. A carrier is tested only with at least four cells per wavelength;
omissions are recorded. Grid-relative two-cell Nyquist references are
separately labelled. The window/support/mean subtraction modify the carrier
spectrum; neither the probes nor their responses are eigenfunctions. With
the original dimensionless state and mass scale, reported static mass gain
is ||dR_mass||_2/||v_state||_2. Its coordinate scaling matters; this is not a
Jacobian condition estimate or a temporal growth rate.

Uniform-state independent checks use the exact symbols of pressure diffusion
and centered density advection. At xi=k*dx, their continuum attenuation
factors are 4*sin(xi/2)^2/xi^2 and sin(xi)/xi. At four cells/wavelength they
are .810569/.636620, at eight .949641/.900316 and at sixteen .987215/.974495.
These are smooth uniform-state symbols, not a model of the loaded nonlinear
front or an explanation of the production pressure pattern.

`active_front_probes` selects a representative adjacent full/cavitated pair
nearest y=0 at each existing +x rupture/reformation. Distinct margins are
required; missing front types and exact P=D tie counts are explicit. The
loaded roots have a selected rupture and no reformation or exact ties.
Probes transfer pressure from the full cell to its neighbor at zero total
force, remove content from the full cell, or add content to the cavitated
cell. West content is fixed and copied east/north/south content-direction
boundaries match the production state map. Pressure/content inequalities
and strict positive film are required. Intermediate diagnostic states may
violate mass/complementarity; they are never accepted physical roots.

For min(P,D), the true one-sided derivative equals dP if P<D, dD if D<P,
and min(dP,dD) at EXACT equality. A selected generalized Jacobian may differ
at a tie for some directions. The unchanged production merit_slope already
uses this directional rule; a separate analytical non-root tie test verifies
it. No near-zero field is rounded into a tie. Non-tied branch-switch limits
bound finite-difference sample sizes; the directions/states are not projected.
Second-order forward differences use two positive feasible steps at fronts;
centered differences apply to pressure-supported carriers. Every step and
error, unchanged-by-roundoff trials and feasibility status are recorded.

Across 43 loaded-root probes, independent seven-term SI mass actions agree
with complete analytic products within 8.94e-15 relative. Best feasible
residual-difference errors are below 9.66e-9, and front-only errors below
1.09e-10. All modes have nonzero action. Tests also cover the exact-tie rule,
reflected reformation selection, copied content boundaries, uniform symbols,
force/edge preservation, explicit sampling omissions and float64 conversion
of integer pressure/content inputs. This does not establish spectral or
temporal stability, continuum pressure-spike accuracy or causality.

## Same-spacing finite-reservoir extension (2026-10-28)

`property_reservoir.extended_reservoir_problem` adds integer original cells,
retaining every original coordinate exactly. The original geometry, material,
scales, supply, pressure/content/film gates and 20,000-node cap are retained.
The separate extension initializer verifies its source root, embeds its
pressure unchanged, sets added pressure to zero and edge-extrapolates content.
It changes no offset and clamps no field. Full target FFT reconstructs film;
shared-source film and integrated force must be preserved to the declared
1e-18 m/1e-12 relative initialization checks. The result is an explicit
PropertyLoadGuess, not a root. The existing repair and all original final
mass/complementarity/force/globalization gates remain mandatory.

This tests existing finite ambient-pressure/flooded-inlet boundary models
at new bounds, not a new infinite-domain condition. Each case starts from
the same accepted 65x97 200 N root; no failed partial seeds another case.
Inlet and outlet each extend four/eight original x cells; both transverse
edges extend four/eight original y cells. Spacings are fixed at dx=a/16,
dy=3a/32. All cases pass separate saved-field/force/history and sampled
direct elastic-source audits. The small-case test additionally compares a
repaired extension with independent cold initialization at the same force.

Largest tested changes are .058302% in sampled minimum film, .156815% in
central film and .062652% in peak pressure. Shared-source pressure changes
are at most .4750% relative L2 and shared film differences at most 2.335 nm.
The material outlet extension leaves the upstream root/fluxes unchanged to
roundoff on these cases. Inlet and transverse boundary mass flows change
substantially despite small inner-film changes; net balance still passes.
This limited sensitivity range is smaller than the existing mesh changes,
but proves neither infinite-domain convergence nor physical film accuracy.

## Isolated donor-capacity transport candidate (2026-10-28)

`transport_candidate` is an experimental CPU component, not a solver option.
With c=rho*h and m=rho*h^3/(12*eta), its +x face flux is
`-average(m)*(p_right-p_left)/dx + Ue*(c*theta)_left`.
The y diffusion flux, arithmetic mobility and finite-volume divergence are
unchanged mathematically. The production reference uses
`Ue*average(c)*theta_left`. Consequently the exact candidate-minus-reference
x-face flux is `-Ue/2*(c_right-c_left)*theta_left`.
Both use single shared face fluxes, so volume-weighted divergence telescopes
to outward boundary mass flow on uniform or nonuniform rectilinear axes.
Public pressure, film and flux units remain Pa, m and kg/(m s); interior
divergence is kg/(m^2 s), fields (ny,nx), entrainment +x. No field is clipped.

At a uniform full-film state, the candidate density-advection pressure symbol
is `Ue*h*rho_p*(1-exp(-i*xi))/dx`, versus the reference
`i*Ue*h*rho_p*sin(xi)/dx`. The donor symbol adds positive real part
`Ue*h*rho_p*(1-cos(xi))/dx` for rho_p>0, including a nonzero two-cell response.
For smooth capacity it adds a leading `-Ue*dx/2 * dxx(c)` term to divergence:
this is first-order numerical diffusion, not physical diffusion. Neither
symbol models the entire elastic/material/cavitation Jacobian or temporal
stability. Constant capacity recovers the reference exactly; stationary
constant-property p=0 and h*theta=constant gives constant donor flux.

Manufactured prescribed fields use linear p and trigonometric positive h,
with either theta=1 or smooth varying theta. Their continuum source is
derived analytically, without discrete differentiation. At zero constitutive
strength, `div(q)=-rho*h^2/(4*eta)*(hx*px+hy*py)
+Ue*rho*(hx*theta+h*theta_x)` for this linear pressure. On 33..257 square
grids the reference is second order for theta=1; the donor is first order.
Both are first order when theta varies. These source-driven prescribed fields
are not complementarity or EHL roots. Analytic donor flux directions include
all material, pressure, film and content variations, verified independently
at constant and full constitutive strength. The loaded-state checks supply
the full unchanged elastic film direction; nonsmooth complementarity and
Newton globalization are outside the candidate component.

The accepted 200 N reference roots fail the candidate mass gate. They cannot
be relabelled candidate solutions. Before a candidate Newton comparison,
implement and verify its own residual, boundary Jacobian, frozen operator,
bordered correction and original globalization/acceptance checks. Current
evidence does not justify changing the reference transport stencil.

## Optional cached CPU elasticity in material force Newton (2026-10-28)

`PropertyLoadSystem`, `solve_property_load_newton`, `continue_property_load`
and `repair_property_load_guess` accept `elasticity_mode="cached"`.
`"reference"` remains the default. Cached mode creates one existing
ElasticityOperator per system using the same analytic integrated kernel,
Nyquist ordering, backward FFT normalization and positive Pa-to-m gap sign.
Zero padding is rounded up to FFT-friendly lengths, still at least the full
linear-convolution size. It reuses the kernel spectrum in film, residual,
elastic JVP, h0 direction, Schur column and feasibility calculations. Active
correction views share this unchanged spectrum. Fresh continuation systems
construct fresh operators; no global cache or cross-step reuse is implied.

This changes arithmetic scheduling and FFT padding, not physical equations
or the Newton method. Strict h>film_floor, the complete material/elastic
Jacobian, force border, original sparse preconditioner, selected active
models, inexact forcing, both feasibility/Armijo stages and stopping gates
are unchanged. Cached reconstruction explicitly checks the original strict
film gate. Final SI mass audits still use the original reference flux;
benchmark audits also recompute reference convolution and sampled direct
source-cell deformation. Six new regressions cover complete products and
dense corrections, Schur fallback, final roots, infeasible film, rejected
trials, active recovery/backup, Krylov failure and continuation retries.

Both modes use NumPy/SciPy on CPU without requiring PyTorch. No CUDA material
or JFO implementation is added. Tiny FFT roundoff differences can change a
Krylov iteration count; equal iteration histories are not assumed or forced.
Matched-field/load convergence is checked with the original tolerances.

## Bounded mesh assessment and linear geometric multigrid (2026-10-28)

Material force systems retain a default 20,000 physical-node limit. An
explicit `maximum_physical_nodes=30000` allows the next 129x193 CPU mesh;
the registered case builder, transfer and repair use the same allowance.
There is no unlimited mode. The refinement runner verifies its initial
sparse factor residual and checks worker-lifetime peak resident memory below
1 GiB before Newton. Nonlinear budgets and every original physical gate are
unchanged. A failed field is retained and independently audited as rejected;
force closure alone never qualifies it as a refined solution.

Mesh comparison uses identical 200 N force, material laws and +/-3a physical
domain. Shared-node pressure norms use only coordinates present in both
grids, with no resampling. Raw profiles and sampled extrema/locations are
retained. Three-point Q(h)=Q0+C*h^p fits allow unequal spacing and require
monotone resolved differences and positive p. They remain provisional, not
asymptotic error certificates; nonmonotone quantities receive no fit.

`ReynoldsMultigrid` constructs the original homogeneous-Dirichlet SPD
`-div(H^3 grad(P))` matrix for prescribed positive dimensionless film. It
does not change face mobilities or flux conservation. Bilinear interior
correction prolongation includes exact zero boundary contributions;
restriction is P.T and coarse matrices are Galerkin P.T A P. Omitting a
constant full-weighting scale is consistent because it cancels between the
restricted RHS and coarse operator. Coarse equations use Cholesky with at
most 64 unknowns by default. Unsupported grid hierarchies raise instead of
silently factoring a large matrix. The foundation is capped at 300,000
interior unknowns before sparse assembly.

Each V-cycle starts at zero and uses equal pre/post weighted-Jacobi sweeps
(two by default). The per-level relaxation factor is at most 2/3, reduced
if necessary using an absolute row-sum bound on the spectrum of D^-1 A.
Transfers, exact coarse solve and symmetric smoothing give a fixed linear
symmetric positive preconditioner for this SPD operator. CG independently
checks its final true residual against its original tolerance. Default
unpreconditioned CG, dimensionless units and pressure edges are unchanged;
no pressure is clipped and there is no silent fallback.

Loaded frozen-mobility studies use H_eff^3=(rho/rho0)*(h/href)^3/(eta/eta0)
to represent positive diffusion coefficients. They omit density/viscosity
directions, nonlocal elasticity, content transport, complementarity and force
borders. Their convergence is not evidence of a complete coupled multigrid
solver. A separate known-pressure check uses symmetric diagonal equilibration
and checks forward error as well as the original true residual, because
extreme mobility contrast can hide weak rows in a global residual norm.
This diagnostic scaling does not alter material Newton stopping rules.

Full material/JFO multigrid requires a separately verified nonsymmetric
block preconditioner and active-front/coarse-constraint treatment. Nonlinear
FAS and CUDA multigrid are not implemented. Multigrid accelerates a discrete
solve; it cannot by itself validate its mesh accuracy or remove a pressure
pattern that already belongs to a converged discrete root.

The signed constant-property mathematical BorderedReynoldsElasticityProblem
now optionally selects `preconditioner="multigrid"`. Its pressure Jacobian
uses the sign/scaling `-A/reynolds_residual_scale`, so the inverse V-cycle
action is `-reynolds_residual_scale*B_MG`. The same fixed operator is reused
for pressure Schur solves and full bordered recovery inside the original
nonsymmetric GMRES path. Default ILU and its legacy explicit recovery remain
unchanged; MG setup/application failures propagate without ILU/Jacobi fallback.
Independent complete-correction, zero/weak/moderate compliance, final
Reynolds/load, strict feasibility, Armijo retry, Krylov-failure and load-step
retry tests cover this optional path. These signed, low-pressure cases remain
mathematical references and do not establish a physical material/JFO solution.

## Diagnostic active pressure/content MG inverse

`property_block_multigrid.PropertyBlockMultigrid` accepts the existing internal
material load system and its dimensionless state `(P,D,h0/H0)`, where
`P=p/p_ref`, `D=1-theta`, and C-order interior fields have shape
`(ny-2,nx-2)`, x along positive entrainment. Physical problem inputs retain
SI pressure, length, viscosity, speed and force conventions. This class is
not selected by any production nonlinear solver.

For a fixed full-film selector F, the frozen positive mobility is
`m=(rho/rho0)*(h/H0)^3/(eta/eta0)`. The existing Reynolds MG hierarchy uses
`cbrt(m/mass_scale)` as its surrogate film, so its diffusion matrix has the
same mass-row normalization as the material block. With A that SPD diffusion
matrix, the masked hierarchy starts from `F A F + (I-F)`. Galerkin coarsening
preserves SPD. Inactive pressure rows/columns have identity values; callers
project a finite cycle's output to impose zero inactive pressure corrections.
This projection applies to correction vectors, never to pressure/film states.

Complementarity rows fix dD on full cells and dP on cavitated cells for every
right-hand side, including nonzero constraint rows. Subtract those known
columns from the frozen mass RHS, apply a projected pressure cycle on full
cells, then solve cavitated donor-content mass rows by lower-triangular
substitution. Transport must be lower triangular with finite nonzero diagonal;
unsupported input fails explicitly. Coupling from unknown cavitated deficits
into full-cell mass rows is omitted, while pressure-to-cavitated-mass coupling
is retained. The resulting fixed linear inverse is nonsymmetric: use GMRES,
not CG, for the complete material/JFO system.

Only the frozen preconditioner is supported. Full material derivatives,
full-source elastic film response and the load border are retained in the
independent outer-operator tests, rather than approximated by this hierarchy.
The verified small-grid Schur and full-bordered corrections do not establish
nonlinear feasibility/globalization, 200 N repair convergence, mesh accuracy,
performance or physical validation. Production integration requires those
additional checks.

## Isolated loaded-state block-preconditioner comparison

`examples.benchmark_property_block` compares full bordered inverse surrogates
with exactly the same complete elastic/material JVP at saved registered 200 N
states. Source pressure, film and force remain in SI; the internal flattened
state/corrections are dimensionless P,D,h0/H0 as above. Accepted sources must
pass the existing independent force/field/film/boundary gates. Failed partials
are separately recomputed and must remain rejected by the original gates;
linear convergence never promotes a partial to a nonlinear root. Summary,
NPZ and packed-state hashes must match between workers, and input bytes and
state vectors are checked again after each worker's computation.

Each source/mode runs in a fresh child process. All modes use the verified
cached CPU elasticity implementation and check the complete JVP and residual
against the original reference before timing solves. Sparse frozen, material
and local modes factor their existing **full bordered** sparse matrices.
The diagnostic MG inverse uses its fixed pressure/content action and an
approximate force Schur complement; an unsafe Schur indicator fails explicitly.
No sparse inverse is substituted for a failed MG setup. Outer GMRES always
uses the original complete material/elastic JVP and original load row.

The two RHS vectors are the saved state's negative residual and J*v for a
fixed-seed normalized known direction. The latter exercises nonzero active
constraint and force components, and reports forward error as well as true
linear residual. Reports retain GMRES info, preconditioned histories, JVP
calls and exhausted solves. JVP counts and GMRES wall time exclude the
independent post-solve validation call. A linear result is verified only when GMRES
succeeds and its independently recomputed true relative residual is <=5e-9.
Only verified directions are compared against the local sparse result.

Inverse setup (including a fresh cached system on the existing physical
problem), inverse application, and GMRES wall times are separate. A setup and
inverse-action warm-up are excluded from those samples. OS worker-lifetime
RSS includes imports, input auditing, warm-up and solves: it is not incremental
preconditioner memory. Factor/array payloads exclude allocation workspaces;
MG stored-array totals can count shared storage more than once. The bounded
restart/cycle settings are diagnostic settings, not changed production limits.
These full-border timings are not timings of the production block-Schur,
recovery, nonlinear globalization or continuation paths. No large-mesh resource
allowance, new pressure root or physical film-height claim follows from them.

## Diagnostic tangent-preserving reduced blocks

`property_tangent_multigrid` adds separate reduced-LU controls and a
nonsymmetric MG prototype. Physical inputs retain SI units and the existing
`(ny,nx)` grid convention with +x entrainment. Internal state/corrections use
P=p/p_ref, D=1-theta and h0/H0, with C-order interior fields. The MG classes remain diagnostic. Exact reduced LU is also available through
the explicit optional field factorization described below.

Write the scaled sparse mass Jacobian as A*dP+B*dD, full-film selector F,
and cavitated selector C=I-F. Complementarity rows prescribe dP_C and dD_F
from any RHS, not just zero constraints. Subtract these known mass columns,
then solve `K*u=b_mass-A*dP_C-B*dD_F`, where `K=A*F+B*C`.
Reconstruct dP=dP_C+F*u and dD=dD_F+C*u. This retains *every* mass coupling
from the existing sparse material/local assembly, including cavitated-content
columns in full-cell mass rows. It does not approximate the nonlinear minimum
or change the chosen active set. Physical pressure/content arrays are never
projected or clipped.

`PropertyTangentLU` explicitly factors this n-by-n sparse matrix as an algebraic
control. With the original complete offset column and force row, exact field
inversion and a scalar Schur reconstruction reproduce the existing 2n+1
sparse bordered surrogate. Only its factorization representation changes;
the complete matrix-free elastic/material Jacobian stays the outer operator.
This remains a diagnostic control and is also an optional integrated field
factorization, never a fallback for MG failure. Standalone diagnostics reject
singular reduced blocks/unsafe surrogate Schur indicators; the integrated path
retains the original explicit full-border recovery.

`PropertyTangentMultigrid` uses the same K, including either material or local
cell-self elastic pressure tangents. Fine signed row/max-column equilibration
produces `D_r*K*D_c`; its inverse action maps back with D_c and D_r. Bilinear
interpolation and transpose restriction act on scaled *mixed corrections*,
with Galerkin coarse matrices. These corrections mix full-cell pressure and
cavitated-cell deficit roles; they are not coarse physical films or states.
No SPD or contraction guarantee applies. Fixed ILU defect correction uses
NATURAL ordering, basic/area dropping, drop_tol=1e-4, explicit fill factor 2
(default, bounded 1..4), and two pre/post sweeps. Exact sparse coarse LU is
bounded to <=64 unknowns by default (maximum configurable 256). Setup errors
identify the failed level and size; no shifts, diagonal regularization,
state clipping or automatic LU fallback conceal factorization failure.
The fixed cycle starts from zero, so outer GMRES sees a linear inverse.

The original frozen `PropertyBlockMultigrid` and SPD `ReynoldsMultigrid`
remain unchanged references. The benchmark's historical four-mode default is
preserved; explicit `--modes` enables material-mg/local-mg and the corresponding
material-reduced/local-reduced LU controls, retaining the local sparse reference.
Fine-block LU success does not establish robustness of a nonsymmetric coarse
hierarchy. All candidate corrections require complete true-JVP/force checks,
and integrated reduced LU preserves strict film feasibility, both line-search
stages, recovery/continuation checks and independent final physical gates.
Robust MG production integration remains planned.

## Optional reduced field factorization in material force Newton

`PropertyLoadSystem`, `solve_property_load_newton`, `continue_property_load`
and `repair_property_load_guess` accept `field_factorization="full"` (the
unchanged default) or `"reduced"`. This choice is independent of the existing
frozen/material/local sparse pressure tangent and reference/cached CPU
elasticity choices. It changes the field-preconditioner LU representation,
not any physical residual, Jacobian, active constraint or convergence gate.
Public pressure/film/force/speed interfaces remain SI, arrays `(ny,nx)` with
+x entrainment; internal P,D,h0/H0 scaling remains unchanged.

The reduced path reuses the already assembled mass/active rows: K=A*F+B*C
is factored with sparse LU and arbitrary mass/complementarity RHS vectors
are reconstructed in the original 2n field space. No second matrix assembly
or experimental MG hierarchy is invoked. For a fixed state and selected active
model, this is algebraically the same field inverse as unreduced sparse LU.
Small floating-point differences remain subject to the original complete
matrix-free true-JVP correction checks.

Original pressure-block and offset-response GMRES solves, the load Schur
indicator and complete correction check are unchanged. A failed reduced field
factor, failed block Krylov solve, unsafe true Schur complement or failed
complete correction uses the existing full 2n+1 sparse-bordered GMRES recovery;
it does not silently build an unreduced field LU. `used_bordered_fallback`
remains explicit. Correction/error diagnostics and material Newton model
histories now retain the field choice, field-setup failure reasons, individual
true linear residuals and Krylov traces, including failures before recovery.
A failed full-border recovery still raises the original linear/nonlinear error
with the last audited state and diagnostics.

Selected active-set views preserve the factorization choice and use their own
selected constraint rows. Full-step and fraction-to-boundary Armijo attempts,
strict film/pressure/content feasibility, best verified lookahead backup,
content-bound pivots, 30-correction/eight-model budgets and load-step retries
are unchanged. Continuation retries preserve the original accepted seed and
factorization choice. Initial guesses still require a separate successful
repair before becoming accepted roots. No pressure/content/film projection,
clipping, constitutive relaxation or new node allowance is introduced.

Example with the existing SI problem/fluid and audited seed:

```python
root = solve_property_load_newton(
    problem, fluid, speed_m_s, target_load_n,
    seed=seed, seed_offset_m=seed_offset_m,
    preconditioner="local", elasticity_mode="cached",
    field_factorization="reduced",
)
```

Measured nonlinear repair parity/performance and its limits are recorded in
validation report section 32. This optional CPU optimization does not establish
cold-continuation performance, a fourth converged mesh, physical film accuracy,
or material/JFO GPU execution. Experimental tangent MG remains diagnostic.

## Replayable Newton diagnostics without changing the step rules

The internal `_solve_property_load_state` accepts an optional `state_observer`.
It receives iteration number, a read-only **copy** of the exact internal
float64 state, and a deep copy of the SI root/iterate fields and history.
Internal state is `(P,D,h0/Href)`, P=p/pref, D=1-theta, with C-order interior
`(ny-2,nx-2)` arrays; physical fields remain `(ny,nx)`, pressure Pa, film and
offset m, force N, +x entrainment. Copies prevent diagnostic changes to arrays,
audits or histories from changing the solver. Default execution has no observer.
An observer exception propagates as `PropertyStateObserverError`, separately
from nonlinear convergence errors. An observed iterate has not necessarily
passed any root gate, including the initial interpolated guess or final
budget-limited partial.

Instrumented model histories retain exact selected-row override indices relative
to the canonical P>=D selector. Failed linear-model diagnostics also retain
iteration/model indices and selected rows. Exact internal state is stored
because reconstructing D from saved SI theta can lose tiny deficits or alter
ties. No state or active set is rounded into a tie for replay.

`newton_diagnostics.affine_feasibility_limits` is a separate, read-only SI
helper. It reports independent uncapped alpha limits for interior pressure
lower, content lower/upper and full-grid strict film lower bounds. Existing
pressure/content tolerances are used, including p>=-1e-10*pref and
-1e-10<=theta<=1+1e-10. Reported equality at the film floor is infeasible.
None denotes no finite float64 bound; overflow in a diagnostic ratio is not
a clipped state. Exact-minimum cell indices/tie counts are reported without
rounding near ties. This helper never chooses or changes a Newton step.

The replay tool also computes the original internal-variable bounds separately,
including the unchanged 0.99 safeguard when a constraint is crossed before
alpha=1. SI theta=1-D can introduce cancellation differences, so SI and internal
bounds are retained separately. Every replay uses recorded selected rows,
recomputes the complete true-JVP correction and original nonsmooth-merit slope,
and checks the accepted next state against its exact snapshot. Sampled full/
bounded trial merit and feasibility are independently recomputed, including
candidates that were deferred or non-descending. These checks describe the
algorithm's recorded restrictions, not physical temporal stability or a
Jacobian spectrum.

`examples.diagnose_mesh_newton` uses the same validated guess fields, fixed
material case, private Newton engine and original acceptance/budget rules.
It logs every evaluated state to new files and performs the existing 1 GiB/
1e-8 sparse-factor preflight. Default physical node cap remains 20,000; the
previously introduced explicit 30,000 allowance is required for 129x193.
`examples.analyze_mesh_newton` independently reopens terminal SI fields,
checks original gates and raw-state hashes, and replays selected or all models.
A converged root is never inferred from force closure or a diagnostic callback.
