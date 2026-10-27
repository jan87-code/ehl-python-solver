# Isothermal pressure-dependent fluid properties

SI pressure p is gauge relative to cavitation pc=0. At a fixed reference
temperature, the independently parameterized reference values are eta0
[Pa s] and rho0 [kg/m^3]. The initial numerical fluid is illustrative:
eta0=.04 Pa s, rho0=850 kg/m^3, z=.68, assigned temperature 313.15 K.
No measured-fluid fit, temperature law or classical EHL validation is implied.

The primary SI source is Hamrock, NASA RP-1255 (1991), printed pp.57,70,
equations 4-10 and 4-19:
https://ntrs.nasa.gov/api/citations/19910021217/downloads/19910021217.pdf

```
R(p)=rho/rho0 = 1 + a p/(1+b p), a=.6e-9 /Pa, b=1.7e-9 /Pa,
V(p)=eta/eta0 = exp(A*((1+p/gamma)^z-1)),
A=ln(eta0/eta_star), eta_star=6.31e-5 Pa s, gamma=1.96e8 Pa.
dR/dp=a/(1+b p)^2,
dV/dp=V*A*z/gamma*(1+p/gamma)^(z-1).
```

The density source uses pressure in GPa; converting its .6/1.7 coefficients
to inverse Pa is essential. R tends mathematically to 1+a/b at infinite
pressure; this is a model limit, not demonstrated physical applicability.
The viscosity formula uses logarithms of a dimensionless viscosity ratio.
`log1p`/`expm1` preserve small-pressure increments; overflow raises instead
of capping viscosity or pressure. Values and derivatives default to float64.

The scanned 1975 NASA TN D-8049, printed p.8, lists eta_star as .00631
N s/m^2 and .0631 cP, which are inconsistent by a factor of 100. Its
preceding density pressure unit is ton/in^2, requiring a declared ton
convention. We inspected the original scan and use the explicit SI
constants above from RP-1255; we do not silently claim exact reproduction
of the older numerical benchmark's fluid constants. Its separately rounded
gamma is 196.09 MPa. Later classical-table matching must record its chosen
fluid constants and this source discrepancy explicitly.

A continuation strength s in [0,1] multiplies R-1 and ln(V), retaining
analytic derivatives. s=0 returns the exact constant-property limit; only
s=1 represents the selected full law. Intermediate strengths are numerical
continuation states. Viscosity index zero or density a zero independently
disables that pressure dependence. A small signed-pressure extension is
available for feasible numerical roundoff/trials within the constitutive
domain; it does not model a liquid supporting tensile load. No values are
clipped. Public `properties` returns dimensional density/viscosity and
pressure derivatives; `ratios` returns R,V,dR/dp,dV/dp.

Standalone verification uses independent 60-digit Decimal evaluation at
0,0.1,10,100,500,1000 MPa, separate centered finite differences, original
GPa density units, origin slope, monotonicity and exact constant-property
limits. Generated curves are under `results/constitutive_20261027_verified/`, with
SI NPZ arrays and runtime/source hashes; generated data are outside Git.
Reproduce with `python -m examples.run_constitutive_verification
--output-directory results/constitutive_new` using the documented single-
thread/MPLCONFIGDIR/PYTHONPATH setup. This milestone does not alter flow.

## Conservative prescribed-film integration

`property_reynolds.py` conserves the physical mass flux
`j=-rho(p) h^3 grad(p)/(12 eta(p)) + Ue rho(p) h theta e_x`,
with p>=0, 0<=theta<=1, p(1-theta)=0. Density is the liquid density;
at cavitation p=0, liquid density is rho0 and theta is transported content.
Dissolved gas, transient squeeze and thermal physics remain absent.

Internally X=x/L,Y=y/L,H=h/href,P=p/pref, R=rho/rho0,V=eta/eta0,
Lambda=12 eta0 Ue L/(href^2 pref). Arithmetic face mobility is
avg(R H^3/V); advective capacity is avg(R H) and theta is west-upwinded.
The physical mass-flux scale is rho0 href^3 pref/(12 eta0 L), in kg/(m s).
Integrated boundary flows are kg/s, not m^3/s. A single reference-density
factor cannot convert variable-density mass flow into physical volume flow.

The bounded coefficient-Picard method solves a complete constrained JFO
active-set system for each frozen mobility/capacity. Density coefficients
are damped arithmetically and viscosity logarithmically, both by factor .5.
Pressure/content fields are never blended or clipped; every accepted inner
state satisfies the frozen complementarity system. Final convergence
independently evaluates the ORIGINAL SI mass flux at unrelaxed constitutive
properties of the returned pressure. Thus coefficient damping cannot make
an incorrect pressure-law state pass. Limits: 20000 nodes, 100 iterations
per constitutive strength and 100 active-set iterations. Exhaustion retains
history and the last candidate. Physical defaults are ambient pressure edges,
flooded west, copied outflow content and no transverse advective flux.
Optional positive Dirichlet pressure data imply wet edges and allow an
independent pressure-driven reference. They do not change default supply.

Acceptance retains max original dimensionless mass defect <=1e-8,
relative local/global mass imbalance <=1e-10 and complementarity/
admissibility <=1e-10. Pressure update <=1e-9 is additional, not sufficient.
The analytic complete smooth mass-flux direction includes h, pressure,
content, density and viscosity derivatives. Independent centered differences
check it; no generalized derivative at front changes is claimed.

The independent pressure-driven channel has constant h=20 um, Ue=0,
x=[0,1] mm, p_west=5 MPa and p_east=1 MPa. F(p)=integral rho/eta dp
is linear in x and mass flux per width equals h^3*(F(p_w)-F(p_e))/(12 L).
Adaptive quadrature and scalar root inversion provide continuum pressure
without a sparse/JFO solve. Exact pressure traces on transverse boundaries
embed this reference in a 2-D rectangular problem. Grid convergence checks
both pressure and physical mass flux. The separate constant-property limit
matches the original flooded/starved JFO solver. A constant high-pressure
transport case checks density in the entrainment term and SI boundary units.

Reproduce `python -m examples.run_property_reynolds_verification
--output-directory results/property_reynolds_new` using single-thread setup.
The verified run is `results/property_reynolds_20261027_verified/`.
Point cases retain the finite +/-8L,+/-8Ly reservoir, h0=.1 um, Rx=10 mm,
Ry=20 mm and Ue=.1 m/s. They demonstrate material pressure dependence,
not a resolved classical-EHL pressure/film benchmark. Their mesh errors
remain separately reported; elasticity is not yet included in this milestone.

## Initial uniform elastic feedback

`property_coupled.py` is a separate prescribed-h0 CPU Picard experiment.
It starts at zero pressure and rigid film, continues the material laws on
the first rigid solve, then increases elastic compliance through
0,.25,.5,.75,1. Each inner call closes the pressure-dependent JFO equations;
all pressure sources on the fixed finite domain enter the existing uniform
zero-padded FFT. Exact film h0+geometry+gamma*w(p), including boundary-node
displacement, is independently audited for original SI mass balance before
stage acceptance. Film iteration is relaxed by .5; returned pressure/content
are never blended or clipped. Final results use full compliance and full
material strength; strength zero is an independent isoviscous limit.

The old uniform CPU problem supplies geometry/elasticity and remains
unchanged. Explicit eta0 and speed must reproduce its Lambda. Uniform cells,
20000-node and 100-iterations-per-compliance-stage caps remain mandatory.
Nonlinear failure preserves outer history, last coupled candidate and any
inner material failure information. Load is measured at prescribed offset;
no new force-controller or claimed classical-EHL load closure is introduced.

Checks include old monolithic parity at material strength zero, and a
separate small complete nonlinear algorithm using SciPy hybrid root. The
latter starts from the old constant-property solution, never from the new
pressure-dependent Picard root. It solves all pressure/content unknowns
with exact full-source film on a 9x13 grid; the verification reference is
capped at 200 interior nodes and 4000 evaluations. Physical mass,
complementarity and field errors are independently audited, rather than
accepting only the root solver's status. This benchmark algorithm is not a
new production/globalized Newton method or a proof through arbitrary fronts.

The small complete reference uses x=[-4L,6L],y=[-4Ly,4Ly]. Larger initial
mesh cases use x=+/-8L,y=+/-8Ly, h0=.1 um,Rx=10 mm,Ry=20 mm,
E*=100 GPa and Ue=.1 m/s, with the same illustrative fluid. All bounds are
finite physical reservoir assumptions, with ambient pressure/flooded west.
SI load is checked by independent rectangular-cell summation against the
trapezoidal integral; direct source-cell deformation at the pressure peak
checks the FFT. Raw density, viscosity, content, pressure, film, deformation,
Newton-reference data, failure candidates and complete histories are saved.

Reproduce `python -m examples.run_property_coupling_verification --include-fine
--output-directory results/property_coupling_new` with the standard CPU
environment. Optional --include-fine adds the bounded 65x65 case. These are
initial coupled numerical checks. Grid/domain accuracy, realistic strong
loads and matched published film heights remain separate subsequent work;
current pressure-dependent fields must not be described as validated
classical EHL merely because conservation and equation checks pass.

## Positive-force control

`property_load.solve_property_at_load(problem, fluid, speed_m_s,
target_load_n, h0_bracket_m=(lower_m, upper_m))` solves for the SI rigid-gap
offset h0 with `integral p dx dy = target_load_n > 0`. Coordinates, physical
reservoir boundaries, radii, E*, fluid, speed, inlet content and all reference
scales remain fixed. Changing h0 does not regenerate the grid or choose a
new natural length. The returned h0 is not the sampled minimum film.

This is a bounded nested reference: each uncached force evaluation cold-
starts the established material/compliance continuation to full coupling.
Only a separately converged coupled state is eligible. The controller
recomputes original SI mass/complementarity, full-source film/displacement
and the pressure force from raw fields, checking the solver's cached load.
Rejected inner candidates are never supplied as root-function values.
Pressure is never rescaled to force closure. Existing nonlinear tolerances,
iteration/node limits, pressure/film feasibility and FFT conventions remain.

Three default offset samples establish a strictly decreasing sampled force
branch. This is evidence on the selected bracket, not a global uniqueness
proof. A sampled state already within the required force tolerance can be
returned, including an endpoint within numerical tolerance; otherwise a
sign-changing bracket is mandatory. Brent iteration uses audited forces
only. Relative force error must be <=1e-8 (or a stricter requested value)
even when offset convergence passes. A deliberately loose offset stopping
rule is tested to ensure it cannot substitute for force convergence.

Limits: at most 24 distinct force evaluations and 24 scalar root iterations;
3..7 monotonicity samples. Defaults use offset tolerance 1e-16 m as a
numerical stopping rule, not physical uncertainty. The scalar cache stores
probe records only; at most best/last coupled roots are retained. Optional
`evaluation_callback(record, solution)` streams each complete accepted
field/history once for reproducibility, including states later revisited
through the cache without a repeat callback.

`PropertyLoadError` retains accepted evaluations and the last audited state,
with a separately identified failed offset, inner exception and failed
coupled candidate. A partial-compliance or otherwise unconverged state is
not a force solution. Nonbracketing targets, budget exhaustion, nondecreasing
sampled response and failed physical/force audits are explicit errors.
Callback failure also stops with diagnostics. There is no automatic bracket
expansion or relaxation of criteria to obtain a root.

The present prescribed-offset solver starts from a positive rigid gap.
Accordingly this initial load API explicitly requires positive h0 brackets.
An actual liquid film may remain positive with signed h0 in other regimes;
that branch requires a separately verified warm-start/nonlinear extension.
A failed positive bracket is not proof that a physical solution is absent.
The method's cost is multiple cold coupled solves; no acceleration claim is
made. Strong-load, signed-offset and global branch robustness remain open.

Reproduce known-offset recovery, an imposed 0.10 N target, a different
bracket for the same target, a fixed-load mesh comparison and retained
failure with `python -m examples.run_property_load_verification
--include-refinement --output-directory results/property_load_new` under
single-thread CPU settings. The verified numerical experiment is
`results/property_load_20261027/`. Load closure does not establish film or
pressure mesh accuracy, domain independence or experimental validation.

## User-supplied 200 N circular benchmark

`ehl_solver.benchmark_200n.Circular200N` fixes the user's SI operating case
and uses 113 GPa in U=eta0 Ue/(E R), W=Q/(E R^2), G=alpha E. The elastic
kernel uses this same 113 GPa contact modulus; E'=2E=226 GPa is recorded
as a conversion only. With eta0=.04 Pa s and alpha=2.21e-8/Pa, the existing
Roelands law is matched at ambient pressure by

```
z = alpha * gamma / log(eta0/eta_star) = .671366671110627
```

Thus d(log eta)/dp at zero is exactly alpha. This specifies an assumed
Roelands curve, not a unique law implied by alpha or a verified match to
the user's reference lubricant at GPa pressure. The existing density model
uses assumed rho0=850 kg/m^3, a=.6e-9/Pa and b=1.7e-9/Pa. No operating
temperature was supplied, and none is invented. Supplied Hamrock-Dowson
film estimates remain correlation targets, not exact solutions.

## Warm signed offsets: bounded analytic reference

The user subsequently confirmed Roelands for the circular case. The existing
density parameters remain assumed, and the matched ambient-alpha definition
is explicit. `property_warm.solve_warm_property_offset` takes a positive-gap
problem template, an independently audited full-material seed, its SI offset
and a new finite signed offset. It does not change the template constructor
or its positive-offset cold-start semantics. All new fields call the existing
free-space FFT with an explicit signed offset override; *total* film must
exceed the original floor everywhere.

The reference solves interior P=p/pref and D=1-theta. Mass rows are original
SI mass-flux divergence divided by rho0 href^3 pref/(12 eta0 L^2) and by
100; complementarity is min(P,D). Its analytic action differentiates both
rho and eta, h through the full-source FFT, face mobility rho h^3/(12 eta),
face transport rho h and upwind content. Dense assembly uses this action
and equilibrates rows/columns; the independently recomputed JVP correction
must have relative residual <=5e-9. True directional derivatives at min ties
enter Armijo; pressure/content bounds and exact positive film are checked
on every trial. Intermediate trials can have mass/complementarity defects;
only roots passing the original independent physical audit are returned.

At most 400 interior nodes (800 unknowns), 30 Newton corrections and 24
backtracks bound this reference. It makes no GPU or scalable production
Newton claim. `continue_warm_property_offset` limits each offset increment
to half the last minimum total film, halves rejected steps and grows accepted
steps by 1.5. Its at-most-60-attempt ledger retains only scalar summaries;
a callback streams complete roots/failures. Retries always start from the
last audited root. It follows offset with measured force, not a new imposed
force algorithm, and does not establish global branch uniqueness. Load
closure remains a separate gate before physical film-height comparisons.

## Matrix-free force Newton and load-directed continuation

`property_load_newton.PropertyLoadSystem` augments the verified analytic
material/elastic action with h0/href and `(integral p dxdy-Qtarget)/Qtarget`.
The h0 column differentiates total film by href; the force row differentiates
SI pressure quadrature. Scales and the finite reservoir stay fixed as force
changes. `solve_property_load_newton` requires a fresh physical/boundary/film
audit of the supplied warm root and accepts only original mass, content,
pressure, complementarity and positive-film criteria plus relative force
error <=1e-8. Signed offsets remain permitted; no seed is silently repaired.

The true operator is matrix-free. A sparse frozen-property/frozen-film JFO
block and analytic offset column precondition the verified Schur/GMRES
correction from `cavitating_newton`. Reconstructed bordered corrections and
all block solves retain their true-JVP checks (5e-9 maximum relative error),
Krylov histories and a complete bordered fallback when Schur reconstruction
is unsafe. An independent dense correction on a small problem checks both
routes. No dense Jacobian is formed by the production load path. The old
dense reference keeps its default 400-interior-node bound; only the shared
analytic system can explicitly opt into the scalable path's allocation cap.

Globalization first tries a full step, then uses a safe affine fraction to
pressure/content bounds and the exact total-film floor before backtracking.
No ratios are formed for bounds that a full step cannot cross, avoiding
irrelevant arithmetic overflow from subnormal direction components. There
is no clipping, projection, film lift or tolerance relaxation. Merit decrease
uses the original min(P,D) equations and their true directional derivative
at ties. Failed searches can try up to eight selected active-constraint
models, pivoting violated P_new=0 or D_new=0 rows. Each selected linear
system has its own fully checked residual; switched rows are not mislabeled
as the old min derivative. Original-equation merit/slope and physical bounds
still govern acceptance. Selected-system algebra is independently verified;
broad robustness across arbitrary active-front switches is not inferred.

Limits: 20,000 physical nodes, 30 Newton corrections, eight active models,
24 line trials per model, GMRES restart 60 and at most ten cycles per block.
Sparse factor fill consumes additional memory; node/Krylov caps are not a
measured total-process memory guarantee. CPU benchmarks use one BLAS/OpenMP
thread. No GPU/runtime dependency is introduced.

`continue_property_load` advances SI force by bounded adaptive increments,
halving rejected steps and growing accepted ones by 1.5. Only a fully audited
force root can seed the next solve; retries roll back to the last accepted
root. Its maximum is 60 attempts per call (benchmark: 12 per requested
checkpoint), with an explicit minimum force increment. Callback-streamed
histories distinguish accepted states, failed partial Newton iterates and
Krylov diagnostics. Nonfinite diagnostics are tagged explicitly in JSON,
not emitted as invalid numeric literals or silently discarded.

## Explicit sparse tangent preconditioners

The unchanged default `preconditioner="frozen"` preserves the earlier path.
Explicit `material` adds the analytic pressure-law tangent at fixed film;
`local` additionally uses the source-cell self compliance as a diagonal
film approximation **inside preconditioning only**. With M=R H^3/V,
C=R H, and c_self=pref*K_self/href, their extra pressure coefficients are

```
dM/dP = pref H^3 (R'/V - R V'/V^2) + 3 R H^2 c_self/V,
dC/dP = pref H R' + R c_self.
```

Primes are pressure derivatives of ratios per Pa; c_self=0 for `material`.
Differentiate arithmetic face coefficients using the actual base pressure
gradients and upstream theta, then assemble the resulting five-point pressure
block sparsely. The offset column, constraints and force row retain the
complete analytic action. Every true operator product still includes the
original Roelands/Dowson-Higginson laws and full-source FFT; there is no
elastic-source cropping, local-elastic physical model or field alteration.

Independent SI finite differences check each approximate sparse tangent,
including the source-cell dimensions. Complete operator/residual equality
across modes and small-grid dense-versus-GMRES correction parity ensure that
only preconditioning changes. Existing node, Newton, line-search, Krylov and
true-residual limits remain unchanged; the optional mode never silently
replaces a failed solve with a looser correction.
