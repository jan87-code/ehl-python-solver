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
