# Hamrock–Dowson benchmark and matching conservative 50 N case

The user's [TriboNet calculator](https://www.tribonet.org/calculators/ehl-film-thickness-calculator-central-and-minimum-elliptical-point-contact/)
is now used as a film-height correlation benchmark alongside the equation,
force, conservation and feasibility checks. The correlation is not an exact
solution or a replacement for those checks. No arbitrary percentage agreement
threshold has been introduced to label a solution physically validated.

## Formula, conventions and applicability

`ehl_solver.film_correlations.hamrock_dowson_point_contact` accepts SI load,
radius, contact modulus, ambient viscosity, mean entrainment speed and declared
pressure-viscosity coefficient. It returns minimum/central film [m] and the
explicit dimensionless parameters. Entrainment follows x, parallel to the
minor contact axis. k is transverse/entrainment Hertz semi-axis ratio; actual
circular k=1 is used, not an approximate geometry-fit value of 1.03.
The fitted ellipticity range is restricted to 1 through 8.

- E'=2 E_contact, with 1/E_contact=sum((1-nu_i²)/E_i).
- U=eta0 Ue/(E' Rx), W=Q/(E' Rx²), G=alpha E'; Ue=(u1+u2)/2.
- hmin/Rx=3.63 U^0.68 G^0.49 W^-0.073 (1-exp(-0.68 k)).
- hc/Rx=2.69 U^0.67 G^0.53 W^-0.067 (1-0.61 exp(-0.73 k)).

The primary source is Hamrock & Dowson, *Isothermal Elastohydrodynamic
Lubrication of Point Contacts: Part III—Fully Flooded Results* (1977),
[DOI 10.1115/1.3453074](https://doi.org/10.1115/1.3453074).
[NASA TP-1342](https://ntrs.nasa.gov/api/citations/19780025504/downloads/19780025504.pdf)
provides the parameter definitions and minimum-film formula (equations 2–8
and 26). These fits concern fully flooded, isothermal, viscous-elastic contact;
roughness, starvation, shear thinning and heating are not included here.
The supplied calculator page was inspected; its interactive calculation was
not executed. Reported numbers come from the tested local formula.

With Q=50 N, Rx=15 mm, E_contact=113 GPa, eta0=0.04 Pa s, Ue=2 m/s,
alpha=2.21e-8 Pa^-1 and k=1, E'=226 GPa and the estimates are
**284.149730 nm minimum and 497.969177 nm central**.
The old supplied 200 N values, 278.494984/477.352974 nm, match substitution
of E_contact directly into the E' formulas. With E'=226 GPa the 200 N
estimates are instead 256.801093/453.799949 nm. Historical result files and
registered case metadata are preserved; they have not been rewritten.

## Matching conservative solver experiment

`examples/compare_conservative_50n.py` starts from the supplied N33 MATLAB P
and signed H0, initially with theta=1 as an explicitly **unaccepted guess**.
It solves the existing pressure/content JFO system at 50 N on [-4a,4a]².
It uses cell-integrated elasticity and arithmetic mobility/capacity faces,
where MATLAB uses a sampled elastic kernel and harmonic mobility faces.
These equation/discretization differences are explicit; matching geometry
and nominal physical inputs does not make the two models identical.

Fluid constants exactly reproduce the MATLAB Roelands law:
p0=1.98e8 Pa, eta_reference=exp(-9.67) Pa s,
z=alpha_fraction*alpha_p*p0/(log(eta0)+9.67). Density is the full
Dowson–Higginson law, with assumed rho0=850 kg/m³. The alpha fraction modifies
the Roelands exponent, not a joint viscosity/density continuation strength.
The HD comparison uses that law's ambient d(log eta)/dp, declared explicitly;
differences from asymptotic or effective pressure-viscosity conventions and
other fitted-fluid assumptions remain modeling uncertainties.

The problem's positive reference offset is separate from the signed
load-controlled unknown. Every trial evaluates the actual total film.
No pressure, content, film or divergence clamp is added. All original force,
mass, complementarity, line-search and Krylov gates remain unchanged; each
run uses at most 30 corrections and 10 Krylov cycles. A direct integrated
elastic convolution separately reconstructs the final film, followed by SI
mass-flux and force audits. Failed guesses are never accepted as roots.

Both `legacy` and `best` existing active-lookahead policies converge in nine
corrections on N33 at alpha_fraction=0.8978125 and at full alpha_fraction=1.
They produce identical saved fields for each fluid. Thus this case does not
show a candidate-selection advantage or require a new production algorithm.
The production core, including its existing selection behavior, is unchanged.
At partial alpha, the conservative N33 film is 155.856/344.646 nm against
HD 269.531/470.317 nm. The original pressure-only MATLAB checkpoint is
463.175/741.209 nm against the same correlation, but fails mass closure.
This quantifies why pressure-only reproduction and physical correctness must
be assessed separately.

## Full-viscosity refinement

Refinement keeps the physical [-4a,4a]² reservoir, load, fluid and supply fixed.
Nested pressure/content interpolation preserves force but is not a solution;
each finer state is repaired and independently audited. Node counts remain
below the default 20,000 physical-node cap, including 129²=16,641.

| Grid | Corrections | Minimum film, nm | Central film, nm | Difference from HD min / central |
| --- | ---: | ---: | ---: | ---: |
| 33×33 | 9 | 171.5854 | 371.6841 | -39.61% / -25.36% |
| 65×65 | 10 | 265.5510 | 479.9217 | -6.55% / -3.62% |
| 129×129 | 9 | 282.9728 | 455.2518 | -0.414% / -8.58% |

All three pass strict discrete numerical acceptance. Agreement with HD is a
quantitative physical benchmark, not proof of mesh convergence: minimum film
changes about 6.56% between the last two meshes and central film changes about
-5.14%. Central film is nonmonotone across this three-grid sequence; no
asymptotic convergence order is asserted. Finite-domain independence, further
spatial resolution, pressure-pattern inspection and fluid-correlation
sensitivity remain open before engineering accuracy is claimed.

## Reproduction

Use a new output directory for each run. Full-viscosity N33:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/compare_conservative_50n.py '/home/jan/GoogleDrive/EHL Solver/EHL_BorderedNewton_Alpha_8978_N33.mat' results/jfo50_NEW --alpha 1 --selection best
```

For the next nested grid, add `--nodes 65 --seed results/jfo50_NEW`; refine
again with `--nodes 129` and the accepted N65 directory as `--seed`.
Use `--alpha 0.8978125` for the checkpoint fluid and `--selection legacy`
for the controlled existing-policy comparison. Seed physics and acceptance
are checked; output records every exact dimensionless state and SI fields,
all Newton diagnostics, source snapshots/hashes and correlation differences.

Local evidence lives in `results/conservative_50n_*_20261029/` and
`results/hd_validation_20261029/`. Generated simulations are Git-ignored.

Pressure accuracy is specifically **not** established. Shared-node pressure
relative L2 differences (normalized by the finer pressure field) are 33.60%
for N33→N65 and 41.93% for N65→N129. The unfiltered pressure maps and central
profiles show an alternating cell pattern, also visible weakly in film.
Direct elastic reconstruction and mass/force closure do not eliminate that
discretization issue. Near-HD minimum film alone must not be treated as
validation of the pressure field. The next numerical investigation should
address the transport/elasticity coupling's alternating mode with separately
benchmarked stencil/derivative changes, rather than tune parameters to the
correlation. Domain and fluid-model sensitivity remain additional checks.

For N129, the separately reconstructed SI field has scaled maximum mass
residual 3.1044e-14, relative local/global balances 7.5721e-15/2.2573e-16,
force 49.99999999999999 N, complementarity 1.6151e-28 and maximum
film/displacement reconstruction error 8.26e-21 m. These are strict discrete
closure results; the pressure-pattern and mesh evidence limits their physical
interpretation. No tolerance or physical-node allowance was increased.

Validation: final full CPU suite 578 passed, 62 optional backend/GPU skips
(337.88 s); 15 focused correlation/convention/fluid-comparison tests pass.
Ruff and whitespace checks pass. All 55 accepted corrections across the six
runs pass independently checked history Armijo/acceptance rules; the maximum
returned true linear residual is 8.995e-10 against the unchanged 5e-9 gate.
No GPU execution or isolated performance speedup is asserted. The raw
film/pressure figure was visually inspected without pressure filtering.

Verified local archive: `results/conservative_hd_evidence_20261029_verified.zip`,
8,428,906 bytes, 95 files; every manifest SHA256 and ZIP CRC passes.
Archive SHA256:
`68bae0deeb9030f1c9e5671c8eaa0ad1cbf4d628abbb11645bd2a2f8715076ca`.
It includes the original MAT input, all six benchmark runs and source versions,
raw states/histories, audits, comparisons and inspected plot. It is Git-ignored;
no off-machine backup is asserted.
