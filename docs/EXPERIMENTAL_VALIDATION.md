# Experimental validation candidates

This note records three user-supplied papers as prospective experimental
benchmarks. The machine-readable source records are in
[`cases/experimental_validation_candidates.json`](../cases/experimental_validation_candidates.json).
For Ciulli et al. (2008), Figure 13 has been digitized into the initial
steady-state, pure-rolling reference profiles at
[`cases/ciulli_2008_figure13_s0_profiles.csv`](../cases/ciulli_2008_figure13_s0_profiles.csv).
The regression tests check source-data integrity, units, and the digitized
reference shape; they do **not** run the EHL solver against the profiles and do
not establish physical validation.

## Candidate cases

| Paper | Experimental evidence captured | Readiness |
|---|---|---|
| Cameron and Gohar (1966), *Theoretical and Experimental Studies of the Oil Film in Lubricated Point Contact*, Figs. 6-9 | Film-thickness measurements reported as dimensionless curves for several oils, loads, and dimensionless speed-viscosity groups. Oil No. 3 composition and two viscosity-temperature values are preserved. | Historical curve data are not digitized into points; the exact oil pressure-viscosity law, test temperature, and pointwise operating conditions are not available in the paper record. |
| Gohar (1971), *Oil Film Thickness and Rolling Friction in Elastohydrodynamic Point Contact*, Fig. 3(a) | One specific glass/tungsten-carbide experiment records a second-black central interference fringe at 7.85 lbf and 11.8 in/s, with the reported viscosity and pressure-viscosity coefficient converted to SI. | The fringe order is experimental evidence, but it is not a film-thickness target without the optical calibration and pressure-dependent refractive index. Material Poisson ratios and the density-pressure law are also missing. |
| Ciulli, Draexl, and Stadler (2008), *Film Thickness Analysis for EHL Contacts under Steady-State and Transient Conditions by Automatic Digital Image Processing*, Fig. 13 | At steady `u_e = 0.2 m/s`, `S = 0`, and `F = 15 N`, both transverse and entrainment-direction mid-plane profiles are digitized at 10 micrometer intervals. | Best initial experiment comparison. The paper supplies the ambient fluid properties and operating point for an isothermal pure-rolling case. The profile is a plot digitization, and the paper did not apply a pressure-dependent refractive-index correction. |

The paper supplied as `902c0a478de8d4af_1794221425.pdf` is Gohar's 1971
article; the PDF's 2009 creation metadata is not its publication year.

## Ciulli benchmark requirements

The paper reports the operating data needed to define an initial **steady,
isothermal, pure-rolling** point-contact calculation. Figure 13 provides
digitizable experimental film profiles. Do not defer the initial comparison
for original interferograms, calibration files, or pressure-dependent optical
properties.

### 1. Inputs required by the EHL solver

| Input | Value/source | Use |
|---|---|---|
| Load | `15 N` | Normal force closure |
| Geometry | AISI 52100 ball diameter `10.319 mm`, glass disc treated as locally flat | Equivalent radius `R = 5.1595 mm` |
| Elastic contact scale | Paper reports Hertz radius about `0.1 mm` and peak pressure about `0.7 GPa` | Existing candidate's effective modulus is inferred from exact reported load, ball radius, and rounded Hertz radius; the rounded pressure is a consistency check |
| Entrainment speed | `u_e = (u_s + u_d)/2 = 0.2 m/s` for Figure 13 | Steady Reynolds transport |
| Slide-to-roll ratio | `S = (u_s - u_d)/u_e = 0` | Pure rolling; use the steady model only |
| Fluid at test temperature | `T = 12 deg C`, `eta_0 = 0.497 Pa s`, `rho_0 = 892 kg/m^3`, `alpha = 2.87e-8 Pa^-1` | Isothermal ambient reference properties and pressure-viscosity slope |
| Supply/thermal assumption | Steady, pure rolling; Figure 13 does not quantify inlet starvation/supply level or contact temperature rise | Declare a fully flooded inlet and ambient isothermal properties as initial model assumptions; do not present them as measured results |

These are solver/case data, not optical-correction data. Once the chosen
isothermal constitutive law is declared, no additional pressure-density curve
is needed to run a **constant-density** reference case; `rho_0` is reported.
The current Ciulli `PointContactCase` does not use constant density: it uses
the assumed Dowson-Higginson coefficients documented below.
The paper's reported Hertz radius is rounded, so the modulus inferred from it
is also approximate. Record that approximation rather than treating the
rounded `0.7 GPa` pressure as an exact material property.

**Viscosity model audit:** [`IsothermalFluid`](../src/ehl_solver/constitutive.py)
implements Roelands viscosity, not Barus. Its equation is

```text
eta(p) = eta_0 exp(log(eta_0 / eta_star) * ((1 + p / p_r)^z - 1))
```

with viscosity in Pa s, pressure in gauge Pa relative to cavitation pressure
`p_c = 0`, `eta_star = 6.31e-5 Pa s`, and `p_r = 1.96e8 Pa`. Since
`-log(eta_star)` is approximately `9.67` when eta is expressed in Pa s, this
is mathematically equivalent to the stated conventional Roelands expression.
The implementation calculates the exponent with `log1p`/`expm1` and its
analytic derivative. For Ciulli, the ambient logarithmic slope gives

```text
z = alpha_p * p_r / log(eta_0 / eta_star) = 0.6269990432
```

The shared conversion helper and tests verify `d(log eta)/dp = alpha_p` at
zero gauge pressure, including an independent finite difference. This matches
only the low-pressure slope; it does not show SN500 follows a Roelands curve
throughout the EHL pressure range. Barus,
`eta = eta_0 exp(alpha_p p)`, is an alternative model using the same reported
ambient slope, not a second experimental dataset; do not silently treat
`alpha_p` and `z` as interchangeable or tune either law to fit Figure 13.

**Density is separate:** the candidate uses `rho_0 = 892 kg/m^3` with the
current assumed pressure-dependent Dowson-Higginson coefficients
`a = 0.6e-9/Pa`, `b = 1.7e-9/Pa`. Ciulli does not report those coefficients or
a pressure-density curve. They are model assumptions, not consequences of
`alpha_p`; a constant-density case and this pressure-dependent-density case
must be identified separately in any comparison.

### 2. Experimental data required for comparison

Figure 13, page 10, plots mid-plane film thickness against transverse `y`
(top row) and entrainment-direction `x` (bottom row), for `S = -1.8`, `0`,
and `+1.8`. The two `S = 0`, `u_e = 0.2 m/s` traces are digitizable and are
stored in
[`cases/ciulli_2008_figure13_s0_profiles.csv`](../cases/ciulli_2008_figure13_s0_profiles.csv).
The paper is Ciulli, Draexl, and Stadler, *Advances in Tribology* (2008),
Article 325187, [doi:10.1155/2008/325187](https://doi.org/10.1155/2008/325187).
Coordinates and film thicknesses are in micrometers; the longitudinal
coordinate increases downstream. The table contains 21 points per direction
from `-100` to `+100 micrometers`, sampled every `10 micrometers`.

The profiles were digitized locally from a 400 dpi render of the supplied
paper, calibrating horizontal positions from the labeled `-100, -50, 0, 50,
100 micrometer` ticks and vertical positions from the labeled `0, 0.2, 0.4,
0.6 micrometer` ticks. Values are rounded to `0.01 micrometer`. Estimated
plot-digitization uncertainty is `+/-0.02 micrometer` over smooth sections,
increased to `+/-0.03-0.04 micrometer` at steep edges/constrictions; the
coordinate read-off uncertainty is approximately `+/-2 micrometers`.
Treat these as digitization estimates, not author-supplied confidence
intervals. The digitized central plateau is about `0.46 micrometer`; the
transverse constriction minima are about `0.25 micrometer`, and the
longitudinal outlet minimum about `0.36 micrometer`.

The authors report a film-measurement method uncertainty of about
`0.01 micrometer` on a Hertz contact and estimate additional deviations of
about the same order in EHL due to pressure dependence of refractive index.
The paper explicitly notes that its reported EHL profiles are not corrected
for that pressure-dependent refractive-index effect. That correction affects
the **experimental reference**, not the EHL solver's governing inputs. The
published profiles are therefore suitable as the initial comparison data,
with the uncorrected optical effect disclosed in the uncertainty budget.

### 3. Optional information to improve reference accuracy/uncertainty

These are useful follow-ups, not prerequisites for the initial isothermal
comparison:

- Original interferograms and the authors' calibration tables/files, to
  replace plot digitization and re-evaluate the reported film map.
- Pressure-dependent lubricant refractive-index data, to correct the
  experimental film-thickness reference.
- More precise elastic constants and the unrounded Hertz contact dimensions,
  to reduce uncertainty in the inferred effective modulus.
- Oil-specific pressure-density measurements, if moving beyond the initial
  constant-density assumption.

## Recommended first comparison

Use the Ciulli steady point only: `S = 0`, `u_e = 0.2 m/s`, and `F = 15 N`.
First document Level 1 load, Reynolds, mass, JFO and film-coupling closure.
For Level 2, run at least three meshes at one fixed physical domain, then a
separate domain-expansion study at approximately fixed grid spacing. Report
central/minimum film and main pressure changes in common physical coordinates;
keep domain and mesh effects separate. Only after these case-specific checks
report a Level 3 comparison of the two digitized mid-plane profiles, alongside
the stated digitization and optical-reference uncertainties. Do not tune fluid
or material parameters to fit Figure 13. Keep constitutive-model uncertainty
(including the current slope-matched Roelands/assumed Dowson-Higginson model
versus any separately implemented Barus/constant-density model) separate from
numerical and experimental uncertainty.

Do not use the paper's 0.25/1 Hz transients or nonzero slide-to-roll ratios as
steady-isothermal validation cases. Digitize Cameron-Gohar and Gohar's
figures as useful trend/fringe checks later; their present records still lack
directly comparable dimensional point data.

The Ciulli profiles now provide a reproducible initial experimental
reference. Cameron-Gohar and Gohar remain at benchmark-preparation/trend-check
readiness. No solver-vs-experiment result has yet been evaluated or
established.
