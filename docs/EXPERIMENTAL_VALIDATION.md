# Experimental validation candidates

This note records three user-supplied papers as prospective experimental
benchmarks. The machine-readable source records are in
[`cases/experimental_validation_candidates.json`](../cases/experimental_validation_candidates.json).
The regression tests check source-data integrity, unit conversions, and one
reconstructed Hertz input. They do **not** run the EHL solver against the
reported film measurements and do not establish physical validation.

## Candidate cases

| Paper | Experimental evidence captured | Readiness |
|---|---|---|
| Cameron and Gohar (1966), *Theoretical and Experimental Studies of the Oil Film in Lubricated Point Contact*, Figs. 6-9 | Film-thickness measurements reported as dimensionless curves for several oils, loads, and dimensionless speed-viscosity groups. Oil No. 3 composition and two viscosity-temperature values are preserved. | Historical curve data are not digitized into points; the exact oil pressure-viscosity law, test temperature, and pointwise operating conditions are not available in the paper record. |
| Gohar (1971), *Oil Film Thickness and Rolling Friction in Elastohydrodynamic Point Contact*, Fig. 3(a) | One specific glass/tungsten-carbide experiment records a second-black central interference fringe at 7.85 lbf and 11.8 in/s, with the reported viscosity and pressure-viscosity coefficient converted to SI. | The fringe order is experimental evidence, but it is not a film-thickness target without the optical calibration and pressure-dependent refractive index. Material Poisson ratios and the density-pressure law are also missing. |
| Ciulli, Draexl, and Stadler (2008), *Film Thickness Analysis for EHL Contacts under Steady-State and Transient Conditions by Automatic Digital Image Processing*, Fig. 13 | A steady pure-rolling point at 15 N and 0.2 m/s has published transverse and entrainment-direction profiles. Approximate center/minimum ranges were digitized from the figure and are stored with explicit uncertainty. | Best first candidate. The SI operating point is largely specified, but the pressure-viscosity curve, pressure-density law, raw film map, and optical index correction are not. The case is not yet a pass/fail solver benchmark. |

The paper supplied as `902c0a478de8d4af_1794221425.pdf` is Gohar's 1971
article; the PDF's 2009 creation metadata is not its publication year.

## Recommendation

1. Start with the Ciulli steady pure-rolling point only: `S = 0`, `u_e = 0.2
   m/s`, `F = 15 N`. Do not use its 0.25/1 Hz transient runs or nonzero
   slide-to-roll cases as steady-isothermal solver tests.
2. Obtain the original interferogram/profile data and exact glass/steel
   elastic properties, SN500 pressure-viscosity and pressure-density curves,
   and pressure-dependent refractive index. Confirm how the 0.06 micrometer
   silica spacer is handled in the published optical calibration.
3. Freeze and publish the input conversions and optical correction before
   running the solver. Compare both mid-plane profiles, center/minimum film,
   load closure, Reynolds residual, mass balance, boundary/domain sensitivity,
   and at least three grids. Keep model error, measurement uncertainty, and
   numerical error as separate quantities. Do not tune fluid parameters to
   the film target.
4. Digitize Cameron-Gohar and Gohar's figures with uncertainty intervals and
   recover their missing lubricant/material inputs before promoting them from
   historical trend checks. Their dimensionless curves and fringe-order
   observation are useful cross-checks, but not yet standalone dimensional
   validation cases.

Until these data gaps are closed, the papers support benchmark preparation
and qualitative/trend checks only. They do not establish that the current
solver predicts physical EHL film thickness.
