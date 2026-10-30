# Standalone prototype and spatial validation, 2026-10-30

The circular isothermal CPU research prototype now runs from SI JSON inputs,
without MATLAB or an accepted checkpoint. Both the 50 N and 200 N donor cases
close the original numerical equations. **Minimum-film accuracy at 200 N is
still limited by discretization uncertainty.** This is a working research
prototype, not a completed physical-validation milestone.

Usage and output definitions are in [PROTOTYPE.md](PROTOTYPE.md). The existing
library arithmetic-capacity default is preserved. The standalone cases select
donor capacity explicitly as the experimental candidate; arithmetic remains
available. No pressure/film filtering, parameter fitting to HD, acceptance
relaxation or default node-budget increase is introduced.

## What was executed

- Added bounded SI case inputs, a Hertz-based unaccepted initial guess,
  accepted-seed re-auditing, mesh interpolation and aligned domain embedding.
- Added a standalone CLI with exact states, source snapshots, hashes, failed
  histories, independently reconstructed elasticity/flux/load audits and plots.
- Reproduced the 50 N mesh sequence without the MATLAB checkpoint.
- Ran 200 N mesh, directional refinement, exit-pressure and fixed-spacing
  directional domain comparisons; re-audited the original arithmetic roots.
- Retained donor as a candidate because the minimum remains mesh sensitive.

All arrays use `(ny,nx)`; mean entrainment is positive x. Runs use float64,
SI public inputs, circular radius 0.015 m, Hertz contact modulus 113 GPa,
ambient viscosity 0.04 Pa s, density 850 kg/m³, mean speed 2 m/s and declared
ambient pressure-viscosity coefficient 2.21e-8 Pa^-1. The 200 N registered
fluid uses Roelands pressure/reference viscosity 1.96e8 Pa / 6.31e-5 Pa s.
The matching 50 N MATLAB-fluid case uses 1.98e8 Pa / exp(-9.67) Pa s instead;
these small constitutive differences are explicit, not mixed between seeds.

Hertz lengths, pressure and approach define the recorded nondimensional
scales. HD uses E_prime=2 E_contact=226 GPa and circular ellipticity k=1.
The coefficient is the declared ambient derivative; correlation/constitutive
coefficient conventions can also affect interpretation. HD is an independent
comparison, not an exact discretization error or acceptance criterion.

## Minimum film and discretization

The 200 N domain is +/-3 dry Hertz radii in each direction unless marked
otherwise. All film values below are unfiltered sampled values in nm.

| Grid (ny,nx) | Minimum | Central | Accepted Newton corrections |
| --- | ---: | ---: | ---: |
| 33,49 | 104.878 | 436.669 | 8 |
| 65,97 | 185.858 | 468.879 | 12 |
| 97,145 | 205.265 | 466.314 | 7 |
| 137,145 | 217.916 | 466.693 | 8 |
| 193,145, extended experiment | 222.014 | 466.883 | 9 |
| 97,193 | 201.230 | 461.598 | 4 |
| 145,193, extended experiment | 214.658 | 462.011 | 5 |
| Hamrock–Dowson | 256.801 | 453.800 | — |

The first five rows refine both axes initially and then y alone. Increasing
y at fixed x raises the sampled minimum; refining x alone lowers it at
ny=97. The last transverse increment changes minimum film by +1.88%, while
central film changes only +0.041%. These are directional sequences, not a
uniform asymptotic convergence study. The two extended minima are still
13.5–16.4% below HD; they do not establish a continuum limit.

The narrow side lobes near |y| about 0.88a make the minimum particularly
sensitive to resolution and the sampled location. Two effects must be
separated: changes in the discrete coupled solution and sampling of its
minimum. Merely missing a smooth continuous minimum would normally make a
sampled minimum too high; it cannot by itself explain these coarse low values.
First-order donor capacity also adds numerical diffusion. Therefore the
observed difference from HD cannot yet be assigned solely to the constitutive
model or correlation. A mesh-extrapolated accuracy claim would be premature.

Next numerical work should refine x and y together at controlled spacing
ratios, resolve the side lobes and exit, and compare both raw minima and
minimum locations. If a less diffusive scheme is investigated, first benchmark
its flux, mass conservation and full coupled Jacobian, preserving both
reference stencils. Do not tune parameters or impose a film floor to match HD.

The default prototype cap remains 20,000 physical nodes. The two 27,985-node
experiments use the existing explicit 30,000-node low-level allowance, separate
from the CLI. Each passed a full bordered sparse-factor preflight with a
known-vector residual below 3.1e-16; conservative memory estimates were below
541 MB and observed peak RSS below 462,000 KiB. Both retained the original
30-correction and 10-Krylov-cycle limits. This is bounded evidence, not a claim
about larger-mesh scalability.

## Pressure and domain checks

For pressure differences, the finer field is linearly interpolated onto the
coarser nodes. The reported norm is ||p_fine_sampled-p_coarse||/||p_fine_sampled||.
This analysis does not modify any solved field. Exit comparisons use
0.5 <= x/a <= 1.5 and |y/a| <= 1.2.

| Refinement | Whole pressure L2 change | Exit pressure L2 change |
| --- | ---: | ---: |
| 33x49 to 65x97 | 6.326% | 10.157% |
| 65x97 to 97x145 | 3.269% | 5.749% |
| 97x145 to 137x145 | 1.477% | 1.895% |
| 137x145 to 193x145 | 1.039% | 1.380% |
| 97x145 to 97x193 | 1.509% | 3.866% |
| 97x193 to 145x193 | 1.577% | 2.529% |

Donor pressure profiles are substantially smoother than the alternating
arithmetic reference. The re-audited arithmetic 97x145 case has minimum/central
film 187.512/396.128 nm, compared with donor 205.265/466.314 nm. This is a
scheme comparison at one mesh, not independent physical validation.

At the 97x145 spacing, extending x from +/-3a to +/-4a (97x193) changes minimum
film +0.0885%, central film +0.3071%, whole pressure 0.1377% and exit pressure
0.2444%. Extending y instead (129x145) changes minimum +0.0515%, central
+0.0099%, whole pressure 0.0322% and exit pressure 0.0569%. Thus tested domain
changes are smaller than mesh effects; additional spacings/domains are still
needed to establish a far-field limit.

## Independent closure and reproducibility

Every accepted standalone result is rechecked for original Reynolds/mass,
complementarity, pressure/theta/film feasibility, boundaries and force. The
separate SI audit recomputes face fluxes and integrates pressure; direct
cell-integrated full-source convolution checks FFT elasticity. Original true
linear-residual and directional-derivative Armijo history checks pass. A
per-run `physical_validation_complete=false` remains explicit.

The independent Hertz-start 50 N sequence gives minimum/central film
241.738/484.818, 300.951/513.601 and 302.699/504.293 nm on 33², 65² and 129².
It reproduces the earlier donor results without the MATLAB checkpoint. HD is
284.150/497.969 nm; that agreement does not resolve the 200 N mesh uncertainty.

Verification: `ruff check .` passes; `pytest -q` reports **604 passed,
62 skipped in 347.70 s**, including 16 prototype tests. Optional unavailable
backends skip gracefully; no GPU performance or accuracy claim is added.
Tests cover an actual cold-start solve, independent seed validation, state
checksum tampering, nonnested interpolation, aligned embedding, incompatible
model rejection, non-overwriting outputs and preserved failed diagnostics.

Raw runs and exact source snapshots are in Git-ignored
`results/prototype_200n_*_20261030` and `results/prototype_50n_*_20261030`.
`results/prototype_plan_analysis_20261030/study.json`, `analyze.py` and
`validation_200n.png` retain the comparison definitions and inspected figure.

The local evidence archive is `results/prototype_validation_20261030.zip`:
49,582,358 bytes, 773 entries, 12 accepted runs plus original arithmetic
references, case files, analysis/reproduction scripts and a SHA256 manifest.
SHA256: `fae2049807c50b830969e306e376be4a188e0ae991468fc9022abd32361b747c`.
Run source/state hashes, archive CRCs and every manifest digest were verified.
The archive remains local under ignored results; it is not included in Git or
claimed to be uploaded elsewhere.

## Follow-up: controlled joint refinement

The fixed-spacing-ratio 200 N sequence is now executed through 141x211;
see [JOINT_REFINEMENT.md](JOINT_REFINEMENT.md) for reproduction, sampling
benchmarks, minimum locations and original independent acceptance. New raw
minima are 207.699/210.640/210.750 nm on 113x169/129x193/141x211. The finest
pair agrees within 0.0523%, but exit-pressure differences remain 3.645%.
Thus film agreement alone does not close mesh convergence. Local sampling
diagnostics are small on the finest meshes; the remaining HD discrepancy
cannot yet be assigned solely to sampling, transport or constitutive laws.

The subsequent [bounded transport benchmark](LIMITED_TRANSPORT.md) validates
an isolated conservative reconstruction, not a new coupled solver. Smooth
accuracy improves, while loaded limiter-switch diagnostics prevent direct
adoption into Newton. Existing prototype predictions/defaults are unchanged;
remaining mesh/physical validation stays open.
