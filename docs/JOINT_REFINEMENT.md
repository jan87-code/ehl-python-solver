# Controlled joint refinement of the 200 N donor candidate

This follow-up separates joint mesh refinement from the earlier directional
experiments. The load, fluid law, domain and transport scheme stay fixed;
no solver equation, acceptance threshold or library default changes.

The square domain is +/-3 dry Hertz radii in x and y. The grid relation is
`nx=1+3(ny-1)/2`, with `ny=4k+1`, so physical `dx/dy=2/3` stays fixed as both
spacings decrease. Arrays are `(ny,nx)`; x is mean entrainment. All public
lengths, pressure, film and force values are SI. The fluid is the registered
200 N Roelands/Dowson–Higginson case, with ambient logarithmic viscosity slope
2.21e-8 Pa^-1. It is not the slightly different 50 N MATLAB-fluid case.

## Reproduction

First produce the accepted 97x145 donor run using the sequence in
[PROTOTYPE.md](PROTOTYPE.md). The joint driver re-audits that prototype seed,
checks its physical parameters/domain, and interpolates it as an **unaccepted**
guess. It preserves the signed offset without force rescaling or a film floor.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python examples/run_joint_refinement.py --seed results/my_200n_fine --output results/my_joint_113 --ny 113
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python examples/run_joint_refinement.py --seed results/my_200n_fine --output results/my_joint_129 --ny 129 --maximum-physical-nodes 30000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python examples/run_joint_refinement.py --seed results/my_200n_fine --output results/my_joint_141 --ny 141 --maximum-physical-nodes 30000
```

The driver default is 20,000 nodes. The larger runs explicitly use the
existing 30,000-node low-level allowance, separately from the prototype CLI.
They have 24,897 and 29,751 nodes respectively. Each performs a full bordered
sparse-factor/known-vector preflight and requires a conservative estimated
peak below 1 GiB. The original 30 Newton corrections and 10 Krylov cycles
remain fixed. Existing output directories are rejected, not overwritten.

Each run retains initial/accepted states, failed nonlinear diagnostics if
present, exact package and driver source, checksums, memory estimates and
original globalization diagnostics. Independent final audits reconstruct
elasticity by direct cell-integrated convolution, recompute SI donor fluxes,
check force, pressure/content/film feasibility and boundary conditions. A
numerically accepted root is not automatically a mesh-converged prediction.

Re-audit and plot the full sequence:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl-cache PYTHONPATH=.:src .venv/bin/python examples/analyze_joint_refinement.py --prototype results/my_200n_coarse --prototype results/my_200n_medium --prototype results/my_200n_fine --joint results/my_joint_113 --joint results/my_joint_129 --joint results/my_joint_141 --output results/my_joint_analysis
```

The analyzer verifies saved joint states against reconstructed raw solver
states, source/state/driver hashes and original final/history audits before
comparing fields. `study.json` records raw minima and locations, spacing,
pressure differences and central film. `joint_refinement.png` shows film,
transverse side-lobe profiles, exit pressure and minimum locations.

## Sampling diagnostic and interpretation

For an interior raw minimum, a local least-squares quadratic is fitted to the
surrounding 3x3 film samples in scaled physical coordinates. Its stationary
point is reported only when the fitted Hessian is positive definite and the
point lies within that stencil. Boundary minima and unsuitable fits return an
explicit status without extrapolation.

This fit is a **diagnostic of sampling the existing discrete film**. It does
not change pressure, content or film, enter Newton acceptance, replace the
raw minimum or estimate total discretization error. An exact off-node
quadratic benchmark verifies its location/value and that input fields remain
unchanged. Nonquadratic side lobes can retain interpolation uncertainty.
Missing a continuous minimum alone would generally raise a sampled minimum;
it cannot explain an excessively low coarse-grid coupled solution.

Pressure norms compare each finer field, linearly interpolated onto coarser
nodes, with the coarse solution; the denominator is the sampled finer norm.
The exit subregion is `0.5<=x/a<=1.5, abs(y/a)<=1.2`. These postprocessing
interpolations never modify a solved field. HD remains an independent film
comparison using E_prime=226 GPa and k=1, not an exact solution or an acceptance
target. No Richardson extrapolation or continuum accuracy is assumed from
these finite meshes.

## Measured results, 2026-10-30

The reused prototype roots and three new joint roots all pass independent
original acceptance. The three new meshes each converge in six corrections.
The 113x169 run stays within the normal 20,000-node budget. The 129x193 and
141x211 runs explicitly select 30,000 nodes. Preflight known-vector relative
residuals are <=3.13e-16; conservative memory estimates are 231/449/492 MB,
and observed peak RSS is 232,416/403,436/407,964 KiB respectively.

| Grid (ny,nx) | Raw minimum [nm] | Central [nm] | x_min/a | abs(y_min)/a |
| --- | ---: | ---: | ---: | ---: |
| 33,49 | 104.878 | 436.669 | 0.1250 | 0.9375 |
| 65,97 | 185.858 | 468.879 | 0.3125 | 0.8438 |
| 97,145 | 205.265 | 466.314 | 0.2083 | 0.8750 |
| 113,169 | 207.699 | 464.072 | 0.1786 | 0.9107 |
| 129,193 | 210.640 | 461.923 | 0.2188 | 0.8906 |
| 141,211 | 210.750 | 460.437 | 0.2000 | 0.9000 |
| HD comparison | 256.801 | 453.800 | — | — |

The last raw minimum changes only +0.0523%, but central film changes -0.3216%.
The finest minimum is 17.93% below HD and central film is 1.46% above HD.
The earlier transverse-only 222 nm result is not part of this fixed-ratio
sequence and must not be treated as its converged limit.

| Refinement | Whole pressure L2 change | Exit pressure L2 change |
| --- | ---: | ---: |
| 33x49 to 65x97 | 6.326% | 10.157% |
| 65x97 to 97x145 | 3.269% | 5.749% |
| 97x145 to 113x169 | 1.857% | 3.739% |
| 113x169 to 129x193 | 1.731% | 3.666% |
| 129x193 to 141x211 | 1.816% | 3.645% |

The fine-mesh minimum agreement alone does **not** establish a converged EHL
solution: pressure differences, especially at the sharp exit, remain visible.
Adjacent meshes have unequal refinement factors, so these percentages are
not a measured convergence order. No mesh-extrapolated error bar is assigned.

Local quadratic diagnostics on the three new meshes give fitted minima
206.287/210.080/210.746 nm, respectively 1.412/0.560/0.004 nm below their raw
minima. Fitted locations cluster near x/a=0.20 and |y|/a=0.90, while the raw
locations move between neighboring nodes. Fit RMS discrepancies are
0.539/0.625/0.329 nm. Some earlier coarse-mesh fits are rejected by curvature
or stencil-location checks; a fit is not universally reliable in these lobes.
These measurements support separating nodal sampling from discrete coupled
solution error. They do not support explaining the remaining roughly 46 nm
HD difference by sampling alone, nor do they prove the cause is constitutive.

The maximum new relative force error is 2.23e-16. Independent scaled mass
residuals are <=1.80e-12, local balance <=5.93e-11 and global balance
<=1.15e-15; all satisfy the unchanged original gates. Direct-vs-FFT film and
displacement differences are <=4.07e-20 m. All 18 accepted corrections satisfy
original directional Armijo and true-linear-residual checks; the maximum
true linear residual is 5.02e-10.

The conclusion remains bounded: discretization explains a substantial part
of the coarse-grid film error, but the available joint meshes do not close
pressure/exit convergence or establish continuum minimum-film accuracy.
Preserve both stencils and the library default. Next work should benchmark
ways to reduce transport diffusion and resolve the exit, then repeat the
joint study; independently audit correlation/constitutive coefficient
conventions before attributing the residual HD difference to either one.

Exact states, source snapshots, audits and preflights are in
`results/joint_200n_{113x169,129x193,141x211}_20261030/`. The final re-audited
comparison and inspected figure are
`results/joint_refinement_complete_20261030/study.json` and
`joint_refinement.png`. All remain local, Git-ignored numerical evidence.

## Verification

`ruff check .` and whitespace checks pass. The full suite reports **609 passed,
62 skipped in 361.10 s**. The five focused joint-study tests also pass on the
final driver (4.71 s), including preservation of rejected Krylov histories
before a later audit raises. Other checks cover fixed grid ratio/node caps,
known off-node quadratic minima, boundary/nonfinite diagnostic rejection,
an actual accepted run, independent elastic/force inconsistency detection
and preservation of existing outputs. No GPU result is claimed.

The local evidence archive is `results/joint_refinement_evidence_20261030.zip`
(28,602,784 bytes, 387 entries). It contains the six source runs, exact
sources/states, drivers, analysis, tests, logs and a SHA256 manifest.
SHA256: `5bc7066fb8eaa8e3013dd76ce2db8bf5f3faf83fd598626c306d1efc7d817e13`.
Source/state/driver hashes, manifest digests and ZIP CRCs were verified.
The archive is local under ignored results; no upload is claimed.

## Follow-up transport benchmark

The isolated bounded-reconstruction study is recorded in
[LIMITED_TRANSPORT.md](LIMITED_TRANSPORT.md). It reduces smooth-source error,
but loaded limiter switches expose a directional-linearization issue that
must be resolved before coupling it to Newton. It supplies no replacement
film result or completed exit-convergence claim.
