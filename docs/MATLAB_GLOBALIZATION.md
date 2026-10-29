# N33 reference globalization investigation

This investigates the pressure-only 50 N MATLAB-equation reproduction in
[MATLAB_REFERENCE.md](MATLAB_REFERENCE.md). The production pressure/content JFO
solver is unchanged. Neither MATLAB execution parity nor physical EHL
validation is claimed.

## Controlled changes

`examples/investigate_matlab_globalization.py` keeps the source residual,
finite-difference Jacobian, active/Schur/direct/LM direction generators,
pressure projection, strict film floor, convergence tolerances and iteration
budgets. Three diagnostic policies isolate the cause:

- `sequential_armijo`: original candidate order, replacing only the source
  relative-decrease test with genuine directional Armijo.
- `compete_reference`: original decrease test, evaluating all available
  candidates and selecting the accepted trial with the lowest original merit.
- `compete_armijo`: the same competition with genuine directional Armijo.

The original compatibility solver remains the default in
`reproduce_matlab_newton.py`. Its continuation driver now accepts an optional
trial-solver callback, allowing identical continuation schedules and rejection
rollback without modifying the baseline algorithm. The active candidate's
original 0.999 merit-ratio gate is preserved.

The Armijo variant uses the analytic one-sided derivative of
`merit(max(P+lambda*dP,0), H0+lambda*dH0)` at lambda=0+. The derivative includes
full nonlocal elastic deformation, density, viscosity, harmonic face mobility,
upwind transport, diagonal residual normalization, unsmoothed FB and the load
border. At zero pressure, the projected tangent uses max(dP,0); signed FD
probes and their derivative-floor approximation are not used to define this
slope. Exact FB, viscosity-cap and diagonal-floor directional kinks are
handled explicitly. Underflowed harmonic face mobility raises an error rather
than claiming a validated derivative. This remains a specific N33 diagnostic.

## Frozen stalled-state checks

Both previously rejected final states were reloaded from SI NPZ fields and
audited without updating them. At alpha 0.9, the primary direction's analytic
slope is -1.996784 times the old merit, consistent with small-step forward
checks. A full primary step increases merit by a factor of 1.047829e8; its
acceptable step remains tiny under both decrease tests. The LM full-step
merit ratio is 0.02334035. At alpha 0.89890625, the LM full-step ratio is
0.05244918. Thus useful alternatives exist at these exact states and are
hidden by sequential acceptance. This does not justify accepting an extra
21st correction as an unchanged-budget continuation success.

## Alpha 0.9 ablation from the original checkpoint

All variants start at the original alpha=0.8978125 checkpoint, without adding
iterations or relaxing its FB=2e-5 and relative-load=1e-4 gates.

| Policy | Accepted corrections | Final relative FB | Accepted |
| --- | ---: | ---: | --- |
| Original compatibility baseline | 20 | 8.8379898e-5 | no |
| Sequential + directional Armijo | 20 | 8.8379898e-5 | no |
| Competition + original decrease test | 2 | 3.4880022e-6 | yes |
| Competition + directional Armijo | 2 | 3.4880022e-6 | yes |

The two successful policies produce identical saved pressure, film,
displacement, offset and coordinate arrays. The Armijo-only policy also
matches the failed baseline's saved arrays exactly on this environment.
This isolates candidate competition, rather than the stronger decrease test,
as the decisive change for this case.

Competition helps before the visible stall, too. At the first correction,
primary and LM merit ratios are 0.4786114 and 0.0002603977; at the second,
0.9509418 and 0.0018986084. Both selected corrections are LM full steps. The
new policy compares actual nonlinear merit; it does not select a candidate
solely by the linearized prediction or by a different residual.

The independent SI verifier accepts the alpha=0.9 result: minimum film is
4.6490463763e-7 m and relative load error -2.33850e-9. Pressure-only cavitation
still permits a global density-normalized mass-flux defect; this is not a
mass-conserving JFO solution.

## Reproduction

Choose a new output directory for every run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python examples/investigate_matlab_globalization.py results/matlab_newton_reproduction_20261029/source/EHL_BorderedNewton_Alpha_8978_N33.mat results/globalization_NEW --policy compete_armijo --target-alpha 0.9 --stalled-results results/matlab_newton_reproduction_20261029
```

Omit `--target-alpha` for the original adaptive schedule toward alpha=1.
Change only `--policy` to reproduce the ablation. Output includes each final
state in SI, candidate/line-search histories, exact source snapshots and
hashes, and an independent direct-convolution/SI-flux audit for each attempted
alpha. A disagreement between solver and independent acceptance raises an
error. Simulation data remain Git-ignored.

Evaluating more candidates and dense LM systems costs extra work. No speedup,
large-grid scalability, production adoption, mesh convergence or physical
film-height accuracy follows from this small-case algorithmic comparison.

## Full continuation result and remaining limitation

Both competing policies complete six accepted continuation stages and reject
four attempts, stopping at the original minimum-alpha-increment condition.
Last verified alpha is **0.9189712524414063**, compared with the baseline's
0.8978125. Neither reaches full viscosity. Their last accepted physical fields
are identical, with these independently recomputed reference diagnostics:

| Quantity | Last accepted state |
| --- | ---: |
| Relative FB measure | 1.9976588015e-5 |
| Relative load error | -1.0147343632e-8 |
| Minimum film | 4.7831235889e-7 m |
| Maximum saved-film/direct-reconstruction difference | 6.78e-21 m |
| Net density-normalized boundary mass flux | 1.1466644607e-7 m³/s |
| Absolute net / sum of absolute boundary contributions | 0.9695423 |

The last target 0.9196388244628907 is rejected after 20 corrections. Its FB
measure is 2.0430813e-5 for directional Armijo and 2.0426927e-5 for the original
decrease test; both exceed 2e-5. Final failed states can differ slightly even
though the accepted endpoint is identical. No partial failed state is recycled.

A separate frozen-state damping probe at rejected alpha 0.9256469726562501
tries LM mu=1e-15, 1e-13, 1e-11, 1e-9 and 1e-7. Reducing damping yields heavily
backtracked directions and no passing trial; it is not a demonstrated cure
and is not adopted in the controlled policies. In the same state, the largest
FB defects occur in weak-pressure outer/inlet cells, including (iy,ix)=(6,5)
and (26,5), where P≈0.001276 and scaled R≈2.997e-5. Two sampled analytic-vs-FD
Jacobian products differ by about 3.4e-6 and 3.7e-6 in relative L2. These sampled
checks do not establish conditioning, root existence or complete Jacobian
accuracy near every active boundary.

The mass audit changes how these reference successes should be interpreted.
Both the pressure-only model and finite diagonally scaled FB tolerances matter;
it would be incorrect to attribute the entire mass defect to cavitation alone.
Partitioning the dimensional divergence at P=1e-8 gives 89.59% of the original
checkpoint's net defect at nodes below that pressure threshold and 10.41%
above. At the last accepted competing state, **all** the net defect is at nodes
above that threshold. These are signed contributions, not film-error estimates
or a physical liquid-content classification. Their summed divergence agrees
with the independent boundary-flux audit. Thus a passing coarse scaled FB/load
test does not establish Reynolds mass closure even at positive-pressure nodes.
The stricter production mass gates must not be replaced by these reference
stopping tests.

The next conservative comparison should use the matching 50 N case with the
production pressure/content equations and independently recomputed mass and
load closure. Candidate competition is a useful tested hypothesis for that
system, not an adopted production fix. More legacy alpha tuning cannot serve
as physical validation; full viscosity, mass conservation, mesh convergence
and film-height validation remain open.

Validation: 563 CPU tests passed, 62 optional backend/GPU tests skipped
(371.67 s, with concurrent reference experiments). Ruff and whitespace checks
pass. The 21 focused audit/reproduction checks include seven new parameterized
cases for exact projected directional slopes, FB kinks, candidate competition,
non-descent rejection and strict-film Armijo trials. No GPU run or isolated
performance improvement is asserted. The standalone comparison plot was
visually inspected; all attempted states were independently re-audited in SI.

Verified local evidence archive:
`results/matlab_globalization_evidence_20261029_verified.zip`, 1,403,685 bytes,
114 files, SHA256
`ae6be900a50d8b530031f0302543c077cfd9fd65e6606839f4321f92a5f31335`.
Every manifest digest and ZIP CRC passes. It retains the previous baseline,
three fixed-alpha ablations, two full schedules, exact sources, independent
SI re-audits, failed trials, damping probes and the inspected plot. The archive
is Git-ignored; no off-machine backup or performance speedup is asserted.
