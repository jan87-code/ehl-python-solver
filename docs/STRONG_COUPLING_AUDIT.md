# Strong-coupling takeover audit (2026-10-25)

The initial worktree was clean on `develop` at
`e9acb5ec28c44e90fd5f98095536aeea43ff1d4b`. A live `git fetch origin`
confirmed no difference from `origin/develop` before this increment.
Copilot co-authored the following published commits:

- `0df4167`: experimental bordered Newton solver, matrix-free Schur/GMRES,
  dual globalization, adaptive signed-load stepping, tests and visualization.
- `9005041`: analytic discrete JVP, frozen-film ILU/Jacobi/identity options,
  true Krylov and reconstructed-correction checks, adaptive forcing,
  explicit feasible initialization, directional Armijo correction and richer
  failed-state diagnostics. Its documentation explicitly reports strong-case
  stagnation and leaves the milestone open.
- `e9acb5e`: prescribed-film cavitation front and starvation verification;
  this does not add cavitation to the monolithic solver.

Git confirms the code, regressions and documentation were committed and
published. It does not prove which commands were executed before those
commits. Independent baseline verification in this takeover returned
`ruff check .` clean and `pytest -q`: 108 passed, 4 skipped (GPU unavailable).
Existing tests cover the complete JVP, bordered correction, Krylov/Schur
fallbacks, strict feasibility, Armijo acceptance, and load-step retries.
Passing these tests does not establish strong-coupling convergence.

This increment adds adaptive compliance stepping at fixed load, independent
starting/candidate verification, complete trial retention on exhaustion, and
a reproducible stress probe. See PHYSICS.md for measured residuals. The
10 MPa result is a moderate mathematical benchmark. The 1 MPa run fails to
reach full compliance even with adaptive stepping and 40 Newton iterations
per trial; its last converged state is retained rather than reported as a
solution at gamma=1.

Remaining numerical work includes diagnosing the branch near the observed
stall (a fold is a hypothesis, not an established result), evaluating
pseudo-arclength continuation, stronger-coupling/grid/domain studies and
energy diagnostics. Pressure-dependent properties, coupling mass-conserving
cavitation to elasticity and load control, physical load closure and
independent film-height validation remain roadmap work. The Picard and
nested signed-load reference implementations are retained unchanged.
