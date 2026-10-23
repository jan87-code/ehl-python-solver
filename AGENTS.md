# Repository instructions for Codex

## Mission
Build a *new*, research-grade, reproducible 2-D point-contact elastohydrodynamic lubrication (EHL) solver in Python. This is not a mechanical MATLAB translation. See `docs/PHYSICS.md`, `docs/ROADMAP.md`, and `docs/MATLAB_LESSONS.md` before numerical changes.

## Non-negotiable requirements
- Distinguish implemented and validated components from planned ones. Never claim a full EHL solution exists until load balance, Reynolds equation, film coupling, cavitation and convergence tests pass.
- Use SI dimensional quantities at public interfaces; state every nondimensionalization explicitly. Label axis order `(ny, nx)`, entrainment direction, pressures, length and force units.
- Default to `float64`. Preserve a reliable NumPy reference implementation, compare GPU results with it, and do not silently move tensors to CPU in iterative loops.
- Compute linear convolutions correctly using zero-padding; no circular wraparound. Explicitly document coordinate conventions and FFT normalization.
- For each physical extension, first add a benchmark: Hertz dry contact, discrete Reynolds residual, FFT vs direct convolution, load closure, grid convergence and eventually EHL film-height comparisons.
- Check positivity, mass conservation, boundary conditions and physically meaningful convergence diagnostics; do not mask divergence with clamps without documenting their impact.
- Prefer small, reviewed numerical increments to monolithic rewrites. Update docs and add regression tests with each change.
- Do not overwrite results/ or change any baseline without explaining it; do not store credentials or large output datasets in Git.
- Before finishing a change, run `ruff check .` and `pytest -q`, noting any tests that could not run. GPU tests are optional and must skip gracefully if no GPU exists.
- Do not install drivers, modify system packages, force-push to GitHub, or perform destructive operations without explicit permission.

## Layout
`src/ehl_solver/backend.py`: NumPy/PyTorch backend selection and FFT primitives
`src/ehl_solver/convolution.py`: tested zero-padded 2-D linear convolution
`src/ehl_solver/hertz.py`: Hertz analytical reference
`src/ehl_solver/`: add geometry, kernels, Reynolds, rheology, load control and solver modules incrementally
`tests/`: CPU tests mandatory; GPU tests conditional

## First development task
Implement a validated *dry elastic point-contact Hertz / half-space deformation module*, then a unit-tested dimensionless Reynolds residual on CPU. Do not attempt a fully coupled nonlinear EHL solver before both are benchmarked.
