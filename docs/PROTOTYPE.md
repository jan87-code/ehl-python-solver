# Running the circular point-contact research prototype

For a practical input-to-results walkthrough, see [USER_GUIDE.md](USER_GUIDE.md).
The future GUI/release and model-extension sequence is in
[PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md).

The CPU prototype runs from SI JSON inputs, without MATLAB, saved solver
results, PyTorch or GPU drivers. It computes pressure, liquid content, elastic
deformation, film and load-controlled signed offset. Donor capacity is the
prototype's candidate selection; arithmetic capacity remains available as a
reference. The established low-level library default remains unchanged.

This is a steady, smooth, fully flooded, isothermal, circular contact with
pressure-dependent Newtonian viscosity and density. Thermal effects,
roughness, shear thinning and experimental validation of a specific real
lubricant are outside the model. Numerical closure is not a physical error
bound; use the grid/domain comparisons below before interpreting accuracy.

## Start and refine

In the existing workspace environment:

```bash
PYTHONPATH=src .venv/bin/python -m ehl_solver.prototype_cli --case cases/circular_200n.json --output results/my_200n_coarse
```

After installing the project with `python -m pip install -e .`, the equivalent
command is `ehl-run --case cases/circular_200n.json --output results/my_200n_coarse`.
Every output directory must be new; existing results are never overwritten.

The coarse case is a starting point, especially for minimum-film resolution.
Use an independently re-audited accepted run as an **unaccepted** spatial guess:

```bash
PYTHONPATH=src .venv/bin/python -m ehl_solver.prototype_cli --case cases/circular_200n_65x97.json --seed results/my_200n_coarse --output results/my_200n_medium
PYTHONPATH=src .venv/bin/python -m ehl_solver.prototype_cli --case cases/circular_200n_97x145.json --seed results/my_200n_medium --output results/my_200n_fine
PYTHONPATH=src .venv/bin/python -m ehl_solver.prototype_cli --case cases/circular_200n_137x145.json --seed results/my_200n_fine --output results/my_200n_transverse
```

The shipped 50 N sequence is `circular_50n.json`, `circular_50n_65.json`,
`circular_50n_129.json`. Its declared fluid constants match the earlier MATLAB
case; its starting state is generated from Hertz, without that MAT file.

For fixed-spacing directional domain checks, use the accepted 200 N 97x145
run as seed with `circular_200n_domain_x.json` or
`circular_200n_domain_y.json`. Both remain below 20,000 physical nodes.
`circular_200n_97x193.json` instead refines x at the original domain; it is a
resolution check, distinct from the wider-domain case with the same shape.

## Inputs and outputs

All physical inputs are SI. `load_n` is force N; `radius_m` is the equivalent
circular radius m; `contact_modulus_pa` follows the Hertz convention
1/E_contact=sum((1-nu_i²)/E_i). Viscosity is Pa s, density kg/m³,
`entrainment_m_s` is the positive-x mean (u1+u2)/2 and `alpha_per_pa` is the
declared ambient d(log eta)/dp. Roelands pressure and viscosity references
are explicit Pa and Pa s. The model uses the stated Dowson–Higginson density
law and the declared ambient density. No temperature is inferred.

`ny,nx` are odd grid counts in array order `(ny,nx)`. Domain half-widths are
meters. If omitted in Python/JSON, they resolve to three dry Hertz radii;
the emitted `case.json` always records the actual SI dimensions. Shipped
JSON widths are fixed numbers: editing the load alone preserves those widths,
so review domain adequacy as well as resolution. The CPU prototype enforces
20,000 physical nodes and at most 30 Newton corrections / 10 Krylov cycles.
`--max-iterations` can lower the correction budget, never raise it.

- `report.md`: film-height comparison and numerical acceptance summary.
- `fields.png`: unfiltered pressure, contact-region film and convergence plots.
  Film colours have a labeled display cap; raw fields are not capped.
- `solution.npz`: exact dimensionless state plus SI pressure, film, displacement,
  offset and axes, and dimensionless liquid content theta. Arrays are `(ny,nx)`.
- `state_*.npz`: exact observed states, including unaccepted initial guesses.
- `summary.json`: inputs, initialization, rejected/accepted trials, audits,
  physical units/nondimensionalization, hashes, runtime and validation limits.
- `source/ehl_solver/`: exact Python package source snapshot for the run.

`--no-plots` skips rendering. An accepted numerical root exits with code 0;
nonconvergence exits with code 1 and preserves its last state and diagnostics.
Invalid input exits with code 2. A rejected run cannot be used as an accepted
seed. State checksums, physical parameters and transport schemes must match;
there is no silent cross-model or backend fallback.

## Acceptance and validation scope

The starting guess uses force-normalized analytical Hertz pressure, theta=1
and an explicit offset giving a positive sampled film. The HD central estimate
helps construct this one initial guess. No final film height is imposed or
fitted, and Newton trials are not clipped or filtered. Spatial interpolation
and domain embedding are also explicitly unaccepted starting states.

All original Reynolds/mass, complementarity, load, film feasibility, true
linear-residual and genuine Armijo gates are retained. The final audit separately
reconstructs full-source elasticity by direct convolution, recomputes SI
face fluxes and integrates pressure for force. Correlation differences are
reported independently: HD is a physical benchmark, not an acceptance substitute.
A per-run `physical_validation_complete=false` remains explicit because one
run cannot establish mesh/domain independence or broad physical accuracy.

Donor transport is first order in advected mass capacity and adds numerical
diffusion. The 50 N pressure-pattern improvement and tests support using it
as the explicit prototype candidate. The 200 N study still exposes minimum-
film resolution/model uncertainty, so the existing library arithmetic default
is preserved rather than silently promoted. Detailed results and default
selection rationale are in [PROTOTYPE_VALIDATION.md](PROTOTYPE_VALIDATION.md).

The subsequent [joint-refinement study](JOINT_REFINEMENT.md) keeps the domain
and spacing ratio fixed, tracks raw/fitted minimum locations and records the
remaining exit-pressure sensitivity. Its separate example driver requires
explicit selection of the existing 30k allowance for meshes beyond 20k;
the standalone CLI cap is unchanged.
