# Everyday use: current research prototype

This guide covers the program that exists today. The planned desktop workflow
and future model choices are in [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md).
A packaged GUI is planned; it is not implemented yet.

Today you put physical inputs in a JSON case file, run one terminal command,
and open the generated report and plots. Routine cases do not require editing
Python or supplying a MATLAB file. The current prototype is steady,
isothermal, circular point contact on CPU. Its physical validation is still
incomplete: numerical convergence alone is not an accuracy guarantee.

## 1. Create your input file

Open a terminal at the repository root. In this workspace:

```bash
cd /home/jan/Projects/EHL-Solver
cp -n cases/circular_200n.json cases/my_case.json
```

Choose a new case name; the copy command preserves an existing destination.
Open `cases/my_case.json` in your editor. Start from one of the shipped
[200 N](../cases/circular_200n.json) or [50 N](../cases/circular_50n.json)
templates rather than changing numerical implementation files. Your case can
also live outside this repository; `--case` accepts its path.

The current JSON is a flat list of input values. JSON uses `null`, not `None`,
and does not allow comments. These are the current field names:

| Field | Meaning and units | Example |
| --- | --- | --- |
| `load_n` | Total normal force, N | `200.0` |
| `radius_m` | Equivalent circular radius, m | `0.015` = 15 mm |
| `contact_modulus_pa` | Hertz contact modulus, Pa | `113000000000.0` = 113 GPa |
| `viscosity_pa_s` | Ambient/reference operating viscosity, Pa s | `0.04` |
| `density_kg_m3` | Ambient density, kg/m³ | `850.0` |
| `entrainment_m_s` | Positive-x mean surface speed (u1+u2)/2, m/s | `2.0` |
| `alpha_per_pa` | Declared ambient d(log viscosity)/dp, Pa^-1 | `2.21e-8` |
| `roelands_pressure_pa` | Roelands pressure reference, Pa | `196000000.0` |
| `viscosity_reference_pa_s` | Roelands law's viscosity reference, Pa s | `6.31e-5` |
| `ny`, `nx` | Odd grid counts; arrays use `(ny,nx)` | `33`, `49` |
| `domain_half_width_x_m` | Half-domain length in x, m; or automatic `null` | `null` |
| `domain_half_width_y_m` | Half-domain length in y, m; or automatic `null` | `null` |
| `transport` | Explicit current numerical candidate/reference | `"donor"` |

Use SI values in the JSON. In particular, entering `113` for a modulus does
not mean 113 GPa. `contact_modulus_pa` follows
1/E_contact=sum((1-nu_i²)/E_i); it is not automatically one body's Young's
modulus. The HD comparison uses E_prime=2*E_contact. Roelands reference
viscosity is distinct from the operating viscosity. Keep the material-law
constants consistent with the fluid being modeled. The shipped 50 N and
200 N cases have slightly different declared reference constants.

For a new load/radius/modulus, setting both domain half-widths to `null` lets
the program start at +/-3 dry Hertz radii. Copied templates otherwise retain
their original fixed physical widths. Automatic bounds are a starting domain,
not proof that boundary effects are negligible. The generated output
`case.json` records the resolved widths in meters.

Keep `"donor"` for the documented prototype workflow. `"arithmetic"` is the
preserved reference. The isolated limited-flux experiments are not selectable
as a production case. The current parser does not accept GUI/thermal/time/
line-contact fields from the future plan.

## 2. Run the case

With the existing workspace environment:

```bash
PYTHONPATH=src .venv/bin/python -m ehl_solver.prototype_cli --case cases/my_case.json --output results/my_case_run_001
```

After a normal project installation, `ehl-run --case cases/my_case.json
--output results/my_case_run_001` provides the same interface. The module
command above works directly in the current workspace.

Every run needs a **new output directory**. For another trial use
`results/my_case_run_002`, preserving the first run for comparison. The program
prints numerical status, minimum/central film in nm, force and mass diagnostics,
and the report path. It creates the output directory itself.

The prototype permits at most 20,000 physical grid nodes, 30 Newton corrections
and 10 Krylov cycles. Arbitrary parameter changes are not guaranteed to converge
or lie in an independently validated range. Do not lower acceptance criteria
to turn a rejected case into an accepted result.

## 3. See the results

In the IDE file explorer, open the new directory under `results/`:

```text
my_case_run_001/
  case.json
  report.md
  fields.png
  summary.json
  solution.npz
  state_000.npz ...
  source/ehl_solver/ ...
```

For a normal completed run:

| Open | What you get |
| --- | --- |
| `report.md` in Markdown preview | Run status, film values, HD differences, force/mass checks and limits |
| `fields.png` in an image viewer | Pressure/film maps, centreline pressure and convergence history |
| `summary.json` | Full inputs, audits, trial/failure diagnostics, units, source hashes and runtime |
| `case.json` | The exact resolved SI inputs used for this run |
| `solution.npz` | Raw numerical arrays for further analysis |

The plot currently displays pressure in GPa, coordinates in mm and film in nm.
Its labeled film colour cap only affects the picture; the saved data are
uncapped. Convergence-iteration plots are numerical iteration histories,
not physical time histories. The current solver has no transient time axis.

Read the status before interpreting a picture:

- `accepted_numerical_root`: the implemented equation/load/feasibility gates
  passed. Check mesh/domain studies separately before using its accuracy.
- `rejected`: inspect the reason and diagnostics. A last saved state can still
  produce a plausible plot; it is not an accepted solution.
- `running` left after an interrupted process: the run has not established
  acceptance. Preserve it as partial diagnostic output.

Exit codes are 0 for numerical acceptance, 1 for nonconvergence, and 2 for
invalid input. Early input errors or unexpected interruptions may leave only
terminal output or partial files. `physical_validation_complete=false` is
currently intentional; it is not changed by a small residual or HD agreement.

## 4. Check resolution before drawing conclusions

The coarse template is a starting point. First run the documented matched
200 N refinement sequence in [PROTOTYPE.md](PROTOTYPE.md) to learn the workflow.
For your own case, copy its input to a second file, keep physical parameters
and bounds identical, and refine `ny,nx` within the node budget. Then run:

```bash
PYTHONPATH=src .venv/bin/python -m ehl_solver.prototype_cli --case cases/my_case_finer.json --seed results/my_case_run_001 --output results/my_case_run_002
```

This command requires an accepted first run and a genuinely compatible finer
case. A transferred field is only a starting guess; the new solve must pass
its own checks. Compare minimum **and central** film, pressure profiles,
minimum location, mass/load diagnostics and domain sensitivity. A stable
minimum alone does not prove the exit pressure is resolved.

Today `--seed` supports spatial refinement or compatible domain embedding
with the same physics. It is not a general warm start for a different load,
speed, radius, fluid or transport scheme. Start such cases without `--seed`.
Keep the old inputs and results intact. Detailed examples and current evidence
are in [PROTOTYPE_VALIDATION.md](PROTOTYPE_VALIDATION.md).

## 5. Access raw data if needed

No Python is needed just to read `report.md` and `fields.png`. For analysis,
NPZ files can be read with NumPy:

```bash
.venv/bin/python - <<'PY'
import numpy as np
with np.load("results/my_case_run_001/solution.npz", allow_pickle=False) as data:
    pressure_pa = data["pressure_pa"]
    film_m = data["film_m"]
    print("Array shape (ny,nx):", pressure_pa.shape)
    print("Sampled minimum film [nm]:", film_m.min() * 1e9)
    print("Peak pressure [GPa]:", pressure_pa.max() * 1e-9)
PY
```

Other keys are `x_m`, `y_m`, `theta`, `displacement_m`, `offset_m` and
`state_dimensionless`. `offset_m` is the signed load-control offset; the actual
gap is `film_m`. Initial and intermediate `state_*.npz` are not automatically
accepted results. Automatic CSV exports, interactive comparisons and a
point-and-click interface belong to the planned release workflow below.

## Planned everyday GUI workflow

The final v1 workflow is intended to be: open a template, enter physical
parameters in labeled unit-aware fields, review domain/mesh, click Run, inspect
pressure/film and validation status, compare runs, and export a reproducible
report/data bundle. The GUI will call the same solver as the command-line
interface. A version-specific walkthrough with screenshots and a worked case
will be part of release acceptance; see [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md).
