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

The default case is limited to 20,000 physical grid nodes; larger runs require
an explicit supported node budget, up to 100,000. The prototype permits at most
100 Newton corrections and 10 Krylov cycles. Arbitrary parameter changes are
not guaranteed to converge or lie in an independently validated range. Do not
lower acceptance criteria to turn a rejected case into an accepted result.

## 3. See the results

In the IDE file explorer, open the new directory under `results/`:

```text
my_case_run_001/
  case.json
  report.md
  fields.png
  point_contact_..._summary.png
  point_contact_..._summary_profiles.csv
  point_contact_..._summary_pressure_profile.csv
  point_contact_..._summary_film_profile.csv
  point_contact_..._summary_metadata.json
  interactive_3d.html
  summary.json
  solution.npz
  state_000.npz ...
  source/ehl_solver/ ...
```

For a normal completed run:

| Open | What you get |
| --- | --- |
| `report.md` in Markdown preview | Run status, film values, HD differences, force/mass checks and limits |
| `*_summary.png` in an image viewer | Standard 2x2 pressure/film maps and raw centreline pressure/film profiles |
| `*_summary_pressure_profile.csv`, `*_summary_film_profile.csv` | Separate raw SI and display-unit data used for the centreline panels |
| `*_summary_profiles.csv` | Combined compatibility export of both centreline profiles |
| `*_summary_metadata.json` | Plot status, field extrema, display ranges, backend/runtime and full-field archive pointer |
| `fields.png` in an image viewer | Compatibility copy of the standard 2x2 summary |
| `interactive_3d.html` in a browser | Offline, rotatable pressure and film surfaces with hover values |
| `summary.json` | Full inputs, audits, trial/failure diagnostics, units, source hashes and runtime |
| `case.json` | The exact resolved SI inputs used for this run |
| `solution.npz` | Raw numerical arrays for further analysis |

The interactive HTML contains its plotting library and works without a network
connection. It shows the full grid in two independent 3D scenes: unfiltered
pressure in GPa and film thickness in nm, over x/y coordinates in mm (x follows
the positive entrainment direction). Drag to rotate, scroll to zoom, and hover
for values. The title reports the numerical run status and explicitly notes
that physical validation is incomplete.

The standard summary displays the full unfiltered pressure field in GPa,
coordinates in mm and film in nm; it annotates the +x entrainment direction,
global and centreline film minima, central film, tied pressure maxima, a
downstream centreline pressure peak when present (the rightmost local maximum
after the centreline maximum with at least 5% of global `p_max` prominence;
this locator is not a resolution test), load audit, backend, and runtime when
available. The lower panels show unsmoothed profiles at y=0, interpolated
linearly when needed. The pressure field is never clipped. The
film map uses a display-only, contact-sensitive colour range when far-field
values would obscure contact structure; the subplot title states the range.
The centreline film panel uses the same range as a viewport and labels it when
profile values fall outside it. Neither the saved field nor the raw profile
data are altered; the separate profile CSVs retain their full SI values and
display-unit values. The
metadata JSON points to the uncapped full-field archive where the run provides
one. Solver convergence histories remain separate diagnostic outputs and are
not substituted for a physical profile.
`--no-plots` skips the optional interactive 3D plot; it does not suppress the
standard summary.

Read the status before interpreting a picture:

- `accepted_numerical_root`: the implemented equation/load/feasibility gates
  passed. Check mesh/domain studies separately before using its accuracy.
- `rejected`: inspect the reason and diagnostics. A last saved state can still
  produce a plausible plot; it is not an accepted solution.
- If nonlinear failure occurs with a saved last valid state, the standard
  summary is generated from that state and marked `FAILED / NOT AN ACCEPTED
  SOLUTION`; it is a diagnostic, not an accepted result. If no valid state
  exists, only the failure report and metadata are produced.
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

## Reproduce the Ciulli saved-field diagnostics

With the accepted local 513x513 archive available, create a **new** output
directory without running the solver (PowerShell example):

```powershell
$env:PYTHONPATH = "src"
$env:OPENBLAS_NUM_THREADS = "1"
@'
from examples.diagnose_ciulli_core_ridge import _read_root, _write_outlet_locus_outputs
run = _read_root(r"results\ciulli_20261111_donor_513_from129")
_write_outlet_locus_outputs(run, r"results\my_new_outlet_diagnostic", film_range_nm=(275, 1000))
'@ | .\.venv\Scripts\python.exe -
```

The local CSV contains cut-wise coordinates, widths and integrals. The film
color cap affects only rendering. To benchmark the existing CUDA elasticity
and the new donor flux/JVP kernels against NumPy, use new output names:

```powershell
.\.venv\Scripts\python.exe -m examples.benchmark_ciulli_elasticity_device --root results\ciulli_20261111_donor_513_from129 --output results\my_elasticity_operator.json
.\.venv\Scripts\python.exe -m examples.benchmark_ciulli_donor_device --root results\ciulli_20261111_donor_513_from129 --output results\my_donor_operator.json
```

Both require the saved accepted archive and a working PyTorch CUDA/ROCm device.
These are resident component comparisons, not a GPU JFO solver or proof of
mesh convergence. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the
measured scope, limitations, and checksums.

To compare the complete donor/JFO residual and bordered products on saved
257x257 CPU Newton checkpoints, use a fresh output filename:

```powershell
$env:PYTHONPATH = "src"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m examples.benchmark_ciulli_jfo_device --root results\ciulli_20261112_cpu_thread_benchmark_v3\threads_1 --output results\my_jfo_device_operator.json
```

The archive must include the recorded case and checkpoints 0, 3, 6 and 7;
the output contains compact error, finite-difference and component timing
data, and refuses to overwrite an existing file. These measurements do not
run a nonlinear CUDA solver or establish CPU/GPU accepted-root equivalence.
In particular, relative residual error is misleading near a roundoff-level
root; inspect the absolute errors as well. Component timings exclude setup
and field transfers, and are not an end-to-end speedup.

To reproduce the bounded CPU/GPU preconditioner screen on the same saved
257x257 root (checkpoints 0, 3 and 6), use a fresh output filename:

```powershell
$env:PYTHONPATH = "src"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m examples.benchmark_ciulli_jfo_preconditioner --root results\ciulli_20261112_cpu_thread_benchmark_v3\threads_1 --checkpoints 0 3 6 --max-iterations 600 --output results\my_jfo_preconditioner_diagnostics.json
```

The script verifies the accepted-root checksum, compares CPU SuperLU
corrections against the experimental device candidate, records independently
recomputed true residuals, iteration/operator counts, setup/apply timings,
peak CUDA allocations and explicit field-transfer estimates, and refuses to
overwrite an output file. It runs no nonlinear solve. The candidate failed
all three production-state residual gates and is not connected to the EHL
solver; passing small-case parity tests is not evidence that it is a useful
production preconditioner. The checked-in compact measurements are in
[`cases/ciulli_20261112_device_preconditioner_diagnostics.json`](../cases/ciulli_20261112_device_preconditioner_diagnostics.json);
see [VALIDATION_REPORT.md](VALIDATION_REPORT.md) and [ROADMAP.md](ROADMAP.md)
for the measured limitations and current gates.
