# Python EHL Solver — research starter

**Status: research foundation, NOT a complete validated EHL solver.** It includes
Hertz references, tested convolution/elasticity, prescribed-film Reynolds/JFO
solvers, and experimental monolithic cavitation/elasticity with positive-load
closure on documented small isoviscous benchmarks. Independent film-height
validation, broad grid/domain/material coverage and pressure-dependent
properties remain unfinished.

The signed full-film Picard, nested load and bordered Newton solvers remain
mathematical references. Their negative-pressure tensile contributions must
not be interpreted as physical cavitation-corrected EHL loads.

## Quick start (Linux Mint / Ubuntu)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
ruff check .
pytest -q
ehl-check --backend numpy
code .
```

## GPU setup
First verify compatibility of your specific GPU model, driver, kernel and PyTorch wheel:
- PyTorch official selector: https://pytorch.org/get-started/locally/
- AMD ROCm system requirements: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
- NVIDIA driver documentation: https://docs.nvidia.com/cuda/

Inside an activated `.venv`, install the matching PyTorch wheel with the **exact command from the PyTorch selector**, choosing either CUDA (NVIDIA) or ROCm (AMD). Do not blindly install both builds into one environment. For CPU-only PyTorch, use the CPU option.

Check with:
```bash
python -m ehl_solver.cli --backend torch --device auto
python -m ehl_solver.cli --backend torch --device cuda
```

On AMD ROCm, `cuda` is **still** the PyTorch device name. The CLI explicitly reports if the device is AMD ROCm or NVIDIA CUDA where detectable. If no working supported GPU, `auto` uses CPU.

### Verified CUDA components (2026-10-26)

CUDA is optional and explicitly selected. On Windows, activate with
`.venv\Scripts\Activate.ps1`; on Linux use `source .venv/bin/activate`.
Install the CPU project with `python -m pip install -e ".[dev]"`, then install
the appropriate PyTorch wheel separately using the selector above. The tested
environment uses Python 3.12.9, PyTorch 2.11.0+cu128, CUDA runtime 12.8 and
NVIDIA driver 596.71 on an RTX A2000 Laptop GPU with 4 GiB VRAM. No CUDA
toolkit build is required by these operators. CPU execution needs no PyTorch.

```python
from ehl_solver.backend import as_float64, make_backend
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.reynolds import reynolds_residual

gpu = make_backend("torch", "cuda")  # Raises if unavailable; no error fallback.
elasticity = ElasticityOperator((ny, nx), dx_m, dy_m, e_star_pa, gpu)
p_gpu = as_float64(pressure_pa, gpu)  # Upload once; pressure in Pa.
w_gpu = elasticity(p_gpu)            # Gap-increasing displacement in m.
# Reuse elasticity and device pressure for repeated evaluations.
r_gpu = reynolds_residual(
    p_gpu / pressure_scale_pa, film_gpu_m / film_scale_m,
    dx_m / length_scale_m, dy_m / length_scale_m, reynolds_parameter,
    backend=gpu,
)
```

Arrays use `(ny,nx)`, with entrainment along x. The residual preserves the
conservative centered full-film stencil and evaluates interior cells; it
does not impose boundary conditions. `parabolic_point_contact_film` also
accepts `backend=gpu`. Elasticity caches a float64 kernel spectrum with a
conservative memory guard. Scalar validity checks synchronize CUDA; resident
fields are not copied back to CPU.

`DeviceFullFilmEvaluator(problem, gpu)` from `ehl_solver.device` snapshots a
CPU bordered problem for device-resident nonlinear residual and analytic
complete JVP evaluation. Recreate it after changing compliance or geometry;
the resident continuation driver manages target-load changes explicitly.
The optional `device_krylov`, `device_bordered` and `device_newton` modules
now provide resident float64 GMRES, frozen-film preconditioning, Schur/full
bordered recovery, dual-line-search Newton and adaptive signed-load stepping.
The NumPy/SciPy solver remains the default reference. JFO cavitation and
physical positive-load EHL validation are not covered by this GPU path.

```python
from ehl_solver.device import DeviceFullFilmEvaluator
from ehl_solver.device_newton import solve_device_full_film_load, continue_device_load

evaluator = DeviceFullFilmEvaluator(problem, gpu)  # problem uses SI geometry/scales
# initial_state = row-major interior dimensionless P followed by h0/h_ref.
state_gpu = as_float64(initial_state, gpu)  # One initial upload.
solution = solve_device_full_film_load(evaluator, state_gpu)
path = continue_device_load(evaluator, solution, next_signed_load_n)
```

Complete signed full-film calculations through 513x513 have converged on
CUDA and passed independent CPU Reynolds/load checks. This demonstrates a
mathematical GPU Newton calculation, not a complete physically validated GPU
EHL solver. All iterative fields and Krylov bases stay resident; Python
receives scalar decisions/diagnostics. GPU failures never fall back to CPU.
The DST preconditioner inverts an explicitly defined mobility-scaled
Dirichlet Laplacian surrogate; it does not emulate SciPy ILU.

```powershell
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
python -m examples.benchmark_cuda --output-directory results/cuda_new --warmup 5 --repeats 15
python -m ruff check .
python -m pytest -q
```

On Linux, set the thread variables with `export`. Output paths must be new.
At 513x513, cached DC-FFT measured 84.802 ms NumPy versus 8.990 ms CUDA
(9.43x); Reynolds measured 20.910 versus 1.256 ms (16.65x).
At 33x33 all measured operations were slower on CUDA. Maximum observed
elasticity difference normalized by CPU peak was 8.46e-16; allocated/reserved
PyTorch peaks were 166.94/202 MiB across the benchmark. See `docs/PHYSICS.md`
for all grids, transfer/setup costs, independent checks and limits.
GPU tests skip without PyTorch/CUDA; failures on an available GPU fail tests.

For Newton/Krylov benchmarks (same thread settings), use:

```bash
python -m examples.benchmark_device_krylov --compare --sizes 33 65 129 --repeats 3 --max-krylov-steps 640 --output-directory results/newton_cuda_new
```

At 65x65/129x129 the measured setup-inclusive speedups over the preserved
NumPy/SciPy solver are 7.32x/20.87x; at 33x33 CUDA is slower. These include
caching and preconditioner changes. Use `--matched-torch-cpu` to compare the
identical cached algorithm on CPU. Both solvers use 1e-10 nonlinear/load
tolerances in this benchmark. Larger CPU attempts are bounded and failed
linear solves are reported without speedup claims. See PHYSICS.md for the
matched comparison, memory, iteration counts and all remaining limits.

With identical algorithms at 513x513, PyTorch CPU/CUDA solve times are
6.947/2.103 s (3.30x), or 2.78x including device setup and transfers in a
warmed process. The first tested setup-inclusive win for that comparison is
257x257; the original reference comparison first wins at 65x65. These are
workload-specific observations. Peak allocated/reserved memory is about
330/390 MiB. CPU-only operation still needs no PyTorch; Linux GPU/ROCm
execution and broad strong-coupling robustness remain unverified.

## GitHub
```bash
git init
git add .
git commit -m 'Initialize Python EHL solver foundation'
git branch -M main
# Create an EMPTY repository at github.com/new, then replace USERNAME:
git remote add origin git@github.com:USERNAME/ehl-python-solver.git
git push -u origin main
```
Configure SSH authentication with GitHub beforehand, or replace the SSH URL with the HTTPS remote and sign in via Git Credential Manager. No GitHub repo is created by downloading this zip.

## VS Code / Codex
Install VS Code Python, Pylance, Ruff and OpenAI Codex extensions. Select `.venv/bin/python` as interpreter. Codex loads `AGENTS.md` at the repository root. Start with `docs/ROADMAP.md` and ask for one milestone at a time.

## Modules
- `backend.py`: minimal NumPy/PyTorch backend selection; GPU is optional
- `convolution.py`: zero-padded full linear convolution with same-shape centered output
- `hertz.py`: independent analytical formulas using *effective plane-strain contact modulus* `E_star` (Pa)
- `reynolds.py`: dimensionless conservative residual and prescribed-film full-film solve, with parabolic verification, physical diagnostics, and sparse direct/iterative comparison; no cavitation model
- `cavitating_continuation.py`: bounded positive-force stepping with verified warm starts, retries and retained failed candidates
- `cavitating_newton.py`: experimental monolithic JFO pressure/content/elasticity and positive-load/h0 Newton solve, analytic JVP, verified Schur/GMRES, feasibility and Armijo safeguards; numerical verification only
- `cavitating_coupled.py`: SI prescribed-offset JFO/elasticity CPU reference; convergence checked on exact coupled film; no prescribed-load control
- `parabolic_reference.py`: independent SI rigid infinite-width parabolic cavitation reference with finite inlet, smooth rupture and conserved flux; not a point-contact EHL solution
- `cavitation.py`: separate fixed-film, stationary JFO pressure/saturation active-set solve with local/global mass-conservation, front-refinement, complementarity and uniform inlet-starvation diagnostics; not experimentally validated or coupled to elasticity/load control
- `coupled.py`: prescribed-`h0` weak Reynolds-elasticity iteration with compliance continuation; no cavitation or load controller
- `load_control.py`: safeguarded `h0` root for monotone, bracketed, converged signed full-film load evaluations; mathematical verification only
- `monolithic.py`: matrix-free bordered Reynolds-elasticity Newton solve with an analytic discrete JVP, Schur/GMRES, frozen-film ILU/Jacobi/identity preconditioners, dual line search with tighter-Krylov recovery, adaptive signed-load/compliance continuation, and small-grid Jacobian-spectrum / experimental pseudo-arclength diagnostics; mathematical verification only
- `pseudo_arclength.py`: bounded dense predictor-corrector continuation prototype with tangent tracking, feasibility, Armijo globalization, retries, and turning-point records; small-system diagnostic only
- `turning_point.py`: bounded tangent-sign bracket localization using the existing dense corrector; retains failures and verifies bracket/tangent tolerances; mathematical branch diagnostic only
- `verification.py`: bounded independent direct-convolution and face-flux audits of SI signed full-film solution fields; no FFT state reconstruction or nonlinear residual calls
- `visualization.py`: independent MPa/µm distributions, interpolated cross sections, Picard/Newton/load/compliance continuation histories (including line-search and inexact-Newton diagnostics), conditioning and branch plots, Krylov/preconditioner comparisons, coupling-ratio diagnostics and accepted-state fields

Run the weakly coupled benchmark and save report figures under
`results/weak_coupling_demo/`:

```bash
python examples/run_coupled_ehl_demo.py
```

Compare the Picard, nested load-control and monolithic Newton formulations,
run adaptive load continuation, and generate Newton/load-history PNGs with:

```bash
python examples/run_monolithic_load_control_demo.py
```

Run the non-plotting compliance stress probe for weak, moderate, 1 MPa, and
bounded 100 kPa cases:

```bash
python -m examples.run_compliance_continuation
```

It reports independently recomputed residuals, Krylov work, line-search
rejections, film/pressure extrema, deformation and continuation-stage status.

Audit the small-grid signed full-film Jacobian, compare fixed-compliance
continuation against the experimental pseudo-arclength branch tracker, and
save conditioning/branch/field figures with:

```bash
python examples/run_pseudo_arclength_diagnostic.py
```

The diagnostic refuses to overwrite
`results/monolithic_pseudo_arclength_audit/`; choose a new path with
`--output-directory` to keep earlier figures. For the 1 MPa exploratory case, the
scaled Jacobian condition estimate rises to `5.33e4` at the last fixed-gamma
state. Pseudo-arclength detects a tangent reversal near gamma `0.179191`, but
does not reach full compliance and follows the branch back down within its
20-step budget. This is evidence about the computed mathematical branch only;
it does not validate strong-coupling or physical EHL behavior.

Reproduce the prescribed-film cavitation wedge, front-refinement, and inlet
starvation verification with:

```bash
python examples/run_cavitation_verification.py
```

The monolithic example and its regression tests establish only mathematical
signed full-film convergence on the documented weak-coupling case. Negative
pressure remains in the equations; cavitation, physical load validation,
and broader strong-coupling branch continuation are not validated. The demo
also compares frozen-film preconditioners, reports a moderate-compliance
reference and records the 1 MPa exploratory line-search failure rather than
presenting it as a converged solution.

The standalone cavitation prototype conserves mass and demonstrates
mesh-refined pressure behavior at selected rupture/reformation fronts and
reduced pressure under a uniform starved inlet. These are dimensionless
prescribed-film checks. Separate CPU cavitation/elasticity and positive-load
solvers pass limited numerical benchmarks; independent physical validation
and domain-independent film heights remain open.

Reproduce the annotated CPU Hertz, analytical cavitation, rigid far-field,
loaded-grid and separate-boundary studies, including SI NPZ/JSON data:

```bash
python -m examples.run_scientific_validation --output-directory results/scientific_validation_new
```

Use one BLAS/OpenMP thread and a fresh output directory. See
[`docs/VALIDATION_REPORT.md`](docs/VALIDATION_REPORT.md) for the equation/unit
audit, exact inputs, independent checks, quantitative discrepancies,
Linux/PowerShell reproduction and the remaining classical-EHL requirements.
This runner requires no PyTorch and preserves the signed full-film CUDA path.

The follow-up mesh/domain/boundary/elasticity error budget is reproduced by:

```bash
python -m examples.run_farfield_budget --output-directory results/farfield_budget_new
```

It retains failed candidates and uses graded prescribed-film far fields plus
unchanged uniform coupled references. Domain independence remains open; see
section 5 of the validation report for the declared criteria and discrepancies.

Analyze the retained inlet fluxes, explain the load changes, and reproduce
the bounded conditioning and independently matched-boundary checks with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_flux_conditioning \
  --input-directory results/farfield_budget_new \
  --output-directory results/flux_conditioning_new
```

Use the fresh output of the preceding far-field runner as input. The analysis
preserves raw rejected states and generates 17 figures plus SI data/provenance.
It accounts for the 1.17% rigid and 13.3% coupled load changes and reduces one
failed physical-flux defect below the original threshold. The largest case
still fails; matched boundary tests verify restricted analytical/numerical
references, and a standalone point-contact outer model remains open. See
section 6 of the validation report. Optional wider-residual correction is
explicitly unavailable on platforms where `longdouble` equals float64;
the original float64 solver remains usable.

The solver does not require plotting or collect snapshots unless
`plot_every` or `snapshot_iterations` is explicitly set. Generated
`results/` output is ignored by Git.

See `docs/PHYSICS.md`, `docs/MATLAB_LESSONS.md`, `docs/ROADMAP.md`.

The finite point-reservoir exterior response can be reproduced independently:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_exterior_response --output-directory results/exterior_response_new
```

It solves exterior pressure/content and accounts for exterior load and
selected-target elastic sources. Numerical cuts match an independent larger
rigid reference; the physical reservoir remains fixed. See section 7 of the
validation report for the formulation, limits and outstanding mesh/coupling work.

The separate fixed-reservoir outer-mesh budget is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_exterior_mesh --output-directory results/exterior_mesh_new
```

It independently refines uniform references and graded exteriors at identical
physical reservoir bounds, retains field-error failures, and audits supply,
fronts and exterior elastic sources. The finest graded load discrepancy is
0.129% against the finest uniform reference. Reference mesh uncertainty and
remaining coupled/infinite-domain limitations are documented in report section 8.

Preparatory exterior film-derivative and all-inner elastic-source checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_exterior_coupling_audit --output-directory results/exterior_coupling_audit_new
```

These compare fixed-active-set film directions with paired nonlinear rigid
solves and restrict a freshly rerun uniform coupled root at its frozen film.
They verify every retained target's exterior-source contribution; they do
not introduce a new nonlinear coupled solver. See report section 9.

A separate uniform finite-reservoir exterior/Picard coupling experiment is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_uniform_exterior_coupling --include-fine \
  --output-directory results/uniform_exterior_coupling_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.reaudit_exterior_coupling \
  --input-directory results/uniform_exterior_coupling_new \
  --output-directory results/uniform_exterior_coupling_audited_new
```

It cold-starts independently of the full monolithic references, updates film
at inner and exterior nodes from every pressure source, and retains failures.
The optional fine case stays within the 20,000-node exterior budget. This is
uniform-grid prescribed-offset verification; graded coupling and physical
infinite-domain accuracy remain open. See validation report section 10.
The fine producer reference fails the stricter local-flow audit despite
passing the original solver defaults. The reaudit preserves that rejection
and tightens the independent reference with one additional Newton correction;
candidate fields are unchanged. Coupled load changes decrease to 1.58%
under refinement, so the 1% mesh target remains open.

Standalone SI pressure-dependent fluid laws and derivatives are documented
in `docs/CONSTITUTIVE.md`. Generate their illustrative verification curves:

```bash
python -m examples.run_constitutive_verification --output-directory results/constitutive_new
```

Verify conservative pressure-dependent prescribed-film flow:

```bash
python -m examples.run_property_reynolds_verification --output-directory results/property_reynolds_new
```

It includes an independent pressure-driven continuum reference and cavitating
point grids. Physical fluxes are now mass flows; point mesh accuracy remains
open. Details and dimensionalization are in `docs/CONSTITUTIVE.md`.

Run initial pressure-dependent density/viscosity **coupled** CPU verification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_property_coupling_verification --include-fine \
  --output-directory results/property_coupling_new
```

This cold-starts elastic/material continuation, compares a small case with
an independent complete nonlinear reference, and reports original mass,
load, deformation and mesh errors. Density and viscosity are active in the
coupled solve; classical benchmark agreement remains subsequent work.

Prescribe a positive force for the pressure-dependent coupled model with
`ehl_solver.property_load.solve_property_at_load`. It returns the solved SI
offset, coupled fields and independently audited force evaluations. Provide
an explicit positive offset bracket on a fixed physical grid; failed inner
solves or out-of-bracket targets raise with retained diagnostics.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ehl-mpl PYTHONPATH=src \
  python -m examples.run_property_load_verification --include-refinement \
  --output-directory results/property_load_new
```

This verifies known-offset recovery and 0.10 N force closure. Negative-offset/
strong-load branches and mesh-resolved physical validation remain open;
see `docs/CONSTITUTIVE.md` and the latest validation-report section.

The user's circular 200 N case is registered in
`ehl_solver.benchmark_200n.Circular200N`, with explicit 113 GPa dimensionless
conventions and declared lubricant assumptions. Run
`python -m examples.run_200n_readiness --output-directory results/200n_new`
(with `PYTHONPATH=src` and one BLAS/OpenMP thread). This is a bounded readiness
assessment; the current positive-offset probes do not achieve 200 N.
See section 15 of `docs/VALIDATION_REPORT.md` for the retained failure,
mesh evidence and signed-offset continuation required next.

A separate warm signed-offset Newton reference is available in
`ehl_solver.property_warm` (maximum 400 interior nodes). Reproduce its
bounded circular branch with `python -m examples.run_signed_200n
--output-directory results/signed_new`, using `PYTHONPATH=src` and one
BLAS/OpenMP thread. It crosses zero rigid offset with positive total film;
the verified branch reaches 8.97 N, while 200 N closure remains open.
See section 16 of `docs/VALIDATION_REPORT.md` for independent verification,
retained retries and the current computational/resolution limits.

`ehl_solver.property_load_newton` adds matrix-free load-directed continuation
with sparse preconditioning and verified bordered corrections. The bounded
initial coarse/refined study reached 100.442 N. An explicitly selected
sparse tangent preconditioner now closes **200 N on 33x49 nodes**; mesh and
physical accuracy remain open (section 18). Matched-force comparisons, preserved warm inputs,
failed states and reproduction commands are in section 17 of
`docs/VALIDATION_REPORT.md`. Run `examples.run_load_directed_200n` with the
specified prior source archives, followed by `examples.audit_load_directed`
to independently check saved fields and globalization histories.

Explicit nested-grid initialization and safeguarded active repair now close
200 N on **65x97 nodes** as well. Initialization alone is never accepted.
Film changes remain large and pressure variation persists; see section 19
of `docs/VALIDATION_REPORT.md` and `examples.run_load_refinement` for the
matched-force evidence, retained failures and bounded reproduction.
