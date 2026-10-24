# Python EHL Solver — research starter

**Status: research foundation, NOT a complete EHL solver.** It includes
analytical Hertz formulas, tested linear convolution and elasticity, a
prescribed-film Reynolds solver, an independent prescribed-film
mass-conserving cavitation prototype, and an experimentally scoped weak
full-film Reynolds-elasticity coupling. Strongly coupled cavitation, load
physical equilibrium, and a fully validated EHL calculation remain future
work. A bracketed root finder exists for signed full-film load closure, but
it is a mathematical verification tool and can include negative-pressure
tensile contributions.

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
- `cavitation.py`: separate fixed-film, stationary JFO pressure/saturation active-set solve with mass-conservation and complementarity diagnostics; not yet coupled to elasticity or load control
- `coupled.py`: prescribed-`h0` weak Reynolds-elasticity iteration with compliance continuation; no cavitation or load controller
- `load_control.py`: safeguarded `h0` root for monotone, bracketed, converged signed full-film load evaluations; mathematical verification only, not physical load control
- `visualization.py`: independent MPa/µm distributions, interpolated cross sections, convergence histories and accepted-iteration snapshots

Run the weakly coupled benchmark and save report figures under
`results/weak_coupling_demo/`:

```bash
python examples/run_coupled_ehl_demo.py
```

The solver does not require plotting or collect snapshots unless
`plot_every` or `snapshot_iterations` is explicitly set. Generated
`results/` output is ignored by Git.

See `docs/PHYSICS.md`, `docs/MATLAB_LESSONS.md`, `docs/ROADMAP.md`.
