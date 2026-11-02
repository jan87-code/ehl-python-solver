@echo off
set PYTHONPATH=.;src
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

mkdir results\limited_mesh_50n_223_final 2>nul

.venv/Scripts/python -u examples/run_limited_transport_path.py \
  --case cases/circular_50n_223.json \
  --output results/limited_mesh_50n_223_final \
  > results/limited_mesh_50n_223_final/run.log 2>&1

echo Run completed at %TIME% %DATE%
