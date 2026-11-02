@echo off
:: Run 223x223 (50k) mesh study with single-threaded BLAS for reproducibility

set PYTHONPATH=.;src
set OUTPUT_DIR=results/limited_mesh_50n_223_20261102
set CASE_FILE=cases/circular_50n_223.json

:: Set single-threaded BLAS to avoid contention
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1

:: Clean output directory
if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%"

:: Start the run
echo Starting 223x223 (50k) mesh study...
echo Output: %OUTPUT_DIR%
echo Case: %CASE_FILE%
echo Time: %TIME% %DATE%
echo.

.venv/Scripts/python examples/run_limited_transport_path.py \
  --case %CASE_FILE% \
  --output %OUTPUT_DIR% \
  --max-attempts 24

echo.
echo Run completed at %TIME% %DATE%
