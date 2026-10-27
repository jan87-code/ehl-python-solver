"""Benchmark the optional HIP elastic-deformation implementation."""

import argparse
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np


def median_milliseconds(function, repeats: int) -> tuple[float, object]:
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ehl_solver.backend import make_backend
    from ehl_solver.elasticity import cell_integrated_kernel, elastic_deformation

    backend = make_backend("hip")
    rng = np.random.default_rng(20261023)
    print(
        "n  gpu_execution_ms  convolution_transfer_ms  elastic_end_to_end_ms"
        "  max_abs_error_m  max_rel_error"
    )
    for size in (129, 257, 513):
        pressure = np.ascontiguousarray(rng.uniform(0.0, 2.0e8, (size, size)))
        spacing = 2.0e-6
        modulus_pa = 113.0e9
        kernel = np.ascontiguousarray(
            cell_integrated_kernel((size, size), spacing, spacing, modulus_pa)
        )

        for _ in range(1):
            backend.native.convolve_same_timed(pressure, kernel)

        convolution_wall_samples = []
        gpu_execution_samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            _, gpu_execution_ms = backend.native.convolve_same_timed(pressure, kernel)
            convolution_wall_samples.append((time.perf_counter() - start) * 1000.0)
            gpu_execution_samples.append(float(gpu_execution_ms))
        convolution_wall_ms = statistics.median(convolution_wall_samples)
        gpu_execution_ms = statistics.median(gpu_execution_samples)
        elastic_wall_ms, elastic_result = median_milliseconds(
            partial(
                elastic_deformation,
                pressure,
                spacing,
                spacing,
                modulus_pa,
                backend=backend,
            ),
            args.repeats,
        )
        reference = elastic_deformation(pressure, spacing, spacing, modulus_pa)
        difference = np.abs(elastic_result - reference)
        max_abs = float(np.max(difference))
        max_rel = float(
            np.max(difference / np.maximum(np.abs(reference), np.max(np.abs(reference)) * 1e-14))
        )
        if not np.allclose(elastic_result, reference, rtol=2e-11, atol=1e-18):
            raise RuntimeError(f"HIP/NumPy mismatch at {size}x{size}")
        print(
            f"{size}  {gpu_execution_ms:.3f}  {convolution_wall_ms:.3f}  "
            f"{elastic_wall_ms:.3f}  {max_abs:.6e}  {max_rel:.6e}"
        )


if __name__ == "__main__":
    main()
