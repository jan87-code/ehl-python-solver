"""Bounded native HIP validation, including host-solver component parity.

Supply a manually built extension path; no package, driver, or runtime is
installed. The solver comparison uses GPU convolutions with explicit NumPy
host transfers per call; all Reynolds, complementarity and Krylov work stays
on CPU. It does not constitute a device-resident EHL implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import convolve2d

from ehl_solver.backend import make_backend
from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.elasticity import cell_integrated_kernel, elastic_deformation
from ehl_solver.verification import audit_cavitating_solution


def load_native(path):
    spec = importlib.util.spec_from_file_location("ehl_solver._hip_native", path)
    if spec is None or spec.loader is None:
        raise ValueError("extension path cannot be loaded")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if native.device_count() < 1:
        raise RuntimeError("no HIP device is visible")
    return native


def assert_close(actual, reference, *, atol=2e-12):
    if actual.dtype != np.float64 or not np.all(np.isfinite(actual)):
        raise RuntimeError("HIP result is not finite float64")
    np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=atol)
    scale = float(np.max(np.abs(reference)))
    return {
        "maximum_absolute_error": float(np.max(np.abs(actual - reference))),
        "maximum_error_over_peak": float(np.max(np.abs(actual - reference)) / scale)
        if scale
        else 0.0,
    }


def direct_cases(native, directory, count=12):
    rng = np.random.default_rng(20261025)
    records = []
    for index in range(count):
        shape = tuple(int(n) for n in rng.integers(1, 34, 2))
        kshape = tuple(int(n) for n in rng.integers(1, 16, 2))
        signal, kernel = rng.normal(size=shape), rng.normal(size=kshape)
        if index == 0:
            signal[:] = 0
        if index % 8 == 1:
            signal[:] = 0
            signal[0, -1] = 1.0
        if index % 8 == 2:
            signal = np.asfortranarray(signal)
        if index % 8 == 3:
            signal = signal[::-1, ::-1]
        full = convolve2d(signal, kernel, mode="full")
        y0, x0 = kshape[0] // 2, kshape[1] // 2
        expected = full[y0 : y0 + shape[0], x0 : x0 + shape[1]]
        print(f"starting direct case {index + 1}/{count}: {shape} * {kshape}", flush=True)
        result = native.convolve_same(np.ascontiguousarray(signal), np.ascontiguousarray(kernel))
        records.append({"shape": shape, "kernel_shape": kshape, **assert_close(result, expected)})
        (directory / "direct_cases.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"direct case {index + 1}/{count}: {shape} * {kshape} passed", flush=True)
    invalid = [np.ones((3, 4), dtype=np.float32), np.ones((3, 4)).T, np.ones((0, 4)), np.ones(4)]
    for values in invalid:
        try:
            native.convolve_same(values, np.ones((3, 3)))
        except (ValueError, BufferError):
            continue
        raise RuntimeError("native HIP accepted an invalid buffer")
    return {"cases": records, "invalid_buffers_rejected": len(invalid)}


def median_call(function, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(1000 * (time.perf_counter() - start))
    return statistics.median(samples), result


def benchmarks(native, sizes, repeats):
    rng = np.random.default_rng(20261025)
    records = []
    cpu = make_backend("numpy")
    for size in sizes:
        # Four times explicit complex/input/output storage leaves workspace room.
        raw_bytes = 32 * (3 * size - 2) ** 2 + 8 * (2 * size**2 + (2 * size - 1) ** 2)
        if 4 * raw_bytes > native.device_info()["free_memory_bytes"]:
            raise RuntimeError("conservative workspace/memory gate refuses the benchmark")
        pressure = np.ascontiguousarray(rng.uniform(0, 2e8, (size, size)))
        kernel = np.ascontiguousarray(cell_integrated_kernel((size, size), 2e-6, 2e-6, 113e9))
        native.convolve_same(pressure, kernel)
        events = []
        wall_samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            output, device_ms = native.convolve_same_timed(pressure, kernel)
            wall_samples.append(1000 * (time.perf_counter() - start))
            events.append(float(device_ms))
        cpu_convolution, reference = median_call(
            lambda values=pressure, coefficients=kernel: fft_convolve_same(
                values, coefficients, cpu
            ),
            repeats,
        )
        parity = assert_close(output, reference, atol=1e-18)
        cpu_full, _ = median_call(
            lambda values=pressure: elastic_deformation(values, 2e-6, 2e-6, 113e9), repeats
        )

        def full_gpu_call(n=size, values=pressure):
            coefficients = np.ascontiguousarray(cell_integrated_kernel((n, n), 2e-6, 2e-6, 113e9))
            return native.convolve_same(values, coefficients)

        gpu_full, result = median_call(full_gpu_call, repeats)
        assert_close(result, reference, atol=1e-18)
        row = {
            "shape": [size, size],
            "gpu_event_ms": statistics.median(events),
            "hip_convolution_wall_ms": statistics.median(wall_samples),
            "numpy_convolution_wall_ms": cpu_convolution,
            "numpy_elasticity_wall_ms": cpu_full,
            "hip_elasticity_wall_ms": gpu_full,
            "explicit_buffer_bytes": raw_bytes,
            **parity,
        }
        records.append(row)
        print(
            f"{size}: GPU compute={row['gpu_event_ms']:.3f} ms, "
            f"HIP convolution wall={row['hip_convolution_wall_ms']:.3f} ms, "
            f"NumPy convolution={cpu_convolution:.3f} ms",
            flush=True,
        )
    pressure = rng.uniform(0, 1, (17, 23))
    kernel = np.ones((33, 45))
    native.convolve_same(pressure, kernel)
    before = native.device_info()["free_memory_bytes"]
    for _ in range(100):
        native.convolve_same(pressure, kernel)
    after = native.device_info()["free_memory_bytes"]
    return {
        "grids": records,
        "repeated_calls": 100,
        "free_memory_before_bytes": before,
        "free_memory_after_bytes": after,
        "memory_note": "desktop activity/caches affect free memory; not a leak proof",
    }


def solver_parity(native, directory):
    records = []
    for modulus in (1e7, 1e6):
        problem = CavitatingElasticityProblem(
            np.linspace(-0.5e-3, 0.75e-3, 13),
            np.linspace(-0.5e-3, 0.5e-3, 9),
            20e-6,
            0.01,
            0.02,
            modulus,
            5e-4,
            20e-6,
            21250,
            12,
        )
        start = time.perf_counter()
        reference = solve_cavitating_newton(problem, target_load_n=0.002)
        cpu_wall = time.perf_counter() - start
        events, calls = [], []

        def convolution(signal, kernel, _backend, wall_samples=calls, device_samples=events):
            begin = time.perf_counter()
            result, elapsed = native.convolve_same_timed(
                np.ascontiguousarray(signal, dtype=np.float64),
                np.ascontiguousarray(kernel, dtype=np.float64),
            )
            wall_samples.append(1000 * (time.perf_counter() - begin))
            device_samples.append(float(elapsed))
            return result

        start = time.perf_counter()
        with (
            patch("ehl_solver.cavitating_coupled.fft_convolve_same", convolution),
            patch("ehl_solver.cavitating_newton.fft_convolve_same", convolution),
        ):
            result = solve_cavitating_newton(problem, target_load_n=0.002)
        wall = time.perf_counter() - start
        actual, expected = result.solution, reference.solution
        np.testing.assert_allclose(actual.pressure_pa, expected.pressure_pa, rtol=2e-7, atol=1e-5)
        np.testing.assert_allclose(
            actual.film_thickness_m, expected.film_thickness_m, rtol=2e-8, atol=1e-14
        )
        np.testing.assert_allclose(
            actual.fractional_film_content, expected.fractional_film_content, rtol=2e-7, atol=1e-9
        )
        audit = audit_cavitating_solution(problem, actual)
        if (
            audit.mass_maximum > 1.001e-8
            or abs(audit.load_n - 0.002) > 2e-11
            or audit.maximum_complementarity > 1e-10
            or audit.minimum_pressure_pa < -1e-10 * problem.pressure_scale_pa
            or audit.minimum_saturation < -1e-10
            or audit.maximum_saturation > 1 + 1e-10
            or audit.pressure_boundary_max_pa != 0
            or audit.inlet_saturation_error > 1e-10
            or audit.film_coupling_max_error_m > 1e-17
        ):
            raise RuntimeError("GPU convolution solver parity fails residual/load acceptance")
        np.savez_compressed(
            directory / f"solver_{int(modulus)}.npz",
            pressure_pa=actual.pressure_pa,
            film_m=actual.film_thickness_m,
            theta=actual.fractional_film_content,
            displacement_m=actual.elastic_displacement_m,
            h0_m=actual.central_film_offset_m,
            x_m=problem.x_m,
            y_m=problem.y_m,
        )
        row = {
            "E_star_pa": modulus,
            "shape": problem.shape,
            "target_load_n": 0.002,
            "numpy_h0_m": expected.central_film_offset_m,
            "hip_convolution_h0_m": actual.central_film_offset_m,
            "numpy_solver_wall_s": cpu_wall,
            "hip_convolution_solver_wall_s": wall,
            "gpu_convolution_calls": len(calls),
            "gpu_compute_sum_ms": sum(events),
            "convolution_transfer_inclusive_sum_ms": sum(calls),
            "newton_iterations_numpy": len(reference.history),
            "newton_iterations_hip_convolution": len(result.history),
            "pressure_relative_difference": float(
                np.linalg.norm(actual.pressure_pa - expected.pressure_pa)
                / np.linalg.norm(expected.pressure_pa)
            ),
            "independent_audit": asdict(audit),
        }
        records.append(row)
        print(
            f"solver E*={modulus:g}: {len(calls)} GPU convolution calls, "
            f"CPU wall={cpu_wall:.3f} s, HIP-convolution wall={wall:.3f} s",
            flush=True,
        )
    return records


def main(extension_path, directory, *, largest_grid=1025, repeats=3):
    directory.mkdir(parents=True, exist_ok=False)
    native = load_native(extension_path)
    report = {
        "extension_path": str(extension_path),
        "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        "device": native.device_info(),
        "numpy": np.__version__,
        "repeats": repeats,
        "scope": "native convolutions with explicit host transfers; CPU Reynolds/Krylov",
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report["direct_validation"] = direct_cases(native, directory)
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report["benchmarks"] = benchmarks(
        native, [n for n in (129, 257, 513, 1025) if n <= largest_grid], repeats
    )
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    report["solver_parity"] = solver_parity(native, directory)
    report["torch_imported"] = "torch" in sys.modules
    if report["torch_imported"]:
        raise RuntimeError("native HIP validation unexpectedly imported PyTorch")
    (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--largest-grid", type=int, choices=(513, 1025), default=1025)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    main(
        args.extension_path,
        args.output_directory,
        largest_grid=args.largest_grid,
        repeats=args.repeats,
    )
