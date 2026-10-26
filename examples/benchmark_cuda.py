"""Reproducible CUDA component benchmark; no GPU coupled-solver claim.

Run: python -m examples.benchmark_cuda --output-directory results/cuda_new
Results directories must be new. Warm, synchronized wall medians include
allocation and scalar validation, but exclude setup and transfers unless named.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.device import DeviceFullFilmEvaluator
from ehl_solver.elasticity import ElasticityOperator, cell_integrated_kernel, elastic_deformation
from ehl_solver.monolithic import BorderedReynoldsElasticityProblem
from ehl_solver.reynolds import reynolds_residual


def timed(call, synchronize, warmup, repeats):
    for _ in range(warmup):
        call()
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        call()
        synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000


def relative_error(actual, expected):
    return float(np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-300))


def benchmark_grid(n, backend, warmup, repeats):
    torch = backend.xp
    synchronize = torch.cuda.synchronize
    cpu = make_backend()
    spacing = 1e-3 / (n - 1)
    shape = (n, n)
    coords = np.linspace(-.5e-3, .5e-3, n)
    x, y = np.meshgrid(coords, coords)
    p = 1e8 * np.exp(-(x**2 + y**2) / (1e-4)**2)
    pd = as_float64(p, backend)
    start = time.perf_counter()
    cpu_op = ElasticityOperator(shape, spacing, spacing, 113e9, cpu)
    cpu_setup_ms = (time.perf_counter() - start) * 1000
    synchronize()
    start = time.perf_counter()
    gpu_op = ElasticityOperator(shape, spacing, spacing, 113e9, backend)
    synchronize()
    gpu_setup_ms = (time.perf_counter() - start) * 1000
    expected = elastic_deformation(p, spacing, spacing, 113e9)
    actual = gpu_op(pd).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-20)
    direct_error = None
    if n == 33:
        kernel = cell_integrated_kernel(shape, spacing, spacing, 113e9)
        full = convolve2d(p, kernel)
        direct = full[n - 1:2*n - 1, n - 1:2*n - 1]
        direct_error = relative_error(actual, direct)
        np.testing.assert_allclose(actual, direct, rtol=2e-11, atol=1e-20)

    problem = BorderedReynoldsElasticityProblem(
        coords, coords, .01, .02, 1e8, 5e-4, 20e-6, 21250., 12., -.001,
        compliance_factor=.6, reynolds_residual_scale=100.,
    )
    evaluator = DeviceFullFilmEvaluator(problem, backend)
    cached_cpu = DeviceFullFilmEvaluator(problem, cpu)
    z = np.r_[(p[1:-1, 1:-1] / 1e10).ravel(), 1.]
    v = np.random.default_rng(52).normal(size=z.shape)
    v /= np.linalg.norm(v)
    zd, vd = as_float64(z, backend), as_float64(v, backend)
    pressure, film, _ = problem.state_fields(z[:-1], z[-1])
    h = film / 20e-6
    pressure_d, film_d = as_float64(pressure, backend), as_float64(h, backend)
    def cpu_r():
        return reynolds_residual(pressure, h, problem.dx, problem.dy, 12.)

    def gpu_r():
        return reynolds_residual(
            pressure_d, film_d, problem.dx, problem.dy, 12., backend=backend
        )
    errors = {
        "elasticity_relative_peak": relative_error(actual, expected),
        "direct_relative_peak": direct_error,
    }
    for name, reference, device in (
        ("reynolds", cpu_r(), gpu_r().cpu().numpy()),
        ("nonlinear_residual", problem.residual(z), evaluator.residual(zd).cpu().numpy()),
        ("complete_jvp", problem.jacobian_vector(z, v),
         evaluator.jacobian_vector(zd, vd).cpu().numpy()),
    ):
        errors[name + "_relative_peak"] = relative_error(device, reference)
        # Cancellation near smooth-field zeros needs an absolute scaled bound.
        np.testing.assert_allclose(device, reference, rtol=2e-9,
                                   atol=2e-10 * max(np.max(np.abs(reference)), 1.))
    timings = {}
    for name, cpu_call, gpu_call in (
        ("dc_fft_cached", lambda: cpu_op(p), lambda: gpu_op(pd)),
        ("reynolds", cpu_r, gpu_r),
        ("nonlinear_residual", lambda: cached_cpu.residual(z), lambda: evaluator.residual(zd)),
        ("complete_jvp", lambda: cached_cpu.jacobian_vector(z, v),
         lambda: evaluator.jacobian_vector(zd, vd)),
    ):
        cpu_ms = timed(cpu_call, lambda: None, warmup, repeats)
        torch.cuda.reset_peak_memory_stats()
        gpu_ms = timed(gpu_call, synchronize, warmup, repeats)
        timings[name] = {
            "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": cpu_ms / gpu_ms,
            "gpu_faster": gpu_ms < cpu_ms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    timings["h2d_ms"] = timed(lambda: as_float64(p, backend), synchronize, warmup, repeats)
    timings["d2h_ms"] = timed(lambda: pd.cpu(), synchronize, warmup, repeats)
    timings["original_cpu_residual_ms"] = timed(
        lambda: problem.residual(z), lambda: None, warmup, repeats
    )
    timings["original_cpu_jvp_ms"] = timed(
        lambda: problem.jacobian_vector(z, v), lambda: None, warmup, repeats
    )
    return {
        "shape_ny_nx": shape, "errors": errors, "timings": timings,
        "cpu_elasticity_setup_ms": cpu_setup_ms, "gpu_elasticity_setup_ms": gpu_setup_ms,
        "fft_shape": gpu_op.fft_shape,
        "workspace_guard_bytes": gpu_op.estimated_workspace_bytes,
        "end_to_end_solver": "unsupported: SciPy Krylov/ILU and Newton control remain CPU",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[33, 65, 129, 257, 513])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or any(n < 3 for n in args.sizes):
        parser.error("warmup/repeats must be positive and grid sizes >= 3")
    backend = make_backend("torch", "cuda")  # Explicit; failures never fall back.
    torch = backend.xp
    args.output_directory.mkdir(parents=True, exist_ok=False)
    report = {
        "python": platform.python_version(), "os": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "total_vram_bytes": torch.cuda.mem_get_info()[1],
        "dtype": "float64", "warmup": args.warmup, "repeats": args.repeats,
        "cpu_threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        "cases": [],
    }
    for n in args.sizes:
        gc.collect()
        torch.cuda.empty_cache()
        case = benchmark_grid(n, backend, args.warmup, args.repeats)
        report["cases"].append(case)
        (args.output_directory / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
