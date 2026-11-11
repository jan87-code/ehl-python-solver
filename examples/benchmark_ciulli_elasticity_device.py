"""Benchmark the existing resident elasticity operator on accepted Ciulli fields.

This is an operator-only check, not a CUDA donor/JFO solve or a GPU EHL result.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.prototype import PointContactCase
from examples.diagnose_ciulli_core_ridge import _read_root


def benchmark(root, repetitions=10):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is unavailable; operator benchmark not run")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    run = _read_root(root)
    case = PointContactCase(**run["summary"]["case"])
    pressure = run["pressure_pa"]
    dx = float(run["x_m"][1] - run["x_m"][0])
    dy = float(run["y_m"][1] - run["y_m"][0])
    cpu = ElasticityOperator(pressure.shape, dx, dy, case.contact_modulus_pa)
    backend = make_backend("torch", "cuda")
    torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    gpu = ElasticityOperator(pressure.shape, dx, dy, case.contact_modulus_pa, backend)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    gpu_pressure = as_float64(pressure, backend)
    torch.cuda.synchronize()
    upload_seconds = time.perf_counter() - start

    reference = cpu(pressure)
    gpu_result = gpu(gpu_pressure)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        reference = cpu(pressure)
    cpu_seconds = (time.perf_counter() - start) / repetitions
    start = time.perf_counter()
    for _ in range(repetitions):
        gpu_result = gpu(gpu_pressure)
    torch.cuda.synchronize()
    gpu_seconds = (time.perf_counter() - start) / repetitions
    start = time.perf_counter()
    actual = gpu_result.cpu().numpy()
    torch.cuda.synchronize()
    download_seconds = time.perf_counter() - start
    delta = actual - reference
    report = {
        "scope": "resident elasticity operator only; not a donor/JFO GPU solve",
        "state_sha256": run["summary"]["state_sha256"],
        "shape_ny_nx": list(pressure.shape),
        "pressure_array_bytes": pressure.nbytes,
        "fft_shape_ny_nx": list(cpu.fft_shape),
        "estimated_operator_workspace_bytes": gpu.estimated_workspace_bytes,
        "repetitions": repetitions,
        "cpu_seconds_per_application": cpu_seconds,
        "gpu_synchronized_seconds_per_application": gpu_seconds,
        "resident_operator_speedup": cpu_seconds / gpu_seconds,
        "gpu_operator_setup_seconds": setup_seconds,
        "upload_seconds": upload_seconds,
        "download_seconds": download_seconds,
        "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
        "rms_error_m": float(np.sqrt(np.mean(delta**2))),
        "linf_error_m": float(np.max(np.abs(delta))),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu_free_before_bytes": free_before,
        "gpu_total_bytes": total,
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "note": (
            "Setup/upload/download excluded from resident per-application timings; "
            "synchronized GPU wall time includes launch and synchronization overhead."
        ),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args(argv)
    report = benchmark(args.root, args.repetitions)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
