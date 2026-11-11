"""Compare resident donor face fluxes and JVP on accepted Ciulli fields.

This is not a GPU JFO solve or a full residual/Jacobian comparison.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.device_donor import DeviceDonorFlux
from ehl_solver.prototype import PointContactCase
from ehl_solver.transport_candidate import donor_flux_direction, donor_fluxes
from examples.diagnose_ciulli_core_ridge import _read_root


def benchmark(root, repetitions=10):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is unavailable")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    run = _read_root(root)
    case = PointContactCase(**run["summary"]["case"])
    h, p, theta = run["film_m"], run["pressure_pa"], run["theta"]
    rng = np.random.default_rng(2026)
    dh = h * 0.001 * rng.standard_normal(h.shape)
    dp = 1e6 * rng.standard_normal(p.shape)
    dt = 0.001 * rng.standard_normal(theta.shape)
    fields = (h, p, theta, dh, dp, dt)
    backend = make_backend("torch", "cuda")
    operator = DeviceDonorFlux(run["x_m"], run["y_m"], case.fluid(), case.entrainment_m_s, backend)
    device_fields = tuple(as_float64(field, backend) for field in fields)
    x, y = run["x_m"], run["y_m"]
    fluid, speed = case.fluid(), case.entrainment_m_s
    results = {}
    for label, cpu_call, device_call in (
        ("flux", lambda: donor_fluxes(x, y, h, p, theta, fluid, speed),
         lambda: operator.fluxes(*device_fields[:3])),
        ("jvp", lambda: donor_flux_direction(x, y, *fields, fluid, speed),
         lambda: operator.direction(*device_fields)),
    ):
        cpu_call()
        device_call()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            reference = cpu_call()
        cpu_seconds = (time.perf_counter() - start) / repetitions
        start = time.perf_counter()
        for _ in range(repetitions):
            actual = device_call()
        torch.cuda.synchronize()
        gpu_seconds = (time.perf_counter() - start) / repetitions
        error = np.concatenate([
            (gpu.cpu().numpy() - cpu).ravel()
            for cpu, gpu in zip(reference, actual, strict=True)
        ])
        norm = np.sqrt(sum(float(np.sum(face**2)) for face in reference))
        results[label] = {
            "cpu_seconds": cpu_seconds,
            "gpu_synchronized_seconds": gpu_seconds,
            "resident_speedup": cpu_seconds / gpu_seconds,
            "relative_l2": float(np.linalg.norm(error) / norm),
            "linf": float(np.max(np.abs(error))),
        }
    return {
        "scope": "face flux and smooth JVP only, not a GPU donor/JFO solve",
        "state_sha256": run["summary"]["state_sha256"],
        "shape_ny_nx": list(h.shape),
        "directions": "seed 2026: dh=0.001*h*N, dp=1e6*N, dtheta=0.001*N",
        "repetitions": repetitions,
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
        "note": "resident times exclude setup and transfers; GPU scalar validation synchronizes",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args(argv)
    report = benchmark(args.root, args.repetitions)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
