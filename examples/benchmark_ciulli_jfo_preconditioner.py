"""Compare an experimental device JFO preconditioner with saved CPU corrections.

The benchmark runs no nonlinear solve and writes only compact timing and
residual data. A failed device Krylov gate is retained as a result, not masked.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.cavitating_newton import solve_cavitating_correction
from ehl_solver.device_jfo import DeviceDonorJFO
from ehl_solver.device_jfo_krylov import (
    DeviceJFOKrylovError,
    solve_device_jfo_correction,
)
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from examples.diagnose_ciulli_core_ridge import _read_root


def _factor_diagnostics(model, state, repetitions=3, solves=20):
    from scipy.sparse.linalg import splu

    start = time.perf_counter()
    matrix = model.frozen_matrix(state)
    assembly_seconds = time.perf_counter() - start
    field = matrix[:2 * model.n, :2 * model.n].tocsc()
    factor_times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        factor = splu(field)
        factor_times.append(time.perf_counter() - start)
    rhs = np.ones(2 * model.n, dtype=np.float64)
    factor.solve(rhs)
    start = time.perf_counter()
    for _ in range(solves):
        factor.solve(rhs)
    apply_seconds = (time.perf_counter() - start) / solves
    return {
        "matrix_assembly_seconds": assembly_seconds,
        "matrix_shape": list(field.shape),
        "matrix_nnz": int(field.nnz),
        "factorization_seconds": factor_times,
        "factorization_median_seconds": float(np.median(factor_times)),
        "factor_application_mean_seconds": apply_seconds,
        "superlu_factor_nnz": int(factor.L.nnz + factor.U.nnz),
    }


def benchmark(root, checkpoint_numbers, *, max_iterations=600):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is unavailable")
    run = _read_root(root)
    case = PointContactCase(**run["summary"]["case"])
    probe = LimitedLoadProbe(case, transport_fraction=0)
    backend = make_backend("torch", "cuda")
    evaluator = DeviceDonorJFO(case, backend)
    results = []
    for number in checkpoint_numbers:
        path = Path(root) / "checkpoints" / f"checkpoint_{number:04d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["case_json"])) != run["summary"]["case"]:
                raise ValueError(f"checkpoint case mismatch: {path}")
            state = archive["state_dimensionless"].copy()
        if not probe.feasible(state):
            raise ValueError(f"checkpoint is infeasible: {path}")
        active_full = state[:probe.n] >= state[probe.n:2 * probe.n]
        model = LimitedLoadLinearization(
            case,
            state,
            np.zeros(probe.size),
            preconditioner="local",
            active_full=active_full,
            transport_fraction=0,
        )
        cpu_preconditioner = _factor_diagnostics(model, state)
        start = time.perf_counter()
        cpu = solve_cavitating_correction(
            model,
            state,
            maximum_krylov_cycles=10,
            gmres_restart=60,
            krylov_rtol=1e-9,
        )
        cpu_seconds = time.perf_counter() - start

        device_state = as_float64(state, backend)
        device_active = torch.as_tensor(active_full, device=backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        torch.cuda.synchronize(backend.device)
        start = time.perf_counter()
        try:
            gpu = solve_device_jfo_correction(
                evaluator,
                device_state,
                device_active,
                rtol=1e-9,
                restart=60,
                max_iterations=max_iterations,
            )
        except DeviceJFOKrylovError as error:
            torch.cuda.synchronize(backend.device)
            preconditioner = error.preconditioner
            results.append({
                "checkpoint": number,
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "active_full_cells": int(active_full.sum()),
                "cpu": {
                    "seconds": cpu_seconds,
                    "true_linear_residual": cpu.true_linear_residual,
                    "gmres_iterations": cpu.linear_iterations,
                    "preconditioner": cpu_preconditioner,
                },
                "device_candidate": {
                    "status": "failed_true_residual_gate",
                    "reason": error.krylov.reason,
                    "seconds": time.perf_counter() - start,
                    "setup_seconds": error.setup_seconds,
                    "gmres_seconds": error.solve_seconds,
                    "iterations": error.krylov.iterations,
                    "operator_applications": error.operator_applications,
                    "preconditioner_applications":
                        preconditioner.application_count,
                    "true_relative_residual": error.krylov.true_relative_residual,
                    "true_residual_history": error.krylov.true_residual_history,
                    "estimated_preconditioned_history_tail":
                        error.krylov.estimated_preconditioned_history[-10:],
                    "preconditioner_device_bytes": preconditioner.estimated_bytes,
                    "pressure_mg_levels": len(
                        preconditioner.pressure_multigrid.levels
                    ),
                    "gpu_peak_allocated_bytes":
                        torch.cuda.max_memory_allocated(backend.device),
                    "gpu_peak_reserved_bytes":
                        torch.cuda.max_memory_reserved(backend.device),
                },
                "explicit_field_transfer_bytes": {
                    "initial_state_host_to_device": int(device_state.numel() * 8),
                    "active_mask_host_to_device": int(active_full.size),
                    "preconditioner_mobility_device_to_host":
                        int(np.prod(evaluator.shape) * 8),
                    "preconditioner_mask_device_to_host": int(active_full.size),
                    "full_field_transfers_per_gmres_iteration": 0,
                },
            })
            continue

        actual = gpu.direction.cpu().numpy()
        delta = actual - cpu.direction
        torch.cuda.synchronize(backend.device)
        results.append({
            "checkpoint": number,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "active_full_cells": int(active_full.sum()),
            "cpu": {
                "seconds": cpu_seconds,
                "true_linear_residual": cpu.true_linear_residual,
                "gmres_iterations": cpu.linear_iterations,
                "preconditioner": cpu_preconditioner,
            },
            "device_candidate": {
                "status": "accepted_linear_correction",
                "seconds": time.perf_counter() - start,
                "setup_seconds": gpu.setup_seconds,
                "gmres_seconds": gpu.solve_seconds,
                "iterations": gpu.krylov.iterations,
                "operator_applications": gpu.operator_applications,
                "preconditioner_applications":
                    gpu.preconditioner.application_count,
                "true_relative_residual": gpu.linear_residual_relative,
                "correction_relative_l2": float(
                    np.linalg.norm(delta) / np.linalg.norm(cpu.direction)
                ),
                "correction_linf": float(np.max(np.abs(delta))),
                "preconditioner_device_bytes": gpu.preconditioner.estimated_bytes,
                "pressure_mg_levels": len(
                    gpu.preconditioner.pressure_multigrid.levels
                ),
                "gpu_peak_allocated_bytes":
                    torch.cuda.max_memory_allocated(backend.device),
                "gpu_peak_reserved_bytes":
                    torch.cuda.max_memory_reserved(backend.device),
            },
            "explicit_field_transfer_bytes": {
                "initial_state_host_to_device": int(device_state.numel() * 8),
                "active_mask_host_to_device": int(active_full.size),
                "preconditioner_mobility_device_to_host": int(np.prod(evaluator.shape) * 8),
                "preconditioner_mask_device_to_host": int(active_full.size),
                "full_field_transfers_per_gmres_iteration": 0,
                "accepted_correction_device_to_host": int(actual.size * 8),
            },
        })
    return {
        "scope": "Saved-state linear correction benchmark; no nonlinear CUDA solve",
        "grid_ny_nx": list(evaluator.shape),
        "accepted_state_sha256": run["summary"]["state_sha256"],
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "float64",
        "cpu_linear_tolerances": {"rtol": 1e-9, "restart": 60, "cycles": 10},
        "device_linear_tolerances": {
            "rtol": 1e-9,
            "restart": 60,
            "maximum_iterations": max_iterations,
        },
        "checkpoints": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 3, 6])
    parser.add_argument("--max-iterations", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(number < 0 for number in args.checkpoints):
        raise ValueError("checkpoint numbers must be nonnegative")
    if args.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    report = benchmark(args.root, args.checkpoints, max_iterations=args.max_iterations)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
