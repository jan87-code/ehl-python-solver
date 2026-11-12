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


def _difference_metrics(reference, actual):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    difference = actual - reference
    reference_norm = float(np.linalg.norm(reference))
    reference_maximum = float(np.max(np.abs(reference)))
    return {
        "relative_l2": float(np.linalg.norm(difference) / max(reference_norm, 1e-300)),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "linf": float(np.max(np.abs(difference))),
        "relative_linf": float(np.max(np.abs(difference)) / max(reference_maximum, 1e-300)),
    }


def _residual_blocks(residual, final_residual, n):
    blocks = {
        "mass": slice(0, n),
        "selected_complementarity": slice(n, 2 * n),
        "load": slice(2 * n, 2 * n + 1),
    }
    return {
        name: {
            "initial_l2": float(np.linalg.norm(residual[selection])),
            "final_l2": float(np.linalg.norm(final_residual[selection])),
            "reduction_factor": float(
                np.linalg.norm(final_residual[selection])
                / max(np.linalg.norm(residual[selection]), 1e-300)
            ),
            "final_linf": float(np.max(np.abs(final_residual[selection]))),
        }
        for name, selection in blocks.items()
    }


def _correction_blocks(reference, actual, n):
    slices = {
        "pressure": slice(0, n),
        "theta_direction": slice(n, 2 * n),
        "bordered_h0_over_href": slice(2 * n, 2 * n + 1),
    }
    return {
        name: _difference_metrics(reference[selection], actual[selection])
        for name, selection in slices.items()
    }


def _boundary_direction_metrics(probe, direction):
    ny, nx = probe.system.problem.shape
    content = np.zeros((ny, nx), dtype=np.float64)
    content[1:-1, 1:-1] = -direction[probe.n:2 * probe.n].reshape(
        ny - 2, nx - 2
    )
    content[1:-1, -1] = content[1:-1, -2]
    content[0], content[-1] = content[1], content[-2]
    return {
        "pressure_boundary_linf": 0.0,
        "west_content_direction_linf": 0.0,
        "east_copy_linf": float(np.max(np.abs(
            content[1:-1, -1] - content[1:-1, -2]
        ))),
        "transverse_copy_linf": float(max(
            np.max(np.abs(content[0] - content[1])),
            np.max(np.abs(content[-1] - content[-2])),
        )),
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
        cpu_matrix = model.frozen_matrix(state)
        cpu_residual = model.residual(state)
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
        device_guide = as_float64(np.zeros_like(state), backend)
        device_operator = evaluator.linearize(
            device_state, device_guide, active_full=device_active
        )
        device_residual = evaluator.linear_residual(device_state, device_active)
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
            gmres_preconditioner_applications = preconditioner.application_count
            peak_allocated_bytes = torch.cuda.max_memory_allocated(backend.device)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(backend.device)
            gpu_direction = error.krylov.solution.detach().cpu().numpy()
            torch.cuda.synchronize(backend.device)
            gpu_jvp = device_operator(error.krylov.solution).detach().cpu().numpy()
            gpu_residual_array = device_residual.detach().cpu().numpy()
            cpu_jvp_on_gpu_direction = model.jacobian_vector(state, gpu_direction)
            gpu_residual_vector = gpu_jvp + gpu_residual_array
            cpu_jvp_on_cpu_direction = model.jacobian_vector(state, cpu.direction)
            gpu_jvp_on_cpu_direction = device_operator(
                as_float64(cpu.direction, backend)
            ).detach().cpu().numpy()
            cpu_residual_vector = cpu_jvp_on_cpu_direction + cpu_residual
            cpu_operator_gpu_correction_residual = (
                cpu_jvp_on_gpu_direction + cpu_residual
            )
            local_rng = np.random.default_rng(20261112 + number)
            local_direction = local_rng.standard_normal(2 * probe.n)
            local_gpu_action = preconditioner.matrix_vector(
                as_float64(local_direction, backend)
            ).detach().cpu().numpy()
            local_cpu_action = (
                cpu_matrix[:2 * probe.n, :2 * probe.n] @ local_direction
            )
            preconditioner_rhs = local_rng.standard_normal(probe.size)
            preconditioned_rhs = preconditioner(
                as_float64(preconditioner_rhs, backend)
            ).detach().cpu().numpy()
            preconditioner_defect = cpu_matrix @ preconditioned_rhs - preconditioner_rhs
            cpu_direction = cpu.direction
            cosine = float(
                np.dot(cpu_direction, gpu_direction)
                / max(
                    np.linalg.norm(cpu_direction) * np.linalg.norm(gpu_direction),
                    1e-300,
                )
            )
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
                    "preconditioner_applications": gmres_preconditioner_applications,
                    "true_relative_residual": error.krylov.true_relative_residual,
                    "true_residual_history": error.krylov.true_residual_history,
                    "estimated_preconditioned_history_tail":
                        error.krylov.estimated_preconditioned_history[-10:],
                    "correction_vs_cpu": {
                        "full": _difference_metrics(cpu_direction, gpu_direction),
                        "blocks": _correction_blocks(cpu_direction, gpu_direction, probe.n),
                        "cosine_similarity": cosine,
                        "angle_degrees": float(np.degrees(np.arccos(
                            np.clip(cosine, -1.0, 1.0)
                        ))),
                    },
                    "residual_blocks": {
                        "cpu_reference_correction": _residual_blocks(
                            cpu_residual, cpu_residual_vector, probe.n
                        ),
                        "gpu_candidate_correction": _residual_blocks(
                            gpu_residual_array, gpu_residual_vector, probe.n
                        ),
                        "cpu_operator_on_gpu_correction": _residual_blocks(
                            cpu_residual,
                            cpu_operator_gpu_correction_residual,
                            probe.n,
                        ),
                    },
                    "independently_recomputed_gpu_true_relative_residual": float(
                        np.linalg.norm(gpu_residual_vector)
                        / max(np.linalg.norm(gpu_residual_array), 1e-300)
                    ),
                    "operator_parity": {
                        "selected_residual_cpu_vs_cuda":
                            _difference_metrics(cpu_residual, gpu_residual_array),
                        "JVP_on_gpu_candidate_correction":
                            _difference_metrics(
                                cpu_jvp_on_gpu_direction, gpu_jvp
                            ),
                        "JVP_on_cpu_reference_correction":
                            _difference_metrics(
                                cpu_jvp_on_cpu_direction, gpu_jvp_on_cpu_direction
                            ),
                        "frozen_local_field_matrix_random_direction":
                            _difference_metrics(local_cpu_action, local_gpu_action),
                    },
                    "bordered_preconditioner": {
                        "sampled_full_matrix_defect_blocks": _residual_blocks(
                            preconditioner_rhs,
                            preconditioner_defect,
                            probe.n,
                        ),
                        "load_row_residual": float(preconditioner_defect[-1]),
                        "cpu_candidate_correction_boundary_extensions":
                            _boundary_direction_metrics(probe, gpu_direction),
                        "active_full_cells": int(active_full.sum()),
                        "preconditioner_active_mask_matches_selected_mask":
                            bool(torch.equal(preconditioner.active_full, device_active)),
                    },
                    "preconditioner_device_bytes": preconditioner.estimated_bytes,
                    "pressure_mg_levels": len(
                        preconditioner.pressure_multigrid.levels
                    ),
                    "gpu_peak_allocated_bytes": peak_allocated_bytes,
                    "gpu_peak_reserved_bytes": peak_reserved_bytes,
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
