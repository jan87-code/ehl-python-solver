"""Compare saved limited-transport corrections on CPU and hybrid CUDA/CPU.

The CPU Newton model remains the reference. This bounded probe checks residual,
frozen JVP, complete bordered correction, and independently recomputed true
residuals on two archived states; it does not resume continuation.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.cavitating_newton import solve_cavitating_correction
from ehl_solver.hybrid_jfo import HybridJFOCorrectionSolver
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.prototype import PointContactCase


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_l2(actual, expected):
    return float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), np.finfo(np.float64).tiny)
    )


def _benchmark_state(case, state_path, fraction, solver, workspace_limit_bytes):
    import torch

    with np.load(state_path, allow_pickle=False) as archive:
        state = archive["state_dimensionless"].astype(np.float64, copy=True)
    probe = LimitedLoadProbe(
        case,
        transport_fraction=fraction,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    if state.shape != (probe.size,) or not np.all(np.isfinite(state)):
        raise ValueError(f"{state_path}: invalid saved state shape or values")
    active_full = state[:probe.n] >= state[probe.n:2 * probe.n]
    guide = np.zeros_like(state)
    rng = np.random.default_rng(20261118)
    direction = rng.normal(size=probe.size) * 1e-5
    cpu_model = LimitedLoadLinearization(
        case,
        state,
        guide,
        preconditioner="local",
        active_full=active_full,
        transport_fraction=fraction,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    cpu_residual = cpu_model.residual(state)
    cpu_jvp = cpu_model.jacobian_vector(state, direction)
    device = solver.evaluator
    device.transport_fraction = fraction
    z = as_float64(state, solver.backend)
    g = as_float64(guide, solver.backend)
    v = as_float64(direction, solver.backend)
    mask = torch.as_tensor(active_full, dtype=torch.bool, device=z.device)
    limiter_branches = (
        None
        if cpu_model.limiter_branches is None
        else tuple(
            torch.tensor(branch.copy(), dtype=torch.int64, device=z.device)
            for branch in cpu_model.limiter_branches
        )
    )

    gpu_residual = device.linear_residual(z, mask).detach().cpu().numpy()
    gpu_jvp = device.linearize(
        z, g, active_full=mask, limiter_branches=limiter_branches
    )(v).detach().cpu().numpy()
    residual_linf = float(np.max(np.abs(gpu_residual - cpu_residual)))
    residual_relative_l2 = _relative_l2(gpu_residual, cpu_residual)
    jvp_linf = float(np.max(np.abs(gpu_jvp - cpu_jvp)))
    jvp_relative_l2 = _relative_l2(gpu_jvp, cpu_jvp)

    cpu_progress = {}

    def observe_cpu(**event):
        cpu_progress.update(event)

    cpu_started = time.perf_counter()
    cpu = solve_cavitating_correction(
        cpu_model,
        state,
        krylov_rtol=1e-9,
        maximum_krylov_cycles=10,
        gmres_restart=60,
        progress_callback=observe_cpu,
    )
    cpu_total_seconds = time.perf_counter() - cpu_started
    cpu_true = cpu_model.jacobian_vector(state, cpu.direction) + cpu_residual
    cpu_true_relative = float(
        np.linalg.norm(cpu_true) / max(np.linalg.norm(cpu_residual), 1e-300)
    )

    warm_operator = device.linearize(
        z, g, active_full=mask, limiter_branches=limiter_branches
    )
    warm_operator(v)
    torch.cuda.synchronize(z.device)
    del warm_operator
    solver_initialization_memory = torch.cuda.memory_allocated(z.device)
    torch.cuda.reset_peak_memory_stats(z.device)
    gpu_progress = []
    hybrid_model = LimitedLoadLinearization(
        case,
        state,
        guide,
        preconditioner="local",
        active_full=active_full,
        transport_fraction=fraction,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    hybrid = solver(
        hybrid_model,
        state,
        krylov_rtol=1e-9,
        maximum_krylov_cycles=10,
        gmres_restart=60,
        progress_callback=lambda **event: gpu_progress.append(event),
    )
    torch.cuda.synchronize(z.device)
    hybrid_direction = hybrid.direction
    gpu_true = hybrid_model.jacobian_vector(state, hybrid_direction) + cpu_residual
    gpu_true_relative = float(
        np.linalg.norm(gpu_true) / max(np.linalg.norm(cpu_residual), 1e-300)
    )
    correction_relative_l2 = _relative_l2(hybrid_direction, cpu.direction)
    correction_relative_linf = float(
        np.max(np.abs(hybrid_direction - cpu.direction))
        / max(np.max(np.abs(cpu.direction)), np.finfo(np.float64).tiny)
    )
    cpu_superlu_seconds = float(
        cpu_progress.get("cumulative_superlu_seconds", 0.0)
    )
    cpu_jvp_seconds = float(cpu_progress.get("cumulative_jvp_seconds", 0.0))
    cuda_peak_allocated = int(torch.cuda.max_memory_allocated(z.device))
    cuda_peak_reserved = int(torch.cuda.max_memory_reserved(z.device))
    cuda_allocated_after = int(torch.cuda.memory_allocated(z.device))
    cuda_reserved_after = int(torch.cuda.memory_reserved(z.device))
    transfer_total_seconds = sum(
        hybrid.timings[key]
        for key in (
            "state_guide_mask_h2d_seconds",
            "limiter_branch_h2d_seconds",
            "linear_rhs_h2d_seconds",
            "d2h_seconds",
            "h2d_preconditioner_seconds",
            "correction_d2h_seconds",
        )
    )

    gates = dict(
        residual_parity=residual_linf <= 1e-10,
        jvp_parity=jvp_linf <= 1e-10 and jvp_relative_l2 <= 1e-10,
        correction_parity=correction_relative_l2 <= 5e-7,
        cpu_true_residual=cpu_true_relative <= 5e-9,
        hybrid_true_residual=gpu_true_relative <= 5e-9,
    )
    result = dict(
        checkpoint=str(state_path),
        checkpoint_sha256=_sha256(state_path),
        transport_fraction=fraction,
        state_shape=list(state.shape),
        active_full_cells=int(np.count_nonzero(active_full)),
        parity=dict(
            residual_linf=residual_linf,
            residual_relative_l2=residual_relative_l2,
            jvp_linf=jvp_linf,
            jvp_relative_l2=jvp_relative_l2,
            correction_relative_l2=correction_relative_l2,
            correction_relative_linf=correction_relative_linf,
            independently_recomputed_cpu_true_relative_residual=cpu_true_relative,
            independently_recomputed_hybrid_true_relative_residual=gpu_true_relative,
        ),
        cpu=dict(
            total_correction_seconds=cpu_total_seconds,
            jvp_seconds=cpu_jvp_seconds,
            superlu_and_preconditioner_seconds=cpu_superlu_seconds,
            linear_iterations=cpu.linear_iterations,
            true_linear_residual=cpu.true_linear_residual,
        ),
        hybrid=dict(
            total_correction_seconds=hybrid.timings["total_correction_seconds"],
            jvp_seconds=hybrid.timings["cuda_jvp_seconds"],
            superlu_and_preconditioner_seconds=(
                hybrid.timings["cpu_preconditioner_setup_seconds"]
                + hybrid.timings["cpu_preconditioner_apply_seconds"]
            ),
            superlu_factorization_seconds=hybrid.timings["cpu_superlu_setup_seconds"],
            superlu_apply_seconds=hybrid.timings["cpu_superlu_solve_seconds"],
            transfer_seconds=dict(
                state_guide_mask_h2d=hybrid.timings["state_guide_mask_h2d_seconds"],
                limiter_branch_h2d=hybrid.timings["limiter_branch_h2d_seconds"],
                cpu_residual_rhs_and_parity_h2d=hybrid.timings[
                    "linear_rhs_h2d_seconds"
                ],
                preconditioner_d2h=hybrid.timings["d2h_seconds"],
                preconditioner_h2d=hybrid.timings["h2d_preconditioner_seconds"],
                correction_d2h=hybrid.timings["correction_d2h_seconds"],
                total_measured=transfer_total_seconds,
            ),
            vram_bytes=dict(
                allocated_before_correction=solver_initialization_memory,
                allocated_after_correction=cuda_allocated_after,
                peak_allocated=cuda_peak_allocated,
                reserved_after_correction=cuda_reserved_after,
                peak_reserved=cuda_peak_reserved,
            ),
            linear_iterations=hybrid.linear_iterations,
            true_linear_residual=hybrid.true_linear_residual,
            progress_events=len(gpu_progress),
        ),
        speedup=dict(
            correction=cpu_total_seconds / hybrid.timings["total_correction_seconds"],
            correction_including_one_time_device_setup=(
                cpu_total_seconds
                / (
                    hybrid.timings["total_correction_seconds"]
                    + solver.initialization_seconds
                )
            ),
            jvp=(
                cpu_jvp_seconds / hybrid.timings["cuda_jvp_seconds"]
                if hybrid.timings["cuda_jvp_seconds"] > 0
                else None
            ),
        ),
        gates=gates,
        passed=all(gates.values()),
    )
    del cpu_model, hybrid_model, probe, device, z, g, v, mask, hybrid, cpu
    gc.collect()
    return result


def benchmark(case_summary, checkpoints, fractions, output, workspace_limit_bytes):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is unavailable; hybrid benchmark cannot run")
    if len(checkpoints) != 2 or len(fractions) != 2:
        raise ValueError("Provide exactly two saved checkpoints and fractions")
    summary = json.loads(case_summary.read_text(encoding="utf-8"))
    case = PointContactCase(**summary["case"])
    backend = make_backend("torch", "cuda")
    solver = HybridJFOCorrectionSolver(
        case, backend, workspace_limit_bytes=workspace_limit_bytes
    )
    results = [
        _benchmark_state(case, path, fraction, solver, workspace_limit_bytes)
        for path, fraction in zip(checkpoints, fractions, strict=True)
    ]
    report = dict(
        status="passed" if all(item["passed"] for item in results) else "failed_gates",
        case=summary["case"],
        backend=str(backend.device),
        hybrid_initialization_seconds=solver.initialization_seconds,
        krylov=dict(
            relative_tolerance=1e-9,
            maximum_cycles=10,
            restart=60,
            preconditioner="existing CPU local SuperLU/Schur",
        ),
        residual_parity_gate=(
            "absolute Linf <= 1e-10; relative L2 is reported but not gated "
            "because the accepted-state residual is near floating-point roundoff"
        ),
        results=results,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--transport-fraction", type=float, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-limit-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.workspace_limit_mib < 1:
        parser.error("--workspace-limit-mib must be positive")
    return benchmark(
        args.case_summary,
        args.checkpoint,
        args.transport_fraction,
        args.output,
        args.workspace_limit_mib * 1024**2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
