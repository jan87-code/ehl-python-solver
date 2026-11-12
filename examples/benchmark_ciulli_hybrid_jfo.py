"""Compare saved 257x257 CPU corrections with the hybrid CUDA/CPU path."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ehl_solver.backend import make_backend
from ehl_solver.cavitating_newton import (
    CavitatingLinearError,
    solve_cavitating_correction,
)
from ehl_solver.hybrid_jfo import HybridJFOCorrectionSolver
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.prototype import PointContactCase


def _relative_difference(reference, actual):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-300)
        ),
        "linf": float(np.max(np.abs(delta))),
    }


def benchmark(root, checkpoint_numbers):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; hybrid benchmark cannot run")
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "accepted_numerical_root":
        raise ValueError(f"{root} is not an accepted numerical root")
    case = PointContactCase(**summary["case"])
    probe = LimitedLoadProbe(case, transport_fraction=0)
    backend = make_backend("torch", "cuda")
    hybrid_solver = HybridJFOCorrectionSolver(case, backend)
    results = []

    for checkpoint_number in checkpoint_numbers:
        checkpoint = root / "checkpoints" / f"checkpoint_{checkpoint_number:04d}.npz"
        with np.load(checkpoint, allow_pickle=False) as archive:
            if json.loads(str(archive["case_json"])) != summary["case"]:
                raise ValueError(f"checkpoint case metadata mismatch: {checkpoint}")
            state = archive["state_dimensionless"].copy()
        if not probe.feasible(state):
            raise ValueError(f"checkpoint is infeasible: {checkpoint}")
        active_full = state[:probe.n] >= state[probe.n:2 * probe.n]
        model = LimitedLoadLinearization(
            case,
            state,
            np.zeros(probe.size),
            preconditioner="local",
            active_full=active_full,
            transport_fraction=0,
        )

        started = time.perf_counter()
        cpu = solve_cavitating_correction(
            model,
            state,
            maximum_krylov_cycles=10,
            gmres_restart=60,
            krylov_rtol=1e-9,
        )
        cpu_seconds = time.perf_counter() - started

        torch.cuda.reset_peak_memory_stats(backend.device)
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        try:
            hybrid = hybrid_solver(model, state)
        except CavitatingLinearError as error:
            torch.cuda.synchronize(backend.device)
            results.append({
                "checkpoint": checkpoint_number,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "status": "failed_linear_gate",
                "reason": str(error),
                "cpu": {
                    "seconds": cpu_seconds,
                    "true_linear_residual": cpu.true_linear_residual,
                    "iterations": cpu.linear_iterations,
                },
                "hybrid": error.diagnostics,
                "gpu_peak_allocated_bytes":
                    torch.cuda.max_memory_allocated(backend.device),
                "gpu_peak_reserved_bytes":
                    torch.cuda.max_memory_reserved(backend.device),
            })
            continue
        torch.cuda.synchronize(backend.device)
        results.append({
            "checkpoint": checkpoint_number,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "status": "accepted_linear_correction",
            "cpu": {
                "seconds": cpu_seconds,
                "true_linear_residual": cpu.true_linear_residual,
                "iterations": cpu.linear_iterations,
                "used_bordered_fallback": cpu.used_bordered_fallback,
            },
            "hybrid": {
                "seconds": time.perf_counter() - started,
                "true_linear_residual": hybrid.true_linear_residual,
                "iterations": hybrid.linear_iterations,
                "used_bordered_fallback": hybrid.used_bordered_fallback,
                "schur_indicator": hybrid.schur_indicator,
                "correction_vs_cpu": _relative_difference(
                    cpu.direction, hybrid.direction
                ),
                "timings": hybrid.timings,
            },
            "gpu_peak_allocated_bytes":
                torch.cuda.max_memory_allocated(backend.device),
            "gpu_peak_reserved_bytes":
                torch.cuda.max_memory_reserved(backend.device),
        })

    return {
        "scope": "Saved-state linear correction comparison; no nonlinear solve",
        "root": str(root),
        "grid_ny_nx": [case.ny, case.nx],
        "accepted_state_sha256": summary["state_sha256"],
        "checkpoint_numbers": checkpoint_numbers,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "float64",
        "linear_acceptance": {
            "relative_true_residual_max": 5e-9,
            "cpu_correction_relative_l2_max": 5e-7,
            "residual_parity_linf_max": 1e-10,
        },
        "hybrid_initialization_seconds": hybrid_solver.initialization_seconds,
        "checkpoints": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 3, 6])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(number < 0 for number in args.checkpoints):
        raise ValueError("checkpoint numbers must be nonnegative")
    report = benchmark(args.root, args.checkpoints)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
