"""Compare saved CPU corrections with configurable hybrid CUDA/CPU settings."""

import argparse
import gc
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


def _difference(reference, actual):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    l2 = float(np.linalg.norm(delta))
    return {
        "l2": l2,
        "relative_l2": l2 / max(float(np.linalg.norm(reference)), 1e-300),
        "linf": float(np.max(np.abs(delta))),
    }


def _peak_memory(torch, device, free_before, reserved_before):
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    reserved_increment = max(0, peak_reserved - reserved_before)
    return {
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "reserved_increment_bytes": reserved_increment,
        "estimated_free_margin_bytes": free_before - reserved_increment,
    }


def _transfer_seconds(timings):
    return sum(
        value
        for key, value in timings.items()
        if key.endswith("_seconds") and ("h2d" in key or "d2h" in key)
    )


def benchmark(
    root,
    checkpoint_numbers,
    *,
    gmres_restarts=(60,),
    fft_workspace_limit_mib=1024,
    maximum_krylov_cycles=10,
):
    import torch

    checkpoint_numbers = tuple(checkpoint_numbers)
    gmres_restarts = tuple(gmres_restarts)
    if not checkpoint_numbers or any(
        type(number) is not int or number < 0 for number in checkpoint_numbers
    ):
        raise ValueError("at least one nonnegative checkpoint number is required")
    if (
        not gmres_restarts
        or any(type(restart) is not int or not 1 <= restart <= 60
               for restart in gmres_restarts)
        or len(set(gmres_restarts)) != len(gmres_restarts)
    ):
        raise ValueError("GMRES restarts must be unique integers from 1 to 60")
    if (
        type(fft_workspace_limit_mib) is not int
        or not 1 <= fft_workspace_limit_mib <= 4096
    ):
        raise ValueError("fft_workspace_limit_mib must be an integer from 1 to 4096")
    if type(maximum_krylov_cycles) is not int or not 1 <= maximum_krylov_cycles <= 10:
        raise ValueError("maximum_krylov_cycles must be an integer from 1 to 10")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; hybrid benchmark cannot run")

    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "accepted_numerical_root":
        raise ValueError(f"{root} is not an accepted numerical root")
    case = PointContactCase(**summary["case"])
    probe = LimitedLoadProbe(case, transport_fraction=0)
    checkpoints = []

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
            maximum_krylov_cycles=maximum_krylov_cycles,
            gmres_restart=60,
            krylov_rtol=1e-9,
        )
        checkpoints.append({
            "number": checkpoint_number,
            "path": checkpoint,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "model": model,
            "state": state,
            "cpu": cpu,
            "cpu_seconds": time.perf_counter() - started,
        })

    backend = make_backend("torch", "cuda")
    device = backend.device
    configurations = []
    for restart in gmres_restarts:
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        free_before, total_bytes = torch.cuda.mem_get_info(device)
        reserved_before = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            hybrid_solver = HybridJFOCorrectionSolver(
                case,
                backend,
                workspace_limit_bytes=fft_workspace_limit_mib * 1024**2,
            )
        except MemoryError as error:
            configurations.append({
                "gmres_restart": restart,
                "fft_workspace_limit_mib": fft_workspace_limit_mib,
                "status": "failed_workspace_admission",
                "reason": str(error),
                "device_free_before_bytes": free_before,
                "device_total_bytes": total_bytes,
                "checkpoints": [],
            })
            continue

        torch.cuda.synchronize(device)
        initialization_memory = _peak_memory(
            torch, device, free_before, reserved_before
        )
        results = []
        for item in checkpoints:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                hybrid = hybrid_solver(
                    item["model"],
                    item["state"],
                    maximum_krylov_cycles=maximum_krylov_cycles,
                    gmres_restart=restart,
                    krylov_rtol=1e-9,
                )
            except CavitatingLinearError as error:
                torch.cuda.synchronize(device)
                results.append({
                    "checkpoint": item["number"],
                    "checkpoint_sha256": item["sha256"],
                    "status": "failed_linear_gate",
                    "reason": str(error),
                    "cpu": {
                        "seconds": item["cpu_seconds"],
                        "true_linear_residual": item["cpu"].true_linear_residual,
                        "iterations": item["cpu"].linear_iterations,
                    },
                    "hybrid": error.diagnostics,
                    "gpu_memory": _peak_memory(
                        torch, device, free_before, reserved_before
                    ),
                })
                continue
            except MemoryError as error:
                torch.cuda.synchronize(device)
                results.append({
                    "checkpoint": item["number"],
                    "checkpoint_sha256": item["sha256"],
                    "status": "failed_workspace_admission",
                    "reason": str(error),
                    "cpu": {
                        "seconds": item["cpu_seconds"],
                        "true_linear_residual": item["cpu"].true_linear_residual,
                        "iterations": item["cpu"].linear_iterations,
                    },
                    "gpu_memory": _peak_memory(
                        torch, device, free_before, reserved_before
                    ),
                })
                continue

            torch.cuda.synchronize(device)
            correction_difference = _difference(
                item["cpu"].direction, hybrid.direction
            )
            residual_parity_linf = hybrid.timings["cpu_residual_parity_linf"]
            accepted = (
                hybrid.true_linear_residual <= 5e-9
                and correction_difference["relative_l2"] <= 5e-7
                and residual_parity_linf <= 1e-10
            )
            results.append({
                "checkpoint": item["number"],
                "checkpoint_sha256": item["sha256"],
                "status": (
                    "accepted_linear_correction" if accepted else "failed_parity_gate"
                ),
                "cpu": {
                    "seconds": item["cpu_seconds"],
                    "true_linear_residual": item["cpu"].true_linear_residual,
                    "iterations": item["cpu"].linear_iterations,
                    "used_bordered_fallback": item["cpu"].used_bordered_fallback,
                },
                "hybrid": {
                    "wall_seconds": time.perf_counter() - started,
                    "complete_correction_seconds":
                        hybrid.timings["total_correction_seconds"],
                    "true_linear_residual": hybrid.true_linear_residual,
                    "residual_reduction_vs_cpu": (
                        item["cpu"].true_linear_residual
                        / max(hybrid.true_linear_residual, 1e-300)
                    ),
                    "iterations": hybrid.linear_iterations,
                    "used_bordered_fallback": hybrid.used_bordered_fallback,
                    "correction_vs_cpu": correction_difference,
                    "residual_parity_linf": residual_parity_linf,
                    "transfer_overhead_seconds": _transfer_seconds(hybrid.timings),
                    "timings": hybrid.timings,
                },
                "gpu_memory": _peak_memory(
                    torch, device, free_before, reserved_before
                ),
            })

        configuration_peak_reserved = max(
            [initialization_memory["peak_reserved_bytes"]]
            + [
                result["gpu_memory"]["peak_reserved_bytes"]
                for result in results
            ]
        )
        configurations.append({
            "gmres_restart": restart,
            "maximum_krylov_cycles": maximum_krylov_cycles,
            "fft_workspace_limit_mib": fft_workspace_limit_mib,
            "device_free_before_bytes": free_before,
            "device_total_bytes": total_bytes,
            "initialization_seconds": hybrid_solver.initialization_seconds,
            "initialization_gpu_memory": initialization_memory,
            "configuration_peak_reserved_bytes": configuration_peak_reserved,
            "projected_1025_reserved_increment_bytes": int(
                max(
                    0,
                    configuration_peak_reserved - reserved_before,
                )
                * (1025**2 / 513**2)
            ),
            "checkpoints": results,
        })
        del hybrid_solver
        gc.collect()

    return {
        "scope": "Saved-state linear corrections; no nonlinear solve",
        "root": str(root),
        "grid_ny_nx": [case.ny, case.nx],
        "accepted_state_sha256": summary["state_sha256"],
        "checkpoint_numbers": checkpoint_numbers,
        "device": torch.cuda.get_device_name(device),
        "device_total_bytes": total_bytes,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "float64",
        "linear_acceptance": {
            "krylov_rtol": 1e-9,
            "relative_true_residual_max": 5e-9,
            "cpu_correction_relative_l2_max": 5e-7,
            "residual_parity_linf_max": 1e-10,
        },
        "configurations": configurations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 3, 6])
    parser.add_argument("--gmres-restarts", type=int, nargs="+", default=[60])
    parser.add_argument("--maximum-krylov-cycles", type=int, default=10)
    parser.add_argument("--fft-workspace-limit-mib", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = benchmark(
        args.root,
        args.checkpoints,
        gmres_restarts=args.gmres_restarts,
        fft_workspace_limit_mib=args.fft_workspace_limit_mib,
        maximum_krylov_cycles=args.maximum_krylov_cycles,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
