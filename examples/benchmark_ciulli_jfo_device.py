"""Verify complete donor/JFO device residual and JVP on archived Newton states.

This is an operator benchmark; no nonlinear GPU solve or acceptance is implied.
Inputs and CPU residual differences are independently recomputed; outputs are
compact scalar diagnostics only. A fresh output filename is required.
"""

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import numpy as np

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.device_jfo import DeviceDonorJFO
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from examples.diagnose_ciulli_core_ridge import _read_root


def _errors(actual, reference):
    delta = actual - reference
    norm = float(np.linalg.norm(reference))
    peak = float(np.max(np.abs(reference)))
    return {
        "relative_l2": float(np.linalg.norm(delta) / max(norm, np.finfo(float).tiny)),
        "linf": float(np.max(np.abs(delta))),
        "relative_linf": float(np.max(np.abs(delta)) / max(peak, np.finfo(float).tiny)),
    }


def _directions(state, probe):
    ny, nx = probe.system.problem.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny - 2), np.linspace(-1, 1, nx - 2), indexing="ij")
    rng = np.random.default_rng(2026)
    weights = rng.normal(size=4)
    smooth = np.zeros(probe.size)
    smooth[:probe.n] = (
        1e-4 * (weights[0] * np.cos(xx * np.pi) * np.cos(yy * np.pi)
                + weights[1] * np.sin(xx * np.pi / 2))
    ).ravel()
    smooth[probe.n:2 * probe.n] = (
        1e-4 * (weights[2] * np.sin(xx * np.pi) + weights[3] * np.cos(yy * np.pi))
    ).ravel()
    pressure = np.zeros(probe.size)
    pressure[(ny // 2 - 1) * (nx - 2) + nx // 2 - 1] = 1e-2
    content = np.zeros(probe.size)
    candidates = np.flatnonzero((state[probe.n:2 * probe.n] > 0.01)
                                & (state[probe.n:2 * probe.n] < 0.9))
    if not candidates.size:
        raise ValueError("state has no interior partially cavitated cell for content probe")
    content[probe.n + candidates[len(candidates) // 2]] = 1e-2
    offset = np.zeros(probe.size)
    offset[-1] = 1e-2
    combined = smooth + pressure + content + offset
    result = {}
    for label, direction in (
        ("smooth", smooth), ("localized_pressure", pressure),
        ("localized_theta", content), ("offset", offset), ("combined", combined),
    ):
        direction = direction.copy()
        for start in (0, probe.n):
            values = state[start:start + probe.n]
            vec = direction[start:start + probe.n]
            vec[values <= 2e-6] = np.maximum(vec[values <= 2e-6], 0)
        deficit = state[probe.n:2 * probe.n]
        vd = direction[probe.n:2 * probe.n]
        vd[deficit >= 1 - 2e-6] = np.minimum(vd[deficit >= 1 - 2e-6], 0)
        result[label] = direction
    stable = result["smooth"].copy()
    near_tie = np.abs(state[:probe.n] - state[probe.n:2 * probe.n]) < 1e-7
    stable[:probe.n][near_tie] = 0
    stable[probe.n:2 * probe.n][near_tie] = 0
    result["smooth_front_masked"] = stable
    return result


def benchmark(root, checkpoint_numbers):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is unavailable")
    run = _read_root(root)
    case = PointContactCase(**run["summary"]["case"])
    probe = LimitedLoadProbe(case, transport_fraction=0)
    backend = make_backend("torch", "cuda")
    device = DeviceDonorJFO(case, backend)
    results = []
    for number in checkpoint_numbers:
        path = Path(root) / "checkpoints" / f"checkpoint_{number:04d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["case_json"])) != run["summary"]["case"]:
                raise ValueError(f"checkpoint case mismatch: {path}")
            state = archive["state_dimensionless"].copy()
        if not probe.feasible(state):
            raise ValueError(f"checkpoint is not a feasible iterate: {path}")
        z = as_float64(state, backend)
        reference_residual = probe.residual(state)
        gpu_residual = device.residual(z).cpu().numpy()
        record = {
            "checkpoint": number,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "residual_cpu_gpu": _errors(gpu_residual, reference_residual),
            "directions": {},
        }
        for label, direction in _directions(state, probe).items():
            v = as_float64(direction, backend)
            cpu_jvp = probe.freeze(state, direction)(direction)
            gpu_jvp = device.linearize(z, v)(v).cpu().numpy()
            true_jvp = probe.derivative(state, direction)
            gpu_true = device.directional_derivative(z, v).cpu().numpy()
            samples = []
            p, d = state[:probe.n], state[probe.n:2 * probe.n]
            vp, vd = direction[:probe.n], direction[probe.n:2 * probe.n]
            selected = (p < d) | ((p == d) & (vp <= vd))
            for step in (1e-2, 1e-3, 1e-4, 1e-5):
                if not probe.feasible(state + step * direction) or not probe.feasible(
                    state + 2 * step * direction
                ):
                    samples.append({"step": step, "status": "infeasible"})
                    continue
                fd = (-3 * reference_residual + 4 * probe.residual(state + step * direction)
                      - probe.residual(state + 2 * step * direction)) / (2 * step)
                p2, d2 = p + 2 * step * vp, d + 2 * step * vd
                switched = np.count_nonzero((selected & (p2 > d2)) | (~selected & (p2 < d2)))
                samples.append({
                    "step": step, "status": "evaluated", "branch_changes_at_2step": int(switched),
                    **_errors(fd, gpu_true),
                })
            record["directions"][label] = {
                "selected_jvp_cpu_gpu": _errors(gpu_jvp, cpu_jvp),
                "directional_cpu_gpu": _errors(gpu_true, true_jvp),
                "one_sided_fd_vs_gpu": samples,
            }
        direction = _directions(state, probe)["combined"]
        v = as_float64(direction, backend)
        cpu_action = probe.freeze(state, direction)
        gpu_action = device.linearize(z, v)
        for label, cpu_call, gpu_call in (
            ("residual", partial(probe.residual, state), partial(device.residual, z)),
            ("jvp", partial(cpu_action, direction), partial(gpu_action, v)),
        ):
            cpu_call()
            gpu_call()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                cpu_call()
            cpu_seconds = (time.perf_counter() - start) / 5
            start = time.perf_counter()
            for _ in range(5):
                gpu_call()
            torch.cuda.synchronize()
            gpu_seconds = (time.perf_counter() - start) / 5
            record[f"{label}_timing"] = {
                "repetitions": 5,
                "cpu_seconds": cpu_seconds,
                "gpu_synchronized_seconds": gpu_seconds,
                "resident_speedup": cpu_seconds / gpu_seconds,
            }
        results.append(record)
    return {
        "scope": "full donor/JFO residual and bordered products, not a nonlinear GPU solve",
        "state_sha256": run["summary"]["state_sha256"],
        "grid_ny_nx": list(device.shape),
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "finite_difference": (
            "second-order one-sided CPU residual; feasible samples only; "
            "selected JVP guide equals direction, including exact min ties; "
            "smooth directions use seed 2026, bounded at near-zero state variables; "
            "smooth_front_masked also excludes near-tied complementarity cells"
        ),
        "checkpoints": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 3, 6, 7])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(n < 0 for n in args.checkpoints):
        raise ValueError("checkpoint numbers must be nonnegative")
    report = benchmark(args.root, args.checkpoints)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
