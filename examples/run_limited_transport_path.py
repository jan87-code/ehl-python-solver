"""Archive a bounded transport path; only full fraction-one closure is success."""

import argparse
import gc
import json
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import splu

from ehl_solver.limited_comparison import file_sha256, load_limited_path
from ehl_solver.limited_continuation import LimitedPathError, continue_limited_transport
from ehl_solver.limited_newton import LimitedNewtonError
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json, digest, load_run


def _interpolate_limited_seed(directory, case):
    source = load_limited_path(directory)
    if source.case.physical_signature() != case.physical_signature():
        raise ValueError("Limited seed and target must use identical physical parameters")
    if (
        source.case.domain_half_width_x_m != case.domain_half_width_x_m
        or source.case.domain_half_width_y_m != case.domain_half_width_y_m
    ):
        raise ValueError("Limited seed and target must use the identical physical domain")
    if case.nx < source.case.nx or case.ny < source.case.ny:
        raise ValueError("Limited seed cannot coarsen either axis")

    old_x = np.linspace(
        -source.case.domain_half_width_x_m, source.case.domain_half_width_x_m, source.case.nx
    )
    old_y = np.linspace(
        -source.case.domain_half_width_y_m, source.case.domain_half_width_y_m, source.case.ny
    )
    new_x = np.linspace(-case.domain_half_width_x_m, case.domain_half_width_x_m, case.nx)
    new_y = np.linspace(-case.domain_half_width_y_m, case.domain_half_width_y_m, case.ny)
    xx, yy = np.meshgrid(new_x, new_y)
    points = np.column_stack((yy.ravel(), xx.ravel()))
    pressure = RegularGridInterpolator((old_y, old_x), source.pressure_pa)(points).reshape(
        case.ny, case.nx
    )
    theta = RegularGridInterpolator((old_y, old_x), source.theta)(points).reshape(
        case.ny, case.nx
    )
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    offset_m = source.state[-1] * source.case.problem().film_scale_m
    state = np.r_[
        (pressure[1:-1, 1:-1] / probe.system.problem.pressure_scale_pa).ravel(),
        (1 - theta[1:-1, 1:-1]).ravel(),
        offset_m / probe.system.problem.film_scale_m,
    ].astype(np.float64, copy=False)
    if not probe.feasible(state):
        raise ValueError("Interpolated limited root is infeasible on the target grid")
    return state, dict(
        kind="independently audited limited MC root; bilinear interpolation",
        directory=str(directory),
        summary_sha256=file_sha256(Path(directory) / "summary.json"),
        source_shape=[source.case.ny, source.case.nx],
        target_shape=[case.ny, case.nx],
        accepted_for_target=False,
        physical_offset_m=float(offset_m),
    )


def _payload_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _available_memory_bytes():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", maxsplit=1)
        fields[name] = int(value.strip().split()[0]) * 1024
    if "MemAvailable" not in fields:
        raise RuntimeError("Linux /proc/meminfo does not report MemAvailable")
    return fields["MemAvailable"]


def resource_preflight(case, state):
    """Measure sparse-factor cost and refuse a large run without bounded headroom."""
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    guide = np.zeros(probe.size, dtype=np.float64)
    model = LimitedLoadLinearization(
        case,
        state,
        guide,
        preconditioner="local",
        transport_fraction=0.0,
    )
    matrix = model.frozen_matrix(state)
    field_matrix = matrix[: 2 * probe.n, : 2 * probe.n].tocsc()
    field_lu = splu(field_matrix)
    bordered_lu = splu(matrix)
    factor_bytes = sum(
        _payload_bytes(lu.L) + _payload_bytes(lu.U) for lu in (field_lu, bordered_lu)
    )
    factor_bytes += sum(
        lu.perm_c.nbytes + lu.perm_r.nbytes for lu in (field_lu, bordered_lu)
    )
    basis_bytes = 2 * probe.size * (60 + 3) * np.dtype(np.float64).itemsize
    fft_workspace_bytes = probe.system.base.elastic_operator.estimated_workspace_bytes
    estimated_peak = (
        2
        * (_payload_bytes(matrix) + _payload_bytes(field_matrix) + factor_bytes + basis_bytes)
        + 2 * fft_workspace_bytes
        + 160 * case.ny * case.nx
    )
    known = np.random.default_rng(17329929).normal(size=probe.size)
    rhs = matrix @ known
    relative_residual = float(
        np.linalg.norm(matrix @ bordered_lu.solve(rhs) - rhs)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
    )
    available = _available_memory_bytes()
    memory_ceiling = min(2**30, 3 * available // 4)
    result = dict(
        physical_nodes=case.ny * case.nx,
        interior_nodes=probe.n,
        matrix_nonzeros=int(matrix.nnz),
        factor_bytes=int(factor_bytes),
        krylov_basis_bytes=int(basis_bytes),
        elastic_fft_workspace_bytes=int(fft_workspace_bytes),
        estimated_peak_with_margin_bytes=int(estimated_peak),
        available_memory_before_run_bytes=int(available),
        memory_ceiling_bytes=int(memory_ceiling),
        known_vector_relative_residual=relative_residual,
        memory_preflight_passed=bool(estimated_peak < memory_ceiling),
        factor_preflight_passed=bool(np.isfinite(relative_residual) and relative_residual < 1e-8),
    )
    result["passed"] = result["memory_preflight_passed"] and result["factor_preflight_passed"]
    del known, rhs, field_lu, bordered_lu, field_matrix, matrix, model, probe
    gc.collect()
    return result


def main(
    case_path,
    seed,
    output,
    maximum_attempts,
    *,
    endpoint_prediction=True,
    limited_seed=None,
):
    if seed is not None and limited_seed is not None:
        raise ValueError("Choose either a prototype seed or a limited MC seed, not both")
    seed_metadata = None
    if limited_seed is not None:
        if case_path is None:
            raise ValueError("A target case is required when refining an MC seed")
        case = PointContactCase(**json.loads(case_path.read_text()))
        initial, seed_metadata = _interpolate_limited_seed(limited_seed, case)
    elif seed is not None:
        case, _, _ = load_run(seed)
        if case_path is not None and PointContactCase(**json.loads(case_path.read_text())) != case:
            raise ValueError("Seed and case must match exactly")
        with np.load(seed / "solution.npz", allow_pickle=False) as saved:
            initial = saved["state_dimensionless"].copy()
            probe = LimitedLoadProbe(case, transport_fraction=0.0)
            p, t, h = probe.fields(initial)
            for actual, key in [(p, "pressure_pa"), (t, "theta"), (h, "film_m")]:
                np.testing.assert_array_equal(actual, saved[key])
    else:
        case = (
            PointContactCase(**json.loads(case_path.read_text()))
            if case_path
            else PointContactCase()
        )
        initial = None
    if limited_seed is None and case.maximum_physical_nodes > 20000:
        raise ValueError(
            "An explicit budget above 20,000 nodes requires an independently audited MC seed"
        )
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        case=asdict(case),
        operator_transport="limited_capacity_target",
        target_transport_fraction=1.0,
        initial_state_accepted_for_target=False,
        seed=(
            seed_metadata
            if limited_seed is not None
            else None
            if seed is None
            else dict(path=str(seed), sha256=digest(seed / "solution.npz"))
        ),
        maximum_attempts=maximum_attempts,
        maximum_physical_nodes=case.maximum_physical_nodes,
        endpoint_prediction_enabled=endpoint_prediction,
        maximum_corrections_per_attempt=30,
        maximum_models_per_correction=8,
        maximum_krylov_cycles=10,
        gmres_restart=60,
        runtime=dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        stages=[],
        physical_validation_complete=False,
        source_sha256={},
    )
    package = Path(__import__("ehl_solver").__file__).parent
    for path in package.glob("*.py"):
        dest = output / "source/ehl_solver" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        report["source_sha256"][path.name] = digest(path)
    shutil.copyfile(__file__, output / "run_limited_transport_path.py")
    report["driver_sha256"] = digest(__file__)

    def write():
        (output / "summary.json").write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False)
        )

    def stage(index, stage_case, fraction, outcome):
        folder = output / f"attempt_{index:03d}"
        folder.mkdir(exist_ok=False)
        probe = LimitedLoadProbe(stage_case, transport_fraction=fraction)
        p, t, h = probe.fields(outcome.state)
        probe.system.sync(outcome.state)
        _, _, _, w, offset = probe.system.base.unpack(outcome.state[:-1])
        np.savez_compressed(
            folder / "state.npz",
            state_dimensionless=outcome.state,
            pressure_pa=p,
            theta=t,
            film_m=h,
            displacement_m=w,
            offset_m=offset,
            x_m=probe.system.problem.x_m,
            y_m=probe.system.problem.y_m,
        )
        info = dict(
            case=asdict(stage_case),
            transport_fraction=fraction,
            status="rejected_stage"
            if isinstance(outcome, LimitedNewtonError)
            else "accepted_stage",
            audit=outcome.audit,
            history=outcome.history,
            state_sha256=digest(folder / "state.npz"),
        )
        if isinstance(outcome, LimitedNewtonError):
            info["reason"] = str(outcome)
        (folder / "summary.json").write_text(
            json.dumps(clean_json(info), indent=2, allow_nan=False)
        )
        report["stages"].append({k: v for k, v in info.items() if k != "history"})
        write()

    write()
    if case.maximum_physical_nodes > 20000:
        try:
            preflight = resource_preflight(case, initial)
        except (MemoryError, OSError, RuntimeError, ValueError) as exc:
            report.update(
                status="rejected_preflight",
                preflight_error=f"{type(exc).__name__}: {exc}",
            )
            write()
            print(report["status"], report["preflight_error"], flush=True)
            return 1
        report["resource_preflight"] = preflight
        if not preflight["passed"]:
            report.update(
                status="rejected_preflight",
                reason="Sparse-factor/resource preflight failed",
            )
            write()
            print(report["status"], report["reason"], flush=True)
            return 1
        write()
    try:
        path = continue_limited_transport(
            case, initial, maximum_attempts=maximum_attempts, stage_observer=stage,
            endpoint_prediction=endpoint_prediction,
        )
        result = path.result
        report.update(
            status="accepted_numerical_root",
            transport_fraction=result.transport_fraction,
            audit=result.audit,
            attempts=path.attempts,
        )
    except LimitedPathError as exc:
        result = exc.last_result
        report.update(
            status="rejected_path",
            reason=str(exc),
            attempts=exc.attempts,
            last_accepted_fraction=None if result is None else result.transport_fraction,
            last_stage_audit=None if result is None else result.audit,
        )
    if result is not None:
        np.savez_compressed(output / "last_accepted_state.npz", state_dimensionless=result.state)
        report["last_accepted_state_sha256"] = digest(output / "last_accepted_state.npz")
    write()
    print(
        report["status"],
        report.get("transport_fraction", report.get("last_accepted_fraction")),
        flush=True,
    )
    return 0 if report["status"] == "accepted_numerical_root" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--limited-seed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, choices=range(1, 25), default=24)
    parser.add_argument(
        "--no-endpoint-prediction", dest="endpoint_prediction", action="store_false"
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            args.case, args.seed, args.output, args.max_attempts,
            endpoint_prediction=args.endpoint_prediction,
            limited_seed=args.limited_seed,
        )
    )
