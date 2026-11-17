"""Archive a bounded transport path; only full fraction-one closure is success."""

import argparse
import gc
import json
import math
import os
import platform
import shutil
import signal
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.fft import next_fast_len
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import splu

from ehl_solver.heartbeat import SolverHeartbeat
from ehl_solver.limited_comparison import file_sha256, load_limited_path
from ehl_solver.limited_continuation import (
    MAXIMUM_STAGE_ATTEMPTS,
    LimitedPathError,
    continue_limited_transport,
)
from ehl_solver.limited_newton import LimitedNewtonError
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.prototype import PointContactCase
from ehl_solver.prototype_cli import clean_json, digest, load_run

_REFERENCE_FACTOR_BYTES = (
    (38_809, 265_243_020),
    (46_225, 310_058_172),
)
_FACTOR_FILL_SAFETY = 1.5
_MATRIX_NNZ_PER_INTERIOR_ESTIMATE = 15
_PREFLIGHT_SAFETY_MARGIN = 1.25
_DEFAULT_MEMORY_LIMIT_BYTES = 2 * 1024**3


class _RunInterrupted(KeyboardInterrupt):
    pass


def _termination_handler(signum, _frame):
    name = signal.Signals(signum).name
    raise _RunInterrupted(f"Received {name}")


def _write_json_atomic(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean_json(data), indent=2, allow_nan=False))
    os.replace(temporary, path)


class _ProgressReporter:
    def __init__(self, interval_seconds):
        if (
            isinstance(interval_seconds, bool)
            or not np.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("Progress interval must be a finite positive number of seconds")
        self.interval_seconds = float(interval_seconds)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._started_at = time.monotonic()
        self._status = "starting"
        self._attempt = None
        self._fraction = None
        self._iteration = None
        self._audit = None
        self._thread = threading.Thread(target=self._run, name="limited-path-progress", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._stop.set()
        self._thread.join()

    def update(self, attempt, fraction, iteration, audit, status):
        with self._lock:
            self._attempt = attempt
            self._fraction = fraction
            self._iteration = iteration
            self._audit = audit
            self._status = status

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            with self._lock:
                status = self._status
                attempt = self._attempt
                fraction = self._fraction
                iteration = self._iteration
                audit = self._audit
            elapsed = time.monotonic() - self._started_at
            details = [f"status={status}", f"elapsed={elapsed:.0f}s"]
            if attempt is not None:
                details.extend(
                    [f"attempt={attempt + 1}", f"target_fraction={fraction:.9g}"]
                )
            if iteration is not None:
                details.append(f"Newton_iteration={iteration}/30")
            if audit is not None:
                mass = audit.get("mass", {})
                if "relative_local_balance" in mass:
                    details.append(f"local_mass={mass['relative_local_balance']:.3e}")
                if "force_n" in audit:
                    details.append(f"force={audit['force_n']:.9g}N")
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"[progress {timestamp}] " + " ".join(details), flush=True)


def _interpolate_limited_seed(directory, case, *, workspace_limit_bytes=2**30):
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
    probe = LimitedLoadProbe(
        case,
        transport_fraction=0.0,
        workspace_limit_bytes=workspace_limit_bytes,
    )
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


def _load_resume_checkpoint(directory, case):
    summary_path = Path(directory) / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") == "accepted_numerical_root":
        raise ValueError("A completed numerical root is not a resumable partial path")
    if PointContactCase(**summary["case"]) != case:
        raise ValueError("Resume checkpoint and target case must match exactly")
    stages = summary.get("stages", [])
    for index in range(len(stages) - 1, -1, -1):
        stage = stages[index]
        if stage.get("status") != "accepted_stage":
            continue
        fraction = stage.get("transport_fraction")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not np.isfinite(fraction)
            or fraction >= 1
            or fraction < 0
            or PointContactCase(**stage["case"]) != case
        ):
            if fraction == 1:
                raise ValueError("Latest completed stage is already an exact endpoint")
            raise ValueError("Latest accepted resume stage has invalid metadata")
        state_path = Path(directory) / f"attempt_{index:03d}" / "state.npz"
        if digest(state_path) != stage.get("state_sha256"):
            raise ValueError("Latest accepted resume checkpoint checksum does not match")
        with np.load(state_path, allow_pickle=False) as saved:
            if "state_dimensionless" not in saved:
                raise ValueError("Resume checkpoint has no dimensionless state")
            state = np.array(saved["state_dimensionless"], dtype=np.float64, copy=True)
        expected_size = 2 * (case.ny - 2) * (case.nx - 2) + 1
        if state.shape != (expected_size,) or not np.all(np.isfinite(state)):
            raise ValueError("Resume checkpoint state has invalid shape or values")
        metadata = dict(
            directory=str(directory),
            source_status=summary.get("status"),
            source_stop_note=(
                "The source summary has no terminal status; its stop cause is unknown."
                if summary.get("status") == "running"
                else None
            ),
            summary_sha256=digest(summary_path),
            stage_index=index,
            transport_fraction=float(fraction),
            state_sha256=stage["state_sha256"],
        )
        return state, float(fraction), metadata
    raise ValueError("Resume directory has no completed accepted stage below fraction one")


def _payload_bytes(matrix):
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _available_memory_bytes():
    """Return available physical memory in bytes, cross-platform."""
    if platform.system() == "Windows":
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    else:
        # Linux / macOS: try /proc/meminfo first
        try:
            fields = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", maxsplit=1)
                fields[name] = int(value.strip().split()[0]) * 1024
            if "MemAvailable" not in fields:
                raise RuntimeError("Linux /proc/meminfo does not report MemAvailable")
            return fields["MemAvailable"]
        except (FileNotFoundError, OSError) as err:
            # Fallback for systems without /proc/meminfo
            raise RuntimeError(
                "Available memory detection requires Linux /proc/meminfo "
                "or Windows GlobalMemoryStatusEx"
            ) from err


def _process_memory_snapshot():
    """Return current and high-water private/working-set bytes for this process."""
    if platform.system() == "Windows":
        import ctypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "current_private_bytes": int(counters.PrivateUsage),
            "peak_private_bytes": int(counters.PeakPagefileUsage),
            "current_working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        }

    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        peak *= 1
    else:
        peak *= 1024
    current = peak
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
                break
    return {
        "current_private_bytes": current,
        "peak_private_bytes": peak,
        "current_working_set_bytes": current,
        "peak_working_set_bytes": peak,
    }


def estimate_resource_requirements(
    case,
    *,
    gmres_restart=60,
    memory_limit_bytes=_DEFAULT_MEMORY_LIMIT_BYTES,
    workspace_limit_bytes=2**30,
    available_memory_bytes=None,
):
    """Estimate limited-path memory without building fields or sparse matrices."""
    if (
        type(gmres_restart) is not int
        or not 1 <= gmres_restart <= 60
        or type(memory_limit_bytes) is not int
        or memory_limit_bytes < 1
        or type(workspace_limit_bytes) is not int
        or workspace_limit_bytes < 1
    ):
        raise ValueError("Invalid GMRES restart, memory limit or workspace limit")
    n = (case.ny - 2) * (case.nx - 2)
    size = 2 * n + 1
    factor_coefficient = max(
        factor_bytes / (reference_n**1.5)
        for reference_n, factor_bytes in _REFERENCE_FACTOR_BYTES
    )
    estimated_factor_bytes = math.ceil(
        _FACTOR_FILL_SAFETY * factor_coefficient * n**1.5
    )
    estimated_matrix_nonzeros = _MATRIX_NNZ_PER_INTERIOR_ESTIMATE * n
    estimated_sparse_payload_bytes = (
        12 * estimated_matrix_nonzeros + 4 * (size + 1)
    )
    basis_bytes = 2 * size * (gmres_restart + 3) * np.dtype(np.float64).itemsize
    fft_shape = tuple(next_fast_len(3 * axis - 2) for axis in (case.ny, case.nx))
    fft_workspace_bytes = 128 * math.prod(fft_shape)
    estimated_peak = (
        2 * (2 * estimated_sparse_payload_bytes + estimated_factor_bytes + basis_bytes)
        + 2 * fft_workspace_bytes
        + 160 * case.ny * case.nx
    )
    adjusted_peak = math.ceil(estimated_peak * _PREFLIGHT_SAFETY_MARGIN)
    available = (
        _available_memory_bytes()
        if available_memory_bytes is None
        else int(available_memory_bytes)
    )
    if available < 0:
        raise ValueError("Available memory cannot be negative")
    memory_ceiling = min(memory_limit_bytes, 3 * available // 4)
    return {
        "physical_nodes": case.ny * case.nx,
        "interior_nodes": n,
        "bordered_unknowns": size,
        "transport_fraction": 1.0,
        "gmres_restart": gmres_restart,
        "matrix_nonzeros_estimate": estimated_matrix_nonzeros,
        "estimated_sparse_payload_bytes": estimated_sparse_payload_bytes,
        "estimated_factor_bytes": estimated_factor_bytes,
        "krylov_basis_bytes": basis_bytes,
        "elastic_fft_workspace_bytes": fft_workspace_bytes,
        "estimated_peak_bytes": estimated_peak,
        "safety_margin": _PREFLIGHT_SAFETY_MARGIN,
        "adjusted_peak_bytes": adjusted_peak,
        "configured_memory_limit_bytes": memory_limit_bytes,
        "configured_workspace_limit_bytes": workspace_limit_bytes,
        "available_memory_before_run_bytes": available,
        "memory_ceiling_bytes": memory_ceiling,
        "memory_preflight_passed": adjusted_peak <= memory_ceiling,
        "workspace_preflight_passed": fft_workspace_bytes <= workspace_limit_bytes,
        "factor_estimator": {
            "growth_exponent": 1.5,
            "safety_multiplier": _FACTOR_FILL_SAFETY,
            "reference_interior_nodes": [n for n, _ in _REFERENCE_FACTOR_BYTES],
        },
    }


def resource_preflight(
    case,
    state,
    *,
    transport_fraction=1.0,
    gmres_restart=60,
    memory_limit_bytes=_DEFAULT_MEMORY_LIMIT_BYTES,
    workspace_limit_bytes=2**30,
    use_measured_peak=True,
):
    """Check a conservative estimate before constructing/factorizing sparse matrices."""
    if (
        isinstance(transport_fraction, bool)
        or not np.isfinite(transport_fraction)
        or not 0 <= transport_fraction <= 1
    ):
        raise ValueError("transport_fraction must be finite in [0,1]")
    estimate = estimate_resource_requirements(
        case,
        gmres_restart=gmres_restart,
        memory_limit_bytes=memory_limit_bytes,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    result = dict(
        **estimate,
        estimated_peak_with_margin_bytes=estimate["estimated_peak_bytes"],
        measured_peak_bytes=None,
        measured_peak_working_set_bytes=None,
        safety_margin_ratio=None,
        factor_bytes=None,
        known_vector_relative_residual=None,
        field_preflight_passed=None,
        factor_preflight_passed=None,
        factor_preflight_transport_fraction=float(transport_fraction),
        use_measured_peak=use_measured_peak,
    )
    if not estimate["memory_preflight_passed"]:
        result.update(
            passed=False,
            status="rejected_preflight",
            reason="Conservative memory estimate exceeds configured ceiling",
        )
        return result
    if not estimate["workspace_preflight_passed"]:
        result.update(
            passed=False,
            status="rejected_preflight",
            reason="Estimated elastic FFT workspace exceeds configured workspace limit",
        )
        return result
    if state is None:
        result.update(passed=True, status="estimate_only")
        return result
    expected_size = 2 * estimate["interior_nodes"] + 1
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (expected_size,) or not np.all(np.isfinite(state)):
        raise ValueError("Finite full P,D,h0/Href state required for factor preflight")
    before = _process_memory_snapshot() if use_measured_peak else None
    probe = LimitedLoadProbe(
        case,
        transport_fraction=transport_fraction,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    guide = np.zeros(probe.size, dtype=np.float64)
    model = LimitedLoadLinearization(
        case,
        state,
        guide,
        preconditioner="local",
        active_full=state[: probe.n] >= state[probe.n : 2 * probe.n],
        transport_fraction=transport_fraction,
        workspace_limit_bytes=workspace_limit_bytes,
    )
    matrix = model.frozen_matrix(state)
    field_matrix = matrix[: 2 * probe.n, : 2 * probe.n].tocsc()
    field_lu = None
    bordered_lu = None
    field_factor_error = None
    try:
        field_lu = splu(field_matrix)
    except RuntimeError as exc:
        field_factor_error = str(exc)
    try:
        bordered_lu = splu(matrix)
    except RuntimeError as exc:
        after = _process_memory_snapshot() if before is not None else None
        measured_peak_bytes = None
        if after is not None:
            measured_peak_bytes = max(
                0, after["peak_private_bytes"] - before["current_private_bytes"]
            )
            result["measured_peak_working_set_bytes"] = max(
                0,
                after["peak_working_set_bytes"] - before["current_working_set_bytes"],
            )
        measured_memory_passed = (
            measured_peak_bytes is None
            or measured_peak_bytes <= estimate["memory_ceiling_bytes"]
        )
        result.update(
            measured_peak_bytes=measured_peak_bytes,
            measured_memory_preflight_passed=measured_memory_passed,
            memory_preflight_passed=bool(
                estimate["memory_preflight_passed"] and measured_memory_passed
            ),
            safety_margin_ratio=(
                None
                if measured_peak_bytes is None
                else measured_peak_bytes / estimate["estimated_peak_bytes"]
            ),
        )
        result.update(
            matrix_nonzeros=int(matrix.nnz),
            field_preflight_passed=field_lu is not None,
            field_preflight_error=field_factor_error,
            factor_preflight_passed=False,
            passed=False,
            status="rejected_preflight",
            reason=(
                f"SuperLU bordered fallback preflight failed: {exc}"
                if field_factor_error is None
                else (
                    "SuperLU field factorization and bordered fallback preflight failed: "
                    f"{field_factor_error}; {exc}"
                )
            ),
        )
        del field_lu, bordered_lu, field_matrix, matrix, model, probe
        gc.collect()
        return result
    factors = tuple(lu for lu in (field_lu, bordered_lu) if lu is not None)
    factor_bytes = sum(
        _payload_bytes(lu.L) + _payload_bytes(lu.U) for lu in factors
    )
    factor_bytes += sum(lu.perm_c.nbytes + lu.perm_r.nbytes for lu in factors)
    known = np.random.default_rng(17329929).normal(size=probe.size)
    rhs = matrix @ known
    relative_residual = float(
        np.linalg.norm(matrix @ bordered_lu.solve(rhs) - rhs)
        / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
    )
    measured_peak_bytes = None
    measured_peak_working_set_bytes = None
    if before is not None:
        after = _process_memory_snapshot()
        measured_peak_bytes = max(
            0, after["peak_private_bytes"] - before["current_private_bytes"]
        )
        measured_peak_working_set_bytes = max(
            0,
            after["peak_working_set_bytes"] - before["current_working_set_bytes"],
        )
    safety_margin_ratio = (
        None
        if measured_peak_bytes is None or estimate["estimated_peak_bytes"] == 0
        else measured_peak_bytes / estimate["estimated_peak_bytes"]
    )
    memory_measured_passed = (
        measured_peak_bytes is None
        or measured_peak_bytes <= estimate["memory_ceiling_bytes"]
    )
    factor_passed = bool(
        np.isfinite(relative_residual) and relative_residual < 1e-8
    )
    result.update(
        matrix_nonzeros=int(matrix.nnz),
        factor_bytes=int(factor_bytes),
        measured_peak_bytes=measured_peak_bytes,
        measured_peak_working_set_bytes=measured_peak_working_set_bytes,
        safety_margin_ratio=safety_margin_ratio,
        known_vector_relative_residual=relative_residual,
        field_preflight_passed=field_lu is not None,
        field_preflight_error=field_factor_error,
        memory_preflight_passed=bool(
            estimate["memory_preflight_passed"] and memory_measured_passed
        ),
        factor_preflight_passed=factor_passed,
        measured_memory_preflight_passed=memory_measured_passed,
        passed=bool(
            estimate["memory_preflight_passed"]
            and memory_measured_passed
            and factor_passed
        ),
        status=(
            "passed"
            if estimate["memory_preflight_passed"] and memory_measured_passed and factor_passed
            else "rejected_preflight"
        ),
    )
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
    resume_from=None,
    use_measured_peak=True,
    progress_interval=300.0,
    gmres_restart=60,
    memory_limit_bytes=_DEFAULT_MEMORY_LIMIT_BYTES,
    workspace_limit_bytes=2**30,
):
    if (
        isinstance(progress_interval, bool)
        or not np.isfinite(progress_interval)
        or progress_interval <= 0
        or type(gmres_restart) is not int
        or not 1 <= gmres_restart <= 60
        or type(memory_limit_bytes) is not int
        or memory_limit_bytes < 1
        or type(workspace_limit_bytes) is not int
        or workspace_limit_bytes < 1
    ):
        raise ValueError("Invalid progress, GMRES, memory or workspace setting")
    if sum(value is not None for value in (seed, limited_seed, resume_from)) > 1:
        raise ValueError(
            "Choose only one of a prototype seed, limited MC seed, or resume checkpoint"
        )
    seed_metadata = None
    resume_fraction = 0.0
    if resume_from is not None:
        summary = json.loads((resume_from / "summary.json").read_text())
        saved_case = PointContactCase(**summary["case"])
        case = (
            PointContactCase(**json.loads(case_path.read_text()))
            if case_path is not None
            else saved_case
        )
    elif limited_seed is not None:
        if case_path is None:
            raise ValueError("A target case is required when refining an MC seed")
        case = PointContactCase(**json.loads(case_path.read_text()))
    elif seed is not None:
        seed_summary = json.loads((seed / "summary.json").read_text())
        case = PointContactCase(**seed_summary["case"])
        if case_path is not None and PointContactCase(**json.loads(case_path.read_text())) != case:
            raise ValueError("Seed and case must match exactly")
    else:
        case = (
            PointContactCase(**json.loads(case_path.read_text()))
            if case_path
            else PointContactCase()
        )

    estimate = None
    if case.maximum_physical_nodes > 20000:
        estimate = estimate_resource_requirements(
            case,
            gmres_restart=gmres_restart,
            memory_limit_bytes=memory_limit_bytes,
            workspace_limit_bytes=workspace_limit_bytes,
        )
        if not estimate["memory_preflight_passed"] or not estimate["workspace_preflight_passed"]:
            output.mkdir(parents=True, exist_ok=False)
            preflight = dict(
                **estimate,
                status="rejected_preflight",
                passed=False,
                preflight_stage="before_seed_loading_or_matrix_construction",
                reason=(
                    "Conservative memory estimate exceeds configured ceiling"
                    if not estimate["memory_preflight_passed"]
                    else "Estimated elastic FFT workspace exceeds configured limit"
                ),
            )
            _write_json_atomic(
                output / "summary.json",
                dict(
                    status="rejected_preflight",
                    case=asdict(case),
                    operator_transport="limited_capacity_target",
                    target_transport_fraction=1.0,
                    gmres_restart=gmres_restart,
                    memory_limit_bytes=memory_limit_bytes,
                    workspace_limit_bytes=workspace_limit_bytes,
                    resource_preflight=preflight,
                ),
            )
            print("rejected_preflight", preflight["reason"], flush=True)
            return 1

    if resume_from is not None:
        initial, resume_fraction, seed_metadata = _load_resume_checkpoint(resume_from, case)
    elif limited_seed is not None:
        initial, seed_metadata = _interpolate_limited_seed(
            limited_seed,
            case,
            workspace_limit_bytes=workspace_limit_bytes,
        )
    elif seed is not None:
        seed_summary = json.loads((seed / "summary.json").read_text())
        if seed_summary.get("equation") == "donor-flux JFO/Elrod-Adams":
            from examples.run_ciulli_donor_grid import _seed_state

            probe = LimitedLoadProbe(
                case,
                transport_fraction=0.0,
                workspace_limit_bytes=workspace_limit_bytes,
            )
            initial, seed_metadata = _seed_state(seed, case, probe)
            seed_metadata["target_transport_fraction"] = 1.0
        else:
            loaded_case, _, _ = load_run(seed)
            if loaded_case != case:
                raise ValueError("Seed and case must match exactly")
            with np.load(seed / "solution.npz", allow_pickle=False) as saved:
                initial = saved["state_dimensionless"].copy()
                probe = LimitedLoadProbe(
                    case,
                    transport_fraction=0.0,
                    workspace_limit_bytes=workspace_limit_bytes,
                )
                p, t, h = probe.fields(initial)
                for actual, key in [(p, "pressure_pa"), (t, "theta"), (h, "film_m")]:
                    np.testing.assert_array_equal(actual, saved[key])
            seed_metadata = dict(
                path=str(seed),
                kind="independently audited prototype root",
                state_sha256=digest(seed / "solution.npz"),
                accepted_for_target=False,
            )
    else:
        initial = None

    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running",
        case=asdict(case),
        operator_transport="limited_capacity_target",
        target_transport_fraction=1.0,
        initial_transport_fraction=resume_fraction,
        initial_state_accepted_for_target=False,
        seed=(
            seed_metadata
            if limited_seed is not None or resume_from is not None or seed_metadata is not None
            else None
            if seed is None
            else dict(path=str(seed), sha256=digest(seed / "solution.npz"))
        ),
        resume_checkpoint=seed_metadata if resume_from is not None else None,
        maximum_attempts=maximum_attempts,
        maximum_physical_nodes=case.maximum_physical_nodes,
        endpoint_prediction_enabled=endpoint_prediction,
        maximum_corrections_per_attempt=30,
        maximum_models_per_correction=8,
        maximum_krylov_cycles=10,
        gmres_restart=gmres_restart,
        memory_limit_bytes=memory_limit_bytes,
        workspace_limit_bytes=workspace_limit_bytes,
        resource_estimate=estimate,
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
        _write_json_atomic(output / "summary.json", report)

    def stage(index, stage_case, fraction, outcome):
        heartbeat.update(
            stage="stage_checkpoint", attempt=index + 1,
            transport_fraction=fraction, emit=True,
        )
        folder = output / f"attempt_{index:03d}"
        folder.mkdir(exist_ok=False)
        probe = LimitedLoadProbe(
            stage_case,
            transport_fraction=fraction,
            workspace_limit_bytes=workspace_limit_bytes,
        )
        p, t, h = probe.fields(outcome.state)
        probe.system.sync(outcome.state)
        _, _, _, w, offset = probe.system.base.unpack(outcome.state[:-1])
        state_path = folder / "state.npz"
        temporary_state_path = folder / "state.npz.tmp"
        with temporary_state_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                state_dimensionless=outcome.state,
                pressure_pa=p,
                theta=t,
                film_m=h,
                displacement_m=w,
                offset_m=offset,
                x_m=probe.system.problem.x_m,
                y_m=probe.system.problem.y_m,
            )
        os.replace(temporary_state_path, state_path)
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
        _write_json_atomic(folder / "summary.json", info)
        report["stages"].append({k: v for k, v in info.items() if k != "history"})
        if info["status"] == "accepted_stage" and fraction < 1:
            checkpoint_path = output / "last_accepted_state.npz"
            temporary_checkpoint_path = output / "last_accepted_state.npz.tmp"
            shutil.copyfile(state_path, temporary_checkpoint_path)
            os.replace(temporary_checkpoint_path, checkpoint_path)
            report["latest_accepted_checkpoint"] = dict(
                stage_index=index,
                transport_fraction=fraction,
                state_file=checkpoint_path.name,
                state_sha256=digest(checkpoint_path),
            )
        write()
        heartbeat.update(stage="stage_checkpoint_complete", emit=True)

    write()
    heartbeat = SolverHeartbeat(output / "heartbeat.jsonl", interval_seconds=progress_interval)
    active_progress = {}
    previous_sigterm = signal.signal(signal.SIGTERM, _termination_handler)
    result = None
    try:
        if case.maximum_physical_nodes > 20000:
            if initial is None:
                interior = (case.ny - 2) * (case.nx - 2)
                initial_for_preflight = np.full(2 * interior + 1, 0.5, dtype=np.float64)
                initial_for_preflight[-1] = 0.1
            else:
                initial_for_preflight = initial
            try:
                preflight = resource_preflight(
                    case,
                    initial_for_preflight,
                    transport_fraction=0.5 * (1.0 + resume_fraction),
                    gmres_restart=gmres_restart,
                    memory_limit_bytes=memory_limit_bytes,
                    workspace_limit_bytes=workspace_limit_bytes,
                    use_measured_peak=use_measured_peak,
                )
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

        def observe_progress(attempt, fraction, iteration, audit, status):
            progress.update(attempt, fraction, iteration, audit, status)
            if status != "Newton correction":
                heartbeat.update(
                    stage=status.replace(" ", "_"),
                    attempt=None if attempt is None else attempt + 1,
                    transport_fraction=fraction,
                    emit=True,
                )
            active_progress.update(
                attempt=attempt,
                transport_fraction=fraction,
                newton_iteration=iteration,
                status=status,
                updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                relative_local_balance=(
                    None
                    if audit is None
                    else audit.get("mass", {}).get("relative_local_balance")
                ),
                force_n=None if audit is None else audit.get("force_n"),
            )

        def observe_heartbeat(attempt, fraction, **event):
            stage_name = event.pop("stage")
            iteration = event.pop("iteration", None)
            emit = event.pop("emit", False)
            if stage_name == "correction_start":
                heartbeat.begin_correction(
                    iteration,
                    maximum_newton_iterations=event.pop("maximum_newton_iterations"),
                    maximum_gmres_iterations=event.pop("maximum_gmres_iterations"),
                )
            if stage_name == "linear_solve":
                event.update(
                    gmres_iterations=0,
                    gmres_restart_count=0,
                    gmres_stagnation_restarts=0,
                    gmres_residual_estimate=None,
                    gmres_true_relative_residual=None,
                    cumulative_jvp_seconds=0.0,
                    cumulative_preconditioner_seconds=0.0,
                    cumulative_superlu_seconds=0.0,
                )
            heartbeat.update(
                stage=stage_name,
                iteration=iteration,
                emit=emit,
                attempt=attempt + 1,
                transport_fraction=fraction,
                gpu_enabled=False,
                **event,
            )

        with _ProgressReporter(progress_interval) as progress:
            observe_progress(None, None, None, None, "continuation starting")
            path = continue_limited_transport(
                case,
                initial,
                maximum_attempts=maximum_attempts,
                stage_observer=stage,
                endpoint_prediction=endpoint_prediction,
                progress_observer=observe_progress,
                heartbeat_observer=observe_heartbeat,
                initial_transport_fraction=resume_fraction,
                gmres_restart=gmres_restart,
                workspace_limit_bytes=workspace_limit_bytes,
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
    except KeyboardInterrupt as exc:
        checkpoint = report.get("latest_accepted_checkpoint")
        report.update(
            status="interrupted",
            reason=str(exc) or "Received KeyboardInterrupt",
            last_progress=active_progress or None,
            last_accepted_fraction=(
                None if checkpoint is None else checkpoint["transport_fraction"]
            ),
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        heartbeat.close(status="failed" if report["status"] == "running" else report["status"])
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
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume at the latest completed accepted stage in an interrupted run",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-attempts",
        type=int,
        choices=range(1, MAXIMUM_STAGE_ATTEMPTS + 1),
        default=24,
    )
    parser.add_argument(
        "--no-endpoint-prediction", dest="endpoint_prediction", action="store_false"
    )
    parser.add_argument(
        "--no-measured-peak", dest="use_measured_peak", action="store_false",
        help="Disable measured peak memory tracking during preflight (uses estimate only)"
    )
    parser.add_argument(
        "--gmres-restart",
        type=int,
        choices=range(1, 61),
        default=60,
        help="GMRES restart length; Krylov tolerances and acceptance gates are unchanged",
    )
    parser.add_argument(
        "--memory-limit-mib",
        type=int,
        default=_DEFAULT_MEMORY_LIMIT_BYTES // 1024**2,
        help="Configured process-memory preflight ceiling in MiB",
    )
    parser.add_argument(
        "--workspace-limit-mib",
        type=int,
        default=1024,
        help="Elastic FFT workspace limit in MiB",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=300.0,
        help="Emit a progress heartbeat every N seconds (default: 300)",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            args.case, args.seed, args.output, args.max_attempts,
            endpoint_prediction=args.endpoint_prediction,
            limited_seed=args.limited_seed,
            resume_from=args.resume_from,
            use_measured_peak=args.use_measured_peak,
            progress_interval=args.progress_interval,
            gmres_restart=args.gmres_restart,
            memory_limit_bytes=args.memory_limit_mib * 1024**2,
            workspace_limit_bytes=args.workspace_limit_mib * 1024**2,
        )
    )
