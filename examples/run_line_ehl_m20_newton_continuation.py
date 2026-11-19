"""Reach the Venner/ten Napel M=20, L=10 line-EHL root by load continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from run_line_ehl_physical_diagnostics import (
    VENNER_FIG5_FILM,
    VENNER_FIG5_FILM_X,
    VENNER_FIG5_PRESSURE,
    VENNER_FIG5_X,
    VENNER_FILM_UNCERTAINTY,
    VENNER_PRESSURE_UNCERTAINTY,
    VENNER_SPIKE_MINIMUM,
    VENNER_SPIKE_MINIMUM_X,
    VENNER_SPIKE_PEAK,
    VENNER_SPIKE_PEAK_X,
    VENNER_X_UNCERTAINTY,
    _plot_diagnostics,
    _profile_csv,
    _reference_profile_errors,
    _venner_case,
)
from scipy.fft import next_fast_len

from ehl_solver.backend import make_backend
from ehl_solver.heartbeat import SolverHeartbeat
from ehl_solver.hertz import line_contact_hertz
from ehl_solver.line_ehl import (
    LineEHLConvergenceError,
    analyze_line_ehl_features,
    solve_line_ehl_at_offset,
)
from ehl_solver.line_ehl_newton import (
    LineEHLNewtonSystem,
    LineJFONewtonError,
    LineJFONewtonSolution,
    solve_line_ehl_newton,
)


class _RunInterrupted(KeyboardInterrupt):
    pass


def _termination_handler(signum, _frame):
    raise _RunInterrupted(f"Received {signal.Signals(signum).name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_state_atomic(path: Path, state: np.ndarray) -> str:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, state_dimensionless=state)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return _sha256(path)


def _state_from_solution(solution: LineJFONewtonSolution) -> np.ndarray:
    return np.array(solution.state, dtype=np.float64, copy=True)


def _case_signature(case, dimensionless, inputs, nodes):
    coordinate_bytes = np.asarray(case.x_m, dtype="<f8").tobytes()
    return {
        "benchmark": "Venner and ten Napel (1990), M=20, L=10",
        "inputs_si": inputs,
        "dimensionless_groups": dimensionless,
        "grid": {
            "nodes": nodes,
            "x_start_m": float(case.x_m[0]),
            "x_end_m": float(case.x_m[-1]),
            "spacing_m": float(case.dx_m),
            "coordinates_sha256": hashlib.sha256(coordinate_bytes).hexdigest(),
        },
        "line_load_n_m": case.line_load_n_m,
        "e_star_pa": case.e_star_pa,
        "fluid": asdict(case.fluid),
        "geometry": asdict(case.geometry),
        "entrainment_speed_m_s": case.entrainment_speed_m_s,
        "length_scale_m": case.length_scale_m,
        "film_scale_m": case.film_scale_m,
        "pressure_scale_pa": case.pressure_scale_pa,
        "elastic_reference_length_m": case.elastic_reference_length_m,
        "inlet_fractional_film_content": case.inlet_fractional_film_content,
    }


def _implementation_signature():
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "src" / "ehl_solver" / "line_ehl_newton.py",
        root / "src" / "ehl_solver" / "line_ehl.py",
        root / "src" / "ehl_solver" / "line_cavitation.py",
        root / "src" / "ehl_solver" / "line_contact.py",
        root / "src" / "ehl_solver" / "constitutive.py",
        root / "src" / "ehl_solver" / "backend.py",
        root / "src" / "ehl_solver" / "device_krylov.py",
        root / "src" / "ehl_solver" / "hybrid_jfo.py",
        root / "src" / "ehl_solver" / "heartbeat.py",
        Path(__file__).with_name("run_line_ehl_physical_diagnostics.py").resolve(),
    )
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in paths
    }


def _control_signature(args):
    return {
        "maximum_newton_iterations": args.maximum_newton_iterations,
        "gmres_restart": args.gmres_restart,
        "maximum_gmres_iterations": args.maximum_gmres_iterations,
        "maximum_krylov_recoveries": 2,
        "krylov_rtol": 1e-8,
        "armijo_constant": 1e-4,
        "compliance_factor": 1.0,
        "constitutive_strength": 1.0,
    }


def _load_resume(
    path: Path,
    signature,
    implementation,
    controls,
    *,
    allow_verified_source_changes=False,
    allow_verified_gmres_restart_change=False,
):
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") == "accepted_numerical_root":
        raise ValueError("An accepted endpoint is not a resumable partial path")
    if summary.get("case_signature") != signature:
        raise ValueError("Resume checkpoint does not match the fixed benchmark and grid")
    prior_implementation = summary.get("implementation_sha256")
    changed_sources = []
    if prior_implementation != implementation:
        if (
            not allow_verified_source_changes
            or not isinstance(prior_implementation, dict)
            or not isinstance(implementation, dict)
        ):
            raise ValueError("Resume checkpoint was produced by different solver sources")
        prior_normalized = {
            key.replace("\\", "/"): value
            for key, value in prior_implementation.items()
        }
        current_normalized = {
            key.replace("\\", "/"): value
            for key, value in implementation.items()
        }
        changed_sources = sorted(
            name
            for name in set(prior_normalized) | set(current_normalized)
            if prior_normalized.get(name) != current_normalized.get(name)
        )
        compatible_sources = {
            "examples/run_line_ehl_m20_newton_continuation.py",
            "src/ehl_solver/line_ehl_newton.py",
            "src/ehl_solver/line_ehl.py",
            "src/ehl_solver/line_cavitation.py",
        }
        if not changed_sources or not set(changed_sources) <= compatible_sources:
            raise ValueError(
                "Resume source changes exceed the explicitly compatible files: "
                f"{changed_sources}"
            )
    prior_controls = summary.get("solver_control_signature")
    changed_controls = []
    if prior_controls != controls:
        if (
            not allow_verified_gmres_restart_change
            or not isinstance(prior_controls, dict)
            or not isinstance(controls, dict)
        ):
            raise ValueError("Resume checkpoint uses different Newton/Krylov controls")
        changed_controls = sorted(
            name
            for name in set(prior_controls) | set(controls)
            if prior_controls.get(name) != controls.get(name)
        )
        if (
            changed_controls != ["gmres_restart"]
            or (
                prior_controls.get("gmres_restart"),
                controls.get("gmres_restart"),
            )
            not in {(60, 120), (120, 240)}
        ):
            raise ValueError(
                "Only reviewed GMRES restart increases from 60 to 120 or "
                "120 to 240 are compatible with resume; changed controls: "
                f"{changed_controls}"
            )
    checkpoint = summary.get("latest_accepted_checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume directory has no accepted load checkpoint")
    state_path = path / checkpoint["state_file"]
    if _sha256(state_path) != checkpoint.get("state_sha256"):
        raise ValueError("Accepted resume checkpoint checksum does not match")
    with np.load(state_path, allow_pickle=False) as saved:
        if "state_dimensionless" not in saved:
            raise ValueError("Resume checkpoint has no dimensionless state")
        state = np.array(saved["state_dimensionless"], dtype=np.float64, copy=True)
    expected_size = 2 * (signature["grid"]["nodes"] - 2) + 1
    if state.shape != (expected_size,) or not np.all(np.isfinite(state)):
        raise ValueError("Resume checkpoint state has invalid shape or values")
    return state, float(checkpoint["load_n_m"]), float(checkpoint["fraction"]), {
        "directory": str(path),
        "summary_sha256": _sha256(summary_path),
        "state_sha256": checkpoint["state_sha256"],
        "stage_index": checkpoint.get("stage_index"),
        "verified_source_changes": changed_sources,
        "verified_control_changes": changed_controls,
    }


def _load_refinement_seed(path: Path, signature):
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_signature = summary.get("case_signature")
    if (
        summary.get("status") != "accepted_numerical_root"
        or summary.get("accepted_endpoint") is not True
        or not isinstance(source_signature, dict)
    ):
        raise ValueError("Refinement seed must be an accepted endpoint root")
    source_grid = source_signature.get("grid", {})
    target_grid = signature["grid"]
    source_nodes = source_grid.get("nodes")
    if (
        not isinstance(source_nodes, int)
        or source_nodes >= target_grid["nodes"]
        or any(
            source_signature.get(key) != signature.get(key)
            for key in set(source_signature) - {"grid"}
        )
        or source_grid.get("x_start_m") != target_grid["x_start_m"]
        or source_grid.get("x_end_m") != target_grid["x_end_m"]
    ):
        raise ValueError("Refinement seed does not match the fixed benchmark/domain")
    checkpoint = summary.get("latest_accepted_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("fraction") != 1.0
        or checkpoint.get("load_n_m") != signature["line_load_n_m"]
    ):
        raise ValueError("Refinement seed is not at the fixed endpoint load")
    state_path = path / checkpoint["state_file"]
    if _sha256(state_path) != checkpoint.get("state_sha256"):
        raise ValueError("Refinement seed checkpoint checksum does not match")
    with np.load(state_path, allow_pickle=False) as saved:
        if "state_dimensionless" not in saved:
            raise ValueError("Refinement seed has no dimensionless state")
        source_state = np.array(
            saved["state_dimensionless"], dtype=np.float64, copy=True
        )

    source_interior = source_nodes - 2
    expected_size = 2 * source_interior + 1
    if (
        source_state.shape != (expected_size,)
        or not np.all(np.isfinite(source_state))
    ):
        raise ValueError("Refinement seed state has invalid shape or values")
    source_x = np.linspace(
        source_grid["x_start_m"],
        source_grid["x_end_m"],
        source_nodes,
    )
    target_x = np.linspace(
        target_grid["x_start_m"],
        target_grid["x_end_m"],
        target_grid["nodes"],
    )
    pressure = np.zeros(source_nodes)
    deficit = np.zeros(source_nodes)
    pressure[1:-1] = source_state[:source_interior]
    deficit[1:-1] = source_state[source_interior : 2 * source_interior]
    target_interior = target_grid["nodes"] - 2
    state = np.concatenate(
        (
            np.interp(target_x[1:-1], source_x, pressure),
            np.interp(target_x[1:-1], source_x, deficit),
            [source_state[-1]],
        )
    )
    return state, float(checkpoint["load_n_m"]), 1.0, {
        "directory": str(path),
        "summary_sha256": _sha256(summary_path),
        "state_sha256": checkpoint["state_sha256"],
        "source_nodes": source_nodes,
        "target_nodes": target_grid["nodes"],
        "target_interior_nodes": target_interior,
        "method": "linear interpolation of dimensionless pressure/content; h0 preserved",
    }


def _preflight(case, hertz, dimensionless, inputs, backend, args):
    import torch

    nodes = case.x_m.size
    state_size = 2 * (nodes - 2) + 1
    gmres_m = min(state_size, args.gmres_restart, args.maximum_gmres_iterations)
    gmres_workspace = (
        16 * state_size * (gmres_m + 5) + 32 * (gmres_m + 1) ** 2
    )
    fft_size = next_fast_len(3 * nodes - 2)
    fft_workspace = 64 * fft_size
    free_gpu_bytes, total_gpu_bytes = torch.cuda.mem_get_info()
    x_over_b = [
        float(case.x_m[0] / hertz.half_width_m),
        float(case.x_m[-1] / hertz.half_width_m),
    ]
    return {
        "status": "preflight_passed",
        "inputs_si": inputs,
        "dimensionless_groups": dimensionless,
        "hertz": {
            "half_width_b_m": hertz.half_width_m,
            "peak_pressure_p0_pa": hertz.peak_pressure_pa,
            "approach_at_reference_m": hertz.approach_m,
            "elastic_reference_length_m": hertz.reference_length_m,
        },
        "scales": {
            "film_scale_b2_over_R_m": case.film_scale_m,
            "digitized_central_film_estimate_m": 0.0866 * case.film_scale_m,
            "domain_x_over_b": x_over_b,
            "elastic_datum_over_b": (
                case.elastic_reference_length_m / hertz.half_width_m
            ),
        },
        "numerical_preflight": {
            "grid_nodes": nodes,
            "state_size": state_size,
            "line_fft_length": fft_size,
            "line_fft_workspace_bytes": fft_workspace,
            "gmres_restart": args.gmres_restart,
            "maximum_gmres_iterations_per_correction": (
                args.maximum_gmres_iterations
            ),
            "gmres_workspace_bytes": gmres_workspace,
            "maximum_attempts": args.maximum_attempts,
            "maximum_newton_iterations_per_attempt": (
                args.maximum_newton_iterations
            ),
            "maximum_krylov_recoveries": 2,
            "maximum_linear_iterations": (
                args.maximum_attempts
                * args.maximum_newton_iterations
                * args.maximum_gmres_iterations
                * 7
            ),
            "estimated_gpu_working_set_bytes": (
                fft_workspace + gmres_workspace + 4 * state_size * 8
            ),
            "conservative_cpu_superlu_dense_factor_bytes": (
                16 * state_size**2
            ),
            "cuda_free_bytes": int(free_gpu_bytes),
            "cuda_total_bytes": int(total_gpu_bytes),
            "backend": f"{backend.platform}/{backend.device}",
            "cpu_superlu_dimension": state_size,
            "runtime_estimate": (
                "The 257-node path has bounded work but no prior accepted "
                "M=20 Newton timing. The driver reports a measured pilot-stage "
                "runtime and extrapolates subsequent stages."
            ),
        },
        "physical_domain_assessment": (
            "The mapped published domain [-4b, 1.5b] is retained unchanged; "
            "no extrapolation beyond the published computational interval."
        ),
    }


def _solution_record(solution, case):
    density, viscosity, _, _ = case.fluid.properties(
        solution.pressure_pa
    )
    maximum_pressure_index = int(np.argmax(solution.pressure_pa))
    return {
        "target_load_n_m": solution.target_load_n_m,
        "calculated_load_n_m": solution.calculated_line_load_n_m,
        "relative_load_error": solution.relative_load_error,
        "h0_m": solution.central_film_offset_m,
        "h_c_m": float(
            np.interp(0.0, case.x_m, solution.film_thickness_m)
        ),
        "h_min_m": float(np.min(solution.film_thickness_m)),
        "p_max_pa": float(solution.pressure_pa[maximum_pressure_index]),
        "p_max_x_m": float(case.x_m[maximum_pressure_index]),
        "maximum_viscosity_pa_s": float(
            np.max(viscosity)
        ),
        "maximum_density_kg_m3": float(np.max(density)),
        "operator_backend": solution.backend,
        "preconditioner": (
            "CPUBlockSchurPreconditioner (SciPy SuperLU; frozen Reynolds matrix)"
        ),
        "audit": solution.audit,
        "newton_iterations": len(solution.history),
    }


def _build_reference_comparison(case, hertz, solution, features):
    reduced_film = (
        solution.film_thickness_m
        * case.geometry.radius_x_m
        / hertz.half_width_m**2
    )
    central_film = float(np.interp(0.0, case.x_m, solution.film_thickness_m))
    minimum_index = int(np.argmin(solution.film_thickness_m))
    return {
        "figure_5_bulk_pressure": _reference_profile_errors(
            VENNER_FIG5_X[3:12],
            VENNER_FIG5_PRESSURE[3:12],
            case.x_m / hertz.half_width_m,
            solution.pressure_pa / hertz.peak_pressure_pa,
            hertz.peak_pressure_pa,
        ),
        "figure_5_bulk_film": _reference_profile_errors(
            VENNER_FIG5_FILM_X[4:12],
            VENNER_FIG5_FILM[4:12],
            case.x_m / hertz.half_width_m,
            reduced_film,
            case.film_scale_m,
        ),
        "figure_5_outlet_film": _reference_profile_errors(
            VENNER_FIG5_FILM_X[12:],
            VENNER_FIG5_FILM[12:],
            case.x_m / hertz.half_width_m,
            reduced_film,
            case.film_scale_m,
        ),
        "film_and_spike": {
            "digitized_h_c_m": float(VENNER_FIG5_FILM[8] * case.film_scale_m),
            "computed_h_c_m": central_film,
            "digitized_h_min_m": float(
                np.min(VENNER_FIG5_FILM) * case.film_scale_m
            ),
            "computed_h_min_m": float(solution.film_thickness_m[minimum_index]),
            "computed_h_min_x_over_b": float(
                case.x_m[minimum_index] / hertz.half_width_m
            ),
            "figure_6_spike_x_over_b": VENNER_SPIKE_PEAK_X,
            "figure_6_spike_pressure_over_hertz": VENNER_SPIKE_PEAK,
            "computed_spike_x_over_b": (
                None
                if features.petrusevich_spike_location_m is None
                else features.petrusevich_spike_location_m
                / hertz.half_width_m
            ),
            "computed_spike_pressure_over_hertz": (
                None
                if features.petrusevich_spike_location_m is None
                else float(
                    np.interp(
                        features.petrusevich_spike_location_m,
                        case.x_m,
                        solution.pressure_pa,
                    )
                    / hertz.peak_pressure_pa
                )
            ),
            "digitization_uncertainty": {
                "figure_5_x_over_b": VENNER_X_UNCERTAINTY,
                "figure_5_pressure_over_hertz": VENNER_PRESSURE_UNCERTAINTY,
                "figure_5_hR_over_b2": VENNER_FILM_UNCERTAINTY,
                "figure_6_peak_x_over_b": 0.0025,
                "figure_6_peak_pressure_over_hertz": 0.001,
                "table_1_pre_spike_x_over_b": VENNER_SPIKE_MINIMUM_X,
                "table_1_pre_spike_pressure_over_hertz": VENNER_SPIKE_MINIMUM,
            },
        },
        "cavitation_limitation": (
            "The paper's pressure calculation uses P>=0 without the current "
            "solver's fractional-film-content JFO closure."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, choices=(257, 513, 1025), default=257)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--refine-from", type=Path)
    parser.add_argument(
        "--allow-verified-source-changes",
        action="store_true",
        help=(
            "Allow reviewed numerical-only changes in the continuation runner, "
            "Newton gate, and feature analysis when resuming a checksummed state."
        ),
    )
    parser.add_argument(
        "--allow-verified-gmres-restart-change",
        action="store_true",
        help=(
            "Allow only a reviewed GMRES restart increase from 60 to 120 "
            "or 120 to 240 when resuming a checksummed accepted state."
        ),
    )
    parser.add_argument("--maximum-attempts", type=int, default=120)
    parser.add_argument("--maximum-newton-iterations", type=int, default=20)
    parser.add_argument("--gmres-restart", type=int, default=120)
    parser.add_argument("--maximum-gmres-iterations", type=int, default=600)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    args = parser.parse_args(argv)
    if (
        args.maximum_attempts < 1
        or args.maximum_newton_iterations < 1
        or args.gmres_restart < 1
        or args.maximum_gmres_iterations < 1
        or not math.isfinite(args.progress_interval)
        or args.progress_interval <= 0.0
    ):
        parser.error("solver limits and progress interval must be positive")
    if args.allow_verified_source_changes and args.resume_from is None:
        parser.error("--allow-verified-source-changes requires --resume-from")
    if (
        args.allow_verified_gmres_restart_change
        and args.resume_from is None
    ):
        parser.error("--allow-verified-gmres-restart-change requires --resume-from")
    if args.resume_from is not None and args.refine_from is not None:
        parser.error("--resume-from and --refine-from are mutually exclusive")
    if args.refine_from is not None and args.nodes <= 257:
        parser.error("--refine-from requires a grid finer than 257 nodes")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing outputs in {output}")

    case, _, dimensionless, inputs, _ = _venner_case(args.nodes)
    hertz = line_contact_hertz(
        case.line_load_n_m,
        case.geometry.radius_x_m,
        case.e_star_pa,
        case.elastic_reference_length_m,
    )
    backend = make_backend("torch", "cuda")
    signature = _case_signature(case, dimensionless, inputs, args.nodes)
    implementation_signature = _implementation_signature()
    control_signature = _control_signature(args)
    if args.refine_from is not None:
        resume_payload = _load_refinement_seed(
            args.refine_from.resolve(), signature
        )
    elif args.resume_from is not None:
        resume_payload = _load_resume(
            args.resume_from.resolve(),
            signature,
            implementation_signature,
            control_signature,
            allow_verified_source_changes=args.allow_verified_source_changes,
            allow_verified_gmres_restart_change=(
                args.allow_verified_gmres_restart_change
            ),
        )
    else:
        resume_payload = None
    preflight = _preflight(case, hertz, dimensionless, inputs, backend, args)
    output.mkdir(parents=True)
    _write_json_atomic(output / "preflight.json", preflight)
    report = {
        "status": "running",
        "case_signature": signature,
        "implementation_sha256": implementation_signature,
        "solver_control_signature": control_signature,
        "preflight": preflight,
        "backend": f"{backend.platform}/{backend.device}",
        "linear_operator_backend": (
            "Torch float64 pressure/content residual, analytic JVP and "
            "zero-padded plane-strain elasticity"
        ),
        "linear_preconditioner": (
            "CPUBlockSchurPreconditioner (SciPy SuperLU; frozen Reynolds matrix)"
        ),
        "continuation_variable": "target line load [N/m]",
        "endpoint_load_n_m": case.line_load_n_m,
        "endpoint_rheology_and_compliance": "unchanged published M=20, L=10",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "attempts": [],
        "latest_accepted_checkpoint": None,
        "physical_validation_complete": False,
    }

    def write_report():
        _write_json_atomic(output / "summary.json", report)

    heartbeat_console = open(os.devnull, "w", encoding="utf-8")
    heartbeat = SolverHeartbeat(
        output / "heartbeat.jsonl",
        interval_seconds=args.progress_interval,
        stream=heartbeat_console,
    )
    previous_sigterm = signal.signal(signal.SIGTERM, _termination_handler)
    started = time.perf_counter()
    result = None
    try:
        if resume_payload is None:
            heartbeat.update(
                stage="fixed_benchmark_seed",
                emit=True,
                backend=f"{backend.platform}/{backend.device}",
                continuation_decision="initialize at prescribed h0 using existing JFO solver",
            )
            seed_started = time.perf_counter()
            try:
                seed = solve_line_ehl_at_offset(
                    case,
                    0.5 * case.film_scale_m,
                    backend=backend,
                )
            except LineEHLConvergenceError as error:
                report.update(
                    status="seed_failed",
                    reason=str(error),
                    seed_history=[asdict(item) for item in error.history],
                )
                raise
            seed_seconds = time.perf_counter() - seed_started
            seed_load = seed.calculated_line_load_n_m
            system = LineEHLNewtonSystem(case, seed_load, backend=backend)
            seed_state = np.concatenate(
                (
                    seed.pressure_pa[1:-1] / case.pressure_scale_pa,
                    1.0 - seed.fractional_film_content[1:-1],
                    [seed.central_film_offset_m / case.film_scale_m],
                )
            )
            seed_root = solve_line_ehl_newton(system, seed_state)
            state = _state_from_solution(seed_root)
            seed_path = output / "initial_seed.npz"
            seed_hash = _save_state_atomic(seed_path, state)
            fraction = seed_load / case.line_load_n_m
            report.update(
                initialization={
                    "method": "existing prescribed-h0 JFO/elasticity CUDA solver",
                    "h0_m": seed.central_film_offset_m,
                    "calculated_seed_load_n_m": seed_load,
                    "fraction_of_endpoint_load": fraction,
                    "elapsed_seconds": seed_seconds,
                    "independent_audit": seed_root.audit,
                },
                latest_accepted_checkpoint={
                    "stage_index": None,
                    "fraction": fraction,
                    "load_n_m": seed_load,
                    "state_file": seed_path.name,
                    "state_sha256": seed_hash,
                },
            )
        else:
            state, start_load, fraction, resume_metadata = resume_payload
            system = LineEHLNewtonSystem(case, start_load, backend=backend)
            report["resume_source"] = resume_metadata
            if args.refine_from is not None:
                report["initialization"] = {
                    "method": "interpolated accepted coarser-grid endpoint state",
                    **resume_metadata,
                }
            _save_state_atomic(output / "initial_seed.npz", state)
            report["latest_accepted_checkpoint"] = {
                "stage_index": None,
                "fraction": fraction,
                "load_n_m": start_load,
                "state_file": "initial_seed.npz",
                "state_sha256": _sha256(output / "initial_seed.npz"),
            }

        write_report()
        start_load = float(report["latest_accepted_checkpoint"]["load_n_m"])
        fraction = float(report["latest_accepted_checkpoint"]["fraction"])
        if not 0.0 < fraction <= 1.0 or (
            fraction == 1.0
            and args.resume_from is None
            and args.refine_from is None
        ):
            raise ValueError("starting accepted load must not exceed the endpoint")
        maximum_step = 0.02
        step = (
            min(0.01, max(1.0e-4, 0.02 * fraction))
            if args.resume_from is not None
            else min(1.0e-4, max(1.0e-6, 0.02 * fraction))
        )
        minimum_step = 1.0e-8
        consecutive_failures = 0
        last_stage_seconds = None
        report["continuation_controls"] = {
            "initial_fraction_step": step,
            "maximum_fraction_step": maximum_step,
            "minimum_fraction_step": minimum_step,
            "consecutive_failure_limit": 8,
            "maximum_attempts": args.maximum_attempts,
            "maximum_newton_iterations": args.maximum_newton_iterations,
            "gmres_restart": args.gmres_restart,
            "maximum_gmres_iterations": args.maximum_gmres_iterations,
            "maximum_krylov_recoveries": 2,
        }
        write_report()

        for attempt in range(args.maximum_attempts):
            trial_fraction = min(1.0, fraction + step)
            trial_load = (
                case.line_load_n_m
                if trial_fraction == 1.0
                else case.line_load_n_m * trial_fraction
            )
            system.target_load_n_m = trial_load
            folder = output / f"attempt_{attempt:03d}"
            folder.mkdir(exist_ok=False)
            attempt_started = time.perf_counter()
            heartbeat.update(
                stage="load_continuation_attempt",
                iteration=attempt,
                emit=True,
                attempt=attempt + 1,
                continuation_fraction=trial_fraction,
                target_parameter_load_n_m=trial_load,
                continuation_decision="try bounded target-load increment",
            )

            def observe(
                _attempt=attempt + 1,
                _fraction=trial_fraction,
                **event,
            ):
                stage = event.pop("stage")
                iteration = event.pop("iteration", None)
                emit = event.pop("emit", False)
                event.setdefault("attempt", _attempt)
                event.setdefault("continuation_fraction", _fraction)
                event.setdefault(
                    "continuation_decision", "Newton stage in progress"
                )
                heartbeat.update(
                    stage=stage,
                    iteration=iteration,
                    emit=emit,
                    **event,
                )

            try:
                candidate = solve_line_ehl_newton(
                    system,
                    state,
                    maximum_newton_iterations=args.maximum_newton_iterations,
                    gmres_restart=args.gmres_restart,
                    maximum_gmres_iterations=args.maximum_gmres_iterations,
                    progress_callback=observe,
                )
            except LineJFONewtonError as error:
                stage_seconds = time.perf_counter() - attempt_started
                rejected_state = np.asarray(error.state, dtype=np.float64)
                state_path = folder / "state.npz"
                state_hash = (
                    _save_state_atomic(state_path, rejected_state)
                    if np.all(np.isfinite(rejected_state))
                    else None
                )
                item = {
                    "status": "rejected_stage",
                    "attempt": attempt + 1,
                    "fraction": trial_fraction,
                    "target_load_n_m": trial_load,
                    "elapsed_seconds": stage_seconds,
                    "reason": str(error),
                    "diagnostics": error.diagnostics,
                    "history": [asdict(entry) for entry in error.history],
                    "state_file": state_path.name if state_hash else None,
                    "state_sha256": state_hash,
                }
                _write_json_atomic(folder / "summary.json", item)
                report["attempts"].append(item)
                step *= 0.5
                consecutive_failures += 1
                heartbeat.update(
                    stage="load_step_rejected",
                    iteration=attempt,
                    emit=True,
                    attempt=attempt + 1,
                    continuation_fraction=trial_fraction,
                    target_parameter_load_n_m=trial_load,
                    continuation_decision="halve load increment; retain last accepted root",
                    failure_reason=str(error),
                    **{
                        key: value
                        for key, value in error.diagnostics.items()
                        if isinstance(value, (str, int, float, bool, type(None)))
                    },
                )
                write_report()
                if consecutive_failures >= 8 or step < minimum_step:
                    report.update(
                        status="continuation_stopped",
                        stop_reason=(
                            "consecutive rejected load steps"
                            if consecutive_failures >= 8
                            else "load step fell below the minimum"
                        ),
                        last_accepted_fraction=fraction,
                        last_accepted_load_n_m=start_load,
                    )
                    break
                continue

            stage_seconds = time.perf_counter() - attempt_started
            last_stage_seconds = stage_seconds
            state = _state_from_solution(candidate)
            fraction = trial_fraction
            start_load = trial_load
            consecutive_failures = 0
            state_path = folder / "state.npz"
            state_hash = _save_state_atomic(state_path, state)
            root = _solution_record(candidate, case)
            item = {
                "status": "accepted_stage",
                "attempt": attempt + 1,
                "fraction": fraction,
                "target_load_n_m": trial_load,
                "elapsed_seconds": stage_seconds,
                "root": root,
                "history": [asdict(entry) for entry in candidate.history],
                "state_file": state_path.name,
                "state_sha256": state_hash,
            }
            _write_json_atomic(folder / "summary.json", item)
            report["attempts"].append(
                {key: value for key, value in item.items() if key != "history"}
            )
            report["latest_accepted_checkpoint"] = {
                "stage_index": attempt,
                "fraction": fraction,
                "load_n_m": trial_load,
                "state_file": str(state_path.relative_to(output)),
                "state_sha256": state_hash,
            }
            if last_stage_seconds is not None:
                report["measured_runtime_estimate"] = {
                    "last_accepted_stage_seconds": last_stage_seconds,
                    "remaining_stage_count_upper_bound": (
                        args.maximum_attempts - attempt - 1
                    ),
                    "remaining_wall_seconds_upper_estimate": (
                        last_stage_seconds
                        * (args.maximum_attempts - attempt - 1)
                    ),
                    "note": (
                        "Conservative linear extrapolation from the latest "
                        "accepted stage; actual adaptive-stage cost varies."
                    ),
                }
            write_report()
            heartbeat.update(
                stage="load_step_accepted",
                iteration=attempt,
                emit=True,
                attempt=attempt + 1,
                continuation_fraction=fraction,
                target_parameter_load_n_m=trial_load,
                continuation_decision="checkpoint accepted root",
                elapsed_seconds=stage_seconds,
                newton_residual=(
                    candidate.history[-1].residual_l2
                    if candidate.history
                    else 0.0
                ),
                load_residual_relative=candidate.relative_load_error,
                pressure_maximum_pa=float(np.max(candidate.pressure_pa)),
                pressure_maximum_location_m=float(
                    case.x_m[np.argmax(candidate.pressure_pa)]
                ),
                h_c_m=float(np.interp(0.0, case.x_m, candidate.film_thickness_m)),
                h_min_m=float(np.min(candidate.film_thickness_m)),
                maximum_viscosity_pa_s=root["maximum_viscosity_pa_s"],
                maximum_density_kg_m3=root["maximum_density_kg_m3"],
                **candidate.audit,
            )
            if fraction >= 1.0:
                result = candidate
                break
            iterations = len(candidate.history)
            rejected_linesearch = sum(
                entry.line_search_rejections for entry in candidate.history
            )
            if iterations <= 4 and rejected_linesearch <= 2:
                step = min(maximum_step, step * 1.5)
            elif iterations >= 10 or rejected_linesearch >= 6:
                step = max(minimum_step, step * 0.75)
        else:
            report.update(
                status="continuation_stopped",
                stop_reason="attempt budget exhausted",
                last_accepted_fraction=fraction,
                last_accepted_load_n_m=start_load,
            )

        if result is not None:
            solution_path = output / "solution.npz"
            _save_state_atomic(solution_path, result.state)
            features = analyze_line_ehl_features(
                case.x_m,
                result.pressure_pa,
                result.film_thickness_m,
                hertz.half_width_m,
            )
            comparison = _build_reference_comparison(
                case, hertz, result, features
            )
            figure_path = output / "diagnostics.png"
            density, viscosity, flux = _plot_diagnostics(
                case, hertz, result, figure_path, True
            )
            profile_path = output / "profiles.csv"
            _profile_csv(
                profile_path, case, hertz, result, density, viscosity, flux
            )
            report.update(
                status="accepted_numerical_root",
                accepted_endpoint=True,
                physical_validation_complete=False,
                root={
                    **_solution_record(result, case),
                    "features": asdict(features),
                },
                published_profile_comparison=comparison,
                artifacts={
                    "solution": solution_path.name,
                    "diagnostic_figure": figure_path.name,
                    "profiles": profile_path.name,
                },
            )
        report["elapsed_seconds"] = time.perf_counter() - started
    except _RunInterrupted as error:
        report.update(
            status="interrupted",
            reason=str(error),
            last_accepted_checkpoint=report.get("latest_accepted_checkpoint"),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        report.update(
            status="failed",
            failure_type=type(error).__name__,
            reason=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            heartbeat.close(status=report["status"])
        finally:
            heartbeat_console.close()
        write_report()

    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
