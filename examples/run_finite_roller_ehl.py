"""Run bounded finite-roller stationary isothermal donor/JFO cases on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ehl_solver.finite_roller_ehl import (
    FiniteRollerEHLCase,
    finite_roller_hertz_seed,
    finite_roller_metrics,
    mirror_finite_roller_seed,
    transfer_finite_roller_seed,
    verify_finite_roller_cuda_parity,
)
from ehl_solver.hybrid_jfo import HybridJFOCorrectionSolver
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import LimitedNewtonError, solve_limited_load
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.visualization import write_run_summary


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.size > 8192:
            finite = value[np.isfinite(value)]
            return {
                "shape": list(value.shape),
                "omitted_large_array": True,
                "finite_l2": float(np.linalg.norm(finite)) if finite.size else None,
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def compact_history(history):
    """Keep Newton/GMRES diagnostics while summarizing full-size search vectors."""
    records = []
    for raw in history:
        record = {
            key: value
            for key, value in raw.items()
            if key not in ("models",)
        }
        models = []
        for model in raw.get("models", []):
            entry = {}
            for key, value in model.items():
                if key in ("guide", "direction", "active_full"):
                    array = np.asarray(value)
                    entry[f"{key}_summary"] = {
                        "shape": list(array.shape),
                        "l2": float(np.linalg.norm(array)),
                    }
                else:
                    entry[key] = value
            models.append(entry)
        record["models"] = models
        records.append(json_value(record))
    return records


def state_diagnostics(case, probe, state):
    state = np.asarray(state, dtype=np.float64)
    n = probe.n
    bad = np.flatnonzero(~np.isfinite(state))
    first_nonfinite = None
    if bad.size:
        index = int(bad[0])
        first_nonfinite = {
            "state_index": index,
            "component": (
                "pressure"
                if index < n
                else "content_deficit"
                if index < 2 * n
                else "film_offset"
            ),
        }
    report = {
        "state_finite": bool(np.all(np.isfinite(state))),
        "first_nonfinite_state_value": first_nonfinite,
        "pressure_state_min_max": (
            [float(np.min(state[:n])), float(np.max(state[:n]))]
            if state.size >= 2 * n and np.all(np.isfinite(state[:n]))
            else None
        ),
        "content_deficit_min_max": (
            [float(np.min(state[n : 2 * n])), float(np.max(state[n : 2 * n]))]
            if state.size >= 2 * n and np.all(np.isfinite(state[n : 2 * n]))
            else None
        ),
        "film_offset_m": (
            float(state[-1] * probe.system.problem.film_scale_m)
            if state.size and np.isfinite(state[-1])
            else None
        ),
    }
    if not np.all(np.isfinite(state)):
        return report
    try:
        pressure, theta, film = probe.fields(state)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        report["field_reconstruction_error"] = f"{type(exc).__name__}: {exc}"
        return report
    report.update(
        pressure_min_max_pa=[float(pressure.min()), float(pressure.max())],
        film_min_max_m=[float(film.min()), float(film.max())],
        theta_min_max=[float(theta.min()), float(theta.max())],
        film_strictly_feasible=bool(film.min() > probe.system.problem.film_floor_m),
    )
    try:
        density, viscosity, _, _ = case.fluid().properties(pressure)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        report["constitutive_evaluation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report.update(
            density_finite=bool(np.all(np.isfinite(density))),
            density_kg_m3_min_max=[float(density.min()), float(density.max())],
            viscosity_finite=bool(np.all(np.isfinite(viscosity))),
            viscosity_pa_s_min_max=[float(viscosity.min()), float(viscosity.max())],
        )
    try:
        residual = probe.residual(state)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        report["residual_evaluation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["residual_finite"] = bool(np.all(np.isfinite(residual)))
        report["residual_linf"] = float(np.max(np.abs(residual)))
    return report


def physical_node_budget(nx, ny):
    node_count = nx * ny
    for budget in (20000, 40000, 100000, 300000, 450000):
        if node_count <= budget:
            return budget
    raise ValueError(f"{node_count} grid nodes exceed the 450000-node budget")


def write_history_and_heartbeat(directory, stem):
    heartbeat_path = directory / f"{stem}_heartbeat.jsonl"
    checkpoint_timing = {"wall_seconds": 0.0, "count": 0}

    def heartbeat_observer(**event):
        if event.get("emit", False):
            with heartbeat_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_value(event), allow_nan=False) + "\n")
                stream.flush()

    def state_observer(iteration, state, audit):
        checkpoint = directory / f"{stem}_checkpoint_{iteration:03d}.npz"
        started = time.perf_counter()
        np.savez_compressed(
            checkpoint,
            state=state,
            audit_json=np.asarray(json.dumps(json_value(audit), allow_nan=False)),
        )
        checkpoint_timing["wall_seconds"] += time.perf_counter() - started
        checkpoint_timing["count"] += 1

    return state_observer, heartbeat_observer, checkpoint_timing


def save_profiles(directory, stem, case, probe, state, metrics, profiles, audit):
    problem = probe.system.problem
    pressure, theta, film = probe.fields(state)
    path = directory / f"{stem}_fields.npz"
    np.savez_compressed(
        path,
        x_m=problem.x_m,
        y_m=problem.y_m,
        pressure_pa=pressure,
        film_thickness_m=film,
        fractional_film_content=theta,
        **{
            key: value
            for key, value in profiles.items()
            if key not in ("pressure_pa", "film_m", "theta")
        },
    )
    np.savetxt(
        directory / f"{stem}_centerline_x.csv",
        np.column_stack(
            (
                problem.x_m,
                profiles["pressure_centerline_pa"],
                profiles["film_centerline_m"],
                profiles["theta_centerline"],
            )
        ),
        delimiter=",",
        header="x_m,pressure_pa,film_thickness_m,fractional_film_content",
        comments="",
    )
    np.savetxt(
        directory / f"{stem}_centerline_y.csv",
        np.column_stack(
            (
                problem.y_m,
                profiles["pressure_axial_pa"],
                profiles["film_axial_m"],
                profiles["theta_axial"],
            )
        ),
        delimiter=",",
        header="y_m,pressure_pa,film_thickness_m,fractional_film_content",
        comments="",
    )
    np.savetxt(
        directory / f"{stem}_centerline_flux_x.csv",
        np.column_stack(
            (profiles["qx_centerline_x_faces_m"], profiles["qx_centerline_kg_m_s"])
        ),
        delimiter=",",
        header="x_face_m,qx_kg_m_s",
        comments="",
    )
    np.savetxt(
        directory / f"{stem}_centerline_flux_y.csv",
        np.column_stack(
            (profiles["qy_centerline_y_faces_m"], profiles["qy_centerline_kg_m_s"])
        ),
        delimiter=",",
        header="y_face_m,qy_kg_m_s",
        comments="",
    )
    figure, centerline = write_run_summary(
        problem.x_m,
        problem.y_m,
        pressure,
        film,
        directory=directory,
        case_name=stem,
        load_n=case.load_n,
        formulation="donor-jfo-roelands-dh",
        status="accepted_numerical_root",
        force_audit=audit,
        newton_corrections=metrics["newton_corrections"],
        backend="CUDA float64 with CPU SuperLU and independent CPU audits",
        runtime_seconds=metrics["runtime_seconds"],
        field_archive_path=path,
    )
    return {
        "field_archive": path.name,
        "run_summary_figure": figure.name,
        "run_summary_profiles": centerline.name,
    }


def solve_case(
    output,
    label,
    nx,
    ny,
    tilt,
    seed_case=None,
    seed_state=None,
    mirror_seed=False,
    spike_window_m=None,
):
    budget = physical_node_budget(nx, ny)
    case = FiniteRollerEHLCase(
        nx=nx,
        ny=ny,
        axial_tilt_rad=tilt,
        maximum_physical_nodes=budget,
    )
    stem = f"{label}_{nx}x{ny}"
    directory = output / stem
    directory.mkdir()
    state_observer, heartbeat_observer, checkpoint_timing = write_history_and_heartbeat(
        directory, stem
    )
    report = {
        "case": json_value(case.physical_signature()),
        "grid_nx_ny": [nx, ny],
        "array_order": "(ny,nx)",
        "entrainment_direction": "+x",
        "load_unit": "N",
        "pressure_unit": "Pa gauge",
        "length_unit": "m",
        "transport": "donor",
        "transport_fraction": 0.0,
        "seed": "pending",
        "status": "running",
    }
    initial_state = None
    probe = None
    seed_transfer_wall_seconds = 0.0
    started = time.perf_counter()
    try:
        probe = LimitedLoadProbe(case, transport_fraction=0.0)
        system = probe.system
        seed_transfer_started = time.perf_counter()
        try:
            if seed_state is None:
                initial_state = finite_roller_hertz_seed(case, system)
                report["seed"] = (
                    "unaccepted elliptical-Hertz pressure, theta=1, "
                    "HD central-film offset"
                )
            elif mirror_seed:
                initial_state = mirror_finite_roller_seed(
                    seed_case, seed_state, case, system
                )
                report["seed"] = "y-mirrored accepted opposite-tilt state"
            else:
                initial_state = transfer_finite_roller_seed(
                    seed_case, seed_state, case, system
                )
                report["seed"] = "interpolated accepted lower-grid or zero-tilt state"
        finally:
            seed_transfer_wall_seconds = time.perf_counter() - seed_transfer_started
        if not probe.feasible(initial_state):
            raise ValueError(
                f"{stem} starting state failed the original strict feasibility gate"
            )
        correction = HybridJFOCorrectionSolver(case, workspace_limit_bytes=2**30)
        root = solve_limited_load(
            case,
            initial_state,
            correction_solver=correction,
            transport_fraction=0.0,
            maximum_iterations=30,
            maximum_krylov_cycles=10,
            gmres_restart=60,
            workspace_limit_bytes=2**30,
            state_observer=state_observer,
            heartbeat_observer=heartbeat_observer,
            local_balance_tolerance=1e-10,
        )
    except LimitedNewtonError as exc:
        np.savez_compressed(directory / f"{stem}_failed_state.npz", state=exc.state)
        report.update(
            status="nonlinear_or_audit_failure",
            failure_reason=str(exc),
            failure_diagnostics=state_diagnostics(case, probe, exc.state),
            last_audit=json_value(exc.audit),
            newton_history=compact_history(exc.history),
            runtime_seconds=time.perf_counter() - started,
            solver_wall_seconds=time.perf_counter() - started,
            seed_transfer_wall_seconds=seed_transfer_wall_seconds,
            checkpoint_wall_seconds=checkpoint_timing["wall_seconds"],
            checkpoint_count=checkpoint_timing["count"],
            complete_wall_seconds=time.perf_counter() - started,
        )
        (directory / f"{stem}_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return None, report
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        report.update(
            status="setup_or_hybrid_failure",
            failure_reason=f"{type(exc).__name__}: {exc}",
            failure_diagnostics=(
                None
                if initial_state is None or probe is None
                else state_diagnostics(case, probe, initial_state)
            ),
            runtime_seconds=time.perf_counter() - started,
            solver_wall_seconds=time.perf_counter() - started,
            seed_transfer_wall_seconds=seed_transfer_wall_seconds,
            checkpoint_wall_seconds=checkpoint_timing["wall_seconds"],
            checkpoint_count=checkpoint_timing["count"],
            complete_wall_seconds=time.perf_counter() - started,
        )
        if initial_state is not None:
            np.savez_compressed(
                directory / f"{stem}_failed_state.npz", state=initial_state
            )
        (directory / f"{stem}_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return None, report
    runtime = time.perf_counter() - started
    audit_started = time.perf_counter()
    independent_cpu_audit = audit_limited_state(probe, root.state, direct=True)
    cpu_audit_wall = time.perf_counter() - audit_started
    metrics, profiles = finite_roller_metrics(
        case,
        probe,
        root.state,
        independent_cpu_audit,
        spike_window_m=spike_window_m,
    )
    if not independent_cpu_audit["numerically_accepted"]:
        report.update(
            status="independent_cpu_audit_failed",
            audit=json_value(root.audit),
            independent_cpu_audit=json_value(independent_cpu_audit),
            metrics=json_value(metrics),
            newton_history=compact_history(root.history),
            solver_wall_seconds=runtime,
            seed_transfer_wall_seconds=seed_transfer_wall_seconds,
            checkpoint_wall_seconds=checkpoint_timing["wall_seconds"],
            checkpoint_count=checkpoint_timing["count"],
            cpu_audit_wall_seconds=cpu_audit_wall,
            complete_wall_seconds=time.perf_counter() - started,
        )
        np.savez_compressed(
            directory / f"{stem}_cpu_audit_failed_state.npz", state=root.state
        )
        (directory / f"{stem}_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return None, report
    try:
        parity_started = time.perf_counter()
        parity = verify_finite_roller_cuda_parity(case, root.state, probe)
        parity_wall = time.perf_counter() - parity_started
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        report.update(
            status="accepted_cpu_root_cuda_parity_failed",
            failure_reason=f"{type(exc).__name__}: {exc}",
            metrics=json_value(metrics),
            audit=json_value(root.audit),
            newton_history=compact_history(root.history),
            runtime_seconds=runtime,
            solver_wall_seconds=runtime,
            seed_transfer_wall_seconds=seed_transfer_wall_seconds,
            checkpoint_wall_seconds=checkpoint_timing["wall_seconds"],
            checkpoint_count=checkpoint_timing["count"],
            cpu_audit_wall_seconds=cpu_audit_wall,
            parity_wall_seconds=time.perf_counter() - parity_started,
            complete_wall_seconds=time.perf_counter() - started,
        )
        np.savez_compressed(directory / f"{stem}_parity_failed_state.npz", state=root.state)
        (directory / f"{stem}_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return None, report
    metrics["newton_corrections"] = len(root.history)
    metrics["runtime_seconds"] = runtime
    metrics["solver_wall_seconds"] = runtime
    metrics["cpu_audit_wall_seconds"] = cpu_audit_wall
    metrics["parity_wall_seconds"] = parity_wall
    summary_started = time.perf_counter()
    summary_files = save_profiles(
        directory, stem, case, probe, root.state, metrics, profiles, root.audit
    )
    summary_wall = time.perf_counter() - summary_started
    metrics["summary_io_wall_seconds"] = summary_wall
    np.savez_compressed(directory / f"{stem}_accepted_state.npz", state=root.state)
    report.update(
        status=(
            "accepted_audited_root"
            if root.audit["numerically_accepted"]
            and root.audit["requested_tolerance_satisfied"]
            else "solver_returned_but_acceptance_gate_failed"
        ),
        metrics=json_value(metrics),
        audit=json_value(root.audit),
        independent_cpu_audit=json_value(independent_cpu_audit),
        cuda_cpu_parity=json_value(parity),
        newton_history=compact_history(root.history),
        summary_files=summary_files,
        runtime_seconds=runtime,
        solver_wall_seconds=runtime,
        seed_transfer_wall_seconds=seed_transfer_wall_seconds,
        checkpoint_wall_seconds=checkpoint_timing["wall_seconds"],
        checkpoint_count=checkpoint_timing["count"],
        cpu_audit_wall_seconds=cpu_audit_wall,
        parity_wall_seconds=parity_wall,
        summary_io_wall_seconds=summary_wall,
        complete_wall_seconds=time.perf_counter() - started,
    )
    (directory / f"{stem}_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if report["status"] != "accepted_audited_root":
        return None, report
    return root, report


def symmetry_metrics(case, root, probe):
    pressure, theta, film = probe.fields(root.state)
    errors = {
        "pressure_relative_linf": float(
            np.max(np.abs(pressure - pressure[::-1])) / max(float(pressure.max()), 1.0)
        ),
        "film_relative_linf": float(
            np.max(np.abs(film - film[::-1])) / probe.system.problem.film_scale_m
        ),
        "theta_linf": float(np.max(np.abs(theta - theta[::-1]))),
        "moment_about_x_n_m": _moment_y(probe, root.state),
    }
    errors["passed"] = bool(
        errors["pressure_relative_linf"] <= 1e-6
        and errors["film_relative_linf"] <= 1e-6
        and errors["theta_linf"] <= 1e-6
        and abs(errors["moment_about_x_n_m"]) <= 1e-8
    )
    return errors


def _moment_y(probe, state):
    from scipy.integrate import trapezoid

    pressure, _, _ = probe.fields(state)
    problem = probe.system.problem
    return float(
        trapezoid(
            trapezoid(pressure * problem.y_m[:, None], x=problem.x_m, axis=1),
            x=problem.y_m,
        )
    )


def mirror_solution_metrics(positive_case, positive_root, negative_case, negative_root):
    positive_probe = LimitedLoadProbe(positive_case, transport_fraction=0.0)
    negative_probe = LimitedLoadProbe(negative_case, transport_fraction=0.0)
    positive = positive_probe.fields(positive_root.state)
    negative = negative_probe.fields(negative_root.state)
    hscale = positive_probe.system.problem.film_scale_m
    errors = {
        "pressure_relative_linf": float(
            np.max(np.abs(positive[0] - negative[0][::-1]))
            / max(float(positive[0].max()), 1.0)
        ),
        "film_relative_linf": float(np.max(np.abs(positive[2] - negative[2][::-1])) / hscale),
        "theta_linf": float(np.max(np.abs(positive[1] - negative[1][::-1]))),
        "positive_moment_about_x_n_m": _moment_y(positive_probe, positive_root.state),
        "negative_moment_about_x_n_m": _moment_y(negative_probe, negative_root.state),
    }
    errors["moment_sign_reversed"] = bool(
        errors["positive_moment_about_x_n_m"]
        * errors["negative_moment_about_x_n_m"]
        < 0
    )
    errors["passed"] = bool(
        errors["pressure_relative_linf"] <= 5e-3
        and errors["film_relative_linf"] <= 5e-3
        and errors["theta_linf"] <= 5e-3
        and errors["moment_sign_reversed"]
    )
    return errors


def refinement_differences(
    coarse_case, coarse_root, fine_case, fine_root, *, spike_window_m=None
):
    coarse_probe = LimitedLoadProbe(coarse_case, transport_fraction=0.0)
    fine_probe = LimitedLoadProbe(fine_case, transport_fraction=0.0)
    coarse_fields = coarse_probe.fields(coarse_root.state)
    fine_fields = fine_probe.fields(fine_root.state)
    cp = coarse_probe.system.problem
    fp = fine_probe.system.problem
    yy, xx = np.meshgrid(fp.y_m, fp.x_m, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    bulk = np.abs(fp.y_m) <= 0.8 * fine_case.half_length_m
    result = {}
    names = ("pressure_pa", "theta", "film_m")
    for name, old, new in zip(names, coarse_fields, fine_fields, strict=True):
        interpolated = RegularGridInterpolator(
            (cp.y_m, cp.x_m), old, bounds_error=True
        )(points).reshape(fp.shape)
        delta = new[bulk, 1:-1] - interpolated[bulk, 1:-1]
        reference = new[bulk, 1:-1]
        scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
        result[f"{name}_bulk_relative_l2_change"] = float(np.linalg.norm(delta) / scale)
        result[f"{name}_bulk_rms_change"] = float(np.sqrt(np.mean(delta**2)))
    result["h_c_relative_change"] = float(
        fine_root.audit["central_film_m"] / coarse_root.audit["central_film_m"] - 1
    )
    result["h_min_relative_change"] = float(
        fine_root.audit["minimum_film_m"] / coarse_root.audit["minimum_film_m"] - 1
    )
    result["force_relative_error_change"] = float(
        fine_root.audit["force_relative_error"] - coarse_root.audit["force_relative_error"]
    )
    coarse_metrics, _ = finite_roller_metrics(
        coarse_case,
        coarse_probe,
        coarse_root.state,
        coarse_root.audit,
        spike_window_m=spike_window_m,
    )
    fine_metrics, _ = finite_roller_metrics(
        fine_case,
        fine_probe,
        fine_root.state,
        fine_root.audit,
        spike_window_m=spike_window_m,
    )
    coarse_spike = coarse_metrics["outlet_centerline_pressure_spike"]
    fine_spike = fine_metrics["outlet_centerline_pressure_spike"]
    result["outlet_spike_location_change_m"] = (
        None
        if coarse_spike is None or fine_spike is None
        else fine_spike["x_m"] - coarse_spike["x_m"]
    )
    result["outlet_spike_width_cells"] = (
        None if fine_spike is None else fine_spike["half_prominence_width_cells"]
    )
    result["outlet_spike_width_m"] = (
        None if fine_spike is None else fine_spike["half_prominence_width_m"]
    )
    if coarse_metrics["fixed_spike_window"] and fine_metrics["fixed_spike_window"]:
        result["fixed_spike_window_centerline_pressure_integral_change_pa_m"] = (
            fine_metrics["fixed_spike_window"]["centerline_pressure_integral_pa_m"]
            - coarse_metrics["fixed_spike_window"]["centerline_pressure_integral_pa_m"]
        )
        result["fixed_spike_window_load_change_n"] = (
            fine_metrics["fixed_spike_window"]["integrated_pressure_load_n"]
            - coarse_metrics["fixed_spike_window"]["integrated_pressure_load_n"]
        )
    result["pressure_centroid_y_change_m"] = float(
        fine_metrics["pressure_centroid_m"][1] - coarse_metrics["pressure_centroid_m"][1]
    )
    result["moment_about_x_relative_change"] = float(
        (fine_metrics["moment_about_x_n_m"] - coarse_metrics["moment_about_x_n_m"])
        / max(fine_case.load_n * fine_case.half_length_m, np.finfo(float).tiny)
    )
    result["end_band_load_absolute_change_n"] = {
        fraction: [
            float(fine_metrics["outer_band_load_n_left_right"][fraction][side])
            - float(coarse_metrics["outer_band_load_n_left_right"][fraction][side])
            for side in range(2)
        ]
        for fraction in ("2", "5", "10")
    }
    result["outlet_constriction_width_cells"] = (
        fine_metrics["outlet_centerline_constriction"]["half_prominence_width_cells"]
    )
    result["pressurized_support_clearance_cells"] = (
        fine_metrics["pressurized_support_axial_clearance_cells"]
    )
    return result


def run_grid(output, nx, ny, previous=None):
    budget = physical_node_budget(nx, ny)
    roots, reports = {}, {}
    source_a = None if previous is None else previous["A"]
    root_a, report_a = solve_case(
        output,
        "A_symmetric",
        nx,
        ny,
        0.0,
        None if source_a is None else source_a[0],
        None if source_a is None else source_a[1].state,
    )
    reports["A_symmetric"] = report_a
    if root_a is None:
        return None, reports
    case_a = FiniteRollerEHLCase(
        nx=nx, ny=ny, axial_tilt_rad=0.0, maximum_physical_nodes=budget
    )
    probe_a = LimitedLoadProbe(case_a, transport_fraction=0.0)
    roots["A"] = (case_a, root_a)
    reports["A_symmetric"]["symmetry"] = json_value(
        symmetry_metrics(case_a, root_a, probe_a)
    )
    if not reports["A_symmetric"]["symmetry"]["passed"]:
        reports["A_symmetric"]["status"] = "accepted_root_symmetry_gate_failed"
        return None, reports

    for label, tilt in (("B_plus_0p00025", 0.00025),):
        case = FiniteRollerEHLCase(
            nx=nx, ny=ny, axial_tilt_rad=tilt, maximum_physical_nodes=budget
        )
        root, report = solve_case(
            output,
            label,
            nx,
            ny,
            tilt,
            case_a,
            root_a.state,
        )
        reports[label] = report
        if root is None:
            return None, reports
        roots["B_plus"] = (case, root)
    plus_case, plus_root = roots["B_plus"]
    minus_case = FiniteRollerEHLCase(
        nx=nx, ny=ny, axial_tilt_rad=-0.00025, maximum_physical_nodes=budget
    )
    minus_root, minus_report = solve_case(
        output,
        "B_minus_0p00025",
        nx,
        ny,
        -0.00025,
        plus_case,
        plus_root.state,
        mirror_seed=True,
    )
    reports["B_minus_0p00025"] = minus_report
    if minus_root is None:
        return None, reports
    roots["B_minus"] = (minus_case, minus_root)
    mirror = mirror_solution_metrics(plus_case, plus_root, minus_case, minus_root)
    reports["tilt_mirror_check"] = json_value(mirror)
    if not mirror["passed"]:
        return None, reports
    return roots, reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--coarse-only", action="store_true")
    parser.add_argument("--engineering-only", action="store_true")
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args(argv)
    if args.coarse_only and args.engineering_only:
        parser.error("--coarse-only cannot be combined with --engineering-only")
    if args.refine and args.coarse_only:
        parser.error("--refine cannot be combined with --coarse-only")
    output = args.output_dir
    if output is None:
        output = Path("results") / (
            "finite_roller_ehl_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    if args.engineering_only:
        grids = [(129, 257)]
    elif args.coarse_only:
        grids = [(65, 129)]
    else:
        grids = [(65, 129), (129, 257)]
    if args.refine:
        grids.append((193, 385))
    previous = None
    previous_reports = None
    for nx, ny in grids:
        roots, grid_reports = run_grid(output, nx, ny, previous)
        reports[f"{nx}x{ny}"] = grid_reports
        (output / "report.json").write_text(
            json.dumps(json_value(reports), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if roots is None:
            print(f"Finite-roller run stopped at {nx}x{ny}; diagnostics: {output}")
            return 2
        if previous is not None:
            for key in ("A", "B_plus", "B_minus"):
                old_case, old_root = previous[key]
                new_case, new_root = roots[key]
                diff = refinement_differences(old_case, old_root, new_case, new_root)
                grid_reports[f"{key}_refinement_from_previous"] = json_value(diff)
                if previous_reports is not None:
                    previous_diff = previous_reports.get(
                        f"{key}_refinement_from_previous"
                    )
                    if previous_diff is not None:
                        grid_reports[f"{key}_refinement_change_trend"] = {
                            metric: (
                                abs(diff[metric]) < abs(previous_diff[metric])
                                if metric.endswith("_relative_change")
                                or metric.endswith("_relative_l2_change")
                                or metric == "pressure_centroid_y_change_m"
                                or metric == "force_relative_error_change"
                                else None
                            )
                            for metric in diff
                        }
        previous, previous_reports = roots, grid_reports
        (output / "report.json").write_text(
            json.dumps(json_value(reports), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"Finite-roller runs completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
