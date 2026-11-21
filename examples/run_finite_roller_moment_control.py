"""Qualify force-and-moment closure for the finite-roller stationary EHL model."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from run_finite_roller_ehl import (
    compact_history,
    json_value,
    physical_node_budget,
    save_profiles,
    write_history_and_heartbeat,
)

from ehl_solver.finite_roller_ehl import (
    FiniteRollerEHLCase,
    finite_roller_metrics,
    transfer_finite_roller_seed,
    verify_finite_roller_cuda_parity,
)
from ehl_solver.hybrid_jfo import HybridJFOCorrectionSolver
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import LimitedNewtonError, solve_limited_load
from ehl_solver.limited_system import (
    MOMENT_ABSOLUTE_TOLERANCE_N_M,
    MOMENT_ANGLE_SCALE_RAD,
    MOMENT_NORMALIZED_TOLERANCE,
    LimitedLoadProbe,
)

NX = 769
NY = 577
MOMENT_SCALE_FRACTION = 0.5
INITIAL_TILT_OFFSET_RAD = 5e-5
SPIKE_WINDOW_HALF_WIDTH_M = 15e-6
DEFAULT_SYMMETRIC_SEED = Path(
    "results/finite_roller_ehl_xref_20261120_152600_retry1"
)
DEFAULT_POSITIVE_SEED = Path(
    "results/finite_roller_ehl_axial_20261120_184013"
)


def _case(tilt_rad):
    return FiniteRollerEHLCase(
        nx=NX,
        ny=NY,
        axial_tilt_rad=tilt_rad,
        maximum_physical_nodes=physical_node_budget(NX, NY),
    )


def _load_state(path):
    with np.load(path, allow_pickle=False) as archive:
        state = archive["state"].copy()
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError(f"saved finite-roller seed is not a finite vector: {path}")
    return state


def _seed_path(directory, label, nx, ny):
    stem = f"{label}_{nx}x{ny}"
    path = directory / stem / f"{stem}_accepted_state.npz"
    if not path.is_file():
        raise FileNotFoundError(f"accepted state archive is missing: {path}")
    return path


def _moment_state(base_state, angle_rad):
    if base_state.ndim != 1:
        raise ValueError("moment-controlled seed must be a one-dimensional state")
    return np.r_[
        base_state,
        angle_rad / MOMENT_ANGLE_SCALE_RAD,
    ]


def _mirrored_moment_state(source_state, nx, ny, angle_rad):
    n = (ny - 2) * (nx - 2)
    if source_state.size not in (2 * n + 1, 2 * n + 2):
        raise ValueError("source state has the wrong finite-roller grid size")
    pressure = source_state[:n].reshape(ny - 2, nx - 2)[::-1].ravel()
    deficit = source_state[n : 2 * n].reshape(ny - 2, nx - 2)[::-1].ravel()
    offset = source_state[2 * n]
    return _moment_state(np.r_[pressure, deficit, offset], angle_rad)


def _performance(history):
    models = [
        model
        for record in history
        for model in record.get("models", ())
    ]
    timings = [
        model["linear_solver_timings"]
        for model in models
        if model.get("status") == "linear_solved"
        and model.get("linear_solver_timings") is not None
    ]
    alpha_updates = [
        float(record["accepted_alpha_correction_rad"])
        for record in history
        if record.get("accepted_alpha_correction_rad") is not None
        and record["accepted_alpha_correction_rad"] != 0.0
    ]
    alpha_signs = [int(np.sign(value)) for value in alpha_updates]
    sign_changes = sum(
        left != right
        for left, right in zip(alpha_signs, alpha_signs[1:], strict=False)
    )
    schur_indicators = [
        float(model["schur_indicator"])
        for model in models
        if model.get("schur_indicator") is not None
        and np.isfinite(model["schur_indicator"])
    ]
    angular_sensitivities = [
        float(timing["force_constrained_moment_sensitivity_n_m_per_rad"])
        for timing in timings
        if timing.get("force_constrained_moment_sensitivity_n_m_per_rad")
        is not None
        and np.isfinite(timing["force_constrained_moment_sensitivity_n_m_per_rad"])
    ]
    normalized_angular_sensitivities = [
        float(timing["normalized_force_constrained_moment_sensitivity"])
        for timing in timings
        if timing.get("normalized_force_constrained_moment_sensitivity") is not None
        and np.isfinite(timing["normalized_force_constrained_moment_sensitivity"])
    ]
    sum_fields = (
        "gmres_iterations",
        "cpu_superlu_setup_seconds",
        "cpu_superlu_solve_seconds",
        "cuda_residual_seconds",
        "cuda_linearization_setup_seconds",
        "cuda_jvp_seconds",
        "state_guide_h2d_seconds",
        "limiter_branch_h2d_seconds",
        "d2h_seconds",
        "h2d_preconditioner_seconds",
        "correction_d2h_seconds",
        "force_moment_border_setup_seconds",
        "force_moment_border_application_seconds",
    )
    totals = {
        key: float(sum(item.get(key, 0.0) for item in timings))
        for key in sum_fields
    }
    totals["total_measured_transfer_seconds"] = sum(
        totals[key]
        for key in (
            "state_guide_h2d_seconds",
            "limiter_branch_h2d_seconds",
            "d2h_seconds",
            "h2d_preconditioner_seconds",
            "correction_d2h_seconds",
        )
    )
    return {
        "newton_iterations": len(history),
        "linear_correction_count": len(timings),
        "force_moment_preconditioner_fallback_count": sum(
            bool(model.get("used_bordered_fallback", False))
            for model in models
            if model.get("status") == "linear_solved"
        ),
        "minimum_force_moment_schur_indicator": (
            min(schur_indicators) if schur_indicators else None
        ),
        "minimum_abs_force_constrained_moment_sensitivity_n_m_per_rad": (
            min(map(abs, angular_sensitivities))
            if angular_sensitivities
            else None
        ),
        "minimum_abs_normalized_force_constrained_moment_sensitivity": (
            min(map(abs, normalized_angular_sensitivities))
            if normalized_angular_sensitivities
            else None
        ),
        "poor_angular_sensitivity_count": sum(
            timing.get("poor_angular_sensitivity") is True
            for timing in timings
        ),
        "accepted_alpha_update_sign_changes": sign_changes,
        "accepted_alpha_update_rad": alpha_updates,
        **totals,
        "maximum_independently_recomputed_linear_true_residual": (
            max(
                (
                    float(model["independently_recomputed_linear_true_residual"])
                    for record in history
                    for model in record.get("models", ())
                    if model.get("status") == "linear_solved"
                    and model.get("independently_recomputed_linear_true_residual")
                    is not None
                ),
                default=None,
            )
        ),
    }


def _write_case_report(output, label, report):
    stem = f"{label}_{NX}x{NY}"
    path = output / stem / f"{stem}_report.json"
    path.write_text(
        json.dumps(json_value(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def solve_case(output, label, case, initial_state, moment_target_n_m):
    stem = f"{label}_{NX}x{NY}"
    directory = output / stem
    directory.mkdir()
    state_observer, heartbeat_observer, checkpoint_timing = (
        write_history_and_heartbeat(directory, stem)
    )
    probe = LimitedLoadProbe(
        case,
        transport_fraction=0.0,
        moment_target_n_m=moment_target_n_m,
        angle_scale_rad=MOMENT_ANGLE_SCALE_RAD,
    )
    report = {
        "case": json_value(case.physical_signature()),
        "label": label,
        "grid_nx_ny": [NX, NY],
        "array_order": "(ny,nx)",
        "entrainment_direction": "+x",
        "load_unit": "N",
        "moment_unit": "N m",
        "pressure_unit": "Pa gauge",
        "length_unit": "m",
        "transport": "donor",
        "transport_fraction": 0.0,
        "moment_target_n_m": moment_target_n_m,
        "moment_scale_n_m": case.load_n * case.half_length_m,
        "moment_absolute_tolerance_n_m": MOMENT_ABSOLUTE_TOLERANCE_N_M,
        "moment_normalized_tolerance": MOMENT_NORMALIZED_TOLERANCE,
        "angle_scale_rad": MOMENT_ANGLE_SCALE_RAD,
        "initial_alpha_rad": probe.angle_rad(initial_state),
        "status": "running",
    }
    started = time.perf_counter()
    solver_started = time.perf_counter()
    try:
        correction = HybridJFOCorrectionSolver(
            case, workspace_limit_bytes=2**30
        )
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
            moment_target_n_m=moment_target_n_m,
            angle_scale_rad=MOMENT_ANGLE_SCALE_RAD,
        )
    except LimitedNewtonError as exc:
        solver_wall = time.perf_counter() - solver_started
        np.savez_compressed(directory / f"{stem}_failed_state.npz", state=exc.state)
        report.update(
            status="nonlinear_or_audit_failure",
            failure_reason=str(exc),
            last_audit=json_value(exc.audit),
            newton_history=compact_history(exc.history),
            solver_wall_seconds=solver_wall,
            checkpoint_wall_seconds=checkpoint_timing["wall_seconds"],
            checkpoint_count=checkpoint_timing["count"],
            complete_wall_seconds=time.perf_counter() - started,
        )
        (directory / f"{stem}_report.json").write_text(
            json.dumps(json_value(report), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return None, report
    solver_wall = time.perf_counter() - solver_started
    independent_started = time.perf_counter()
    independent_audit = audit_limited_state(probe, root.state, direct=True)
    cpu_audit_wall = time.perf_counter() - independent_started
    if not independent_audit["numerically_accepted"]:
        status = "independent_cpu_audit_failed"
        parity = None
    else:
        parity_started = time.perf_counter()
        parity = verify_finite_roller_cuda_parity(case, root.state, probe)
        parity_wall = time.perf_counter() - parity_started
        status = (
            "accepted_audited_root"
            if parity["passed"]
            and root.audit["requested_tolerance_satisfied"]
            else "cuda_cpu_parity_or_acceptance_failed"
        )
    if parity is None:
        parity_wall = 0.0
    preliminary_metrics, _ = finite_roller_metrics(
        case, probe, root.state, independent_audit
    )
    spike = preliminary_metrics["outlet_centerline_pressure_spike"]
    spike_window = (
        None
        if spike is None
        else (
            spike["x_m"] - SPIKE_WINDOW_HALF_WIDTH_M,
            spike["x_m"] + SPIKE_WINDOW_HALF_WIDTH_M,
        )
    )
    metrics, profiles = finite_roller_metrics(
        case,
        probe,
        root.state,
        independent_audit,
        spike_window_m=spike_window,
    )
    spike_metrics = metrics["outlet_centerline_pressure_spike"]
    metrics["p_max_unresolved_warning"] = bool(
        spike_metrics is None
        or spike_metrics["half_prominence_width_cells"] <= 3.0
    )
    metrics["p_max_interpretation"] = (
        "Nodal outlet pressure maximum is unresolved and is not an acceptance gate."
    )
    metrics["newton_corrections"] = len(root.history)
    metrics["runtime_seconds"] = solver_wall
    metrics["solver_wall_seconds"] = solver_wall
    metrics["cpu_audit_wall_seconds"] = cpu_audit_wall
    metrics["cuda_cpu_parity_wall_seconds"] = parity_wall
    metrics["performance"] = _performance(root.history)
    summary_started = time.perf_counter()
    summary_files = save_profiles(
        directory, stem, case, probe, root.state, metrics, profiles, root.audit
    )
    summary_wall = time.perf_counter() - summary_started
    np.savez_compressed(directory / f"{stem}_accepted_state.npz", state=root.state)
    report.update(
        status=status,
        recovered_alpha_rad=probe.angle_rad(root.state),
        h0_m=probe.offset_m(root.state),
        audit=json_value(root.audit),
        independent_cpu_audit=json_value(independent_audit),
        cuda_cpu_parity=json_value(parity),
        metrics=json_value(metrics),
        performance=_performance(root.history),
        newton_history=compact_history(root.history),
        summary_files=summary_files,
        solver_wall_seconds=solver_wall,
        checkpoint_wall_seconds=checkpoint_timing["wall_seconds"],
        checkpoint_count=checkpoint_timing["count"],
        cpu_audit_wall_seconds=cpu_audit_wall,
        cuda_cpu_parity_wall_seconds=parity_wall,
        summary_io_wall_seconds=summary_wall,
        complete_wall_seconds=time.perf_counter() - started,
    )
    (directory / f"{stem}_report.json").write_text(
        json.dumps(json_value(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if status != "accepted_audited_root":
        return None, report
    return root, report


def _mirror_check(positive_case, positive_state, negative_case, negative_state):
    positive_probe = LimitedLoadProbe(
        positive_case,
        transport_fraction=0.0,
        moment_target_n_m=None,
    )
    negative_probe = LimitedLoadProbe(
        negative_case,
        transport_fraction=0.0,
        moment_target_n_m=None,
    )
    positive = positive_probe.fields(positive_state[:-1])
    negative = negative_probe.fields(negative_state[:-1])
    scale = positive_probe.system.problem.film_scale_m
    pressure_error = float(
        np.max(np.abs(positive[0] - negative[0][::-1]))
        / max(float(positive[0].max()), 1.0)
    )
    film_error = float(
        np.max(np.abs(positive[2] - negative[2][::-1])) / scale
    )
    theta_error = float(np.max(np.abs(positive[1] - negative[1][::-1])))
    positive_centroid = float(
        np.sum(positive[0] * positive_probe.system.problem.y_m[:, None])
        * positive_probe.system.problem.dx_m
        * positive_probe.system.problem.dy_m
        / positive_case.load_n
    )
    negative_centroid = float(
        np.sum(negative[0] * negative_probe.system.problem.y_m[:, None])
        * negative_probe.system.problem.dx_m
        * negative_probe.system.problem.dy_m
        / negative_case.load_n
    )
    positive_moment = -positive_centroid * positive_case.load_n
    negative_moment = -negative_centroid * negative_case.load_n
    return {
        "pressure_relative_linf": pressure_error,
        "film_relative_linf": film_error,
        "theta_linf": theta_error,
        "pressure_centroid_y_positive_m": positive_centroid,
        "pressure_centroid_y_negative_m": negative_centroid,
        "pressure_centroid_sign_reversed": positive_centroid * negative_centroid < 0,
        "contact_moment_x_positive_n_m": positive_moment,
        "contact_moment_x_negative_n_m": negative_moment,
        "contact_moment_sign_reversed": positive_moment * negative_moment < 0,
        "recovered_alpha_sum_rad": (positive_state[-1] + negative_state[-1])
        * MOMENT_ANGLE_SCALE_RAD,
        "h_c_difference_m": float(positive[2][NY // 2, NX // 2] - negative[2][NY // 2, NX // 2]),
        "h_min_difference_m": float(positive[2].min() - negative[2].min()),
        "passed": bool(
            pressure_error <= 5e-3
            and film_error <= 5e-3
            and theta_error <= 5e-3
            and positive_centroid * negative_centroid < 0
            and positive_moment * negative_moment < 0
            and abs(positive_state[-1] + negative_state[-1]) <= 1e-8
        ),
    }


def _zero_moment_symmetry(case, state, moment_target_n_m):
    probe = LimitedLoadProbe(
        case, transport_fraction=0.0, moment_target_n_m=moment_target_n_m
    )
    pressure, theta, film = probe.fields(state)
    pressure_scale = max(float(np.max(pressure)), 1.0)
    metrics, _ = finite_roller_metrics(case, probe, state)
    pressure_error = float(np.max(np.abs(pressure - pressure[::-1])) / pressure_scale)
    film_error = float(
        np.max(np.abs(film - film[::-1])) / probe.system.problem.film_scale_m
    )
    theta_error = float(np.max(np.abs(theta - theta[::-1])))
    centroid_y = float(metrics["pressure_centroid_m"][1])
    return {
        "pressure_relative_linf": pressure_error,
        "film_relative_linf": film_error,
        "theta_linf": theta_error,
        "pressure_centroid_y_m": centroid_y,
        "contact_moment_x_n_m": metrics["physical_contact_moment_x_n_m"],
        "recovered_alpha_rad": probe.angle_rad(state),
        "passed": bool(
            pressure_error <= 5e-3
            and film_error <= 5e-3
            and theta_error <= 5e-3
            and abs(centroid_y) <= 1e-12
            and abs(probe.angle_rad(state)) <= 1e-8
        ),
    }


def _prescribed_tilt_reproduction(case, reference_state, controlled_state):
    reference_probe = LimitedLoadProbe(case, transport_fraction=0.0)
    reference_fields = reference_probe.fields(reference_state)
    reference_moment = reference_probe.physical_contact_moment_x_n_m(
        reference_fields[0]
    )
    controlled_probe = LimitedLoadProbe(
        case,
        transport_fraction=0.0,
        moment_target_n_m=reference_moment,
    )
    controlled_fields = controlled_probe.fields(controlled_state)
    reference_metrics, _ = finite_roller_metrics(
        case, reference_probe, reference_state
    )
    controlled_metrics, _ = finite_roller_metrics(
        case, controlled_probe, controlled_state
    )
    pressure_rel_l2 = float(
        np.linalg.norm(controlled_fields[0] - reference_fields[0])
        / max(np.linalg.norm(reference_fields[0]), np.finfo(float).tiny)
    )
    film_rel_l2 = float(
        np.linalg.norm(controlled_fields[2] - reference_fields[2])
        / max(np.linalg.norm(reference_fields[2]), np.finfo(float).tiny)
    )
    theta_linf = float(np.max(np.abs(controlled_fields[1] - reference_fields[1])))
    hc_relative = float(
        controlled_metrics["h_c_m"] / reference_metrics["h_c_m"] - 1.0
    )
    hmin_relative = float(
        controlled_metrics["h_min_m"] / reference_metrics["h_min_m"] - 1.0
    )
    centroid_difference = float(
        controlled_metrics["pressure_centroid_m"][1]
        - reference_metrics["pressure_centroid_m"][1]
    )
    centroid_scale = max(
        abs(reference_metrics["pressure_centroid_m"][1]),
        case.half_length_m * 1e-6,
    )
    return {
        "pressure_relative_l2": pressure_rel_l2,
        "film_relative_l2": film_rel_l2,
        "theta_linf": theta_linf,
        "h_c_relative_change": hc_relative,
        "h_min_relative_change": hmin_relative,
        "pressure_centroid_y_difference_m": centroid_difference,
        "pressure_centroid_y_relative_change": centroid_difference / centroid_scale,
        "h_min_coordinates_reference_m": reference_metrics["h_min_coordinates_m"],
        "h_min_coordinates_controlled_m": controlled_metrics["h_min_coordinates_m"],
        "reference_physical_contact_moment_x_n_m": (
            reference_metrics["physical_contact_moment_x_n_m"]
        ),
        "controlled_physical_contact_moment_x_n_m": (
            controlled_metrics["physical_contact_moment_x_n_m"]
        ),
        "passed": bool(
            pressure_rel_l2 <= 1e-2
            and film_rel_l2 <= 1e-2
            and theta_linf <= 5e-3
            and abs(hc_relative) < 0.01
            and abs(hmin_relative) < 0.02
            and abs(centroid_difference) <= 0.03 * centroid_scale
        ),
    }


def run(seed_symmetric_dir, seed_positive_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "milestone": "stationary_finite_roller_force_and_moment_closure",
        "grid_nx_ny": [NX, NY],
        "backend": "hybrid CUDA float64 JFO/JVP/FFT with CPU SuperLU and independent CPU audits",
        "moment_sign_convention": (
            "Mx = -integral(y*p dA); +y is positive physical axial coordinate"
        ),
        "seed_directories": {
            "symmetric": str(seed_symmetric_dir),
            "positive_0p5_mrad": str(seed_positive_dir),
        },
        "cases": {},
    }
    symmetric_seed_path = _seed_path(
        seed_symmetric_dir, "A_symmetric", NX, 513
    )
    source_symmetric_case = FiniteRollerEHLCase(
        nx=NX,
        ny=513,
        axial_tilt_rad=0.0,
        maximum_physical_nodes=physical_node_budget(NX, 513),
    )
    target_zero_case = _case(0.0)
    symmetric_seed = _load_state(symmetric_seed_path)
    symmetric_initial = transfer_finite_roller_seed(
        source_symmetric_case,
        symmetric_seed,
        target_zero_case,
    )
    zero_root, zero_report = solve_case(
        output,
        "A_zero_moment",
        target_zero_case,
        _moment_state(symmetric_initial, 0.0),
        0.0,
    )
    report["cases"]["A_zero_moment"] = zero_report
    if zero_root is None:
        return report
    zero_symmetry = _zero_moment_symmetry(
        target_zero_case, zero_root.state, 0.0
    )
    zero_report["zero_moment_symmetry"] = zero_symmetry
    if not zero_symmetry["passed"]:
        zero_report["status"] = "zero_moment_symmetry_failed"
        report["cases"]["A_zero_moment"] = zero_report
        _write_case_report(output, "A_zero_moment", zero_report)
        return report
    _write_case_report(output, "A_zero_moment", zero_report)

    positive_case = _case(5e-4)
    positive_seed_path = _seed_path(
        seed_positive_dir, "D_plus_000500urad", NX, NY
    )
    positive_seed = _load_state(positive_seed_path)
    positive_reference_probe = LimitedLoadProbe(
        positive_case, transport_fraction=0.0
    )
    reference_audit = audit_limited_state(
        positive_reference_probe, positive_seed, direct=True
    )
    if not reference_audit["numerically_accepted"]:
        raise ValueError("published +0.5 mrad seed failed independent CPU audit")
    reference_pressure, _, _ = positive_reference_probe.fields(positive_seed)
    positive_target = positive_reference_probe.physical_contact_moment_x_n_m(
        reference_pressure
    )
    if positive_target <= 0.0:
        raise ValueError("positive tilt seed has the wrong physical moment sign")
    positive_initial = _moment_state(
        positive_seed, 5e-4 - INITIAL_TILT_OFFSET_RAD
    )
    positive_root, positive_report = solve_case(
        output,
        "B_positive_0p5_mrad",
        positive_case,
        positive_initial,
        positive_target,
    )
    positive_report["prescribed_tilt_reference_audit"] = json_value(reference_audit)
    positive_report["target_source"] = (
        "independently measured physical contact moment of the accepted prescribed +0.5 mrad root"
    )
    positive_report["baseline_reproduction"] = (
        None
        if positive_root is None
        else _prescribed_tilt_reproduction(
            positive_case, positive_seed, positive_root.state
        )
    )
    if positive_root is not None and not positive_report["baseline_reproduction"]["passed"]:
        positive_report["status"] = "prescribed_tilt_reproduction_failed"
        report["cases"]["B_positive_0p5_mrad"] = positive_report
        _write_case_report(output, "B_positive_0p5_mrad", positive_report)
        return report
    report["cases"]["B_positive_0p5_mrad"] = positive_report
    _write_case_report(output, "B_positive_0p5_mrad", positive_report)
    if positive_root is None:
        return report

    negative_case = _case(-5e-4)
    negative_initial = _mirrored_moment_state(
        positive_seed, NX, NY, -5e-4 + INITIAL_TILT_OFFSET_RAD
    )
    negative_root, negative_report = solve_case(
        output,
        "C_negative_0p5_mrad",
        negative_case,
        negative_initial,
        -positive_target,
    )
    report["cases"]["C_negative_0p5_mrad"] = negative_report
    if negative_root is None:
        return report
    mirror = _mirror_check(
        positive_case,
        positive_root.state,
        negative_case,
        negative_root.state,
    )
    report["positive_negative_mirror_check"] = mirror
    negative_report["positive_negative_mirror_check"] = mirror
    _write_case_report(output, "C_negative_0p5_mrad", negative_report)
    if not mirror["passed"]:
        return report

    midpoint_case = positive_case
    midpoint_initial = 0.5 * (zero_root.state + positive_root.state)
    midpoint_root, midpoint_report = solve_case(
        output,
        "D_half_moment",
        midpoint_case,
        midpoint_initial,
        MOMENT_SCALE_FRACTION * positive_target,
    )
    midpoint_report["target_fraction_of_qualified_moment"] = MOMENT_SCALE_FRACTION
    midpoint_report["seed_method"] = (
        "componentwise interpolation of accepted zero- and positive-moment states"
    )
    if midpoint_root is not None:
        midpoint_report["between_zero_and_positive_tilt"] = bool(
            0.0 < midpoint_root.state[-1] * MOMENT_ANGLE_SCALE_RAD < 5e-4
        )
        midpoint_report["pressure_centroid_between_zero_and_positive"] = bool(
            zero_report["metrics"]["pressure_centroid_m"][1]
            > midpoint_report["metrics"]["pressure_centroid_m"][1]
            > positive_report["metrics"]["pressure_centroid_m"][1]
        )
        midpoint_report["h_min_between_zero_and_positive"] = bool(
            zero_report["metrics"]["h_min_m"]
            > midpoint_report["metrics"]["h_min_m"]
            > positive_report["metrics"]["h_min_m"]
        )
        midpoint_report["interpolation_trend_passed"] = bool(
            midpoint_report["between_zero_and_positive_tilt"]
            and midpoint_report["pressure_centroid_between_zero_and_positive"]
            and midpoint_report["h_min_between_zero_and_positive"]
        )
    report["cases"]["D_half_moment"] = midpoint_report
    _write_case_report(output, "D_half_moment", midpoint_report)
    (output / "report.json").write_text(
        json.dumps(json_value(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetric-seed-dir", type=Path, default=DEFAULT_SYMMETRIC_SEED)
    parser.add_argument("--positive-seed-dir", type=Path, default=DEFAULT_POSITIVE_SEED)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir
    if output is None:
        output = Path("results") / (
            "finite_roller_moment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    result = run(args.symmetric_seed_dir, args.positive_seed_dir, output)
    (output / "report.json").write_text(
        json.dumps(json_value(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for label, item in result["cases"].items():
        print(
            f"{label}: {item['status']} "
            f"(alpha={item.get('recovered_alpha_rad')}, "
            f"complete={item.get('complete_wall_seconds')} s)"
        )
    return 0 if all(
        result["cases"][name]["status"] == "accepted_audited_root"
        for name in ("A_zero_moment", "B_positive_0p5_mrad", "C_negative_0p5_mrad")
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
