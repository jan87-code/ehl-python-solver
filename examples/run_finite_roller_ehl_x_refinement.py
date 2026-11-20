"""Refine finite-roller EHL in x, then conditionally continue prescribed tilt."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from run_finite_roller_ehl import (
    json_value,
    mirror_solution_metrics,
    physical_node_budget,
    refinement_differences,
    solve_case,
)

from ehl_solver.finite_roller_ehl import (
    FiniteRollerEHLCase,
    finite_roller_metrics,
    mirror_finite_roller_seed,
)
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_system import LimitedLoadProbe

NY = 513
NX_SEQUENCE = (385, 769)
BASE_NX = 193
BASE_TILT = 0.00025
SPIKE_WINDOW_HALF_WIDTH_M = 15e-6
AXIAL_QUALIFICATION_NY = 577
AXIAL_QUALIFICATION_NX = 769
AXIAL_QUALIFICATION_TILT = 0.0005


def _read_accepted_seed(seed_dir, label, nx, tilt):
    case = FiniteRollerEHLCase(
        nx=nx,
        ny=NY,
        axial_tilt_rad=tilt,
        maximum_physical_nodes=physical_node_budget(nx, NY),
    )
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    path = seed_dir / f"{label}_{nx}x{NY}" / f"{label}_{nx}x{NY}_accepted_state.npz"
    with np.load(path) as archive:
        state = archive["state"].copy()
    audit = audit_limited_state(probe, state, direct=True)
    if not audit["numerically_accepted"]:
        raise ValueError(f"saved seed failed independent CPU audit: {path}")
    return case, SimpleNamespace(state=state, audit=audit), probe


def _write_report(output, report):
    report["complete_wall_seconds"] = time.perf_counter() - report["_started"]
    serializable = {key: value for key, value in report.items() if key != "_started"}
    (output / "report.json").write_text(
        json.dumps(json_value(serializable), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _new_output(path):
    output = path or Path("results") / (
        "finite_roller_ehl_xref_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    return output


def _baseline_spike_window(case, root, probe):
    metrics, _ = finite_roller_metrics(case, probe, root.state, root.audit)
    spike = metrics["outlet_centerline_pressure_spike"]
    if spike is None:
        raise RuntimeError("accepted 193x513 symmetric root has no detected outlet spike")
    center = spike["x_m"]
    window = (center - SPIKE_WINDOW_HALF_WIDTH_M, center + SPIKE_WINDOW_HALF_WIDTH_M)
    metrics, _ = finite_roller_metrics(
        case, probe, root.state, root.audit, spike_window_m=window
    )
    return window, metrics


def run_rolling_refinement(seed_dir, output):
    started = time.perf_counter()
    reports = {
        "_started": started,
        "phase": "rolling_direction_refinement",
        "grid_order": "Nx x Ny",
        "ny_fixed": NY,
        "nx_sequence": [BASE_NX, *NX_SEQUENCE],
        "seed_directory": str(seed_dir),
        "cases": {},
        "refinements": {},
    }

    seed_a = _read_accepted_seed(seed_dir, "A_symmetric", BASE_NX, 0.0)
    spike_window, baseline_metrics = _baseline_spike_window(*seed_a)
    reports["fixed_spike_window_m"] = list(spike_window)
    reports["baseline_A_193x513"] = json_value(baseline_metrics)
    reports["baseline_A_independent_cpu_audit"] = json_value(seed_a[1].audit)
    seed_b = _read_accepted_seed(
        seed_dir, "B_plus_0p00025", BASE_NX, BASE_TILT
    )
    reports["baseline_B_plus_0p00025_193x513"] = json_value(
        finite_roller_metrics(
            seed_b[0],
            seed_b[2],
            seed_b[1].state,
            seed_b[1].audit,
            spike_window_m=spike_window,
        )[0]
    )
    reports["baseline_B_independent_cpu_audit"] = json_value(seed_b[1].audit)
    _write_report(output, reports)

    for key, label, initial in (
        ("A", "A_symmetric", seed_a),
        ("B_plus", "B_plus_0p00025", seed_b),
    ):
        source_case, source_root, _ = initial
        for nx in NX_SEQUENCE:
            root, case_report = solve_case(
                output,
                label,
                nx,
                NY,
                source_case.axial_tilt_rad,
                source_case,
                source_root.state,
                spike_window_m=spike_window,
            )
            stage = f"{key}_{nx}x{NY}"
            reports["cases"][stage] = case_report
            if root is None or case_report["status"] != "accepted_audited_root":
                reports["stopped_at"] = stage
                _write_report(output, reports)
                return 2
            new_case = FiniteRollerEHLCase(
                nx=nx,
                ny=NY,
                axial_tilt_rad=source_case.axial_tilt_rad,
                maximum_physical_nodes=physical_node_budget(nx, NY),
            )
            reports["refinements"][stage] = refinement_differences(
                source_case,
                source_root,
                new_case,
                root,
                spike_window_m=spike_window,
            )
            reports["cases"][stage]["refinement_from_previous"] = json_value(
                reports["refinements"][stage]
            )
            source_case, source_root = new_case, root
            _write_report(output, reports)
    reports["rolling_refinement_complete"] = True
    reports["recommended_highest_nx"] = 769
    _write_report(output, reports)
    return 0


def _read_accepted_half_mrad_seed(seed_dir):
    label = "C_plus_000500urad_769x513"
    directory = seed_dir / label
    state_path = directory / f"{label}_accepted_state.npz"
    report_path = directory / f"{label}_report.json"
    seed_report = json.loads(report_path.read_text(encoding="utf-8"))
    if seed_report.get("status") != "accepted_audited_root":
        raise ValueError(f"0.5 mrad seed is not an accepted audited root: {report_path}")
    case = FiniteRollerEHLCase(
        nx=769,
        ny=NY,
        axial_tilt_rad=AXIAL_QUALIFICATION_TILT,
        maximum_physical_nodes=450000,
    )
    if seed_report.get("case") != json_value(case.physical_signature()):
        raise ValueError("0.5 mrad seed physical inputs do not match this qualification")
    if not seed_report.get("cuda_cpu_parity", {}).get("passed", False):
        raise ValueError("0.5 mrad seed did not pass its CUDA/CPU parity gate")
    with np.load(state_path) as archive:
        state = archive["state"].copy()
    probe = LimitedLoadProbe(case, transport_fraction=0.0)
    audit_started = time.perf_counter()
    audit = audit_limited_state(probe, state, direct=True)
    audit_wall = time.perf_counter() - audit_started
    if not audit["numerically_accepted"]:
        raise ValueError(f"saved 0.5 mrad seed failed independent CPU audit: {state_path}")
    root = SimpleNamespace(state=state, audit=audit)
    spike_window = tuple(
        json.loads((seed_dir / "report.json").read_text(encoding="utf-8"))[
            "fixed_spike_window_m"
        ]
    )
    metrics, profiles = finite_roller_metrics(
        case, probe, state, audit, spike_window_m=spike_window
    )
    return case, root, probe, metrics, profiles, seed_report, audit_wall, spike_window


def _signed_change(source, refined):
    difference = float(refined - source)
    return {
        "source": float(source),
        "refined": float(refined),
        "absolute_change": difference,
        "relative_change": (
            None if source == 0.0 else difference / abs(float(source))
        ),
    }


def _boundary_cavitation(probe, profiles):
    pressure = profiles["pressure_pa"]
    theta = profiles["theta"]
    threshold = 1e-8 * probe.system.problem.pressure_scale_pa
    mask = (pressure <= threshold) & (theta < 1.0 - 1e-8)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    rows = (1, pressure.shape[0] - 2)
    return {
        "negative_end": {
            "present": bool(np.any(mask[rows[0]])),
            "fraction_of_interior_x_cavitated": float(np.mean(mask[rows[0], 1:-1])),
        },
        "positive_end": {
            "present": bool(np.any(mask[rows[1]])),
            "fraction_of_interior_x_cavitated": float(np.mean(mask[rows[1], 1:-1])),
        },
    }


def _axial_metric_comparison(
    source_case,
    source_metrics,
    source_profiles,
    target_case,
    target_metrics,
    target_profiles,
    field_differences,
    source_probe,
    target_probe,
):
    source_hmin_xy = np.asarray(source_metrics["h_min_coordinates_m"])
    target_hmin_xy = np.asarray(target_metrics["h_min_coordinates_m"])
    source_centroid = np.asarray(source_metrics["pressure_centroid_m"])
    target_centroid = np.asarray(target_metrics["pressure_centroid_m"])
    loaded_end_clearance = lambda case, coordinates: float(  # noqa: E731
        coordinates[1] + case.half_length_m
    )
    support_key = "0.01_pmax"
    source_support = source_metrics["pressurized_support_clearance_by_peak_fraction"][
        "left"
    ][support_key]
    target_support = target_metrics["pressurized_support_clearance_by_peak_fraction"][
        "left"
    ][support_key]

    def extent_change(key):
        source = source_metrics[key]
        target = target_metrics[key]
        if source is None or target is None:
            return {"source": source, "refined": target, "change_m": None}
        source_array = np.asarray(source, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        return {
            "source_m": source_array.tolist(),
            "refined_m": target_array.tolist(),
            "change_m": (target_array - source_array).tolist(),
        }

    bands = {}
    for band in ("2", "5", "10"):
        source_loads = np.asarray(source_metrics["outer_band_load_n_left_right"][band])
        target_loads = np.asarray(target_metrics["outer_band_load_n_left_right"][band])
        bands[band] = {
            "source_load_n_negative_positive": source_loads.tolist(),
            "refined_load_n_negative_positive": target_loads.tolist(),
            "change_n_negative_positive": (target_loads - source_loads).tolist(),
            "refined_percent_total_load_negative_positive": (
                100.0 * target_loads / target_case.load_n
            ).tolist(),
        }

    source_spike_window = source_metrics["fixed_spike_window"]
    target_spike_window = target_metrics["fixed_spike_window"]
    source_constriction = source_metrics["outlet_centerline_constriction"]
    target_constriction = target_metrics["outlet_centerline_constriction"]
    source_spike = source_metrics["outlet_centerline_pressure_spike"]
    target_spike = target_metrics["outlet_centerline_pressure_spike"]
    source_loaded_clearance = loaded_end_clearance(source_case, source_hmin_xy)
    target_loaded_clearance = loaded_end_clearance(target_case, target_hmin_xy)
    centroid_relative = float(
        np.linalg.norm(target_centroid - source_centroid)
        / max(np.linalg.norm(source_centroid), np.finfo(float).tiny)
    )
    moment_change = _signed_change(
        source_metrics["moment_about_x_n_m"],
        target_metrics["moment_about_x_n_m"],
    )
    source_boundary_cavitation = _boundary_cavitation(source_probe, source_profiles)
    target_boundary_cavitation = _boundary_cavitation(target_probe, target_profiles)
    newly_cavitated_end = any(
        target_boundary_cavitation[side]["present"]
        and not source_boundary_cavitation[side]["present"]
        for side in ("negative_end", "positive_end")
    )
    outermost_fraction = max(target_metrics["outermost_axial_row_load_fraction"])
    outer_2_percent = sum(target_metrics["outer_band_load_n_left_right"]["2"])
    outer_2_percent_fraction = outer_2_percent / target_case.load_n
    min_support_clearance = min(
        target_support["left_clearance_m"], target_support["right_clearance_m"]
    )
    end_film_positive = bool(
        np.all(np.isfinite(target_metrics["axial_end_row_h_min_m"]))
        and min(target_metrics["axial_end_row_h_min_m"]) > 0.0
    )
    no_new_boundary_feature = bool(
        target_metrics["boundary_pressure_max_pa"] == 0.0
        and end_film_positive
        and min_support_clearance > 0.0
        and target_loaded_clearance > 0.0
        and not newly_cavitated_end
    )
    audits_pass = bool(
        target_metrics["audit"]["numerically_accepted"]
        and abs(target_metrics["load_relative_error"]) <= 1e-8
        and target_metrics["theta_min_max"][0] >= 0.0
        and target_metrics["theta_min_max"][1] <= 1.0
        and source_case.load_n == target_case.load_n
    )
    criteria = {
        "h_c_change_below_1_percent": abs(
            target_metrics["h_c_m"] / source_metrics["h_c_m"] - 1.0
        )
        < 0.01,
        "h_min_change_below_2_percent": abs(
            target_metrics["h_min_m"] / source_metrics["h_min_m"] - 1.0
        )
        < 0.02,
        "bulk_film_relative_l2_below_1_percent": (
            field_differences["film_m_bulk_relative_l2_change"] < 0.01
        ),
        "centroid_and_moment_changes_below_3_percent": (
            centroid_relative < 0.03
            and abs(moment_change["relative_change"]) < 0.03
        ),
        "h_min_and_main_support_detached": (
            target_loaded_clearance > 0.0
            and target_support["left_clearance_m"] > 0.0
            and target_support["right_clearance_m"] > 0.0
        ),
        "outermost_row_load_negligible": outermost_fraction <= 1e-6,
        "outer_2_percent_load_below_0p1_percent": (
            outer_2_percent_fraction < 0.001
        ),
        "no_new_axial_boundary_feature": no_new_boundary_feature,
        "all_solver_audits_pass": audits_pass,
    }
    return {
        "scalar_changes": {
            "h_c_m": _signed_change(
                source_metrics["h_c_m"], target_metrics["h_c_m"]
            ),
            "h_min_m": _signed_change(
                source_metrics["h_min_m"], target_metrics["h_min_m"]
            ),
            "centerline_h_min_m": _signed_change(
                source_metrics["centerline_h_min_m"],
                target_metrics["centerline_h_min_m"],
            ),
            "loaded_end_outer_10_percent_h_min_m": _signed_change(
                source_metrics["end_10_percent_h_min_m"][0],
                target_metrics["end_10_percent_h_min_m"][0],
            ),
            "p_max_pa": _signed_change(
                source_metrics["p_max_pa"], target_metrics["p_max_pa"]
            ),
            "loaded_end_h_min_clearance_m": _signed_change(
                source_loaded_clearance, target_loaded_clearance
            ),
        },
        "coordinates": {
            "h_min_source_m": source_hmin_xy.tolist(),
            "h_min_refined_m": target_hmin_xy.tolist(),
            "h_min_coordinate_change_m": (
                target_hmin_xy - source_hmin_xy
            ).tolist(),
            "p_max_source_m": source_metrics["p_max_coordinates_m"],
            "p_max_refined_m": target_metrics["p_max_coordinates_m"],
            "p_max_coordinate_change_m": (
                np.asarray(target_metrics["p_max_coordinates_m"])
                - np.asarray(source_metrics["p_max_coordinates_m"])
            ).tolist(),
            "pressure_centroid_source_m": source_centroid.tolist(),
            "pressure_centroid_refined_m": target_centroid.tolist(),
            "pressure_centroid_change_m": (
                target_centroid - source_centroid
            ).tolist(),
            "pressure_centroid_relative_change": centroid_relative,
        },
        "pressure_support_1_percent_pmax": {
            "source_clearance_m_left_right": [
                source_support["left_clearance_m"],
                source_support["right_clearance_m"],
            ],
            "refined_clearance_m_left_right": [
                target_support["left_clearance_m"],
                target_support["right_clearance_m"],
            ],
            "refined_clearance_cells_left_right": [
                target_support["left_clearance_cells"],
                target_support["right_clearance_cells"],
            ],
        },
        "outermost_row_load_n_source_negative_positive": source_metrics[
            "outermost_axial_row_load_n"
        ],
        "outermost_row_load_n_refined_negative_positive": target_metrics[
            "outermost_axial_row_load_n"
        ],
        "outermost_row_load_fraction_refined": target_metrics[
            "outermost_axial_row_load_fraction"
        ],
        "outer_band_loads": bands,
        "moment_about_x_n_m": moment_change,
        "cavitation_bounds": {
            "axial": extent_change("cavitated_interior_axial_extent_m"),
            "rolling": extent_change("cavitated_interior_rolling_extent_m"),
            "boundary_adjacent_source": source_boundary_cavitation,
            "boundary_adjacent_refined": target_boundary_cavitation,
            "new_boundary_adjacent_cavitation": newly_cavitated_end,
        },
        "fixed_spike_window": {
            "source": source_spike_window,
            "refined": target_spike_window,
            "centerline_pressure_integral_change_pa_m": (
                target_spike_window["centerline_pressure_integral_pa_m"]
                - source_spike_window["centerline_pressure_integral_pa_m"]
            ),
            "integrated_pressure_load_change_n": (
                target_spike_window["integrated_pressure_load_n"]
                - source_spike_window["integrated_pressure_load_n"]
            ),
        },
        "outlet_pressure_spike": {
            "source": source_spike,
            "refined": target_spike,
            "amplitude_change_does_not_gate_axial_qualification": True,
            "unresolved": bool(
                target_spike is not None
                and target_spike["half_prominence_width_cells"] < 3.0
            ),
        },
        "outlet_constriction": {
            "source": source_constriction,
            "refined": target_constriction,
            "x_change_m": (
                target_constriction["x_m"] - source_constriction["x_m"]
            ),
            "film_change_m": (
                target_constriction["film_m"] - source_constriction["film_m"]
            ),
            "width_change_cells": (
                target_constriction["half_prominence_width_cells"]
                - source_constriction["half_prominence_width_cells"]
            ),
        },
        "bulk_field_changes": field_differences,
        "qualification_criteria": criteria,
        "refinement_qualified": bool(all(criteria.values())),
    }


def _solver_performance(report):
    timings = {}
    gmres_iterations = 0
    true_residuals = []
    true_relative_residuals = []
    model_attempts = 0
    for iteration in report.get("newton_history", []):
        for model in iteration.get("models", []):
            linear = model.get("linear_solver_timings", {})
            if not linear:
                continue
            model_attempts += 1
            for name in (
                "device_operator_initialization_seconds",
                "cuda_residual_seconds",
                "cuda_jvp_seconds",
                "cpu_superlu_setup_seconds",
                "cpu_superlu_solve_seconds",
                "cpu_preconditioner_apply_wall_seconds",
                "gmres_wall_seconds",
                "state_guide_mask_h2d_seconds",
                "cpu_residual_parity_h2d_seconds",
                "h2d_preconditioner_seconds",
                "d2h_seconds",
                "total_correction_seconds",
            ):
                timings[name] = timings.get(name, 0.0) + float(linear.get(name, 0.0))
            gmres_iterations += int(model.get("linear_iterations", 0))
            true_residuals.extend(
                float(value)
                for value in model.get("linear_true_residuals", [])
                if np.isfinite(value)
            )
            relative = linear.get("cpu_recomputed_true_relative_residual")
            if relative is not None and np.isfinite(relative):
                true_relative_residuals.append(float(relative))
    return {
        "solver_wall_seconds": report.get("solver_wall_seconds"),
        "complete_wall_seconds": report.get("complete_wall_seconds"),
        "newton_corrections": len(report.get("newton_history", [])),
        "linear_model_attempts_including_recoveries": model_attempts,
        "gmres_iterations": gmres_iterations,
        "linear_true_residual_norms": true_residuals,
        "maximum_linear_true_residual_norm": (
            max(true_residuals) if true_residuals else None
        ),
        "cpu_true_relative_residuals": true_relative_residuals,
        "maximum_cpu_true_relative_residual": (
            max(true_relative_residuals) if true_relative_residuals else None
        ),
        "seed_transfer_wall_seconds": report.get("seed_transfer_wall_seconds"),
        "checkpoint_wall_seconds": report.get("checkpoint_wall_seconds"),
        "checkpoint_count": report.get("checkpoint_count"),
        "cpu_audit_wall_seconds": report.get("cpu_audit_wall_seconds"),
        "parity_wall_seconds": report.get("parity_wall_seconds"),
        "summary_io_wall_seconds": report.get("summary_io_wall_seconds"),
        "summed_linear_component_seconds": timings,
        "timing_note": (
            "Components overlap; CUDA JVP includes FFT elasticity operations. "
            "No standalone FFT or matched CPU-only full-solve timer is available."
        ),
    }


def run_axial_qualification(seed_dir, output):
    started = time.perf_counter()
    source = _read_accepted_half_mrad_seed(seed_dir)
    (
        source_case,
        source_root,
        source_probe,
        source_metrics,
        source_profiles,
        source_report,
        source_audit_wall,
        spike_window,
    ) = source
    report = {
        "_started": started,
        "phase": "axial_refinement_qualification",
        "grid_order": "Nx x Ny",
        "source_grid_nx_ny": [source_case.nx, source_case.ny],
        "target_grid_nx_ny": [AXIAL_QUALIFICATION_NX, AXIAL_QUALIFICATION_NY],
        "fixed_spike_window_m": list(spike_window),
        "seed_directory": str(seed_dir),
        "source_seed_status": source_report["status"],
        "source_independent_cpu_audit": json_value(source_root.audit),
        "source_independent_cpu_audit_wall_seconds": source_audit_wall,
        "source_metrics": json_value(source_metrics),
        "status": "running",
    }
    _write_report(output, report)
    target_root, target_report = solve_case(
        output,
        "D_plus_000500urad",
        AXIAL_QUALIFICATION_NX,
        AXIAL_QUALIFICATION_NY,
        AXIAL_QUALIFICATION_TILT,
        source_case,
        source_root.state,
        spike_window_m=spike_window,
    )
    report["target_case"] = target_report
    if target_root is None or target_report["status"] != "accepted_audited_root":
        report["status"] = "axial_refined_root_not_accepted"
        _write_report(output, report)
        return 2

    target_case = FiniteRollerEHLCase(
        nx=AXIAL_QUALIFICATION_NX,
        ny=AXIAL_QUALIFICATION_NY,
        axial_tilt_rad=AXIAL_QUALIFICATION_TILT,
        maximum_physical_nodes=450000,
    )
    target_probe = LimitedLoadProbe(target_case, transport_fraction=0.0)
    target_metrics, target_profiles = finite_roller_metrics(
        target_case,
        target_probe,
        target_root.state,
        target_report["independent_cpu_audit"],
        spike_window_m=spike_window,
    )
    field_differences = refinement_differences(
        source_case,
        source_root,
        target_case,
        target_root,
        spike_window_m=spike_window,
    )
    comparison = _axial_metric_comparison(
        source_case,
        source_metrics,
        source_profiles,
        target_case,
        target_metrics,
        target_profiles,
        field_differences,
        source_probe,
        target_probe,
    )
    comparison["qualification_criteria"]["all_solver_audits_pass"] = bool(
        comparison["qualification_criteria"]["all_solver_audits_pass"]
        and target_report.get("independent_cpu_audit", {}).get(
            "numerically_accepted", False
        )
        and target_report.get("cuda_cpu_parity", {}).get("passed", False)
        and source_root.audit.get("numerically_accepted", False)
        and source_report.get("cuda_cpu_parity", {}).get("passed", False)
    )
    comparison["refinement_qualified"] = bool(
        all(comparison["qualification_criteria"].values())
    )
    report["target_metrics_recomputed"] = json_value(target_metrics)
    report["comparison"] = json_value(comparison)
    report["performance"] = _solver_performance(target_report)

    negative_case = FiniteRollerEHLCase(
        nx=AXIAL_QUALIFICATION_NX,
        ny=AXIAL_QUALIFICATION_NY,
        axial_tilt_rad=-AXIAL_QUALIFICATION_TILT,
        domain_half_width_x_m=target_case.domain_half_width_x_m,
        maximum_physical_nodes=450000,
    )
    negative_probe = LimitedLoadProbe(negative_case, transport_fraction=0.0)
    negative_seed = mirror_finite_roller_seed(
        target_case, target_root.state, negative_case, negative_probe.system
    )
    mirror_audit_started = time.perf_counter()
    negative_audit = audit_limited_state(negative_probe, negative_seed, direct=True)
    mirror_audit_wall = time.perf_counter() - mirror_audit_started
    negative_root = SimpleNamespace(state=negative_seed, audit=negative_audit)
    negative_metrics, negative_profiles = finite_roller_metrics(
        negative_case,
        negative_probe,
        negative_seed,
        negative_audit,
        spike_window_m=spike_window,
    )
    mirror = mirror_solution_metrics(
        target_case, target_root, negative_case, negative_root
    )
    centroid = target_metrics["pressure_centroid_m"]
    negative_centroid = negative_metrics["pressure_centroid_m"]
    scalar_film_relative = max(
        abs(target_metrics["h_c_m"] - negative_metrics["h_c_m"]),
        abs(target_metrics["h_min_m"] - negative_metrics["h_min_m"]),
        abs(target_metrics["centerline_h_min_m"] - negative_metrics["centerline_h_min_m"]),
    ) / target_case.correlation().central_film_m
    centroid_reflection_error = [
        abs(centroid[0] - negative_centroid[0]),
        abs(centroid[1] + negative_centroid[1]),
    ]
    moment_reflection_error = abs(
        target_metrics["moment_about_x_n_m"]
        + negative_metrics["moment_about_x_n_m"]
    )
    mirror["pressure_centroid_reflection_error_m"] = centroid_reflection_error
    mirror["moment_reflection_error_n_m"] = moment_reflection_error
    mirror["scalar_film_relative_difference"] = float(scalar_film_relative)
    mirror["independent_cpu_audit_passed"] = bool(
        negative_audit["numerically_accepted"]
    )
    mirror["passed"] = bool(
        mirror["passed"]
        and negative_audit["numerically_accepted"]
        and max(centroid_reflection_error) < 1e-10
        and moment_reflection_error < 1e-10
        and scalar_film_relative < 1e-10
    )
    np.savez_compressed(
        output / "E_minus_000500urad_mirrored_fields.npz",
        state=negative_seed,
        pressure_pa=negative_profiles["pressure_pa"],
        film_thickness_m=negative_profiles["film_m"],
        fractional_film_content=negative_profiles["theta"],
    )
    report["mirrored_negative_tilt"] = {
        "independent_cpu_audit": json_value(negative_audit),
        "audit_wall_seconds": mirror_audit_wall,
        "metrics": json_value(negative_metrics),
        "comparison": json_value(mirror),
    }
    report["status"] = (
        "refinement_qualified_and_mirrored"
        if comparison["refinement_qualified"] and mirror["passed"]
        else "accepted_root_not_refinement_qualified_or_mirror_failed"
    )
    _write_report(output, report)
    return 0 if report["status"] == "refinement_qualified_and_mirrored" else 2


def _tilt_gate_metrics(metrics):
    clearance = metrics["pressurized_support_clearance_by_peak_fraction"]["left"][
        "0.01_pmax"
    ]
    return {
        "minimum_1pct_pressure_support_clearance_m": min(
            clearance["left_clearance_m"], clearance["right_clearance_m"]
        ),
        "h_min_clearance_m": metrics["h_min_boundary_clearance_m"],
        "outer_band_load_n_left_right": metrics["outer_band_load_n_left_right"],
        "outermost_axial_row_load_n": metrics["outermost_axial_row_load_n"],
        "film_minimum_is_interior": metrics["h_min_boundary_clearance_m"] > 0.0,
    }


def _summarize_interrupted_attempt(heartbeat_path):
    events = [
        json.loads(line)
        for line in heartbeat_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed = [
        event for event in events if event.get("stage") == "correction_complete"
    ]
    gmres = [event for event in events if event.get("stage") == "gmres"]
    return {
        "status": "stopped before the 0.5 mrad root; checkpoints and heartbeat retained",
        "heartbeat_path": str(heartbeat_path),
        "completed_newton_corrections": len(completed),
        "last_completed_correction": completed[-1] if completed else None,
        "latest_gmres_event": gmres[-1] if gmres else None,
        "last_event": events[-1] if events else None,
    }


def run_tilt_continuation(
    seed_dir, output, *, continue_only=False, direct_attempt_heartbeat=None
):
    started = time.perf_counter()
    reports = {
        "_started": started,
        "phase": "prescribed_tilt_0p0005",
        "grid_nx_ny": [769, NY],
        "seed_directory": str(seed_dir),
        "tilt_stages": [],
        "direct_attempt_skipped": continue_only,
    }
    if direct_attempt_heartbeat is not None:
        reports["prior_direct_attempt"] = _summarize_interrupted_attempt(
            direct_attempt_heartbeat
        )
    seed_case, seed_root, _ = _read_accepted_seed(
        seed_dir, "B_plus_0p00025", 769, BASE_TILT
    )
    source_report = json.loads((seed_dir / "report.json").read_text(encoding="utf-8"))
    spike_window = tuple(source_report["fixed_spike_window_m"])
    reports["fixed_spike_window_m"] = list(spike_window)
    candidates = (
        []
        if continue_only
        else [(0.0005, "C_plus_0p0005_direct")]
    )
    reports["tilt_strategy"] = "direct first; bounded 0.00005-rad increments after failure"
    if continue_only:
        reports["tilt_strategy"] = "bounded 0.00005-rad increments from 0.00025 to 0.0005 rad"
    accepted = None
    _write_report(output, reports)
    for tilt, label in candidates:
        root, case_report = solve_case(
            output,
            label,
            769,
            NY,
            tilt,
            seed_case,
            seed_root.state,
            spike_window_m=spike_window,
        )
        reports["tilt_stages"].append(case_report)
        _write_report(output, reports)
        if root is not None and case_report["status"] == "accepted_audited_root":
            accepted = (
                FiniteRollerEHLCase(
                    nx=769,
                    ny=NY,
                    axial_tilt_rad=tilt,
                    maximum_physical_nodes=physical_node_budget(769, NY),
                ),
                root,
                case_report,
            )
            break
    if accepted is None:
        continuation = (0.00030, 0.00035, 0.00040, 0.00045, 0.00050)
        for index, tilt in enumerate(continuation):
            label = f"C_plus_{int(round(tilt * 1e6)):06d}urad"
            root, case_report = solve_case(
                output,
                label,
                769,
                NY,
                tilt,
                seed_case,
                seed_root.state,
                spike_window_m=spike_window,
            )
            reports["tilt_stages"].append(case_report)
            _write_report(output, reports)
            if root is None or case_report["status"] != "accepted_audited_root":
                reports["stopped_at_tilt_rad"] = tilt
                _write_report(output, reports)
                return 2
            seed_case = FiniteRollerEHLCase(
                nx=769,
                ny=NY,
                axial_tilt_rad=tilt,
                maximum_physical_nodes=physical_node_budget(769, NY),
            )
            seed_root = root
            if index == len(continuation) - 1:
                accepted = (seed_case, seed_root, case_report)
    if accepted is None:
        reports["status"] = "0.5 mrad solver root not accepted"
        _write_report(output, reports)
        return 2

    positive_case, positive_root, positive_report = accepted
    positive_metrics = positive_report["metrics"]
    reports["positive_tilt_gate_metrics"] = _tilt_gate_metrics(positive_metrics)
    reports["positive_tilt_detached"] = bool(
        reports["positive_tilt_gate_metrics"][
            "minimum_1pct_pressure_support_clearance_m"
        ]
        > 0.0
        and reports["positive_tilt_gate_metrics"]["film_minimum_is_interior"]
    )
    if not reports["positive_tilt_detached"]:
        reports["status"] = "audited root accepted; axial detachment gate failed"
        _write_report(output, reports)
        return 0

    negative_case = FiniteRollerEHLCase(
        nx=769,
        ny=NY,
        axial_tilt_rad=-0.0005,
        maximum_physical_nodes=physical_node_budget(769, NY),
    )
    negative_probe = LimitedLoadProbe(negative_case, transport_fraction=0.0)
    negative_state = mirror_finite_roller_seed(
        positive_case, positive_root.state, negative_case, negative_probe.system
    )
    negative_audit = audit_limited_state(negative_probe, negative_state, direct=True)
    negative_root = SimpleNamespace(state=negative_state, audit=negative_audit)
    negative_metrics, _ = finite_roller_metrics(
        negative_case,
        negative_probe,
        negative_state,
        negative_audit,
        spike_window_m=spike_window,
    )
    mirror = mirror_solution_metrics(
        positive_case, positive_root, negative_case, negative_root
    )
    np.savez_compressed(
        output / "B_minus_0p0005_mirrored_fields.npz",
        state=negative_state,
        pressure_pa=negative_probe.fields(negative_state)[0],
        film_thickness_m=negative_probe.fields(negative_state)[2],
        fractional_film_content=negative_probe.fields(negative_state)[1],
    )
    reports["negative_mirror_verification"] = {
        "independent_cpu_audit": json_value(negative_audit),
        "metrics": json_value(negative_metrics),
        "mirror_comparison": json_value(mirror),
    }
    reports["status"] = (
        "accepted_audited_detached_mirrored"
        if negative_audit["numerically_accepted"] and mirror["passed"]
        else "positive accepted; mirrored CPU audit or symmetry check failed"
    )
    _write_report(output, reports)
    return 0 if reports["status"] == "accepted_audited_detached_mirrored" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("rolling", "tilt", "axial"), required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--continue-only", action="store_true")
    parser.add_argument("--direct-attempt-heartbeat", type=Path)
    args = parser.parse_args(argv)
    if args.phase != "tilt" and (
        args.continue_only or args.direct_attempt_heartbeat is not None
    ):
        parser.error("--continue-only and --direct-attempt-heartbeat require --phase tilt")
    if args.continue_only and args.direct_attempt_heartbeat is None:
        parser.error("--continue-only requires --direct-attempt-heartbeat")
    import torch

    if not torch.cuda.is_available():
        parser.error("production finite-roller solves require the verified CUDA backend")
    output = _new_output(args.output_dir)
    if args.phase == "rolling":
        return run_rolling_refinement(args.seed_dir, output)
    if args.phase == "axial":
        return run_axial_qualification(args.seed_dir, output)
    return run_tilt_continuation(
        args.seed_dir,
        output,
        continue_only=args.continue_only,
        direct_attempt_heartbeat=args.direct_attempt_heartbeat,
    )


if __name__ == "__main__":
    raise SystemExit(main())
