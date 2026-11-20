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
    parser.add_argument("--phase", choices=("rolling", "tilt"), required=True)
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
    return run_tilt_continuation(
        args.seed_dir,
        output,
        continue_only=args.continue_only,
        direct_attempt_heartbeat=args.direct_attempt_heartbeat,
    )


if __name__ == "__main__":
    raise SystemExit(main())
