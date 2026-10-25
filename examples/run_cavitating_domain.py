"""Fixed-positive-force domain sensitivity using verified coupled warm starts.

Use python -m examples.run_cavitating_domain with one BLAS/OpenMP thread.
The domains add boundary nodes while preserving baseline dx,dy and SI load.
Cold-start iteration limits are not raised; a coupled reference supplies the
initial root for bounded adaptive force continuation on each domain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ehl_solver.cavitating_continuation import CavitatingContinuationError, continue_cavitating_load
from ehl_solver.cavitating_coupled import (
    CavitatingCouplingError,
    CavitatingElasticityProblem,
    solve_cavitating_elasticity,
)
from ehl_solver.verification import audit_cavitating_solution
from examples.run_cavitating_elasticity import save_plots
from examples.run_cavitating_load_continuation import save_path_plot


def make_domain_problem(extra):
    return make_boundary_problem(inlet_nodes=extra, outlet_nodes=extra, transverse_nodes=extra)


def make_boundary_problem(*, inlet_nodes=0, outlet_nodes=0, transverse_nodes=0):
    """Extend selected boundaries in whole cells; preserve baseline SI nodes.

    Transverse extensions add the same number of nodes on both y edges.
    x follows entrainment; arrays have shape (ny, nx).
    """
    for extra in (inlet_nodes, outlet_nodes, transverse_nodes):
        if not isinstance(extra, int) or isinstance(extra, bool) or extra < 0:
            raise ValueError("extra boundary nodes must be a nonnegative integer")
    dx, dy = 0.00125 / 16, 0.001 / 12
    return CavitatingElasticityProblem(
        -0.0005 + np.arange(-inlet_nodes, 17 + outlet_nodes) * dx,
        -0.0005 + np.arange(-transverse_nodes, 13 + transverse_nodes) * dy,
        20e-6,
        0.01,
        0.02,
        1e7,
        5e-4,
        20e-6,
        21250,
        12,
    )


def _save_state(path, problem, solution):
    np.savez_compressed(
        path,
        x_m=problem.x_m,
        y_m=problem.y_m,
        pressure_pa=solution.pressure_pa,
        theta=solution.fractional_film_content,
        film_m=solution.film_thickness_m,
        displacement_m=solution.elastic_displacement_m,
        h0_m=solution.central_film_offset_m,
    )


def _audit(problem, solution, target=None):
    audit = audit_cavitating_solution(problem, solution)
    if (
        audit.mass_maximum > 1.001e-8
        or audit.maximum_complementarity > 1e-10
        or audit.minimum_pressure_pa < -1e-10 * problem.pressure_scale_pa
        or audit.minimum_saturation < -1e-10
        or audit.maximum_saturation > 1 + 1e-10
        or audit.pressure_boundary_max_pa != 0
        or audit.inlet_saturation_error > 1e-10
        or audit.film_coupling_max_error_m > 1e-17
        or (target is not None and abs(audit.load_n - target) / target > 1e-8)
    ):
        raise RuntimeError(f"domain-state verification failed: {audit}")
    return audit


def run_case(extra, directory, *, plots=False, target=0.002):
    problem = make_domain_problem(extra)
    name = f"domain_{problem.shape[0]}_{problem.shape[1]}"
    entry = run_problem(problem, directory, name=name, plots=plots, target=target)
    entry["extra_nodes_per_edge"] = extra
    return entry


def run_problem(problem, directory, *, name, plots=False, target=0.002):
    """Solve and independently audit one domain; retain failed candidates."""
    entry = {
        "name": name,
        "shape": problem.shape,
        "x_limits_m": [float(problem.x_m[0]), float(problem.x_m[-1])],
        "y_limits_m": [float(problem.y_m[0]), float(problem.y_m[-1])],
        "dx_m": problem.dx_m,
        "dy_m": problem.dy_m,
        "target_load_n": target,
        "E_star_pa": problem.e_star_pa,
    }
    try:
        initial = solve_cavitating_elasticity(problem)
    except CavitatingCouplingError as error:
        entry.update(
            {
                "termination": "reference failed: " + str(error),
                "target_reached": False,
                "reference_history": [asdict(row) for row in error.history],
            }
        )
        if error.last_solution is not None:
            _save_state(directory / f"{name}_failed_reference.npz", problem, error.last_solution)
        return entry
    entry["initial_reference_audit"] = asdict(_audit(problem, initial))
    entry["initial_reference_iterations"] = len(initial.history)
    entry["initial_reference_force_n"] = initial.load_n
    entry["initial_reference_history"] = [asdict(row) for row in initial.history]
    entry["rigid_reference_force_n"] = next(
        row.load_n for row in reversed(initial.history) if row.compliance_factor == 0
    )
    _save_state(directory / f"{name}_initial.npz", problem, initial)
    failed = None
    try:
        path = continue_cavitating_load(problem, initial, target)
        solution, trials = path.solution, path.trials
    except CavitatingContinuationError as error:
        failed, solution, trials = str(error), error.last_solution, error.trials
    trial_records = []
    for number, trial in enumerate(trials):
        record = {
            "target_load_n": trial.target_load_n,
            "accepted": trial.accepted,
            "reason": trial.reason,
            "diagnostics": trial.diagnostics,
            "newton_history": [asdict(row) for row in trial.newton_history],
        }
        if trial.solution is not None:
            _save_state(directory / f"{name}_trial_{number:02d}.npz", problem, trial.solution)
        if trial.accepted:
            record["audit"] = asdict(_audit(problem, trial.solution, trial.target_load_n))
        trial_records.append(record)
    entry.update(
        {
            "termination": failed or "target reached",
            "target_reached": failed is None,
            "final_h0_m": solution.central_film_offset_m,
            "peak_pressure_pa": float(solution.pressure_pa.max()),
            "final_audit": asdict(_audit(problem, solution, target if failed is None else None)),
            "trials": trial_records,
        }
    )
    _save_state(directory / f"{name}_final.npz", problem, solution)
    if plots:
        save_plots(directory, name, problem, solution)
        save_path_plot(directory, name, trials)
    return entry


def main(directory, *, plots=False):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "method": "coupled fixed-point warm start + adaptive monolithic positive-load continuation",
        "inputs": {
            "target_load_n": 0.002,
            "E_star_pa": 1e7,
            "reference_h0_m": 20e-6,
            "Rx_m": 0.01,
            "Ry_m": 0.02,
            "L_m": 5e-4,
            "h_ref_m": 20e-6,
            "p_ref_pa": 21250,
            "Lambda": 12,
            "inlet_theta": 1,
        },
        "cases": [],
    }
    for extra in (0, 4, 8):
        entry = run_case(extra, directory, plots=plots)
        report["cases"].append(entry)
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{entry['shape']}: {entry['termination']}; h0={entry.get('final_h0_m')}; "
            f"load trials={len(entry.get('trials', []))}",
            flush=True,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/cavitating_domain"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    report = main(args.output_directory, plots=args.plots)
    raise SystemExit(0 if all(entry["target_reached"] for entry in report["cases"]) else 1)
