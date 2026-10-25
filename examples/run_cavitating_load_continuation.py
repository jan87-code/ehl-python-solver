"""Bounded bidirectional positive-force continuation with independent SI audits."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ehl_solver.cavitating_continuation import (
    CavitatingContinuationError,
    continue_cavitating_load,
)
from ehl_solver.cavitating_coupled import CavitatingElasticityProblem, solve_cavitating_elasticity
from ehl_solver.cavitating_newton import solve_cavitating_newton
from ehl_solver.verification import audit_cavitating_solution
from examples.run_cavitating_elasticity import save_plots


def main(directory, *, plots=False):
    directory.mkdir(parents=True, exist_ok=False)
    problem = CavitatingElasticityProblem(
        np.linspace(-0.5e-3, 0.75e-3, 17),
        np.linspace(-0.5e-3, 0.5e-3, 13),
        20e-6,
        0.01,
        0.02,
        1e7,
        5e-4,
        20e-6,
        21250,
        12,
    )
    initial = solve_cavitating_newton(problem, target_load_n=0.002).solution
    report = {
        "initial_force_n": 0.002,
        "shape": problem.shape,
        "E_star_pa": problem.e_star_pa,
        "settings": {
            "initial_step_fraction": 0.1,
            "minimum_step_fraction": 0.005,
            "maximum_step_fraction": 0.25,
            "maximum_attempts": 30,
        },
        "cases": [],
    }
    for force in (0.001, 0.005, 0.02):
        failed = None
        try:
            result = continue_cavitating_load(problem, initial, force)
            solution, trials = result.solution, result.trials
        except CavitatingContinuationError as error:
            solution, trials, failed = error.last_solution, error.trials, str(error)
        name = f"target_{force}"
        trial_records = []
        for number, trial in enumerate(trials):
            record = {
                "target_load_n": trial.target_load_n,
                "accepted": trial.accepted,
                "step_fraction": trial.step_fraction,
                "reason": trial.reason,
                "diagnostics": trial.diagnostics,
                "newton_history": [asdict(row) for row in trial.newton_history],
            }
            if trial.solution is not None:
                s = trial.solution
                np.savez_compressed(
                    directory / f"{name}_trial_{number:02d}.npz",
                    pressure_pa=s.pressure_pa,
                    theta=s.fractional_film_content,
                    film_m=s.film_thickness_m,
                    displacement_m=s.elastic_displacement_m,
                    h0_m=s.central_film_offset_m,
                )
                record["candidate_h0_m"] = s.central_film_offset_m
            if trial.accepted:
                audit = audit_cavitating_solution(problem, trial.solution)
                if (
                    audit.mass_maximum > 1.001e-8
                    or audit.maximum_complementarity > 1e-10
                    or abs(audit.load_n - trial.target_load_n) / trial.target_load_n > 1e-8
                    or audit.minimum_pressure_pa < -1e-10 * problem.pressure_scale_pa
                    or audit.minimum_saturation < -1e-10
                    or audit.maximum_saturation > 1 + 1e-10
                    or audit.film_coupling_max_error_m > 1e-17
                ):
                    raise RuntimeError(f"independent continuation-state audit failed: {audit}")
                record["audit"] = asdict(audit)
            trial_records.append(record)
        if force == 0.02 and failed is None:
            reference = solve_cavitating_elasticity(
                replace(problem, central_film_offset_m=solution.central_film_offset_m),
                mass_tolerance=1e-10,
            )
            reference_difference = float(
                np.linalg.norm(solution.pressure_pa - reference.pressure_pa)
                / np.linalg.norm(reference.pressure_pa)
            )
            if reference_difference > 1e-6:
                raise RuntimeError("stronger-coupling reference pressure mismatch")
            (directory / "strong_reference.json").write_text(
                json.dumps(
                    {
                        "pressure_relative_difference": reference_difference,
                        "outer_iterations": len(reference.history),
                        "reference_audit": asdict(audit_cavitating_solution(problem, reference)),
                    },
                    indent=2,
                )
                + "\n"
            )
        report["cases"].append(
            {
                "target_load_n": force,
                "termination": failed or "target reached",
                "final_h0_m": solution.central_film_offset_m,
                "final_load_n": solution.load_n,
                "maximum_displacement_m": float(np.max(solution.elastic_displacement_m)),
                "final_audit": asdict(audit_cavitating_solution(problem, solution)),
                "trials": trial_records,
            }
        )
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if plots:
            save_plots(directory, name, problem, solution)
            save_path_plot(directory, name, trials)
        print(
            f"{force} N: {failed or 'target reached'}; h0={solution.central_film_offset_m:.9g} m; "
            f"{len(trials)} trials, {sum(not t.accepted for t in trials)} rejected",
            flush=True,
        )
        if failed:
            raise RuntimeError(failed)
    return report


def save_path_plot(directory, name, trials):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    for index, trial in enumerate(trials):
        marker, color = ("o", "tab:green") if trial.accepted else ("x", "tab:red")
        axes[0].plot(index + 1, trial.target_load_n, marker, color=color)
        if trial.solution is not None:
            axes[1].plot(index + 1, trial.solution.central_film_offset_m * 1e6, marker, color=color)
        axes[2].plot(index + 1, len(trial.newton_history), marker, color=color)
    axes[0].set_ylabel("Attempted positive force [N]")
    axes[1].set_ylabel("Candidate h0 [um]")
    axes[2].set(xlabel="Load trial", ylabel="Accepted Newton corrections")
    figure.suptitle("Green: verified roots; red: rejected trials")
    figure.savefig(directory / f"{name}_continuation.png", dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/cavitating_continuation")
    )
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    main(args.output_directory, plots=args.plots)
