"""Independent positive-force monolithic JFO/elasticity verification.

Run with python -m examples.run_cavitating_newton and one BLAS/OpenMP thread.
Pressure/saturation/film are solved together with h0 for one fixed SI force.
The nested reference brackets h0 with separate converged fixed-point solves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem, solve_cavitating_elasticity
from ehl_solver.cavitating_newton import CavitatingNewtonError, solve_cavitating_newton
from ehl_solver.verification import audit_cavitating_solution
from examples.run_cavitating_elasticity import save_plots


def nested_reference(problem, target_load_n):
    """Independent scalar h0 root; each force evaluation solves full coupling."""
    evaluations = []

    def evaluate(offset):
        solution = solve_cavitating_elasticity(
            replace(problem, central_film_offset_m=offset), mass_tolerance=1e-10
        )
        evaluations.append(
            {"h0_m": offset, "force_n": solution.load_n, "outer_iterations": len(solution.history)}
        )
        return solution.load_n - target_load_n

    offset = brentq(evaluate, 18e-6, 26e-6, xtol=1e-14, rtol=1e-12, maxiter=20)
    solution = solve_cavitating_elasticity(
        replace(problem, central_film_offset_m=offset), mass_tolerance=1e-10
    )
    return solution, evaluations


def save_newton_plot(directory, label, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8, 11), constrained_layout=True)
    axes[0].semilogy(
        [r.mass_maximum_after for r in result.history], label="Post-step max mass residual"
    )
    axes[0].semilogy([r.merit_after for r in result.history], label="Post-step merit")
    axes[0].legend()
    axes[1].plot([r.central_film_offset_m_after * 1e6 for r in result.history], label="h0 [um]")
    axes[1].plot([r.minimum_film_m * 1e6 for r in result.history], label="Minimum film [um]")
    axes[1].legend()
    axes[2].plot([r.step_length for r in result.history], label="Accepted step")
    axes[2].plot([r.linear_iterations for r in result.history], label="Krylov iterations")
    axes[2].set_xlabel("Accepted Newton correction")
    axes[2].legend()
    work = 0
    for row in result.history:
        for trace, measured in zip(row.krylov_histories, row.linear_true_residuals, strict=True):
            axes[3].semilogy(range(work + 1, work + len(trace) + 1), trace, color="tab:blue")
            work += len(trace)
            axes[3].semilogy([work], [measured], "x", color="tab:red")
    axes[3].set(xlabel="Cumulative Krylov iteration", ylabel="Preconditioned / true residual")
    fig.savefig(directory / f"{label}_newton_convergence.png", dpi=120)
    plt.close(fig)


def main(directory, *, plots=False, target_load_n=0.002):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "target_load_n": target_load_n,
        "E_star_pa": 1e7,
        "model": "isoviscous incompressible JFO/elasticity numerical verification",
        "settings": {
            "x_limits_m": [-0.5e-3, 0.75e-3],
            "y_limits_m": [-0.5e-3, 0.5e-3],
            "initial_h0_m": 20e-6,
            "Rx_m": 0.01,
            "Ry_m": 0.02,
            "L_m": 5e-4,
            "h_ref_m": 20e-6,
            "p_ref_pa": 21250,
            "Lambda": 12,
        },
        "cases": [],
    }
    for shape in ((9, 13), (17, 25), (25, 37)):
        label = f"grid_{shape[0]}_{shape[1]}"
        problem = CavitatingElasticityProblem(
            np.linspace(-0.5e-3, 0.75e-3, shape[1]),
            np.linspace(-0.5e-3, 0.5e-3, shape[0]),
            20e-6,
            0.01,
            0.02,
            1e7,
            5e-4,
            20e-6,
            21250,
            12,
        )
        try:
            result = solve_cavitating_newton(problem, target_load_n=target_load_n)
        except CavitatingNewtonError as error:
            (directory / f"{label}_failure.json").write_text(
                json.dumps(
                    {
                        "reason": str(error),
                        "diagnostics": error.diagnostics,
                        "history": [asdict(row) for row in error.history],
                    },
                    indent=2,
                )
                + "\n"
            )
            raise
        solution = result.solution
        np.savez_compressed(
            directory / f"{label}.npz",
            x_m=problem.x_m,
            y_m=problem.y_m,
            pressure_pa=solution.pressure_pa,
            theta=solution.fractional_film_content,
            film_m=solution.film_thickness_m,
            displacement_m=solution.elastic_displacement_m,
        )
        audit = audit_cavitating_solution(problem, solution)
        if (
            audit.mass_maximum > 1.001e-8
            or audit.maximum_complementarity > 1e-10
            or abs(audit.load_n - target_load_n) / target_load_n > 1e-8
            or audit.minimum_pressure_pa < -1e-10 * problem.pressure_scale_pa
            or audit.minimum_saturation < -1e-10
            or audit.maximum_saturation > 1 + 1e-10
            or audit.pressure_boundary_max_pa != 0
            or audit.inlet_saturation_error > 1e-10
            or audit.film_coupling_max_error_m > 1e-17
        ):
            raise RuntimeError(f"independent loaded-state audit failed: {audit}")
        entry = {
            "shape": shape,
            "h0_m": solution.central_film_offset_m,
            "peak_pressure_pa": float(solution.pressure_pa.max()),
            "audit": asdict(audit),
            "history": [asdict(row) for row in result.history],
        }
        if shape == (9, 13):
            reference, evaluations = nested_reference(problem, target_load_n)
            entry["nested_reference"] = {
                "h0_m": reference.central_film_offset_m,
                "audit": asdict(audit_cavitating_solution(problem, reference)),
                "force_evaluations": evaluations,
                "pressure_relative_difference": float(
                    np.linalg.norm(solution.pressure_pa - reference.pressure_pa)
                    / np.linalg.norm(reference.pressure_pa)
                ),
            }
            np.savez_compressed(
                directory / f"{label}_nested_reference.npz",
                x_m=problem.x_m,
                y_m=problem.y_m,
                pressure_pa=reference.pressure_pa,
                film_m=reference.film_thickness_m,
                theta=reference.fractional_film_content,
                displacement_m=reference.elastic_displacement_m,
                h0_m=reference.central_film_offset_m,
            )
            np.testing.assert_allclose(
                solution.pressure_pa, reference.pressure_pa, rtol=1e-6, atol=1e-4
            )
        report["cases"].append(entry)
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if plots:
            save_plots(directory, label, problem, solution)
            save_newton_plot(directory, label, result)
        print(
            f"{shape}: h0={solution.central_film_offset_m:.12g} m; "
            f"force error={result.load_residual_n:.3e} N; "
            f"mass max={audit.mass_maximum:.3e}; {len(result.history)} Newton corrections",
            flush=True,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/cavitating_newton"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--target-load-n", type=float, default=0.002)
    args = parser.parse_args()
    main(args.output_directory, plots=args.plots, target_load_n=args.target_load_n)
