"""Reproducible positive-load prescribed-offset JFO/elasticity CPU benchmark."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem, solve_cavitating_elasticity
from ehl_solver.verification import audit_cavitating_solution


def save_plots(directory, name, problem, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ehl_solver.visualization import (
        plot_field_cross_section,
        plot_film_thickness_distribution,
        plot_pressure_distribution,
    )

    figures = {
        "pressure": plot_pressure_distribution(problem.x_m, problem.y_m, result.pressure_pa)[0],
        "film": plot_film_thickness_distribution(problem.x_m, problem.y_m, result.film_thickness_m)[
            0
        ],
    }
    for axis in ("x", "y"):
        for label, values in (("pressure", result.pressure_pa), ("film", result.film_thickness_m)):
            figures[f"{label}_{axis}"] = plot_field_cross_section(
                problem.x_m, problem.y_m, values, axis=axis, coordinate_m=0, quantity=label
            )[0]
    for label, values, units in (
        ("saturation", result.fractional_film_content, "Fractional film content theta"),
        ("displacement", result.elastic_displacement_m * 1e6, "Displacement [um]"),
    ):
        figure, axis = plt.subplots(constrained_layout=True)
        artist = axis.pcolormesh(problem.x_m * 1e3, problem.y_m * 1e3, values, shading="auto")
        figure.colorbar(artist, ax=axis, label=units)
        axis.set(xlabel="x [mm] (entrainment)", ylabel="y [mm]", title=label)
        figures[label] = figure
    if result.history:
        figure, axes = plt.subplots(3, 1, constrained_layout=True, figsize=(8, 9))
        axes[0].semilogy(
            [r.metrics.mass_maximum for r in result.history], label="Maximum mass residual"
        )
        axes[0].semilogy(
            [r.film_defect_relative for r in result.history], label="Relative film defect"
        )
        axes[0].semilogy(
            [r.pressure_update_relative for r in result.history], label="Relative p update"
        )
        axes[0].legend()
        axes[1].plot([r.load_n for r in result.history])
        axes[1].set_ylabel("Pressure force [N]; prescribed h0")
        axes[2].plot([r.compliance_factor for r in result.history], label="Compliance gamma")
        axes[2].plot(
            [r.active_set_iterations for r in result.history], label="Inner active-set iterations"
        )
        axes[2].set_xlabel("Outer iteration")
        axes[2].legend()
        figures["convergence"] = figure
    for label, figure in figures.items():
        figure.savefig(directory / f"{name}_{label}.png", dpi=120)
        plt.close(figure)


def main(directory: Path, *, modulus_pa=1e7, plots=False):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "model": "prescribed-offset isoviscous JFO/elasticity numerical reference",
        "axis_order": "(ny,nx), x entrainment",
        "modulus_pa": modulus_pa,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            "h0_m": 20e-6,
            "Rx_m": 0.01,
            "Ry_m": 0.02,
            "L_m": 5e-4,
            "h_ref_m": 20e-6,
            "p_ref_pa": 21250,
            "Lambda": 12,
            "x_limits_m": [-0.5e-3, 0.75e-3],
            "y_limits_m": [-0.5e-3, 0.5e-3],
        },
        "cases": [],
    }
    for ny, nx, supply in ((9, 13, 1.0), (17, 25, 1.0), (25, 37, 1.0), (17, 25, 0.99)):
        name = f"grid_{ny}_{nx}_supply_{supply}"
        problem = CavitatingElasticityProblem(
            np.linspace(-0.5e-3, 0.75e-3, nx),
            np.linspace(-0.5e-3, 0.5e-3, ny),
            20e-6,
            0.01,
            0.02,
            modulus_pa,
            5e-4,
            20e-6,
            21250,
            12,
            inlet_fractional_film_content=supply,
        )
        result = solve_cavitating_elasticity(problem)
        np.savez_compressed(
            directory / f"{name}.npz",
            x_m=problem.x_m,
            y_m=problem.y_m,
            pressure_pa=result.pressure_pa,
            saturation=result.fractional_film_content,
            film_m=result.film_thickness_m,
            displacement_m=result.elastic_displacement_m,
        )
        audit = audit_cavitating_solution(problem, result)
        if (
            audit.mass_maximum > 1.001e-8
            or audit.maximum_complementarity > 1e-10
            or audit.minimum_pressure_pa < -1e-10 * problem.pressure_scale_pa
            or audit.minimum_saturation < -1e-10
            or audit.maximum_saturation > 1 + 1e-10
            or audit.pressure_boundary_max_pa != 0
            or audit.inlet_saturation_error > 1e-10
            or audit.film_coupling_max_error_m > 1e-17
            or audit.load_n <= 0
            or abs(audit.net_boundary_flux - audit.summed_volume_balance) > 1e-11
        ):
            raise RuntimeError(f"independent coupled state audit failed: {audit}")
        report["cases"].append(
            {
                "shape": [ny, nx],
                "inlet_fraction": supply,
                "load_n": result.load_n,
                "peak_pressure_pa": float(result.pressure_pa.max()),
                "audit": asdict(audit),
                "history": [asdict(row) for row in result.history],
            }
        )
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if plots:
            save_plots(directory, name, problem, result)
        print(
            f"{name}: force={result.load_n:.12g} N; min film={audit.minimum_film_m:.6g} m; "
            f"theta_min={audit.minimum_saturation:.6g}; "
            f"max mass residual={audit.mass_maximum:.3e}; "
            f"{len(result.history)} outer iterations",
            flush=True,
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/cavitating_elasticity")
    )
    parser.add_argument("--modulus-pa", type=float, default=1e7)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    main(args.output_directory, modulus_pa=args.modulus_pa, plots=args.plots)
