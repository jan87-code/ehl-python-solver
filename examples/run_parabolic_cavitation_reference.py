"""Independent finite-inlet parabolic reference vs wide 2-D rigid JFO grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.parabolic_reference import parabolic_jfo_reference


def compare_grid(nx, directory=None, *, plots=False):
    h0, radius, viscosity, speed = 20e-6, 0.01, 0.04, 0.425
    length = np.sqrt(2 * radius * h0)
    x, y = np.linspace(-3 * length, 3 * length, nx), np.linspace(-8 * length, 8 * length, 17)
    reference = parabolic_jfo_reference(x, h0, radius, viscosity, speed)
    film = np.broadcast_to(reference.film_thickness_m / h0, (y.size, nx)).copy()
    solution = solve_cavitating_reynolds(film, (x[1] - x[0]) / length, (y[1] - y[0]) / length, 1.0)
    pressure = solution.pressure[y.size // 2] * reference.pressure_scale_pa
    theta = solution.fractional_film_content[y.size // 2]
    force = float(trapezoid(pressure, x))
    cavitated = np.flatnonzero((x > 0) & (theta < 1 - 1e-8))
    rupture = float(x[cavitated[0]]) if cavitated.size else None
    entry = {
        "shape": film.shape,
        "x_limits_m": [float(x[0]), float(x[-1])],
        "y_limits_m": [float(y[0]), float(y[-1])],
        "Lambda": 1.0,
        "length_scale_m": float(length),
        "pressure_scale_pa": reference.pressure_scale_pa,
        "pressure_relative_l2": float(
            np.linalg.norm(pressure - reference.pressure_pa) / np.linalg.norm(reference.pressure_pa)
        ),
        "theta_maximum_absolute_error": float(
            np.max(np.abs(theta - reference.fractional_film_content))
        ),
        "force_per_width_n_m": force,
        "reference_force_per_width_n_m": reference.force_per_width_n_m,
        "relative_force_error": abs(force - reference.force_per_width_n_m)
        / reference.force_per_width_n_m,
        "first_cavitated_node_m": rupture,
        "analytical_rupture_m": reference.rupture_location_m,
        "mass_maximum": solution.maximum_mass_balance_residual,
        "complementarity_maximum": solution.maximum_complementarity_violation,
        "net_boundary_flux": solution.net_boundary_flux,
    }
    if directory is not None:
        np.savez_compressed(
            directory / f"grid_17_{nx}.npz",
            x_m=x,
            y_m=y,
            pressure_pa=solution.pressure * reference.pressure_scale_pa,
            theta=solution.fractional_film_content,
            reference_pressure_pa=reference.pressure_pa,
            reference_theta=reference.fractional_film_content,
        )
        if plots:
            save_plot(directory, nx, x, pressure, theta, reference)
    return entry


def save_plot(directory, nx, x, pressure, theta, reference):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(8, 7))
    axes[0].plot(x * 1e3, reference.pressure_pa / 1e3, label="Analytical infinite-width reference")
    axes[0].plot(x * 1e3, pressure / 1e3, ".--", label="Wide 2-D JFO centerline")
    axes[0].set_ylabel("Pressure [kPa]")
    axes[0].legend()
    axes[1].plot(x * 1e3, reference.fractional_film_content, label="Analytical content")
    axes[1].plot(x * 1e3, theta, ".--", label="Wide 2-D centerline")
    axes[1].axvline(reference.rupture_location_m * 1e3, color="grey", linestyle=":")
    axes[1].set(xlabel="x [mm] (entrainment)", ylabel="Fractional film content")
    axes[1].legend()
    figure.savefig(directory / f"reference_17_{nx}.png", dpi=120)
    plt.close(figure)


def main(directory, *, plots=False):
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "rigid parabolic infinite-width limit, no elastic or point-contact validation",
        "SI_inputs": {"h0_m": 20e-6, "radius_m": 0.01, "eta_pa_s": 0.04, "Ue_m_s": 0.425},
        "cases": [],
        "inlet_study": [],
    }
    for nx in (33, 65, 129):
        entry = compare_grid(nx, directory, plots=plots)
        report["cases"].append(entry)
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"17x{nx}: pressure relative error={entry['pressure_relative_l2']:.6g}, "
            f"force relative error={entry['relative_force_error']:.6g}, "
            f"max mass={entry['mass_maximum']:.3e}",
            flush=True,
        )
    pressure_errors = [row["pressure_relative_l2"] for row in report["cases"]]
    force_errors = [row["relative_force_error"] for row in report["cases"]]
    if (
        not all(b < a for a, b in zip(pressure_errors, pressure_errors[1:], strict=False))
        or not all(b < a for a, b in zip(force_errors, force_errors[1:], strict=False))
        or pressure_errors[-1] > 0.02
        or force_errors[-1] > 0.02
        or max(row["mass_maximum"] for row in report["cases"]) > 1e-8
    ):
        raise RuntimeError("wide rigid JFO comparison failed convergence/accuracy criteria")
    length = np.sqrt(2 * 0.01 * 20e-6)
    for inlet in (-0.5, -1, -2, -4, -8, -16):
        reference = parabolic_jfo_reference(
            np.linspace(inlet * length, 3 * length, 129), 20e-6, 0.01, 0.04, 0.425
        )
        report["inlet_study"].append(
            {
                "inlet_over_length": inlet,
                "rupture_over_length": reference.rupture_location_m / length,
                "force_per_width_n_m": reference.force_per_width_n_m,
                "flux_per_width_m2_s": reference.volume_flux_per_width_m2_s,
            }
        )
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/parabolic_cavitation_reference")
    )
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    main(args.output_directory, plots=args.plots)
