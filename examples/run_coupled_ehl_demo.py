"""Run the existing weakly coupled benchmark and write visualization PNGs."""

import logging
from pathlib import Path

import numpy as np

from ehl_solver.coupled import solve_coupled_reynolds_elasticity
from ehl_solver.reynolds import dimensionless_reynolds_parameter
from ehl_solver.visualization import save_coupled_figures


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    viscosity_pa_s = 0.04
    surface_speed_1_m_s = 0.85
    surface_speed_2_m_s = 0.0
    length_scale_m = 5e-4
    film_scale_m = 20e-6
    pressure_scale_pa = (
        viscosity_pa_s
        * (surface_speed_1_m_s + surface_speed_2_m_s)
        / 2.0
        * length_scale_m
        / film_scale_m**2
    )
    reynolds_parameter = dimensionless_reynolds_parameter(
        viscosity_pa_s,
        surface_speed_1_m_s,
        surface_speed_2_m_s,
        length_scale_m,
        film_scale_m,
        pressure_scale_pa,
    )
    x_m = np.linspace(-0.5e-3, 0.75e-3, 17)
    y_m = np.linspace(-0.5e-3, 0.5e-3, 13)
    solution = solve_coupled_reynolds_elasticity(
        x_m,
        y_m,
        central_film_offset_m=film_scale_m,
        radius_x_m=0.01,
        radius_y_m=0.02,
        e_star_pa=1e8,
        length_scale_m=length_scale_m,
        film_scale_m=film_scale_m,
        pressure_scale_pa=pressure_scale_pa,
        reynolds_parameter=reynolds_parameter,
        pressure_relaxation=0.6,
        film_relaxation=0.6,
        pressure_tolerance=1e-6,
        film_tolerance=1e-6,
        reynolds_tolerance=1e-8,
        max_iterations_per_stage=100,
        plot_every=20,
        log_every=10,
    )
    output = Path("results")
    paths = save_coupled_figures(
        x_m,
        y_m,
        solution,
        case_name="weak_coupling_demo",
        results_directory=output,
    )
    print(f"Compliance stages: {len(set(row.compliance_factor for row in solution.history))}")
    print(f"Accepted outer iterations: {len(solution.history)}")
    print(f"Saved figures to {output / 'weak_coupling_demo'}:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
