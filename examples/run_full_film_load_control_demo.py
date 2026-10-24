"""Run a signed-full-film load-control verification case (not physical EHL)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.coupled import solve_coupled_reynolds_elasticity
from ehl_solver.load_control import solve_full_film_at_load
from ehl_solver.reynolds import dimensionless_reynolds_parameter
from ehl_solver.visualization import plot_load_control_history

VISCOSITY_PA_S = 0.04
SURFACE_SPEED_1_M_S = 0.85
LENGTH_SCALE_M = 5e-4
FILM_SCALE_M = 20e-6
PRESSURE_SCALE_PA = (
    VISCOSITY_PA_S
    * SURFACE_SPEED_1_M_S
    * 0.5
    * LENGTH_SCALE_M
    / FILM_SCALE_M**2
)
REYNOLDS_PARAMETER = dimensionless_reynolds_parameter(
    VISCOSITY_PA_S,
    SURFACE_SPEED_1_M_S,
    0.0,
    LENGTH_SCALE_M,
    FILM_SCALE_M,
    PRESSURE_SCALE_PA,
)


def main() -> None:
    x_m = np.linspace(-0.5e-3, 0.75e-3, 17)
    y_m = np.linspace(-0.5e-3, 0.5e-3, 13)
    known_offset_m = 20e-6
    known_state = solve_coupled_reynolds_elasticity(
        x_m,
        y_m,
        known_offset_m,
        0.01,
        0.02,
        1e8,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        continuation_factors=(0.0,),
        pressure_relaxation=1.0,
        film_relaxation=1.0,
        pressure_tolerance=1e-8,
        film_tolerance=1e-8,
        reynolds_tolerance=1e-10,
        inner_rtol=1e-12,
        inner_atol=1e-14,
    )
    target_signed_load_n = float(
        trapezoid(trapezoid(known_state.pressure_pa, x_m, axis=1), y_m)
    )
    result = solve_full_film_at_load(
        x_m,
        y_m,
        target_signed_load_n,
        0.01,
        0.02,
        1e8,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        h0_bracket_m=(15e-6, 25e-6),
        load_tolerance=1e-6,
        h0_tolerance_m=1e-12,
        coupling_options={
            "continuation_factors": (0.0,),
            "pressure_relaxation": 1.0,
            "film_relaxation": 1.0,
            "pressure_tolerance": 1e-8,
            "film_tolerance": 1e-8,
            "reynolds_tolerance": 1e-10,
            "inner_rtol": 1e-12,
            "inner_atol": 1e-14,
        },
    )

    output = Path("results/full_film_load_verification")
    output.mkdir(parents=True, exist_ok=True)
    figure, _ = plot_load_control_history(result)
    path = output / "load_vs_h0_and_relative_error.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print("Mathematical signed-full-film verification only; negative pressure is retained.")
    print(f"Target signed load: {result.target_load_n:.9e} N")
    print(f"Calculated signed load: {result.calculated_signed_load_n:.9e} N")
    print(f"Relative load error: {result.relative_load_error:.3e}")
    print(f"Recovered h0: {result.central_film_offset_m * 1e6:.6f} µm")
    print(f"Converged h0 evaluations: {len(result.evaluations)}")
    print(f"Saved load history: {path}")


if __name__ == "__main__":
    main()
