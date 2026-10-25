"""Reproduce prescribed-film JFO cavitation front and starvation checks."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.cavitation import solve_cavitating_reynolds


def _double_wedge(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(-0.5, 0.5, ny)
    xx, _ = np.meshgrid(x, y, indexing="xy")
    return x, y, 1.5 + 0.05 * np.cos(4.0 * np.pi * xx)


def _pressure_integral(
    pressure: np.ndarray, x: np.ndarray, y: np.ndarray
) -> float:
    return float(trapezoid(trapezoid(pressure, y, axis=0), x))


def main() -> None:
    print(
        "Prescribed-film stationary JFO verification only; "
        "dimensionless fields, no elasticity or load control."
    )
    print("Repeated-wedge rupture/reformation front refinement:")
    print(
        "nx x ny | front x coordinates | max full-cell pressure at front / "
        "centerline peak | local mass residual"
    )
    for nx, ny in ((41, 21), (61, 31), (101, 51)):
        x, y, film = _double_wedge(nx, ny)
        solution = solve_cavitating_reynolds(
            film, x[1] - x[0], y[1] - y[0], 2.0
        )
        center = ny // 2
        pressure = solution.pressure[center]
        saturation = solution.fractional_film_content[center]
        cavitated = saturation < 1.0 - 1e-8
        fronts = np.flatnonzero(cavitated[1:] != cavitated[:-1])
        front_pressure = np.array(
            [
                pressure[front + 1] if cavitated[front] else pressure[front]
                for front in fronts
            ]
        )
        print(
            f"{nx} x {ny} | "
            f"{', '.join(f'{x[front + 1]:.6f}' for front in fronts)} | "
            f"{np.max(front_pressure) / np.max(pressure):.6e} | "
            f"{solution.maximum_mass_balance_residual:.6e}"
        )

    x = np.linspace(0.0, 1.0, 81)
    y = np.linspace(-0.5, 0.5, 33)
    xx, _ = np.meshgrid(x, y, indexing="xy")
    film = 1.4 - 0.4 * xx + 0.8 * xx**2
    print("Inlet starvation sensitivity on the converging-diverging wedge:")
    print("inlet theta | pressure integral | peak pressure | theta range | net flux")
    for inlet_content in (1.0, 0.99, 0.9):
        solution = solve_cavitating_reynolds(
            film,
            x[1] - x[0],
            y[1] - y[0],
            2.0,
            inlet_fractional_film_content=inlet_content,
        )
        print(
            f"{inlet_content:.2f} | "
            f"{_pressure_integral(solution.pressure, x, y):.8e} | "
            f"{np.max(solution.pressure):.8e} | "
            f"[{np.min(solution.fractional_film_content):.6f}, "
            f"{np.max(solution.fractional_film_content):.6f}] | "
            f"{solution.net_boundary_flux:.6e}"
        )


if __name__ == "__main__":
    main()
