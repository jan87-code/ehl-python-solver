"""Fixed signed-load compliance stress probe; writes no simulation outputs."""

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    MonolithicNewtonError,
    continue_monolithic_compliance,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import reynolds_residual, solve_reynolds_pressure


def main():
    x = np.linspace(-0.5e-3, 0.75e-3, 17)
    y = np.linspace(-0.5e-3, 0.5e-3, 13)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    film_scale = 20e-6
    pressure_scale = 21250.0
    dx, dy = (x[1] - x[0]) / 5e-4, (y[1] - y[0]) / 5e-4
    rigid = film_scale + xx**2 / 0.02 + yy**2 / 0.04
    pressure = solve_reynolds_pressure(
        rigid / film_scale, dx, dy, 12.0, rtol=1e-12, atol=1e-14
    ).pressure
    target = float(trapezoid(trapezoid(pressure * pressure_scale, x, axis=1), y))
    for modulus in (1e7, 1e6):
        problem = BorderedReynoldsElasticityProblem(
            x, y, 0.01, 0.02, modulus, 5e-4, film_scale, pressure_scale,
            12.0, target, compliance_factor=0.0, reynolds_residual_scale=100.,
            krylov_max_iterations=300,
        )
        initial = solve_monolithic_full_film_load(problem, pressure, film_scale)
        try:
            result = continue_monolithic_compliance(
                problem, initial, newton_options={"max_iterations": 40}
            )
            solution, steps = result.solution, result.steps
            status = "reached target"
        except MonolithicNewtonError as error:
            solution = error.diagnostics["last_accepted_solution"]
            steps = error.diagnostics["compliance_steps"]
            status = "FAILED to reach target"
        residual = reynolds_residual(
            solution.pressure_dimensionless,
            solution.film_thickness_m / film_scale, dx, dy, 12.0,
        )
        load = float(trapezoid(trapezoid(solution.pressure_pa, x, axis=1), y))
        print(f"E*={modulus:g} Pa: {status}; gamma={problem.compliance_factor:.9g}; "
              f"attempts={len(steps)}, rejected={sum(not s.accepted for s in steps)}")
        print(f"independent scaled Reynolds RMS={np.sqrt(np.mean(residual**2))/100:.6e}; "
              f"load error={load-target:.6e} N; "
              f"minimum film={solution.film_thickness_m.min():.6e} m")


if __name__ == "__main__":
    main()
