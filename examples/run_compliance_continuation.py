"""Fixed signed-load compliance stress probe; writes no simulation outputs."""

import time

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    MonolithicNewtonError,
    continue_monolithic_compliance,
    make_feasible_initial_state,
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
    for modulus in (1e8, 1e7, 1e6):
        problem = BorderedReynoldsElasticityProblem(
            x, y, 0.01, 0.02, modulus, 5e-4, film_scale, pressure_scale,
            12.0, target, compliance_factor=0.0, reynolds_residual_scale=100.,
            krylov_max_iterations=300,
        )
        initial = solve_monolithic_full_film_load(problem, pressure, film_scale)
        initial_state = np.concatenate(
            (initial.pressure_dimensionless[1:-1, 1:-1].ravel(),
             [initial.central_film_offset_m / film_scale])
        )
        initial_residual = problem.residual(initial_state)
        initial_reynolds_rms = (
            np.linalg.norm(initial_residual[:-1])
            / np.sqrt(problem.pressure_size)
        )
        started_at = time.perf_counter()
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
        elapsed_seconds = time.perf_counter() - started_at
        residual = reynolds_residual(
            solution.pressure_dimensionless,
            solution.film_thickness_m / film_scale, dx, dy, 12.0,
        )
        load = float(trapezoid(trapezoid(solution.pressure_pa, x, axis=1), y))
        final_state = np.concatenate(
            (solution.pressure_dimensionless[1:-1, 1:-1].ravel(),
             [solution.central_film_offset_m / film_scale])
        )
        final_residual = problem.residual(final_state)
        final_reynolds_rms = (
            np.linalg.norm(final_residual[:-1])
            / np.sqrt(problem.pressure_size)
        )
        print(
            f"E*={modulus:g} Pa: {status}; gamma={problem.compliance_factor:.9g}; "
            f"attempts={len(steps)}, rejected stages={sum(not s.accepted for s in steps)}, "
            f"elapsed={elapsed_seconds:.3f} s"
        )
        print(
            f"scaled nonlinear norm {np.hypot(initial_reynolds_rms, initial_residual[-1]):.6e}"
            f" -> {np.hypot(final_reynolds_rms, final_residual[-1]):.6e}; "
            f"independent Reynolds RMS={np.sqrt(np.mean(residual**2))/100:.6e}; "
            f"load error={load-target:.6e} N"
        )
        print(
            f"p range=({solution.pressure_pa.min():.6e}, "
            f"{solution.pressure_pa.max():.6e}) Pa; "
            f"minimum film={solution.film_thickness_m.min():.6e} m; "
            f"maximum |deformation|={np.max(np.abs(solution.elastic_displacement_m)):.6e} m"
        )
        maximum_true_residual = max(
            (step.maximum_true_linear_residual for step in steps), default=0.0
        )
        maximum_bordered_residual = max(
            (step.maximum_bordered_correction_residual for step in steps), default=0.0
        )
        print(
            f"continuation Newton iterations="
            f"{sum(step.newton_iterations for step in steps)}; "
            f"Krylov iterations={sum(step.krylov_iterations for step in steps)}; "
            f"line-search rejections="
            f"{sum(step.rejected_line_search_trials for step in steps)}; "
            f"max true Krylov residual={maximum_true_residual:.6e}; "
            f"max bordered correction residual={maximum_bordered_residual:.6e}"
        )
        for step in steps:
            print(
                f"  gamma={step.compliance_factor:.9g} "
                f"{'accepted' if step.accepted else 'rejected'}: "
                f"Newton={step.newton_iterations}, Krylov={step.krylov_iterations}, "
                f"line-search rejects={step.rejected_line_search_trials}; {step.reason}"
            )

    extreme_problem = BorderedReynoldsElasticityProblem(
        x, y, 0.01, 0.02, 1e5, 5e-4, film_scale, pressure_scale,
        12.0, target, compliance_factor=0.0, reynolds_residual_scale=100.0,
        krylov_max_iterations=300,
    )
    extreme_initial = solve_monolithic_full_film_load(
        extreme_problem, pressure, film_scale
    )
    extreme_started_at = time.perf_counter()
    try:
        extreme_result = continue_monolithic_compliance(
            extreme_problem,
            extreme_initial,
            target_compliance_factor=0.2,
            initial_step=0.1,
            minimum_step=0.025,
            maximum_step=0.1,
            maximum_retries_per_step=1,
            newton_options={"max_iterations": 20},
        )
        extreme_steps = extreme_result.steps
        extreme_status = "reached gamma=0.2"
    except MonolithicNewtonError as error:
        extreme_steps = error.diagnostics["compliance_steps"]
        extreme_status = (
            f"stopped at gamma={error.diagnostics['last_accepted_compliance_factor']:.9g}"
        )
    print(
        f"E*=100 kPa bounded continuation: {extreme_status}; "
        f"attempts={len(extreme_steps)}, "
        f"elapsed={time.perf_counter() - extreme_started_at:.3f} s"
    )
    for step in extreme_steps:
        print(
            f"  gamma={step.compliance_factor:.9g} "
            f"{'accepted' if step.accepted else 'rejected'}: "
            f"Newton={step.newton_iterations}, Krylov={step.krylov_iterations}; "
            f"{step.reason}"
        )
    extreme_problem.compliance_factor = 1.0
    extreme_feasible = make_feasible_initial_state(
        extreme_problem, pressure, film_scale
    )
    print(
        "E*=100 kPa full-compliance initialization-only h0 raise="
        f"{extreme_feasible.h0_adjustment_m:.6e} m; minimum film "
        f"{extreme_feasible.original_minimum_film_thickness_m:.6e} -> "
        f"{extreme_feasible.adjusted_minimum_film_thickness_m:.6e} m"
    )


if __name__ == "__main__":
    main()
