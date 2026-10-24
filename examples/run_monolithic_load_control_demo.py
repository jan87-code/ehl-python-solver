"""Compare signed full-film load-control solvers on the weak-coupling case."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.coupled import solve_coupled_reynolds_elasticity
from ehl_solver.load_control import solve_full_film_at_load
from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    MonolithicNewtonError,
    compare_bordered_preconditioners,
    continue_monolithic_load,
    make_feasible_initial_state,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import (
    dimensionless_reynolds_parameter,
    reynolds_residual,
    solve_reynolds_pressure,
)
from ehl_solver.visualization import save_monolithic_figures

VISCOSITY_PA_S = 0.04
SURFACE_SPEED_M_S = 0.85
LENGTH_SCALE_M = 5e-4
FILM_SCALE_M = 20e-6
PRESSURE_SCALE_PA = (
    VISCOSITY_PA_S
    * SURFACE_SPEED_M_S
    * 0.5
    * LENGTH_SCALE_M
    / FILM_SCALE_M**2
)
REYNOLDS_PARAMETER = dimensionless_reynolds_parameter(
    VISCOSITY_PA_S,
    SURFACE_SPEED_M_S,
    0.0,
    LENGTH_SCALE_M,
    FILM_SCALE_M,
    PRESSURE_SCALE_PA,
)
RADIUS_X_M = 0.01
RADIUS_Y_M = 0.02
E_STAR_PA = 1e8


def _integrated_load(pressure_pa: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> float:
    return float(trapezoid(trapezoid(pressure_pa, x_m, axis=1), y_m))


def _prescribed_pressure(
    x_m: np.ndarray, y_m: np.ndarray, h0_m: float
) -> np.ndarray:
    xx, yy = np.meshgrid(x_m, y_m, indexing="xy")
    film = (
        h0_m + xx**2 / (2.0 * RADIUS_X_M) + yy**2 / (2.0 * RADIUS_Y_M)
    ) / FILM_SCALE_M
    return solve_reynolds_pressure(
        film,
        (x_m[1] - x_m[0]) / LENGTH_SCALE_M,
        (y_m[1] - y_m[0]) / LENGTH_SCALE_M,
        REYNOLDS_PARAMETER,
        rtol=1e-12,
        atol=1e-14,
    ).pressure


def _elapsed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def main() -> None:
    x_m = np.linspace(-0.5e-3, 0.75e-3, 17)
    y_m = np.linspace(-0.5e-3, 0.5e-3, 13)
    coupling_options = {
        "pressure_relaxation": 0.6,
        "film_relaxation": 0.6,
        "pressure_tolerance": 1e-6,
        "film_tolerance": 1e-6,
        "reynolds_tolerance": 1e-8,
        "inner_rtol": 1e-10,
        "inner_atol": 1e-12,
        "log_every": 1000,
    }

    reference, reference_seconds = _elapsed(
        lambda: solve_coupled_reynolds_elasticity(
            x_m,
            y_m,
            FILM_SCALE_M,
            RADIUS_X_M,
            RADIUS_Y_M,
            E_STAR_PA,
            LENGTH_SCALE_M,
            FILM_SCALE_M,
            PRESSURE_SCALE_PA,
            REYNOLDS_PARAMETER,
            **coupling_options,
        )
    )
    target_load_n = _integrated_load(reference.pressure_pa, x_m, y_m)

    initial_pressure = _prescribed_pressure(x_m, y_m, 18e-6)
    problem = BorderedReynoldsElasticityProblem(
        x_m,
        y_m,
        RADIUS_X_M,
        RADIUS_Y_M,
        E_STAR_PA,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        target_load_n,
        reynolds_residual_scale=100.0,
        krylov_rtol=1e-8,
        krylov_max_iterations=300,
    )
    starting_state = np.concatenate(
        (
            initial_pressure[1:-1, 1:-1].ravel(),
            [18e-6 / FILM_SCALE_M],
        )
    )
    preconditioner_results = compare_bordered_preconditioners(
        problem, starting_state
    )
    monolithic, monolithic_seconds = _elapsed(
        lambda: solve_monolithic_full_film_load(
            problem,
            initial_pressure,
            18e-6,
            reynolds_tolerance=1e-7,
            load_tolerance=1e-7,
        )
    )
    moderate_problem = BorderedReynoldsElasticityProblem(
        x_m,
        y_m,
        RADIUS_X_M,
        RADIUS_Y_M,
        1e7,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        target_load_n,
        reynolds_residual_scale=100.0,
        krylov_rtol=1e-8,
        krylov_max_iterations=300,
    )
    moderate, moderate_seconds = _elapsed(
        lambda: solve_monolithic_full_film_load(
            moderate_problem,
            initial_pressure,
            18e-6,
            reynolds_tolerance=1e-7,
            load_tolerance=1e-7,
        )
    )
    strong_problem = BorderedReynoldsElasticityProblem(
        x_m,
        y_m,
        RADIUS_X_M,
        RADIUS_Y_M,
        1e6,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        target_load_n,
        reynolds_residual_scale=100.0,
        krylov_rtol=1e-8,
        krylov_max_iterations=300,
    )
    strong_solution = None
    try:
        strong_solution = solve_monolithic_full_film_load(
            strong_problem,
            initial_pressure,
            18e-6,
            reynolds_tolerance=1e-7,
            load_tolerance=1e-7,
            max_iterations=40,
        )
    except MonolithicNewtonError as error:
        diagnostics = error.diagnostics
        failed_krylov = error.diagnostics.get("krylov_diagnostics", ())
        final_krylov = failed_krylov[-1] if failed_krylov else None
        print(
            "Exploratory E*=1 MPa failure retained: "
            f"{error}; iteration={diagnostics.get('iteration')}; "
            f"scaled Reynolds residual="
            f"{diagnostics.get('reynolds_residual_scaled')}; "
            f"load residual [N]={diagnostics.get('load_residual_n')}; "
            f"p_range [Pa]=({diagnostics.get('minimum_pressure_pa')}, "
            f"{diagnostics.get('maximum_pressure_pa')}); "
            f"h_min [m]={diagnostics.get('minimum_film_thickness_m')}; "
            f"deformation/rigid-gap="
            f"{diagnostics.get('elastic_deformation_to_rigid_gap_ratio')}; "
            f"last GMRES true residual="
            f"{getattr(final_krylov, 'true_relative_residual', None)}; "
            f"GMRES termination="
            f"{getattr(final_krylov, 'termination_reason', None)}; "
            f"line-search slope="
            f"{diagnostics.get('merit_directional_derivative')}; "
            f"merit={diagnostics.get('current_merit')} -> "
            f"{diagnostics.get('last_trial_merit')} "
            f"(limit {diagnostics.get('armijo_merit_limit')})"
        )
        strong_failure_diagnostics = diagnostics
    else:
        strong_failure_diagnostics = {}

    extreme_problem = BorderedReynoldsElasticityProblem(
        x_m,
        y_m,
        RADIUS_X_M,
        RADIUS_Y_M,
        1e5,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        target_load_n,
        reynolds_residual_scale=100.0,
        krylov_rtol=1e-8,
        krylov_max_iterations=300,
    )
    feasible_initial = make_feasible_initial_state(
        extreme_problem, initial_pressure, 18e-6
    )
    print(
        "E*=100 kPa feasible-start adjustment (initialization only): "
        f"h0 increase={feasible_initial.h0_adjustment_m:.6e} m; "
        f"minimum film {feasible_initial.original_minimum_film_thickness_m:.6e} -> "
        f"{feasible_initial.adjusted_minimum_film_thickness_m:.6e} m"
    )

    nested, nested_seconds = _elapsed(
        lambda: solve_full_film_at_load(
            x_m,
            y_m,
            target_load_n,
            RADIUS_X_M,
            RADIUS_Y_M,
            E_STAR_PA,
            LENGTH_SCALE_M,
            FILM_SCALE_M,
            PRESSURE_SCALE_PA,
            REYNOLDS_PARAMETER,
            h0_bracket_m=(15e-6, 25e-6),
            load_tolerance=1e-6,
            h0_tolerance_m=1e-12,
            coupling_options=coupling_options,
        )
    )

    continuation, continuation_seconds = _elapsed(
        lambda: continue_monolithic_load(
            problem,
            monolithic,
            target_load_n * 0.8,
            initial_step_fraction=0.5,
            minimum_step_fraction=1e-3,
            maximum_step_fraction=0.5,
            newton_options={
                "reynolds_tolerance": 1e-7,
                "load_tolerance": 1e-7,
            },
        )
    )

    comparison = (
        (
            "Picard at prescribed h0",
            reference.pressure_pa,
            reference.film_thickness_m,
            reference.elastic_displacement_m,
            target_load_n,
            target_load_n,
            reference_seconds,
            len(reference.history),
            sum(entry.linear_iterations for entry in reference.history),
        ),
        (
            "Nested h0 load controller",
            nested.coupled_solution.pressure_pa,
            nested.coupled_solution.film_thickness_m,
            nested.coupled_solution.elastic_displacement_m,
            nested.calculated_signed_load_n,
            target_load_n,
            nested_seconds,
            sum(item.nonlinear_iterations for item in nested.evaluations),
            len(nested.evaluations),
        ),
        (
            "Bordered Newton",
            monolithic.pressure_pa,
            monolithic.film_thickness_m,
            monolithic.elastic_displacement_m,
            monolithic.calculated_load_n,
            target_load_n,
            monolithic_seconds,
            len(monolithic.history),
            monolithic.operator_evaluations,
        ),
        (
            "Bordered Newton (E*=10 MPa)",
            moderate.pressure_pa,
            moderate.film_thickness_m,
            moderate.elastic_displacement_m,
            moderate.calculated_load_n,
            target_load_n,
            moderate_seconds,
            len(moderate.history),
            moderate.operator_evaluations,
        ),
    )
    print("Signed full-film mathematical verification only; negative pressure retained.")
    print(f"Reference target signed load: {target_load_n:.9e} N")
    print(
        "Frozen-film preconditioner comparison (includes sparse setup/factorization):"
    )
    for result in preconditioner_results:
        print(
            f"  {result.preconditioner}: used={result.preconditioner_used}; "
            f"converged={result.converged}; Krylov={result.krylov_iterations}; "
            f"true residual={result.maximum_true_relative_residual:.3e}; "
            f"bordered residual={result.bordered_residual_relative:.3e}; "
            f"time={result.elapsed_seconds:.6f} s; {result.termination_reason}"
        )
    print(
        "method | relative pressure L2 | film RMS difference [m] | "
        "peak |w| [m] | |w|max/min rigid gap | scaled Reynolds RMS | "
        "load residual [N] | "
        "nonlinear work | reported solver work | wall time [s]"
    )
    for (
        name,
        pressure_pa,
        film_m,
        displacement_m,
        calculated_load_n,
        expected_load_n,
        elapsed,
        iterations,
        work_count,
    ) in comparison:
        pressure_difference = pressure_pa - reference.pressure_pa
        relative_pressure = float(
            np.linalg.norm(pressure_difference)
            / max(np.linalg.norm(reference.pressure_pa), np.finfo(float).tiny)
        )
        film_rms = float(np.sqrt(np.mean((film_m - reference.film_thickness_m) ** 2)))
        dimensionless_residual = reynolds_residual(
            pressure_pa / PRESSURE_SCALE_PA,
            film_m / FILM_SCALE_M,
            (x_m[1] - x_m[0]) / LENGTH_SCALE_M,
            (y_m[1] - y_m[0]) / LENGTH_SCALE_M,
            REYNOLDS_PARAMETER,
        )
        scaled_reynolds_rms = (
            np.linalg.norm(dimensionless_residual)
            / np.sqrt(problem.pressure_size)
            / problem.reynolds_residual_scale
        )
        rigid_gap_m = film_m - displacement_m
        deformation_ratio = float(
            np.max(np.abs(displacement_m))
            / max(np.min(rigid_gap_m), np.finfo(float).tiny)
        )
        print(
            f"{name} | {relative_pressure:.3e} | {film_rms:.3e} | "
            f"{np.max(np.abs(displacement_m)):.3e} | "
            f"{deformation_ratio:.3e} | {scaled_reynolds_rms:.3e} | "
            f"{calculated_load_n - expected_load_n:.3e} | {iterations} | "
            f"{work_count} | {elapsed:.3f}"
        )

    final_reynolds = reynolds_residual(
        monolithic.pressure_dimensionless,
        monolithic.film_thickness_m / FILM_SCALE_M,
        (x_m[1] - x_m[0]) / LENGTH_SCALE_M,
        (y_m[1] - y_m[0]) / LENGTH_SCALE_M,
        REYNOLDS_PARAMETER,
    )
    print(
        f"Bordered Newton final: h0={monolithic.central_film_offset_m * 1e6:.9f} µm; "
        f"p=[{monolithic.pressure_pa.min():.6e}, "
        f"{monolithic.pressure_pa.max():.6e}] Pa; "
        f"h_min={monolithic.film_thickness_m.min() * 1e6:.9f} µm; "
        f"independent Reynolds RMS="
        f"{np.linalg.norm(final_reynolds) / np.sqrt(problem.pressure_size):.6e}; "
        f"load residual={monolithic.load_residual_n:.6e} N; "
        f"residual evaluations={monolithic.operator_evaluations}"
    )
    print("Newton iteration history (residual at state before accepted correction):")
    for iteration in monolithic.history:
        print(
            f"  {iteration.iteration}: R_P={iteration.reynolds_residual_scaled:.3e}; "
            f"R_Q={iteration.load_residual_n:.3e} N; "
            f"alpha_feasible={iteration.feasible_step_limit:.3e}; "
            f"alpha={iteration.accepted_step_length:.3e}; "
            f"rejected={iteration.rejected_line_search_trials}; "
            f"GMRES={iteration.krylov_iterations}; "
            f"GMRES-relres={iteration.krylov_residual_relative:.3e}; "
            f"Schur-scale={iteration.schur_condition_indicator:.3e}"
        )

    output = Path("results/monolithic_signed_load_demo")
    coupling_ratios = [
        np.max(np.abs(reference.elastic_displacement_m))
        / np.min(reference.film_thickness_m - reference.elastic_displacement_m),
        np.max(np.abs(moderate.elastic_displacement_m))
        / np.min(moderate.film_thickness_m - moderate.elastic_displacement_m),
    ]
    coupling_moduli = [E_STAR_PA, 1e7]
    coupling_converged = [True, True]
    if strong_solution is not None:
        coupling_moduli.append(1e6)
        coupling_ratios.append(
            float(
                np.max(np.abs(strong_solution.elastic_displacement_m))
                / np.min(
                    strong_solution.film_thickness_m
                    - strong_solution.elastic_displacement_m
                )
            )
        )
        coupling_converged.append(True)
    elif "elastic_deformation_to_rigid_gap_ratio" in strong_failure_diagnostics:
        coupling_moduli.append(1e6)
        coupling_ratios.append(
            float(
                strong_failure_diagnostics[
                    "elastic_deformation_to_rigid_gap_ratio"
                ]
            )
        )
        coupling_converged.append(False)
    paths = save_monolithic_figures(
        x_m,
        y_m,
        continuation.solution,
        case_name=output.name,
        results_directory=output.parent,
        continuation=continuation,
        preconditioner_comparison=preconditioner_results,
        effective_modulus_pa=coupling_moduli,
        deformation_to_rigid_gap_ratio=coupling_ratios,
        coupling_converged=coupling_converged,
    )
    print(f"Load-continuation wall time: {continuation_seconds:.3f} s")
    print("Continuation attempts:")
    for step in continuation.steps:
        print(
            f"  {step.attempt}: {'accepted' if step.accepted else 'rejected'} "
            f"load={step.target_load_n:.9e} N; "
            f"{step.newton_iterations} Newton iterations; {step.reason}"
        )
    print("Generated figures:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
