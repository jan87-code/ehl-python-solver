"""Compare fixed-gamma and pseudo-arclength continuation on the 1 MPa case."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    MonolithicNewtonError,
    compare_bordered_preconditioners,
    continue_monolithic_compliance,
    continue_monolithic_compliance_pseudo_arclength,
    solve_bordered_correction,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import reynolds_residual, solve_reynolds_pressure
from ehl_solver.visualization import (
    plot_film_thickness_distribution,
    plot_monolithic_compliance_continuation,
    plot_monolithic_conditioning,
    plot_pressure_distribution,
    plot_pseudo_arclength_branch,
    plot_pseudo_arclength_diagnostics,
)

OUTPUT_DIRECTORY = Path("results/monolithic_pseudo_arclength_audit")


def _case(compliance_factor: float = 0.0):
    x = np.linspace(-0.5e-3, 0.75e-3, 17)
    y = np.linspace(-0.5e-3, 0.5e-3, 13)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    film_scale = 20e-6
    pressure_scale = 21250.0
    length_scale = 5e-4
    reynolds_parameter = 12.0
    dx = (x[1] - x[0]) / length_scale
    dy = (y[1] - y[0]) / length_scale
    rigid_film = (
        film_scale + xx**2 / 0.02 + yy**2 / 0.04
    ) / film_scale
    pressure = solve_reynolds_pressure(
        rigid_film,
        dx,
        dy,
        reynolds_parameter,
        rtol=1e-12,
        atol=1e-14,
    ).pressure
    target_load = float(
        trapezoid(trapezoid(pressure * pressure_scale, x, axis=1), y)
    )
    problem = BorderedReynoldsElasticityProblem(
        x,
        y,
        0.01,
        0.02,
        1e6,
        length_scale,
        film_scale,
        pressure_scale,
        reynolds_parameter,
        target_load,
        compliance_factor=compliance_factor,
        reynolds_residual_scale=100.0,
        krylov_max_iterations=300,
    )
    initial = solve_monolithic_full_film_load(problem, pressure, film_scale)
    return problem, initial, x, y, dx, dy


def _state(solution, film_scale):
    return np.concatenate(
        (
            solution.pressure_dimensionless[1:-1, 1:-1].ravel(),
            [solution.central_film_offset_m / film_scale],
        )
    )


def _conditioning_at(problem, z, gamma):
    problem.compliance_factor = gamma
    identity = np.eye(z.size, dtype=np.float64)
    matrix = np.column_stack(
        [
            problem.jacobian_vector(z, identity[:, index])
            for index in range(z.size)
        ]
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    smallest = float(singular_values[-1])
    spectrum = (
        float(singular_values[0]),
        smallest,
        float(singular_values[0] / smallest) if smallest else float("inf"),
    )
    pressure_block = matrix[:-1, :-1]
    h0_column = matrix[:-1, -1]
    load_row = matrix[-1, :-1]
    h0_response = np.linalg.solve(pressure_block, h0_column)
    schur = -float(np.dot(load_row, h0_response))
    schur_angle = abs(schur) / max(
        np.linalg.norm(load_row) * np.linalg.norm(h0_response),
        np.finfo(np.float64).tiny,
    )
    residual = problem.residual(z)
    pressure_step = np.linalg.solve(pressure_block, -residual[:-1])
    h0_step = (-residual[-1] - np.dot(load_row, pressure_step)) / schur
    schur_correction = np.concatenate(
        (pressure_step - h0_response * h0_step, [h0_step])
    )
    direct_correction = np.linalg.solve(matrix, -residual)
    denominator = max(np.linalg.norm(residual), np.finfo(np.float64).tiny)
    schur_residual = np.linalg.norm(matrix @ schur_correction + residual) / denominator
    direct_residual = np.linalg.norm(matrix @ direct_correction + residual) / denominator
    correction_difference = np.linalg.norm(
        schur_correction - direct_correction
    ) / max(np.linalg.norm(direct_correction), np.finfo(np.float64).tiny)
    return (
        spectrum,
        schur,
        schur_angle,
        schur_residual,
        direct_residual,
        correction_difference,
    )


def main(output_directory: Path = OUTPUT_DIRECTORY) -> None:
    if output_directory.exists():
        raise FileExistsError(
            f"refusing to overwrite existing diagnostic output: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    problem, _, x, y, dx, dy = _case()
    film_scale = problem.film_scale_m

    conventional_problem, conventional_initial, *_ = _case()
    started = time.perf_counter()
    try:
        conventional = continue_monolithic_compliance(
            conventional_problem,
            conventional_initial,
            newton_options={"max_iterations": 40},
        )
        conventional_solution = conventional.solution
        conventional_steps = conventional.steps
        conventional_reason = "target_reached"
    except MonolithicNewtonError as error:
        conventional_solution = error.diagnostics["last_accepted_solution"]
        conventional_steps = error.diagnostics["compliance_steps"]
        conventional_reason = str(error)
    conventional_seconds = time.perf_counter() - started

    conventional_spectra = []
    conventional_schur = []
    for step in conventional_steps:
        if not step.accepted or step.solution is None:
            continue
        z = _state(step.solution, film_scale)
        (
            spectrum,
            schur,
            schur_angle,
            schur_residual,
            direct_residual,
            correction_difference,
        ) = _conditioning_at(
            conventional_problem, z, step.compliance_factor
        )
        conventional_spectra.append((step.compliance_factor, spectrum))
        conventional_schur.append(schur)
        print(
            f"fixed-gamma γ={step.compliance_factor:.9f}: "
            f"condition={spectrum[2]:.6e}, sigma_min={spectrum[1]:.6e}, "
            f"Schur={schur:.6e}, Schur angle={schur_angle:.6e}, "
            f"direct residual={direct_residual:.3e}, "
            f"Schur residual={schur_residual:.3e}, "
            f"Schur-vs-direct correction={correction_difference:.3e}"
        )
    accepted_conventional = [
        step for step in conventional_steps if step.accepted and step.solution is not None
    ]
    if not accepted_conventional:
        raise RuntimeError("fixed-gamma continuation produced no accepted state")
    last_conventional_step = accepted_conventional[-1]
    conventional_problem.compliance_factor = last_conventional_step.compliance_factor
    conventional_state = _state(conventional_solution, film_scale)
    conventional_pressure = conventional_solution.pressure_dimensionless
    conventional_film = conventional_solution.film_thickness_m / film_scale
    conventional_reynolds = reynolds_residual(
        conventional_pressure,
        conventional_film,
        dx,
        dy,
        conventional_problem.reynolds_parameter,
    )
    conventional_reynolds_rms = float(
        np.sqrt(np.mean(conventional_reynolds**2))
        / conventional_problem.reynolds_residual_scale
    )
    conventional_load = float(
        trapezoid(
            trapezoid(conventional_solution.pressure_pa, x, axis=1), y
        )
    )
    conventional_residual_rms = float(
        np.linalg.norm(conventional_problem.residual(conventional_state))
        / np.sqrt(conventional_state.size)
    )
    print(
        f"fixed-gamma final accepted state: gamma="
        f"{last_conventional_step.compliance_factor:.9f}; "
        f"independent scaled Reynolds RMS={conventional_reynolds_rms:.6e}; "
        f"independent load error="
        f"{conventional_load - conventional_problem.target_load_n:.6e} N; "
        f"minimum film={conventional_solution.film_thickness_m.min():.6e} m; "
        f"state residual RMS={conventional_residual_rms:.6e}"
    )
    print(
        f"fixed-gamma continuation: {conventional_reason}; "
        f"furthest gamma={conventional_problem.compliance_factor:.9f}; "
        f"accepted={sum(step.accepted for step in conventional_steps)}, "
        f"rejected={sum(not step.accepted for step in conventional_steps)}, "
        f"Newton={sum(step.newton_iterations for step in conventional_steps)}, "
        f"Krylov={sum(step.krylov_iterations for step in conventional_steps)}, "
        f"line-search rejects="
        f"{sum(step.rejected_line_search_trials for step in conventional_steps)}, "
        f"elapsed={conventional_seconds:.3f} s"
    )

    if accepted_conventional:
        preconditioner_step = min(
            accepted_conventional,
            key=lambda item: abs(item.compliance_factor - 0.175),
        )
        preconditioner_problem, _, *_ = _case()
        preconditioner_problem.compliance_factor = preconditioner_step.compliance_factor
        preconditioner_state = _state(preconditioner_step.solution, film_scale)
        comparisons = compare_bordered_preconditioners(
            preconditioner_problem, preconditioner_state
        )
        exact_schur = _conditioning_at(
            preconditioner_problem,
            preconditioner_state,
            preconditioner_step.compliance_factor,
        )[1]
        for result in comparisons:
            print(
                f"preconditioner γ={preconditioner_step.compliance_factor:.9f} "
                f"{result.preconditioner}: converged={result.converged}, "
                f"Krylov={result.krylov_iterations}, "
                f"true residual={result.maximum_true_relative_residual:.6e}, "
                f"bordered residual={result.bordered_residual_relative:.6e}, "
                f"time={result.elapsed_seconds:.3f} s"
            )
        iterative_schur = solve_bordered_correction(
            preconditioner_problem,
            preconditioner_state,
            preconditioner_strategy="ilu",
            krylov_rtol=1e-8,
        )
        schur_relative_error = (
            abs(iterative_schur.schur_denominator - exact_schur)
            / abs(exact_schur)
        )
        print(
            f"ILU Schur sensitivity at γ={preconditioner_step.compliance_factor:.9f}: "
            f"iterative denominator={iterative_schur.schur_denominator:.6e}, "
            f"exact denominator={exact_schur:.6e}, relative error="
            f"{schur_relative_error:.3e}, pressure true residual="
            f"{iterative_schur.pressure_krylov_residual_relative:.3e}, "
            f"bordered true residual="
            f"{iterative_schur.bordered_residual_relative:.3e}"
        )

    pseudo_problem, pseudo_initial, *_ = _case()
    started = time.perf_counter()
    pseudo = continue_monolithic_compliance_pseudo_arclength(
        pseudo_problem,
        pseudo_initial,
        target_compliance_factor=1.0,
        initial_step=0.015,
        minimum_step=0.001,
        maximum_step=0.03,
        maximum_steps=20,
        maximum_retries_per_step=5,
        residual_tolerance=1e-7,
        load_tolerance=1e-7,
    )
    pseudo_seconds = time.perf_counter() - started
    accepted_pseudo = [step for step in pseudo.branch.steps if step.accepted]
    rejected_pseudo = [step for step in pseudo.branch.steps if not step.accepted]
    maximum_gamma_step = max(accepted_pseudo, key=lambda step: step.parameter)
    independent_reynolds = reynolds_residual(
        pseudo.solution.pressure_dimensionless,
        pseudo.solution.film_thickness_m / film_scale,
        dx,
        dy,
        problem.reynolds_parameter,
    )
    independent_reynolds_rms = float(
        np.sqrt(np.mean(independent_reynolds**2))
        / problem.reynolds_residual_scale
    )
    independent_load = float(
        trapezoid(trapezoid(pseudo.solution.pressure_pa, x, axis=1), y)
    )
    print(
        f"pseudo-arclength: {pseudo.branch.termination_reason}; "
        f"furthest gamma={maximum_gamma_step.parameter:.9f}; "
        f"final gamma={pseudo.branch.parameter:.9f}; "
        f"turning attempts={pseudo.branch.turning_point_attempts}; "
        f"accepted={len(accepted_pseudo)}, rejected={len(rejected_pseudo)}, "
        f"Newton={sum(step.newton_iterations for step in pseudo.branch.steps)}, "
        f"Krylov=0 (dense direct correctors), "
        f"line-search rejects="
        f"{sum(step.rejected_line_search_trials for step in pseudo.branch.steps)}, "
        f"elapsed={pseudo_seconds:.3f} s"
    )
    print(
        f"pseudo-arclength final state: independent scaled Reynolds RMS="
        f"{independent_reynolds_rms:.6e}; independent load error="
        f"{independent_load - problem.target_load_n:.6e} N; "
        f"minimum film={pseudo.solution.film_thickness_m.min():.6e} m; "
        f"pressure range=({pseudo.solution.pressure_pa.min():.6e}, "
        f"{pseudo.solution.pressure_pa.max():.6e}) Pa; "
        f"max |deformation|="
        f"{np.max(np.abs(pseudo.solution.elastic_displacement_m)):.6e} m"
    )

    pseudo_conditioning = [
        (
            step,
            step.jacobian_condition_number,
            step.smallest_jacobian_singular_value,
        )
        for step in accepted_pseudo
    ]
    selected_gamma_targets = (
        0.1,
        0.175,
        maximum_gamma_step.parameter,
        pseudo.branch.parameter,
    )
    selected_attempts = {
        min(accepted_pseudo, key=lambda item: abs(item.parameter - target)).attempt
        for target in selected_gamma_targets
    }
    pseudo_schur = [np.nan] * len(pseudo_conditioning)
    for index, (step, _condition, _smallest) in enumerate(pseudo_conditioning):
        if step.attempt not in selected_attempts:
            continue
        z = np.asarray(step.state, dtype=np.float64)
        (
            _spectrum,
            schur,
            schur_angle,
            schur_residual,
            direct_residual,
            correction_difference,
        ) = _conditioning_at(
            pseudo_problem, z, step.parameter
        )
        pseudo_schur[index] = schur
        if step is maximum_gamma_step:
            print(
                f"pseudo-arclength turning state γ={step.parameter:.9f}: "
                f"condition={_spectrum[2]:.6e}, "
                f"sigma_min={_spectrum[1]:.6e}, "
                f"Schur={schur:.6e}, Schur angle={schur_angle:.6e}, "
                f"direct residual={direct_residual:.3e}, "
                f"Schur residual={schur_residual:.3e}, "
                f"Schur-vs-direct correction={correction_difference:.3e}"
            )

    figure, _ = plot_pseudo_arclength_branch(
        pseudo.branch, state_label="h0 / h_ref"
    )
    figure.savefig(output_directory / "pseudo_arclength_branch.png", dpi=160)
    plt.close(figure)
    figure, _ = plot_pseudo_arclength_diagnostics(pseudo.branch)
    figure.savefig(output_directory / "pseudo_arclength_diagnostics.png", dpi=160)
    plt.close(figure)
    figure, _ = plot_monolithic_compliance_continuation(
        tuple(conventional_steps)
    )
    figure.savefig(output_directory / "fixed_gamma_continuation.png", dpi=160)
    plt.close(figure)
    figure, _ = plot_monolithic_conditioning(
        [gamma for gamma, _spectrum in conventional_spectra],
        [spectrum[2] for _gamma, spectrum in conventional_spectra],
        [spectrum[1] for _gamma, spectrum in conventional_spectra],
        conventional_schur,
    )
    figure.savefig(output_directory / "fixed_gamma_conditioning.png", dpi=160)
    plt.close(figure)
    figure, _ = plot_monolithic_conditioning(
        [step.parameter for step, _, _ in pseudo_conditioning],
        [condition for _, condition, _ in pseudo_conditioning],
        [smallest for _, _, smallest in pseudo_conditioning],
        pseudo_schur,
    )
    figure.savefig(output_directory / "pseudo_arclength_conditioning.png", dpi=160)
    plt.close(figure)

    selected_steps = {
        min(
            accepted_pseudo,
            key=lambda item: abs(item.parameter - target),
        ).attempt: target
        for target in selected_gamma_targets
    }
    for step in accepted_pseudo:
        if step.attempt not in selected_steps:
            continue
        z = np.asarray(step.state, dtype=np.float64)
        pseudo_problem.compliance_factor = step.parameter
        pressure, film, _ = pseudo_problem.state_fields(z[:-1], float(z[-1]))
        pressure_pa = pressure * pseudo_problem.pressure_scale_pa
        label = f"gamma_{step.parameter:.6f}_step_{step.attempt:02d}"
        for filename, (figure, _axes) in (
            (
                "pressure",
                plot_pressure_distribution(x, y, pressure_pa),
            ),
            (
                "film",
                plot_film_thickness_distribution(x, y, film),
            ),
        ):
            figure.savefig(output_directory / f"{filename}_{label}.png", dpi=160)
            plt.close(figure)

    print(f"Diagnostic figures: {output_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="new directory for diagnostic figures (existing directories are refused)",
    )
    main(parser.parse_args().output_directory)
