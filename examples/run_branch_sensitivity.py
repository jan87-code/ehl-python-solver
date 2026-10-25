"""Bounded CPU grid/domain and modulus sensitivity of a signed branch.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src.
All cases share the baseline rigid-reference signed load; no results are
physical EHL solutions. Output directories must be new. Dense state size,
continuation attempts, retries and Newton corrections are explicitly capped.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import trapezoid

from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    continue_monolithic_compliance_pseudo_arclength,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import solve_reynolds_pressure
from ehl_solver.turning_point import localize_parameter_turn
from ehl_solver.verification import audit_full_film_solution


@dataclass(frozen=True)
class Case:
    name: str
    ny: int
    nx: int
    x_limits_m: tuple[float, float] = (-0.5e-3, 0.75e-3)
    y_limits_m: tuple[float, float] = (-0.5e-3, 0.5e-3)
    e_star_pa: float = 1e6


CASES = (
    Case("baseline", 13, 17),
    Case("coarse", 7, 9),
    Case("fine", 19, 25),
    # Two extra nodes at each edge preserve the baseline dx and dy.
    Case("expanded", 17, 21, (-0.65625e-3, 0.90625e-3), (-2e-3 / 3, 2e-3 / 3)),
)


MODULUS_CASES = (CASES[0], Case("soft_100kPa", 13, 17, e_star_pa=1e5))


def make_case(case: Case, target_load_n: float | None = None):
    """Build a rigid initial root; target_load_n is SI N and is never rescaled."""
    x = np.linspace(*case.x_limits_m, case.nx)
    y = np.linspace(*case.y_limits_m, case.ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    film_scale_m, length_scale_m, pressure_scale_pa = 20e-6, 5e-4, 21250.0
    film = (film_scale_m + xx**2 / 0.02 + yy**2 / 0.04) / film_scale_m
    pressure = solve_reynolds_pressure(
        film, (x[1] - x[0]) / length_scale_m,
        (y[1] - y[0]) / length_scale_m, 12.0, rtol=1e-12, atol=1e-14,
    ).pressure
    if target_load_n is None:
        target_load_n = float(trapezoid(trapezoid(pressure * pressure_scale_pa, x, axis=1), y))
    problem = BorderedReynoldsElasticityProblem(
        x, y, 0.01, 0.02, case.e_star_pa, length_scale_m, film_scale_m, pressure_scale_pa,
        12.0, target_load_n, compliance_factor=0.0, reynolds_residual_scale=100.0,
        krylov_max_iterations=50,
    )
    initial = solve_monolithic_full_film_load(problem, pressure, film_scale_m, max_iterations=12)
    return problem, initial


def _solution_at(problem, initial, step):
    z = np.asarray(step.state)
    saved_gamma = problem.compliance_factor
    try:
        problem.compliance_factor = step.parameter
        pressure, film, displacement = problem.state_fields(z[:-1], float(z[-1]))
    finally:
        problem.compliance_factor = saved_gamma
    return replace(
        initial,
        pressure_dimensionless=pressure,
        pressure_pa=pressure * problem.pressure_scale_pa,
        film_thickness_m=film,
        elastic_displacement_m=displacement,
        central_film_offset_m=float(z[-1]) * problem.film_scale_m,
        calculated_load_n=float(trapezoid(
            trapezoid(pressure * problem.pressure_scale_pa, problem.x, axis=1), problem.y
        )),
        load_residual_n=float(trapezoid(
            trapezoid(pressure * problem.pressure_scale_pa, problem.x, axis=1), problem.y
        )) - problem.target_load_n,
        history=(),
    )


def _require_verified(audit, problem):
    # A combined RMS corrector tolerance permits a small sqrt((N+1)/N)
    # factor in Reynolds RMS; verify each component separately here.
    if (
        audit.reynolds_rms_scaled > 1.01e-8
        or abs(audit.load_residual_scaled) > 1e-6
        or audit.minimum_film_m <= problem.film_floor_m
        or audit.boundary_pressure_max_pa != 0.0
        or audit.pressure_scaling_max_error_pa > 1e-10
        or audit.film_coupling_max_error_m > 1e-12 * problem.film_scale_m
        or audit.displacement_max_error_m > 1e-12 * problem.film_scale_m
        or audit.maximum_local_flux_balance > 1e-6
        or abs(audit.net_boundary_flux) > 1e-6
        or abs(audit.net_boundary_flux - audit.summed_local_flux_balance) > 1e-10
    ):
        raise RuntimeError(f"independent branch-state audit failed: {audit}")


def run_case(
    case, problem, initial, directory, *, maximum_steps=12, plots=False,
    compliance_reference_modulus_pa=None, localize_turn=False, turn_bracket_tolerance=1e-6,
):
    started = time.perf_counter()
    result = continue_monolithic_compliance_pseudo_arclength(
        problem, initial, initial_step=0.03, maximum_step=0.05,
        maximum_steps=maximum_steps, maximum_retries_per_step=3,
        maximum_state_size=512, residual_tolerance=1e-8,
        compliance_reference_modulus_pa=compliance_reference_modulus_pa,
    )
    # Save all attempted states before checking acceptance independently.
    np.savez_compressed(
        directory / f"{case.name}_states.npz", x_m=problem.x, y_m=problem.y,
        states=np.asarray([step.state for step in result.branch.steps]),
        parameters=np.asarray([step.parameter for step in result.branch.steps]),
        accepted=np.asarray([step.accepted for step in result.branch.steps]),
    )
    accepted = [step for step in result.branch.steps if step.accepted]
    if not accepted:
        raise RuntimeError(f"no accepted branch states for {case.name}")
    audits = []
    for step in accepted:
        audit = audit_full_film_solution(
            problem, _solution_at(problem, initial, step), compliance_factor=step.parameter
        )
        audits.append(audit)
        _require_verified(audit, problem)
    peak_index = max(range(len(accepted)), key=lambda index: accepted[index].parameter)
    peak = accepted[peak_index]
    reversals = [
        (before, after)
        for before, after in zip(accepted, accepted[1:], strict=False)
        if before.tangent_parameter_component * after.tangent_parameter_component < 0
    ]
    summary = {
        "case": asdict(case), "target_load_n": problem.target_load_n,
        "dx_m": problem.dx_m, "dy_m": problem.dy_m,
        "initial_h0_m": initial.central_film_offset_m,
        "termination_reason": result.branch.termination_reason,
        "maximum_sampled_compliance": peak.parameter,
        "maximum_sampled_compliance_per_pa": peak.parameter / problem.e_star_pa,
        "effective_modulus_at_sampled_peak_pa": problem.e_star_pa / peak.parameter,
        "peak_state_audit": asdict(audits[peak_index]),
        "peak_jacobian_condition_number": peak.jacobian_condition_number,
        "peak_smallest_jacobian_singular_value": peak.smallest_jacobian_singular_value,
        "tangent_reversal_brackets": [
            {"attempts": [before.attempt, after.attempt],
             "compliance": [before.parameter, after.parameter],
             "h0_m": [before.state[-1] * problem.film_scale_m,
                      after.state[-1] * problem.film_scale_m],
             "tangent_parameter_components": [before.tangent_parameter_component,
                                              after.tangent_parameter_component]}
            for before, after in reversals
        ],
        "maximum_reynolds_rms_scaled": max(a.reynolds_rms_scaled for a in audits),
        "maximum_abs_load_error_n": max(abs(a.load_residual_n) for a in audits),
        "maximum_local_flux_balance": max(a.maximum_local_flux_balance for a in audits),
        "maximum_abs_net_boundary_flux": max(abs(a.net_boundary_flux) for a in audits),
        "steps": [asdict(step) for step in result.branch.steps],
        "audits": [asdict(audit) for audit in audits],
        "elapsed_seconds": time.perf_counter() - started,
    }
    if localize_turn:
        summary["turning_point"] = (
            _localize_turn(problem, initial, *reversals[0], directory, case,
                           compliance_reference_modulus_pa, turn_bracket_tolerance, plots)
            if reversals else {"termination_reason": "no_tangent_reversal"}
        )
    summary["elapsed_seconds"] = time.perf_counter() - started
    if plots:
        _save_plots(directory, case, problem, initial, peak, result.branch)
    return summary


def _branch_callbacks(problem):
    def residual(state, gamma):
        problem.compliance_factor = gamma
        return problem.residual(state)

    def jacobian(state, gamma):
        problem.compliance_factor = gamma
        identity = np.eye(state.size)
        return np.column_stack([
            problem.jacobian_vector(state, identity[:, i]) for i in range(state.size)
        ])

    def derivative(state, gamma):
        problem.compliance_factor = gamma
        return problem.compliance_derivative(state)

    def feasible(state, gamma):
        if not 0 <= gamma <= 1:
            return False
        problem.compliance_factor = gamma
        _, film, _ = problem.state_fields(state[:-1], state[-1])
        return bool(np.all(np.isfinite(film)) and np.all(film > problem.film_floor_m))

    return residual, jacobian, derivative, feasible


def _localize_turn(problem, initial, left, right, directory, case, reference_modulus,
                   bracket_tolerance, plots):
    # Numerical callbacks use interior P, h0/h_ref, and dimensionless gamma.
    # Geometry/material inputs and reconstructed solution fields remain SI.
    original_gamma = problem.compliance_factor
    weights = np.ones(problem.pressure_size + 1)
    weights[:-1] = 1 / problem.pressure_size
    parameter_weight = (
        1.0 if reference_modulus is None else (reference_modulus / problem.e_star_pa)**2
    )

    residual, jacobian, derivative, feasible = _branch_callbacks(problem)

    try:
        turn = localize_parameter_turn(
            residual, jacobian, derivative, np.asarray(left.state), left.parameter,
            np.asarray(right.state), right.parameter, state_weights=weights,
            parameter_weight=parameter_weight, feasible=feasible,
            bracket_tolerance=bracket_tolerance, residual_tolerance=1e-10,
        )
        # Preserve attempted correctors before the independent state audits.
        (directory / f"{case.name}_turning_point.json").write_text(
            json.dumps(_json_finite(asdict(turn)), indent=2, allow_nan=False) + "\n"
        )
        audits = []
        for probe in turn.probes:
            if probe.steps and probe.steps[-1].accepted:
                audit = audit_full_film_solution(
                    problem, _solution_at(problem, initial, probe),
                    compliance_factor=probe.parameter
                )
                _require_verified(audit, problem)
                if audit.reynolds_rms_scaled > 1.01e-10:
                    raise RuntimeError("localized probe fails independent Reynolds tolerance")
                audits.append(asdict(audit))
        best_audit = audit_full_film_solution(
            problem, _solution_at(problem, initial, turn.best),
            compliance_factor=turn.best.parameter
        )
        _require_verified(best_audit, problem)
        if best_audit.reynolds_rms_scaled > 1.01e-10:
            raise RuntimeError("localized best state fails independent Reynolds tolerance")
        if plots:
            _save_turn_plot(directory, case, turn, parameter_weight)
        return {
            "termination_reason": turn.termination_reason,
            "parameter": turn.best.parameter,
            "normalized_compliance_per_pa": turn.best.parameter / problem.e_star_pa,
            "bracket_width": turn.bracket_width,
            "bracket_widths": turn.bracket_widths,
            "normalized_parameter_tangent": turn.normalized_parameter_tangent,
            "refinements": len(turn.probes),
            "best_audit": asdict(best_audit), "probe_audits": audits,
        }
    finally:
        problem.compliance_factor = original_gamma


def _save_turn_plot(directory, case, turn, parameter_weight):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, constrained_layout=True)
    axes[0].semilogy(turn.bracket_widths, "o-")
    axes[0].set_ylabel("Weighted state/parameter bracket width")
    axes[1].semilogy(
        range(1, len(turn.probes) + 1),
        [abs(math.sqrt(parameter_weight) * probe.tangent[-1]) for probe in turn.probes], "o-",
    )
    axes[1].set(xlabel="Refinement", ylabel="|Normalized probe parameter tangent|")
    figure.suptitle(f"{case.name}: {turn.termination_reason}; gamma={turn.best.parameter:.10f}")
    figure.savefig(directory / f"{case.name}_turn_localization.png", dpi=120)
    plt.close(figure)


def _save_plots(directory, case, problem, initial, peak, branch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ehl_solver.visualization import (
        plot_field_cross_section,
        plot_film_thickness_distribution,
        plot_monolithic_conditioning,
        plot_pressure_distribution,
        plot_pseudo_arclength_branch,
        plot_pseudo_arclength_diagnostics,
    )
    solution = _solution_at(problem, initial, peak)
    accepted = [step for step in branch.steps if step.accepted]
    figures = {
        "branch": plot_pseudo_arclength_branch(branch, state_label="h0 / h_ref")[0],
        "correctors": plot_pseudo_arclength_diagnostics(branch)[0],
        "conditioning": plot_monolithic_conditioning(
            [step.parameter for step in accepted],
            [step.jacobian_condition_number for step in accepted],
            [step.smallest_jacobian_singular_value for step in accepted],
            [math.nan] * len(accepted),
        )[0],
        "pressure": plot_pressure_distribution(problem.x, problem.y, solution.pressure_pa)[0],
        "film": plot_film_thickness_distribution(
            problem.x, problem.y, solution.film_thickness_m
        )[0],
    }
    for axis in ("x", "y"):
        for field, values in (
            ("pressure", solution.pressure_pa),
            ("film", solution.film_thickness_m),
        ):
            figures[f"{field}_{axis}"] = plot_field_cross_section(
                problem.x, problem.y, values, axis=axis, coordinate_m=0.0,
                quantity=field,
            )[0]
    figure, axis = plt.subplots(constrained_layout=True)
    mesh = axis.pcolormesh(
        problem.x * 1e3, problem.y * 1e3, solution.elastic_displacement_m * 1e6,
        shading="auto", cmap="coolwarm",
    )
    axis.set(xlabel="x [mm] (entrainment)", ylabel="y [mm]", title="Unscaled displacement w")
    figure.colorbar(mesh, ax=axis, label="w [µm]; film uses gamma*w")
    figures["displacement"] = figure
    for name, figure in figures.items():
        figure.savefig(directory / f"{case.name}_{name}.png", dpi=120)
        plt.close(figure)


def _json_finite(value):
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(
    output_directory: Path, *, maximum_steps=12, plots=False, study="mesh",
    compliance_reference_modulus_pa=None, localize_turn=False, turn_bracket_tolerance=1e-6,
):
    cases = CASES if study == "mesh" else MODULUS_CASES
    output_directory.mkdir(parents=True, exist_ok=False)
    report = {
        "model": "signed full-film mathematical branch; no coupled cavitation",
        "study": study,
        "compliance_reference_modulus_pa": compliance_reference_modulus_pa,
        "localization": {"enabled": localize_turn, "bracket_tolerance": turn_bracket_tolerance,
                         "tangent_tolerance": 1e-6, "maximum_refinements": 24,
                         "residual_tolerance": 1e-10},
        "axis_order": "(ny, nx); x is entrainment",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "settings": {"maximum_steps": maximum_steps, "initial_step": 0.03,
                     "maximum_step": 0.05, "maximum_retries_per_step": 3,
                     "maximum_state_size": 512, "residual_tolerance": 1e-8,
                     "length_scale_m": 5e-4,
                     "film_scale_m": 20e-6, "pressure_scale_pa": 21250.0,
                     "reynolds_parameter": 12.0, "reynolds_residual_scale": 100.0},
        "cases": [],
    }
    baseline_problem, baseline_initial = make_case(CASES[0])
    target_load_n = baseline_problem.target_load_n
    for case in cases:
        problem, initial = (
            (baseline_problem, baseline_initial) if case is CASES[0]
            else make_case(case, target_load_n)
        )
        summary = run_case(
            case, problem, initial, output_directory, maximum_steps=maximum_steps, plots=plots,
            compliance_reference_modulus_pa=compliance_reference_modulus_pa,
            localize_turn=localize_turn, turn_bracket_tolerance=turn_bracket_tolerance,
        )
        report["cases"].append(summary)
        (output_directory / "summary.json").write_text(
            json.dumps(_json_finite(report), indent=2, allow_nan=False) + "\n"
        )
        print(f"{case.name} ({case.ny}, {case.nx}): "
              f"sample max gamma={summary['maximum_sampled_compliance']:.9f}; "
              f"reversal brackets={len(summary['tangent_reversal_brackets'])}; "
              f"{summary['elapsed_seconds']:.2f} s", flush=True)
        if "turning_point" in summary:
            turn = summary["turning_point"]
            print(f"  localization: {turn['termination_reason']}; "
                  f"gamma={turn.get('parameter')}; width={turn.get('bracket_width')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("results/branch_sensitivity"))
    parser.add_argument("--maximum-steps", type=int, default=12, choices=range(1, 41))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--study", choices=("mesh", "modulus"), default="mesh")
    parser.add_argument("--compliance-reference-modulus-pa", type=float)
    parser.add_argument("--localize-turn", action="store_true")
    parser.add_argument("--turn-bracket-tolerance", type=float, default=1e-6)
    options = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    main(options.output_directory, maximum_steps=options.maximum_steps,
         plots=options.plots, study=options.study,
         compliance_reference_modulus_pa=options.compliance_reference_modulus_pa,
         localize_turn=options.localize_turn, turn_bracket_tolerance=options.turn_bracket_tolerance)
