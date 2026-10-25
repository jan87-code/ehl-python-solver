"""Bounded adaptive positive-load stepping for the monolithic JFO solver."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ehl_solver.cavitating_coupled import CavitatingElasticitySolution
from ehl_solver.cavitating_newton import CavitatingNewtonError, solve_cavitating_newton


@dataclass(frozen=True)
class CavitatingLoadTrial:
    target_load_n: float
    accepted: bool
    step_fraction: float
    reason: str
    solution: CavitatingElasticitySolution | None
    newton_history: tuple
    diagnostics: dict


@dataclass(frozen=True)
class CavitatingLoadContinuation:
    solution: CavitatingElasticitySolution
    trials: tuple[CavitatingLoadTrial, ...]


class CavitatingContinuationError(RuntimeError):
    def __init__(self, message, last_solution, trials, target_load_n):
        super().__init__(message)
        self.last_solution = last_solution
        self.trials = tuple(trials)
        self.target_load_n = target_load_n


def _verify_root(problem, solution, target, options):
    mass_tolerance = options.get("mass_tolerance", 1e-8)
    admissibility = options.get("admissibility_tolerance", 1e-10)
    load_tolerance = options.get("load_tolerance", 1e-8)
    metrics = problem.metrics(
        solution.pressure_pa, solution.fractional_film_content, solution.film_thickness_m
    )
    exact, displacement = problem.fields(
        solution.pressure_pa,
        compliance_factor=solution.compliance_factor,
        central_film_offset_m=solution.central_film_offset_m,
    )
    load = problem.load_n(solution.pressure_pa)
    minimum_residual = np.minimum(
        solution.pressure_pa[1:-1, 1:-1] / problem.pressure_scale_pa,
        1 - solution.fractional_film_content[1:-1, 1:-1],
    )
    if (
        np.max(np.abs(minimum_residual)) > admissibility
        or metrics.mass_maximum > mass_tolerance
        or metrics.maximum_relative_local_balance > mass_tolerance
        or max(
            metrics.maximum_complementarity,
            metrics.pressure_violation,
            metrics.saturation_violation,
            metrics.inlet_saturation_error,
        )
        > admissibility
        or metrics.boundary_pressure_max_pa != 0
        or abs(load - target) / target > load_tolerance
        or abs(load - solution.load_n) / target > load_tolerance
        or np.max(np.abs(exact - solution.film_thickness_m)) > 1e-12 * problem.film_scale_m
        or np.max(np.abs(displacement - solution.elastic_displacement_m))
        > 1e-12 * problem.film_scale_m
    ):
        raise ValueError("candidate fails independent mass/film/complementarity/load verification")
    return load


def continue_cavitating_load(
    problem,
    initial_solution,
    target_load_n,
    *,
    solver_options=None,
    initial_step_fraction=0.1,
    minimum_step_fraction=0.005,
    maximum_step_fraction=0.25,
    maximum_attempts=30,
    maximum_retries_per_stage=6,
):
    """Adapt force steps, retaining accepted roots and rejected trial diagnostics.

    Force inputs are SI N. Step fractions refer to the original force span.
    Only the last independently verified root supplies the next warm start.
    Rejections halve the step; accepted steps grow by 1.5 within configured
    bounds. Exhaustion raises with the last converged root and every trial.
    The problem and supplied initial solution are not mutated. Gamma stays
    equal to the initial solution's compliance factor throughout the path.
    """
    if not math.isfinite(target_load_n) or target_load_n <= 0:
        raise ValueError("target_load_n must be finite and positive [N]")
    if not (
        all(
            math.isfinite(x)
            for x in (initial_step_fraction, minimum_step_fraction, maximum_step_fraction)
        )
        and 0 < minimum_step_fraction <= initial_step_fraction <= maximum_step_fraction <= 1
    ):
        raise ValueError("load step fractions must satisfy 0 < minimum <= initial <= maximum <= 1")
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1
        for n in (maximum_attempts, maximum_retries_per_stage)
    ):
        raise ValueError("attempt and retry limits must be positive integers")
    options = dict(solver_options or {})
    if {"target_load_n", "initial_solution", "compliance_factor"} & options.keys():
        raise ValueError("solver_options cannot override load, warm start or compliance")
    initial_load = problem.load_n(initial_solution.pressure_pa)
    if initial_load <= 0:
        raise ValueError("initial root must have positive integrated pressure force")
    _verify_root(problem, initial_solution, initial_load, options)
    last, accepted_load = initial_solution, initial_load
    if abs(target_load_n - initial_load) / target_load_n <= options.get("load_tolerance", 1e-8):
        return CavitatingLoadContinuation(last, ())
    span = abs(target_load_n - initial_load)
    direction = math.copysign(1.0, target_load_n - initial_load)
    step, retries, trials = initial_step_fraction, 0, []

    def exhausted(reason):
        raise CavitatingContinuationError(reason, last, trials, target_load_n)

    for _ in range(maximum_attempts):
        remaining = abs(target_load_n - accepted_load)
        trial_load = (
            target_load_n if step * span >= remaining else accepted_load + direction * step * span
        )
        actual_fraction = abs(trial_load - accepted_load) / span
        candidate, trial_history = None, ()
        try:
            result = solve_cavitating_newton(
                problem,
                initial_solution=last,
                target_load_n=trial_load,
                compliance_factor=initial_solution.compliance_factor,
                **options,
            )
            candidate, trial_history = result.solution, result.history
            if result.solution.compliance_factor != initial_solution.compliance_factor:
                raise ValueError("candidate changed continuation compliance")
            _verify_root(problem, result.solution, trial_load, options)
        except (CavitatingNewtonError, ValueError) as error:
            trials.append(
                CavitatingLoadTrial(
                    trial_load,
                    False,
                    actual_fraction,
                    str(error),
                    getattr(error, "last_solution", candidate),
                    getattr(error, "history", trial_history),
                    getattr(error, "diagnostics", {}),
                )
            )
            retries += 1
            step = actual_fraction * 0.5
            if step < minimum_step_fraction:
                exhausted("minimum force step exhausted")
            if retries >= maximum_retries_per_stage:
                exhausted("force-step retry budget exhausted")
            continue
        last, accepted_load = result.solution, trial_load
        trials.append(
            CavitatingLoadTrial(
                trial_load, True, actual_fraction, "verified root", last, result.history, {}
            )
        )
        if trial_load == target_load_n:
            return CavitatingLoadContinuation(last, tuple(trials))
        retries = 0
        step = min(maximum_step_fraction, 1.5 * step)
    exhausted("force continuation attempt budget exhausted")
