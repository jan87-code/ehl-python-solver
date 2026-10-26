"""Separate bounded uniform-grid finite-reservoir exterior/Picard experiment.

Prescribed h0, full source support, no load controller and no nonuniform FFT.
The original coupled/reference/GPU implementations remain independent.
"""

from dataclasses import dataclass

import numpy as np

from ehl_solver.cavitating_coupled import (
    CavitatingCoupledIteration,
    CavitatingCouplingError,
    CavitatingElasticitySolution,
)
from ehl_solver.exterior import solve_exterior_response
from ehl_solver.rectilinear_cavitation import flux_audit


@dataclass(frozen=True)
class ExteriorCoupledResult:
    solution: CavitatingElasticitySolution
    audit: dict
    exterior_history: tuple
    retained_mask: np.ndarray


class ExteriorCouplingError(CavitatingCouplingError):
    def __init__(self, message, history, last_solution, exterior_history):
        super().__init__(message, history, last_solution)
        self.exterior_history = tuple(exterior_history)


def solve_uniform_exterior_coupling(
    problem,
    viscosity_pa_s,
    speed_m_s,
    *,
    core_bounds_m,
    maximum_iterations_per_stage=100,
):
    """Damped prescribed-offset Picard with exact exterior response each step.

    `problem` is the existing uniform SI CavitatingElasticityProblem. eta
    [Pa s] and speed [m/s] must reproduce its Lambda. Start from zero pressure,
    use fixed compliance stages 0,.25,.5,.75,1 and film relaxation .5. Every
    step reconstructs ALL exterior pressure/content, then applies the existing
    full-source zero-padded uniform FFT to update film at ALL physical nodes.
    Default caps and original physical tolerances are not relaxed. This is
    a separate restricted-model experiment, not physical EHL validation.
    """
    if (
        not isinstance(maximum_iterations_per_stage, int)
        or isinstance(maximum_iterations_per_stage, bool)
        or not 1 <= maximum_iterations_per_stage <= 100
    ):
        raise ValueError("outer stage budget must be in 1..100")
    if not all(np.isfinite(v) and v > 0 for v in (viscosity_pa_s, speed_m_s)):
        raise ValueError("positive finite SI viscosity and speed required")
    parameter = (
        12
        * viscosity_pa_s
        * speed_m_s
        * problem.length_scale_m
        / (problem.film_scale_m**2 * problem.pressure_scale_pa)
    )
    if not np.isclose(parameter, problem.reynolds_parameter, rtol=1e-12, atol=0):
        raise ValueError("SI viscosity/speed disagree with problem Reynolds parameter")
    # Recheck uniformity: never rely on an unrelated object's dx/dy attributes.
    for axis in (problem.x_m, problem.y_m):
        if not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-10, atol=0):
            raise ValueError("exterior coupling requires a uniform physical source grid")
    if np.prod(problem.shape) > 20000:
        raise ValueError("uniform exterior coupling exceeds 20000 nodes")
    p = np.zeros(problem.shape)
    film, _ = problem.fields(p, compliance_factor=0)
    history, traces, last = [], [], None
    for gamma in (0, 0.25, 0.5, 0.75, 1):
        for iteration in range(1, maximum_iterations_per_stage + 1):
            try:
                inner = solve_exterior_response(
                    problem.x_m,
                    problem.y_m,
                    film,
                    viscosity_pa_s,
                    speed_m_s,
                    core_bounds_m=core_bounds_m,
                    length_scale_m=problem.length_scale_m,
                    film_scale_m=problem.film_scale_m,
                    pressure_scale_pa=problem.pressure_scale_pa,
                    inlet_content=problem.inlet_fractional_film_content,
                )
                candidate = inner.pressure_pa
                theta = inner.fractional_film_content
                exact, w = problem.fields(candidate, compliance_factor=gamma)
                metrics = problem.metrics(candidate, theta, exact)
                audit = flux_audit(
                    problem.x_m,
                    problem.y_m,
                    exact,
                    candidate,
                    theta,
                    viscosity_pa_s,
                    speed_m_s,
                    length_scale_m=problem.length_scale_m,
                    film_scale_m=problem.film_scale_m,
                    pressure_scale_pa=problem.pressure_scale_pa,
                )
            except (RuntimeError, ValueError) as exc:
                raise ExteriorCouplingError(str(exc), history, last, traces) from exc
            change = float(
                np.linalg.norm(candidate - p) / max(np.linalg.norm(candidate), np.finfo(float).tiny)
            )
            defect = float(np.linalg.norm(exact - film) / np.linalg.norm(exact))
            history.append(
                CavitatingCoupledIteration(
                    gamma,
                    iteration,
                    change,
                    defect,
                    float(exact.min()),
                    problem.load_n(candidate),
                    metrics,
                    len(inner.history),
                )
            )
            traces.append(
                {
                    "compliance_factor": gamma,
                    "iteration": iteration,
                    "exact_film_audit": audit,
                    "exterior_active_sets": inner.history,
                }
            )
            last = CavitatingElasticitySolution(
                candidate,
                theta,
                exact,
                w,
                problem.central_film_offset_m,
                gamma,
                problem.load_n(candidate),
                tuple(history),
            )
            if (
                max(change, defect) <= 1e-8
                and audit["mass_maximum"] <= 1e-8
                and audit["relative_local_balance"] <= 1e-10
                and audit["relative_global_balance"] <= 1e-10
                and max(
                    metrics.maximum_complementarity,
                    metrics.pressure_violation,
                    metrics.saturation_violation,
                    metrics.inlet_saturation_error,
                )
                <= 1e-10
                and metrics.boundary_pressure_max_pa == 0
            ):
                p, film = candidate, exact
                break
            p = candidate
            film = film + 0.5 * (exact - film)
        else:
            raise ExteriorCouplingError(
                f"exterior coupling stage {gamma} exhausted", history, last, traces
            )
    return ExteriorCoupledResult(last, audit, tuple(traces), inner.retained_mask)
