"""Uniform CPU prescribed-offset elasticity with compressible/piezoviscous JFO.

Separate bounded reference experiment; no force controller or GPU port.
All exterior pressure sources on the stated finite reservoir enter the FFT.
"""

from dataclasses import dataclass

import numpy as np

from ehl_solver.property_reynolds import admissible, mass_audit, solve_property_reynolds


@dataclass(frozen=True)
class PropertyCoupledSolution:
    pressure_pa: np.ndarray
    fractional_film_content: np.ndarray
    film_thickness_m: np.ndarray
    elastic_displacement_m: np.ndarray
    load_n: float
    audit: dict
    history: tuple
    constitutive_strength: float


class PropertyCouplingError(RuntimeError):
    def __init__(self, message, history, last_solution, inner_failure=None):
        super().__init__(message)
        self.history = tuple(history)
        self.last_solution = last_solution
        self.inner_failure = inner_failure


def solve_property_coupling(
    problem, fluid, speed_m_s, *, constitutive_strength=1.0, maximum_iterations_per_stage=100
):
    """SI uniform problem and fluid, explicit speed [m/s], prescribed h0.

    Start from rigid film/zero pressure. First rigid solve continues the laws;
    then compliance 0,.25,.5,.75,1 updates FULL-domain film using the verified
    full-source uniform FFT. Only the film iterate is relaxed by .5. Stopping
    evaluates original pressure-dependent SI mass on exact coupled film.
    No pressure/content/film clipping. At most 20000 nodes, 100 iterations
    per compliance stage and inherited bounded inner property/JFO solves.
    Constitutive strength zero is an independent constant-property limit.
    """
    if (
        not np.isfinite(speed_m_s)
        or speed_m_s <= 0
        or not np.isfinite(constitutive_strength)
        or not 0 <= constitutive_strength <= 1
    ):
        raise ValueError("invalid speed or constitutive strength")
    if (
        not isinstance(maximum_iterations_per_stage, int)
        or isinstance(maximum_iterations_per_stage, bool)
        or not 1 <= maximum_iterations_per_stage <= 100
    ):
        raise ValueError("coupled stage budget must be in 1..100")
    if np.prod(problem.shape) > 20000:
        raise ValueError("coupled node budget exceeds 20000")
    for axis in (problem.x_m, problem.y_m):
        if not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-10, atol=0):
            raise ValueError("uniform physical cells required for coupled FFT")
    parameter = (
        12
        * fluid.viscosity_pa_s
        * speed_m_s
        * problem.length_scale_m
        / (problem.film_scale_m**2 * problem.pressure_scale_pa)
    )
    if not np.isclose(parameter, problem.reynolds_parameter, rtol=1e-12, atol=0):
        raise ValueError("SI fluid/speed do not reproduce problem Lambda")
    scales = dict(
        length_scale_m=problem.length_scale_m,
        film_scale_m=problem.film_scale_m,
        pressure_scale_pa=problem.pressure_scale_pa,
    )
    p = np.zeros(problem.shape)
    film, _ = problem.fields(p, compliance_factor=0)
    history = []
    last = None
    for gamma in (0, 0.25, 0.5, 0.75, 1):
        for iteration in range(1, maximum_iterations_per_stage + 1):
            try:
                strengths = (constitutive_strength,)
                if not history and constitutive_strength > 0:
                    strengths = tuple(s * constitutive_strength for s in (0, 0.25, 0.5, 0.75, 1))
                inner = solve_property_reynolds(
                    problem.x_m,
                    problem.y_m,
                    film,
                    fluid,
                    speed_m_s,
                    strengths=strengths,
                    initial_pressure_pa=p,
                    inlet_content=problem.inlet_fractional_film_content,
                    **scales,
                )
                candidate = inner.pressure_pa
                theta = inner.fractional_film_content
                exact, w = problem.fields(candidate, compliance_factor=gamma)
                audit = mass_audit(
                    problem.x_m,
                    problem.y_m,
                    exact,
                    candidate,
                    theta,
                    fluid,
                    speed_m_s,
                    strength=constitutive_strength,
                    **scales,
                )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                raise PropertyCouplingError(str(exc), history, last, exc) from exc
            update = float(
                np.linalg.norm(candidate - p) / max(np.linalg.norm(candidate), np.finfo(float).tiny)
            )
            defect = float(np.linalg.norm(exact - film) / np.linalg.norm(exact))
            history.append(
                {
                    "compliance_factor": gamma,
                    "iteration": iteration,
                    "pressure_update": update,
                    "film_defect": defect,
                    "minimum_film_m": float(exact.min()),
                    "load_n": problem.load_n(candidate),
                    "audit": audit,
                    "inner_history": inner.history,
                }
            )
            last = PropertyCoupledSolution(
                candidate,
                theta,
                exact,
                w,
                problem.load_n(candidate),
                audit,
                tuple(history),
                constitutive_strength,
            )
            if max(update, defect) <= 1e-8 and admissible(
                audit, candidate, theta, problem.pressure_scale_pa
            ):
                p, film = candidate, exact
                break
            p = candidate
            film = film + 0.5 * (exact - film)
        else:
            raise PropertyCouplingError(f"property coupling stage {gamma} exhausted", history, last)
    return last
