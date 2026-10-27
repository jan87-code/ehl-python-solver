"""Verified fixed-reservoir transfer of an accepted root to an unaccepted guess.

Interpolation preserves uniform nested-grid trapezoidal force, not Reynolds
or film equations. Only the separate repair solve may return an audited root.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ehl_solver.property_load_newton import (
    PropertyLoadSystem,
    _solve_property_load_state,
    verify_seed,
)


@dataclass(frozen=True)
class PropertyLoadGuess:
    pressure_pa: np.ndarray
    fractional_film_content: np.ndarray
    central_film_offset_m: float
    diagnostics: dict


def prolongate_property_guess(
    source_problem, target_problem, fluid, speed_m_s, *, seed, seed_offset_m
):
    """Bilinear SI pressure/content initialization on the same physical reservoir.

    Same geometry, scales, fluid, entrainment and supply are required. No
    pressure/content/film clipping, offset adjustment or source-field mutation.
    New full-source target FFT determines feasibility and initial residual.
    """
    for name in (
        "radius_x_m",
        "radius_y_m",
        "e_star_pa",
        "length_scale_m",
        "film_scale_m",
        "pressure_scale_pa",
        "reynolds_parameter",
        "inlet_fractional_film_content",
        "film_floor_m",
    ):
        if not np.isclose(
            getattr(source_problem, name), getattr(target_problem, name), rtol=1e-12, atol=0
        ):
            raise ValueError("transfer requires the same geometry, supply and SI scales")
    for source, target in (
        (source_problem.x_m, target_problem.x_m),
        (source_problem.y_m, target_problem.y_m),
    ):
        if not np.array_equal(source[[0, -1]], target[[0, -1]]):
            raise ValueError("transfer requires identical physical reservoir bounds")
        if len(target) < len(source):
            raise ValueError("target grid must not coarsen either axis")
        if not all(
            np.any(np.isclose(target, v, rtol=0, atol=1e-12 * source_problem.length_scale_m))
            for v in source
        ):
            raise ValueError("nested uniform refinement required")
    system = PropertyLoadSystem(source_problem, fluid, speed_m_s, 1.0)
    z = verify_seed(system, seed, seed_offset_m)
    accepted = system.solution(z).solution
    target = PropertyLoadSystem(target_problem, fluid, speed_m_s, accepted.load_n)
    y, x = np.meshgrid(target_problem.y_m, target_problem.x_m, indexing="ij")
    points = np.c_[y.ravel(), x.ravel()]

    def interpolate(values):
        return RegularGridInterpolator(
            (source_problem.y_m, source_problem.x_m), values, bounds_error=True
        )(points).reshape(target_problem.shape)

    p, t = interpolate(seed.pressure_pa), interpolate(seed.fractional_film_content)
    guessed = target.pack(SimpleNamespace(pressure_pa=p, fractional_film_content=t), seed_offset_m)
    initial = target.solution(guessed).solution  # Feasible total film required, mass may fail.
    diagnostics = {
        "accepted": False,
        "source_load_n": accepted.load_n,
        "initial_load_n": initial.load_n,
        "relative_load_difference": abs(initial.load_n / accepted.load_n - 1),
        "initial_audit": initial.audit,
        "minimum_film_m": float(initial.film_thickness_m.min()),
        "source_shape_ny_nx": list(source_problem.shape),
        "target_shape_ny_nx": list(target_problem.shape),
        "offset_adjustment_m": 0.0,
        "interpolation": "bilinear on physical y,x; no extrapolation",
    }
    if diagnostics["relative_load_difference"] > 1e-12:
        raise ValueError("nested interpolation failed independent force-preservation check")
    return PropertyLoadGuess(
        initial.pressure_pa, initial.fractional_film_content, float(seed_offset_m), diagnostics
    )


def repair_property_load_guess(
    problem,
    fluid,
    speed_m_s,
    target_load_n,
    guess,
    *,
    maximum_iterations=30,
    preconditioner="local",
    active_lookahead=True,
):
    """Explicitly unaccepted initialization; same original Newton/physical gates.

    Unlike a warm-root continuation call, this entry point permits initial
    mass/complementarity defects. All fields must remain within the original
    pressure/content bounds and positive-film domain. A returned root passes
    exactly the original mass/constraint/force criteria; zero budget cannot
    silently promote an interpolated non-root to an accepted solution.
    """
    if (
        not isinstance(maximum_iterations, int)
        or isinstance(maximum_iterations, bool)
        or not 0 <= maximum_iterations <= 30
    ):
        raise ValueError("Newton correction budget must lie in 0..30")
    p = np.asarray(guess.pressure_pa, dtype=float)
    t = np.asarray(guess.fractional_film_content, dtype=float)
    if (
        p.shape != problem.shape
        or t.shape != problem.shape
        or not np.all(np.isfinite(p))
        or not np.all(np.isfinite(t))
        or p.min() < -1e-10 * problem.pressure_scale_pa
        or t.min() < -1e-10
        or t.max() > 1 + 1e-10
        or np.any(np.r_[p[0], p[-1], p[:, 0], p[:, -1]] != 0)
    ):
        raise ValueError("initial guess fails original shape/pressure/content/boundary bounds")
    system = PropertyLoadSystem(
        problem, fluid, speed_m_s, target_load_n, preconditioner=preconditioner
    )
    z = system.pack(guess, guess.central_film_offset_m)
    system.solution(z)  # Checks the exact signed-offset film and original law domain.
    return _solve_property_load_state(
        system, z, maximum_iterations, 10, 1e-10, True, active_lookahead=active_lookahead
    )
