"""Independent, bounded CPU audits of signed full-film mathematical states."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import convolve2d

from ehl_solver.monolithic import BorderedReynoldsElasticityProblem, MonolithicLoadSolution


@dataclass(frozen=True)
class FullFilmAudit:
    """Residual scales follow the problem; flux balances are dimensionless.

    Fluxes use X=x/L, Y=y/L, H=h/h_ref, P=p/p_ref and
    q=-H**3 grad(P)+Lambda*H*e_x. Net flux surrounds the interior control
    volumes, with their outer faces halfway to the boundary nodes.
    """

    reynolds_rms_scaled: float
    load_residual_n: float
    load_residual_scaled: float
    minimum_film_m: float
    minimum_pressure_pa: float
    maximum_pressure_pa: float
    boundary_pressure_max_pa: float
    pressure_scaling_max_error_pa: float
    film_coupling_max_error_m: float
    displacement_max_error_m: float
    maximum_local_flux_balance: float
    net_boundary_flux: float
    summed_local_flux_balance: float


def audit_full_film_solution(
    problem: BorderedReynoldsElasticityProblem,
    solution: MonolithicLoadSolution,
    *,
    compliance_factor: float | None = None,
    maximum_grid_points: int = 1024,
) -> FullFilmAudit:
    """Audit SI solution fields without invoking the nonlinear residual or FFT.

    Direct spatial convolution reuses the separately validated cell-integrated
    coefficients. Face-flux differences independently measure the Reynolds
    residual and local/global conservation. The supplied SI film is audited
    as supplied, so inconsistent film and displacement cannot hide behind a
    reconstruction. This small-grid reference intentionally rejects larger
    grids rather than silently switching to FFT. No state is mutated.
    """
    gamma = problem.compliance_factor if compliance_factor is None else compliance_factor
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("compliance_factor must be finite and in [0, 1]")
    shape = (problem.y.size, problem.x.size)
    if (
        not isinstance(maximum_grid_points, int)
        or isinstance(maximum_grid_points, bool)
        or maximum_grid_points < 1
        or np.prod(shape) > maximum_grid_points
    ):
        raise ValueError("direct audit grid exceeds maximum_grid_points")
    fields = [np.asarray(field, dtype=np.float64) for field in (
        solution.pressure_pa, solution.film_thickness_m,
        solution.elastic_displacement_m, solution.pressure_dimensionless,
    )]
    if any(field.shape != shape or not np.all(np.isfinite(field)) for field in fields):
        raise ValueError("solution fields must be finite arrays with shape (ny, nx)")
    pressure_pa, film_m, displacement_m, pressure_dimensionless = fields
    if np.any(film_m <= problem.film_floor_m):
        raise ValueError("solution film must exceed the problem film floor")
    if not math.isfinite(solution.central_film_offset_m):
        raise ValueError("solution offset must be finite [m]")

    direct_displacement = convolve2d(pressure_pa, problem.elastic_kernel, mode="same")
    xx, yy = np.meshgrid(problem.x, problem.y, indexing="xy")
    reconstructed_film = (
        solution.central_film_offset_m
        + xx**2 / (2.0 * problem.radius_x_m)
        + yy**2 / (2.0 * problem.radius_y_m)
        + gamma * direct_displacement
    )
    p = pressure_pa / problem.pressure_scale_pa
    h = film_m / problem.film_scale_m
    h3 = h**3
    qx = (
        -0.5 * (h3[:, 1:] + h3[:, :-1]) * np.diff(p, axis=1) / problem.dx
        + problem.reynolds_parameter * 0.5 * (h[:, 1:] + h[:, :-1])
    )
    qy = -0.5 * (h3[1:, :] + h3[:-1, :]) * np.diff(p, axis=0) / problem.dy
    divergence = (
        np.diff(qx[1:-1, :], axis=1) / problem.dx
        + np.diff(qy[:, 1:-1], axis=0) / problem.dy
    )
    local_balances = divergence * problem.dx * problem.dy
    net_flux = (
        np.sum(qx[1:-1, -1] - qx[1:-1, 0]) * problem.dy
        + np.sum(qy[-1, 1:-1] - qy[0, 1:-1]) * problem.dx
    )
    load_n = float(trapezoid(trapezoid(pressure_pa, problem.x, axis=1), problem.y))
    boundary = np.concatenate((pressure_pa[0], pressure_pa[-1],
                               pressure_pa[1:-1, 0], pressure_pa[1:-1, -1]))
    return FullFilmAudit(
        reynolds_rms_scaled=float(np.sqrt(np.mean(divergence**2)))
        / problem.reynolds_residual_scale,
        load_residual_n=load_n - problem.target_load_n,
        load_residual_scaled=(load_n - problem.target_load_n) / problem.load_residual_scale_n,
        minimum_film_m=float(film_m.min()),
        minimum_pressure_pa=float(pressure_pa.min()),
        maximum_pressure_pa=float(pressure_pa.max()),
        boundary_pressure_max_pa=float(np.max(np.abs(boundary))),
        pressure_scaling_max_error_pa=float(np.max(np.abs(
            pressure_pa - pressure_dimensionless * problem.pressure_scale_pa
        ))),
        film_coupling_max_error_m=float(np.max(np.abs(film_m - reconstructed_film))),
        displacement_max_error_m=float(np.max(np.abs(displacement_m - direct_displacement))),
        maximum_local_flux_balance=float(np.max(np.abs(local_balances))),
        net_boundary_flux=float(net_flux),
        summed_local_flux_balance=float(np.sum(local_balances)),
    )


@dataclass(frozen=True)
class CavitatingStateAudit:
    """Independent supplied-field checks; balances use dimensionless JFO flux."""

    mass_rms: float
    mass_maximum: float
    maximum_complementarity: float
    minimum_pressure_pa: float
    minimum_saturation: float
    maximum_saturation: float
    pressure_boundary_max_pa: float
    inlet_saturation_error: float
    net_boundary_flux: float
    summed_volume_balance: float
    film_coupling_max_error_m: float
    displacement_max_error_m: float
    minimum_film_m: float
    load_n: float
    load_metadata_error_n: float
    west_boundary_flux: float
    east_boundary_flux: float
    south_boundary_flux: float
    north_boundary_flux: float


def audit_cavitating_solution(problem, solution, *, maximum_grid_points=1024):
    """Audit SI fields with direct elasticity and the fixed-film JFO stencil.

    Does not call problem.fields, problem.metrics, or the coupled residual.
    The explicit opposing-face reference balances are independent of the
    coupled implementation's whole-face arrays and np.diff operations.
    """
    from ehl_solver.cavitation import _mass_balance

    if (not isinstance(maximum_grid_points, int) or isinstance(maximum_grid_points, bool)
            or maximum_grid_points < 1 or np.prod(problem.shape) > maximum_grid_points):
        raise ValueError("direct audit grid exceeds maximum_grid_points")
    fields = [np.asarray(values, dtype=np.float64) for values in
              (solution.pressure_pa, solution.fractional_film_content,
               solution.film_thickness_m, solution.elastic_displacement_m)]
    if any(values.shape != problem.shape or not np.all(np.isfinite(values)) for values in fields):
        raise ValueError("audit fields must be finite with shape (ny,nx)")
    p_pa, theta, film, displacement = fields
    if np.any(film <= problem.film_floor_m):
        raise ValueError("audit film must exceed film_floor_m")
    if (not math.isfinite(solution.compliance_factor) or not 0 <= solution.compliance_factor <= 1
            or not math.isfinite(solution.central_film_offset_m)):
        raise ValueError("audit compliance/offset is invalid")
    p = p_pa / problem.pressure_scale_pa
    mass, fluxes = _mass_balance(film / problem.film_scale_m, p, theta,
                                problem.dx, problem.dy, problem.reynolds_parameter)
    direct_w = convolve2d(p_pa, problem.elastic_kernel, mode="same")
    xx, yy = np.meshgrid(problem.x_m, problem.y_m, indexing="xy")
    exact_film = (solution.central_film_offset_m + xx**2 / (2*problem.radius_x_m)
                  + yy**2 / (2*problem.radius_y_m) + solution.compliance_factor * direct_w)
    edges = np.r_[p_pa[0], p_pa[-1], p_pa[1:-1, 0], p_pa[1:-1, -1]]
    load = float(trapezoid(trapezoid(p_pa, problem.x_m, axis=1), problem.y_m))
    return CavitatingStateAudit(
        float(np.sqrt(np.mean(mass**2))), float(np.max(np.abs(mass))),
        float(np.max(np.abs(p[1:-1, 1:-1] * (1-theta[1:-1, 1:-1])))),
        float(p_pa.min()), float(theta.min()), float(theta.max()),
        float(np.max(np.abs(edges))),
        float(np.max(np.abs(theta[1:-1, 0] - problem.inlet_fractional_film_content))),
        float(sum(fluxes)), float(np.sum(mass) * problem.dx * problem.dy),
        float(np.max(np.abs(film - exact_film))),
        float(np.max(np.abs(displacement - direct_w))), float(film.min()), load,
        load - solution.load_n,
        *(float(value) for value in fluxes),
    )
