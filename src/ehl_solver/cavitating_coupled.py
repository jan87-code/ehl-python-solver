"""CPU prescribed-offset cavitation/elasticity reference with SI interfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid

from ehl_solver.backend import make_backend
from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.elasticity import cell_integrated_kernel


def _uniform_axis(values, name):
    axis = np.array(values, dtype=np.float64, copy=True)
    if (
        axis.ndim != 1
        or axis.size < 3
        or not np.all(np.isfinite(axis))
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError(f"{name} must have at least 3 finite ascending 1-D coordinates [m]")
    spacing = float(axis[1] - axis[0])
    atol = 32 * np.finfo(float).eps * max(abs(spacing), float(np.max(np.abs(axis))))
    if not np.allclose(np.diff(axis), spacing, rtol=1e-10, atol=atol):
        raise ValueError(f"{name} must be uniformly spaced [m]")
    return axis, spacing


def _fluxes(p, h, theta, dx, dy, parameter):
    """Dimensionless staggered fluxes, independently differenced by callers."""
    h3 = h**3
    qx = (
        -0.5 * (h3[:, 1:] + h3[:, :-1]) * np.diff(p, axis=1) / dx
        + parameter * 0.5 * (h[:, 1:] + h[:, :-1]) * theta[:, :-1]
    )
    qy = -0.5 * (h3[1:, :] + h3[:-1, :]) * np.diff(p, axis=0) / dy
    return qx, qy


@dataclass(frozen=True)
class CavitatingMetrics:
    mass_rms: float
    mass_maximum: float
    maximum_relative_local_balance: float
    net_boundary_flux: float
    maximum_complementarity: float
    pressure_violation: float
    saturation_violation: float
    boundary_pressure_max_pa: float
    inlet_saturation_error: float


@dataclass
class CavitatingElasticityProblem:
    """Isoviscous/incompressible point geometry, units SI, array order (ny,nx).

    X=x/L, H=h/h_ref, P=p/p_ref and Lambda=12*eta*Ue*L/(h_ref**2*p_ref).
    x follows positive entrainment. Modulus uses the Hertz E* convention.
    Uniform west saturation and zero gauge pressure on all edges are fixed.
    This model has no load controller or pressure-dependent rheology.
    """

    x_m: ArrayLike
    y_m: ArrayLike
    central_film_offset_m: float
    radius_x_m: float
    radius_y_m: float
    e_star_pa: float
    length_scale_m: float
    film_scale_m: float
    pressure_scale_pa: float
    reynolds_parameter: float
    inlet_fractional_film_content: float = 1.0
    film_floor_m: float = 1e-12
    dx_m: float = field(init=False)
    dy_m: float = field(init=False)
    rigid_geometry_m: NDArray[np.float64] = field(init=False, repr=False)
    elastic_kernel: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        self.x_m, self.dx_m = _uniform_axis(self.x_m, "x_m")
        self.y_m, self.dy_m = _uniform_axis(self.y_m, "y_m")
        positives = (
            self.central_film_offset_m,
            self.radius_x_m,
            self.radius_y_m,
            self.e_star_pa,
            self.length_scale_m,
            self.film_scale_m,
            self.pressure_scale_pa,
            self.reynolds_parameter,
            self.film_floor_m,
        )
        if not all(math.isfinite(value) and value > 0 for value in positives):
            raise ValueError("offset, radii, modulus, scales, Lambda and floor must be positive")
        if not math.isfinite(self.inlet_fractional_film_content) or not (
            0 <= self.inlet_fractional_film_content <= 1
        ):
            raise ValueError("inlet_fractional_film_content must lie in [0,1]")
        xx, yy = np.meshgrid(self.x_m, self.y_m, indexing="xy")
        self.rigid_geometry_m = xx**2 / (2 * self.radius_x_m) + yy**2 / (2 * self.radius_y_m)
        self.elastic_kernel = cell_integrated_kernel(
            self.shape, self.dx_m, self.dy_m, self.e_star_pa, make_backend("numpy")
        )

    @property
    def shape(self):
        return (self.y_m.size, self.x_m.size)

    @property
    def dx(self):
        return self.dx_m / self.length_scale_m

    @property
    def dy(self):
        return self.dy_m / self.length_scale_m

    def _array(self, values, name):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite array of shape (ny,nx)={self.shape}")
        return array

    def fields(self, pressure_pa, *, compliance_factor=1.0, central_film_offset_m=None):
        """Return exact coupled film and unscaled displacement [m], for p [Pa]."""
        pressure = self._array(pressure_pa, "pressure_pa")
        if not math.isfinite(compliance_factor) or not 0 <= compliance_factor <= 1:
            raise ValueError("compliance_factor must lie in [0,1]")
        offset = (
            self.central_film_offset_m if central_film_offset_m is None else central_film_offset_m
        )
        if not math.isfinite(offset):
            raise ValueError("central_film_offset_m must be finite [m]")
        displacement = fft_convolve_same(pressure, self.elastic_kernel, make_backend("numpy"))
        film = offset + self.rigid_geometry_m + compliance_factor * displacement
        if not np.all(np.isfinite(film)) or np.any(film <= self.film_floor_m):
            raise ValueError("coupled film is nonfinite or at/below film_floor_m")
        return film, displacement

    def mass_residual(self, pressure_pa, fractional_film_content, film_thickness_m):
        """Dimensionless div(q) on interior volumes, evaluated on supplied SI film."""
        p = self._array(pressure_pa, "pressure_pa") / self.pressure_scale_pa
        theta = self._array(fractional_film_content, "fractional_film_content")
        h = self._array(film_thickness_m, "film_thickness_m") / self.film_scale_m
        if np.any(h <= self.film_floor_m / self.film_scale_m):
            raise ValueError("film must exceed film_floor_m")
        qx, qy = _fluxes(p, h, theta, self.dx, self.dy, self.reynolds_parameter)
        return np.diff(qx[1:-1], axis=1) / self.dx + np.diff(qy[:, 1:-1], axis=0) / self.dy

    def metrics(self, pressure_pa, fractional_film_content, film_thickness_m):
        """Conservation and admissibility of supplied fields; no reconstruction."""
        p = self._array(pressure_pa, "pressure_pa") / self.pressure_scale_pa
        theta = self._array(fractional_film_content, "fractional_film_content")
        h = self._array(film_thickness_m, "film_thickness_m") / self.film_scale_m
        residual = self.mass_residual(pressure_pa, theta, film_thickness_m)
        qx, qy = _fluxes(p, h, theta, self.dx, self.dy, self.reynolds_parameter)
        local_scale = (np.abs(qx[1:-1, 1:]) + np.abs(qx[1:-1, :-1])) * self.dy + (
            np.abs(qy[1:, 1:-1]) + np.abs(qy[:-1, 1:-1])
        ) * self.dx
        boundary = np.r_[p[0], p[-1], p[1:-1, 0], p[1:-1, -1]]
        net = (
            np.sum(qx[1:-1, -1] - qx[1:-1, 0]) * self.dy
            + np.sum(qy[-1, 1:-1] - qy[0, 1:-1]) * self.dx
        )
        return CavitatingMetrics(
            float(np.sqrt(np.mean(residual**2))),
            float(np.max(np.abs(residual))),
            float(
                np.max(
                    np.abs(residual)
                    * self.dx
                    * self.dy
                    / np.maximum(local_scale, np.finfo(float).tiny)
                )
            ),
            float(net),
            float(np.max(np.abs(p[1:-1, 1:-1] * (1 - theta[1:-1, 1:-1])))),
            max(0.0, -float(p.min())),
            max(0.0, -float(theta.min()), float(theta.max()) - 1),
            float(np.max(np.abs(boundary))) * self.pressure_scale_pa,
            float(np.max(np.abs(theta[1:-1, 0] - self.inlet_fractional_film_content))),
        )

    def load_n(self, pressure_pa):
        """Trapezoidal integrated pressure force [N], with zero pressure edges."""
        return float(
            trapezoid(
                trapezoid(self._array(pressure_pa, "pressure_pa"), self.x_m, axis=1), self.y_m
            )
        )


@dataclass(frozen=True)
class CavitatingCoupledIteration:
    compliance_factor: float
    iteration: int
    pressure_update_relative: float
    film_defect_relative: float
    minimum_film_m: float
    load_n: float
    metrics: CavitatingMetrics
    active_set_iterations: int


@dataclass(frozen=True)
class CavitatingElasticitySolution:
    pressure_pa: NDArray[np.float64]
    fractional_film_content: NDArray[np.float64]
    film_thickness_m: NDArray[np.float64]
    elastic_displacement_m: NDArray[np.float64]
    central_film_offset_m: float
    compliance_factor: float
    load_n: float
    history: tuple[CavitatingCoupledIteration, ...]


class CavitatingCouplingError(RuntimeError):
    def __init__(self, message, history, last_solution):
        super().__init__(message)
        self.history = tuple(history)
        self.last_solution = last_solution


def solve_cavitating_elasticity(
    problem: CavitatingElasticityProblem,
    *,
    continuation_factors=(0.0, 0.25, 0.5, 0.75, 1.0),
    film_relaxation=0.5,
    update_tolerance=1e-8,
    mass_tolerance=1e-8,
    admissibility_tolerance=1e-10,
    maximum_iterations_per_stage=100,
    maximum_active_set_iterations=100,
) -> CavitatingElasticitySolution:
    """Reference fixed-point coupling at prescribed h0; no load root is solved.

    Only the film iterate is relaxed. Each pressure/saturation pair is from
    one JFO solve, preserving its complementarity. Convergence checks use the
    exact film h=h0+geometry+gamma*w(p), not the old inner-solve film. Require
    both update/film defects and mass/admissibility residuals to pass.
    Results contain SI p [Pa], h/w [m], force [N] and dimensionless theta.
    Failed iterations preserve the last candidate and complete history; a
    last candidate carried by an exception is not a converged solution.
    """
    if (
        not continuation_factors
        or continuation_factors[0] != 0.0
        or any(not math.isfinite(g) or not 0 <= g <= 1 for g in continuation_factors)
        or any(b <= a for a, b in zip(continuation_factors, continuation_factors[1:], strict=False))
    ):
        raise ValueError("continuation_factors must start at zero and increase strictly in [0,1]")
    if not math.isfinite(film_relaxation) or not 0 < film_relaxation <= 1:
        raise ValueError("film_relaxation must lie in (0,1]")
    if not all(
        math.isfinite(t) and t > 0
        for t in (update_tolerance, mass_tolerance, admissibility_tolerance)
    ):
        raise ValueError("nonlinear tolerances must be finite and positive")
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1
        for n in (maximum_iterations_per_stage, maximum_active_set_iterations)
    ):
        raise ValueError("iteration limits must be positive integers")
    p = np.zeros(problem.shape)
    film, _ = problem.fields(p, compliance_factor=0)
    history, last = [], None
    for gamma in continuation_factors:
        for iteration in range(1, maximum_iterations_per_stage + 1):
            try:
                inner = solve_cavitating_reynolds(
                    film / problem.film_scale_m,
                    problem.dx,
                    problem.dy,
                    problem.reynolds_parameter,
                    pressure_tolerance=admissibility_tolerance,
                    saturation_tolerance=admissibility_tolerance,
                    max_active_set_iterations=maximum_active_set_iterations,
                    inlet_fractional_film_content=problem.inlet_fractional_film_content,
                )
                candidate_p = inner.pressure * problem.pressure_scale_pa
                exact_film, displacement = problem.fields(candidate_p, compliance_factor=gamma)
                metrics = problem.metrics(candidate_p, inner.fractional_film_content, exact_film)
            except (RuntimeError, ValueError) as error:
                raise CavitatingCouplingError(str(error), history, last) from error
            p_update = float(
                np.linalg.norm(candidate_p - p)
                / max(np.linalg.norm(candidate_p), np.finfo(float).tiny)
            )
            film_defect = float(np.linalg.norm(exact_film - film) / np.linalg.norm(exact_film))
            history.append(
                CavitatingCoupledIteration(
                    gamma,
                    iteration,
                    p_update,
                    film_defect,
                    float(exact_film.min()),
                    problem.load_n(candidate_p),
                    metrics,
                    inner.active_set_iterations,
                )
            )
            last = CavitatingElasticitySolution(
                candidate_p,
                inner.fractional_film_content,
                exact_film,
                displacement,
                problem.central_film_offset_m,
                gamma,
                problem.load_n(candidate_p),
                tuple(history),
            )
            if (
                max(p_update, film_defect) <= update_tolerance
                and metrics.mass_rms <= mass_tolerance
                and metrics.mass_maximum <= mass_tolerance
                and metrics.maximum_relative_local_balance <= mass_tolerance
                and max(
                    metrics.maximum_complementarity,
                    metrics.pressure_violation,
                    metrics.saturation_violation,
                    metrics.inlet_saturation_error,
                )
                <= admissibility_tolerance
                and metrics.boundary_pressure_max_pa == 0.0
            ):
                p, film = candidate_p, exact_film
                break
            p = candidate_p
            film = film + film_relaxation * (exact_film - film)
        else:
            raise CavitatingCouplingError(
                f"cavitation/elasticity stage gamma={gamma} exhausted; metrics={history[-1]}",
                history,
                last,
            )
    return last
