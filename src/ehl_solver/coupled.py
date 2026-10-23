"""Stationary full-film Reynolds-elasticity fixed-point coupling."""

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.reynolds import reynolds_residual, solve_reynolds_pressure


@dataclass(frozen=True)
class CoupledIteration:
    """Convergence diagnostics for one outer nonlinear iteration."""

    global_iteration: int
    compliance_factor: float
    iteration: int
    pressure_relaxation: float
    film_relaxation: float
    pressure_update_relative: float
    film_update_relative: float
    reynolds_residual_rms: float
    reynolds_residual_relative: float
    minimum_film_thickness_m: float
    minimum_pressure_pa: float
    maximum_pressure_pa: float
    linear_iterations: int


@dataclass(frozen=True)
class CoupledSnapshot:
    """Copied pressure and film fields from one accepted outer iteration."""

    global_iteration: int
    compliance_factor: float
    iteration: int
    pressure_pa: NDArray[np.float64]
    film_thickness_m: NDArray[np.float64]


@dataclass(frozen=True)
class CoupledReynoldsSolution:
    """Converged dimensionless pressure and dimensional coupled film."""

    pressure_dimensionless: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    film_thickness_m: NDArray[np.float64]
    elastic_displacement_m: NDArray[np.float64]
    history: tuple[CoupledIteration, ...]
    snapshots: tuple[CoupledSnapshot, ...]


def solve_coupled_reynolds_elasticity(
    x_m: ArrayLike,
    y_m: ArrayLike,
    central_film_offset_m: float,
    radius_x_m: float,
    radius_y_m: float,
    e_star_pa: float,
    length_scale_m: float,
    film_scale_m: float,
    pressure_scale_pa: float,
    reynolds_parameter: float,
    *,
    pressure_relaxation: float = 0.5,
    film_relaxation: float = 0.5,
    pressure_tolerance: float = 1e-6,
    film_tolerance: float = 1e-6,
    reynolds_tolerance: float = 1e-8,
    max_iterations_per_stage: int = 100,
    continuation_factors: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    inner_rtol: float = 1e-10,
    inner_atol: float = 1e-12,
    inner_max_iterations: int | None = None,
    log_every: int = 10,
    plot_every: int | None = None,
    snapshot_iterations: tuple[int, ...] | None = None,
) -> CoupledReynoldsSolution:
    """Solve a prescribed-offset Reynolds-elasticity problem in SI units.

    ``x_m`` and ``y_m`` are uniformly spaced, strictly ascending 1-D
    coordinates in metres; x is the entrainment direction. The base rigid gap
    is ``h0 + x**2/(2*Rx) + y**2/(2*Ry)`` and elastic displacement is added
    to the gap: ``h = rigid_gap + gamma * w``. Here ``h0`` is the prescribed
    central rigid-gap offset, not a load-controlled unknown. Pressure in Pa
    is converted to a positive gap-increasing displacement in metres by the
    existing cell-integrated half-space kernel, with
    ``1/E* = sum((1-nu_i**2)/E_i)`` as documented by the Hertz convention.

    The Reynolds equation uses ``H=h/film_scale_m``,
    ``P=p/pressure_scale_pa``, dimensionless spacings
    ``dx=delta_x/length_scale_m`` and ``dy=delta_y/length_scale_m``, and
    ``Lambda=12*eta0*Ue*L/(h_ref**2*p_ref)`` with mean surface speed
    ``Ue=(u1+u2)/2``. The caller supplies this already-computed dimensionless
    parameter. Homogeneous zero-pressure Dirichlet boundaries and the
    unconstrained full-film formulation are preserved; pressure is not clipped.

    A relaxed Picard iteration calls the existing conjugate-gradient
    Reynolds solver at each step. Compliance continuation runs the listed
    factors in ascending order from zero toward (and by default through) one,
    warm-starting pressure and film at each stage. Independent convergence
    checks cover relative RMS pressure update, relative RMS film update and a
    normalized Reynolds-equation residual. A small update alone is never
    considered convergence.
    Every proposed film must remain finite and strictly positive; otherwise
    the solve fails without clipping or modifying it. Exhausting a stage's
    iteration limit raises ``RuntimeError`` with the final residual metrics.

    Snapshot collection is disabled by default. Set ``plot_every`` to copy
    every Nth accepted global outer iterate, or provide specific accepted
    ``snapshot_iterations`` (global, one-based). If both are given, their
    selections are combined. Snapshots contain copied dimensional pressure
    [Pa] and film thickness [m], and do not require or import plotting code.

    No load controller, cavitation, pressure-dependent material properties,
    thermal physics, or GPU backend is included.
    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 3:
        raise ValueError("x_m and y_m must be 1-D coordinate arrays with at least 3 points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinates must contain only finite values")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("coordinates must be strictly ascending")
    dx_m = float(np.diff(x)[0])
    dy_m = float(np.diff(y)[0])
    x_spacing_atol = 32 * np.finfo(np.float64).eps * max(
        abs(dx_m), float(np.max(np.abs(x))), np.finfo(np.float64).tiny
    )
    y_spacing_atol = 32 * np.finfo(np.float64).eps * max(
        abs(dy_m), float(np.max(np.abs(y))), np.finfo(np.float64).tiny
    )
    if not np.allclose(np.diff(x), dx_m, rtol=1e-10, atol=x_spacing_atol):
        raise ValueError("x_m must be uniformly spaced")
    if not np.allclose(np.diff(y), dy_m, rtol=1e-10, atol=y_spacing_atol):
        raise ValueError("y_m must be uniformly spaced")

    positive_parameters = (
        central_film_offset_m,
        radius_x_m,
        radius_y_m,
        e_star_pa,
        length_scale_m,
        film_scale_m,
        pressure_scale_pa,
    )
    if not all(math.isfinite(value) and value > 0 for value in positive_parameters):
        raise ValueError("film offset, radii, modulus and scales must be finite and positive")
    if not math.isfinite(reynolds_parameter) or reynolds_parameter < 0:
        raise ValueError("reynolds_parameter must be finite and nonnegative")
    if not 0 < pressure_relaxation <= 1 or not 0 < film_relaxation <= 1:
        raise ValueError("pressure and film relaxation factors must be in (0, 1]")
    tolerances = (pressure_tolerance, film_tolerance, reynolds_tolerance)
    if not all(math.isfinite(value) and value > 0 for value in tolerances):
        raise ValueError("nonlinear tolerances must be finite and positive")
    if (
        not isinstance(max_iterations_per_stage, int)
        or isinstance(max_iterations_per_stage, bool)
        or max_iterations_per_stage < 1
    ):
        raise ValueError("max_iterations_per_stage must be a positive integer")
    if not isinstance(log_every, int) or isinstance(log_every, bool) or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    if plot_every is not None and (
        not isinstance(plot_every, int) or isinstance(plot_every, bool) or plot_every < 1
    ):
        raise ValueError("plot_every must be a positive integer or None")
    if snapshot_iterations is not None and any(
        not isinstance(step, int) or isinstance(step, bool) or step < 1
        for step in snapshot_iterations
    ):
        raise ValueError("snapshot_iterations must contain positive global iteration numbers")
    if snapshot_iterations is not None and len(set(snapshot_iterations)) != len(
        snapshot_iterations
    ):
        raise ValueError("snapshot_iterations must not contain duplicates")
    selected_iterations = set(snapshot_iterations or ())
    if (
        not continuation_factors
        or continuation_factors[0] != 0.0
        or any(
            not math.isfinite(factor) or factor < 0 or factor > 1
            for factor in continuation_factors
        )
        or any(
            next_factor <= factor
            for factor, next_factor in zip(
                continuation_factors, continuation_factors[1:], strict=False
            )
        )
    ):
        raise ValueError(
            "continuation_factors must start at 0.0 and increase strictly up to 1.0"
        )
    if not math.isfinite(inner_rtol) or inner_rtol < 0:
        raise ValueError("inner_rtol must be finite and nonnegative")
    if not math.isfinite(inner_atol) or inner_atol < 0:
        raise ValueError("inner_atol must be finite and nonnegative")
    if inner_rtol == 0 and inner_atol == 0:
        raise ValueError("at least one inner linear tolerance must be positive")

    xx, yy = np.meshgrid(x, y, indexing="xy")
    rigid_gap_m = (
        central_film_offset_m + xx**2 / (2.0 * radius_x_m) + yy**2 / (2.0 * radius_y_m)
    )
    if np.any(~np.isfinite(rigid_gap_m)) or np.any(rigid_gap_m <= 0):
        raise ValueError("rigid reference gap must remain finite and strictly positive")

    grid_shape = rigid_gap_m.shape
    dimensionless_dx = dx_m / length_scale_m
    dimensionless_dy = dy_m / length_scale_m
    backend = make_backend("numpy")
    elastic_kernel = cell_integrated_kernel(
        grid_shape, dx_m, dy_m, e_star_pa, backend
    )
    pressure = np.zeros(grid_shape, dtype=np.float64)
    film_m = rigid_gap_m.copy()
    history: list[CoupledIteration] = []
    snapshots: list[CoupledSnapshot] = []
    global_iteration = 0
    logger = logging.getLogger(__name__)

    for compliance_factor in continuation_factors:
        for iteration in range(1, max_iterations_per_stage + 1):
            film_dimensionless = film_m / film_scale_m
            reynolds_solution = solve_reynolds_pressure(
                film_dimensionless,
                dimensionless_dx,
                dimensionless_dy,
                reynolds_parameter,
                rtol=inner_rtol,
                atol=inner_atol,
                max_iterations=inner_max_iterations,
                initial_pressure=pressure,
            )
            pressure_target = reynolds_solution.pressure
            pressure_new = pressure + pressure_relaxation * (pressure_target - pressure)
            pressure_pa = pressure_new * pressure_scale_pa
            displacement_m = np.asarray(
                fft_convolve_same(pressure_pa, elastic_kernel, backend),
                dtype=np.float64,
            )
            film_target_m = rigid_gap_m + compliance_factor * displacement_m
            if np.any(~np.isfinite(film_target_m)) or np.any(film_target_m <= 0):
                raise RuntimeError(
                    "elastic coupling proposed a non-finite or nonpositive film thickness"
                )
            film_new_m = film_m + film_relaxation * (film_target_m - film_m)
            if np.any(~np.isfinite(film_new_m)) or np.any(film_new_m <= 0):
                raise RuntimeError(
                    "relaxed coupled iterate has a non-finite or nonpositive film thickness"
                )

            pressure_update = float(
                np.linalg.norm(pressure_new - pressure)
                / max(np.linalg.norm(pressure_new), np.finfo(np.float64).tiny)
            )
            film_update = float(
                np.linalg.norm((film_new_m - film_m) / film_scale_m)
                / max(
                    np.linalg.norm(film_new_m / film_scale_m),
                    np.finfo(np.float64).tiny,
                )
            )
            film_new_dimensionless = film_new_m / film_scale_m
            equation_residual = reynolds_residual(
                pressure_new,
                film_new_dimensionless,
                dimensionless_dx,
                dimensionless_dy,
                reynolds_parameter,
            )
            forcing = (
                reynolds_parameter
                * (
                    film_new_dimensionless[1:-1, 2:]
                    - film_new_dimensionless[1:-1, :-2]
                )
                / (2.0 * dimensionless_dx)
            )
            residual_rms = float(np.sqrt(np.mean(equation_residual**2)))
            forcing_rms = float(np.sqrt(np.mean(forcing**2)))
            diffusion_rms = float(np.sqrt(np.mean((equation_residual + forcing) ** 2)))
            residual_relative = residual_rms / max(
                forcing_rms + diffusion_rms, np.finfo(np.float64).tiny
            )
            global_iteration += 1
            entry = CoupledIteration(
                global_iteration=global_iteration,
                compliance_factor=compliance_factor,
                iteration=iteration,
                pressure_relaxation=pressure_relaxation,
                film_relaxation=film_relaxation,
                pressure_update_relative=pressure_update,
                film_update_relative=film_update,
                reynolds_residual_rms=residual_rms,
                reynolds_residual_relative=residual_relative,
                minimum_film_thickness_m=float(np.min(film_new_m)),
                minimum_pressure_pa=float(np.min(pressure_pa)),
                maximum_pressure_pa=float(np.max(pressure_pa)),
                linear_iterations=reynolds_solution.iterations,
            )
            history.append(entry)
            if iteration == 1 or iteration % log_every == 0:
                logger.info(
                    "Reynolds-elasticity gamma=%.3f iter=%d "
                    "dP=%.3e dH=%.3e R=%.3e (relative %.3e) min_h=%.6e m",
                    compliance_factor,
                    iteration,
                    pressure_update,
                    film_update,
                    residual_rms,
                    residual_relative,
                    entry.minimum_film_thickness_m,
                )

            pressure = pressure_new
            film_m = film_new_m
            if (
                (plot_every is not None and global_iteration % plot_every == 0)
                or global_iteration in selected_iterations
            ):
                snapshots.append(
                    CoupledSnapshot(
                        global_iteration=global_iteration,
                        compliance_factor=compliance_factor,
                        iteration=iteration,
                        pressure_pa=pressure_pa.copy(),
                        film_thickness_m=film_m.copy(),
                    )
                )
            if (
                pressure_update <= pressure_tolerance
                and film_update <= film_tolerance
                and residual_relative <= reynolds_tolerance
            ):
                logger.info(
                    "Reynolds-elasticity continuation stage gamma=%.3f converged "
                    "in %d iterations",
                    compliance_factor,
                    iteration,
                )
                break
        else:
            raise RuntimeError(
                "Reynolds-elasticity iteration failed at "
                f"compliance_factor={compliance_factor:.3f} after "
                f"{max_iterations_per_stage} iterations: "
                f"pressure_update={pressure_update:.3e}, "
                f"film_update={film_update:.3e}, "
                f"reynolds_residual_relative={residual_relative:.3e}"
            )

    pressure_pa = pressure * pressure_scale_pa
    displacement_m = np.asarray(
        fft_convolve_same(pressure_pa, elastic_kernel, backend),
        dtype=np.float64,
    )
    return CoupledReynoldsSolution(
        pressure_dimensionless=pressure,
        pressure_pa=pressure_pa,
        film_thickness_m=film_m,
        elastic_displacement_m=displacement_m,
        history=tuple(history),
        snapshots=tuple(snapshots),
    )
