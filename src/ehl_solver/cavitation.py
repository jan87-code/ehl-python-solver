"""Mass-conserving stationary Reynolds cavitation using JFO complementarity."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ehl_solver.reynolds import assemble_reynolds_system


@dataclass(frozen=True)
class CavitatingReynoldsSolution:
    """Pressure and fractional film content from a stationary JFO solve.

    Pressure is dimensionless, and `fractional_film_content` is the
    dimensionless saturation field in [0, 1]. Both arrays use shape (ny, nx),
    with x along entrainment. Boundary pressure is zero.
    """

    pressure: NDArray[np.float64]
    fractional_film_content: NDArray[np.float64]
    active_set_iterations: int
    maximum_mass_balance_residual: float
    rms_mass_balance_residual: float
    maximum_complementarity_violation: float
    west_boundary_flux: float
    east_boundary_flux: float
    south_boundary_flux: float
    north_boundary_flux: float
    inlet_fractional_film_content: float

    @property
    def net_boundary_flux(self) -> float:
        """Return total outward dimensionless mass flux."""
        return (
            self.west_boundary_flux
            + self.east_boundary_flux
            + self.south_boundary_flux
            + self.north_boundary_flux
        )


def solve_cavitating_reynolds(
    film: ArrayLike,
    dx: float,
    dy: float,
    reynolds_parameter: float,
    *,
    pressure_tolerance: float = 1e-10,
    saturation_tolerance: float = 1e-10,
    max_active_set_iterations: int = 100,
    inlet_fractional_film_content: float = 1.0,
    equilibrate_linear_system: bool = True,
) -> CavitatingReynoldsSolution:
    """Solve stationary incompressible JFO Reynolds flow for a prescribed film.

    The complementarity conditions are ``P >= 0``, ``0 <= theta <= 1``, and
    ``P * (1 - theta) = 0``. The dimensionless mass flux is
    ``q = -H**3 grad(P) + Lambda * H * theta * e_x``. Pressure diffusion uses
    the same arithmetic face conductance and homogeneous Dirichlet pressure
    boundaries as :func:`ehl_solver.reynolds.solve_reynolds_pressure`.

    Saturation is upwinded in the positive x direction while H retains the
    validated arithmetic face average. Consequently, theta=1 reproduces the
    existing full-film Reynolds stencil exactly. The west inlet has uniform
    prescribed ``inlet_fractional_film_content`` in [0, 1], defaulting to a
    fully supplied value of 1; values below 1 provide a simple
    inlet-starvation boundary condition. The east is an outflow, and there
    is no transverse advective saturation flux. This is a stationary, isothermal,
    incompressible JFO/Elrod-Adams mass balance with a fixed cavitation
    pressure of zero gauge. It does not model dissolved gas, transient squeeze
    flow, lateral inlet variation, or elastic/load coupling.

    An active-set method solves the linear mass balances and pressure/
    saturation constraints as one sparse system per active set. Invalid
    saturation or a cycling/nonconvergent active set raises ``RuntimeError``;
    pressure and saturation are never clipped.

    By default, positive diagonal row/column equilibration balances the
    sparse mass/constraint system before direct solution. This changes only
    linear algebra coordinates, not the stencil, active-set tolerances or
    returned physical variables. False retains the unscaled reference path.
    Reported mass residuals always use the original unscaled face fluxes.
    """
    h = np.asarray(film, dtype=np.float64)
    if not isinstance(equilibrate_linear_system, bool):
        raise ValueError("equilibrate_linear_system must be a boolean")
    if h.ndim != 2 or min(h.shape) < 3:
        raise ValueError("film must be a 2-D array with at least 3 points per axis")
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        raise ValueError("film must contain only finite, strictly positive values")
    if not math.isfinite(dx) or dx <= 0.0 or not math.isfinite(dy) or dy <= 0.0:
        raise ValueError("dx and dy must be finite and positive")
    if not math.isfinite(reynolds_parameter) or reynolds_parameter <= 0.0:
        raise ValueError("reynolds_parameter must be finite and positive for cavitating flow")
    if (
        not math.isfinite(pressure_tolerance)
        or pressure_tolerance <= 0.0
        or not math.isfinite(saturation_tolerance)
        or saturation_tolerance <= 0.0
    ):
        raise ValueError("active-set tolerances must be finite and positive")
    if (
        not isinstance(max_active_set_iterations, int)
        or isinstance(max_active_set_iterations, bool)
        or max_active_set_iterations < 1
    ):
        raise ValueError("max_active_set_iterations must be a positive integer")
    if (
        not math.isfinite(inlet_fractional_film_content)
        or not 0.0 <= inlet_fractional_film_content <= 1.0
    ):
        raise ValueError("inlet_fractional_film_content must be finite and in [0, 1]")

    full_matrix, _ = assemble_reynolds_system(h, dx, dy, reynolds_parameter)
    ny, nx = h.shape
    interior_nx = nx - 2
    interior_ny = ny - 2
    count = interior_nx * interior_ny
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    known_inlet_flux = np.zeros(count, dtype=np.float64)
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            row = (j - 1) * interior_nx + i - 1
            h_east = 0.5 * (h[j, i] + h[j, i + 1])
            h_west = 0.5 * (h[j, i - 1] + h[j, i])
            rows.append(row)
            columns.append(row)
            values.append(reynolds_parameter * h_east / dx)
            if i == 1:
                known_inlet_flux[row] = (
                    -reynolds_parameter
                    * h_west
                    * inlet_fractional_film_content
                    / dx
                )
            else:
                west = (j - 1) * interior_nx + i - 2
                rows.append(row)
                columns.append(west)
                values.append(-reynolds_parameter * h_west / dx)
    saturation_operator = csr_matrix(
        (values, (rows, columns)), shape=(count, count), dtype=np.float64
    )
    mass_matrix = hstack((full_matrix, saturation_operator), format="csr")
    full_active = np.ones(count, dtype=bool)
    seen_active_sets: set[bytes] = set()

    for active_iteration in range(1, max_active_set_iterations + 1):
        active_key = np.packbits(full_active).tobytes()
        if active_key in seen_active_sets:
            raise RuntimeError("cavitation active set cycled without convergence")
        seen_active_sets.add(active_key)

        cavitated = ~full_active
        pressure_constraints = diags(cavitated.astype(np.float64), format="csr")
        saturation_constraints = diags(full_active.astype(np.float64), format="csr")
        constraints = hstack(
            (pressure_constraints, saturation_constraints), format="csr"
        )
        system = vstack((mass_matrix, constraints), format="csr")
        constraint_rhs = np.where(full_active, 1.0, 0.0)
        rhs = np.concatenate((-known_inlet_flux, constraint_rhs))
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                state = _solve_active_system(system, rhs, equilibrate_linear_system)
            except (MatrixRankWarning, RuntimeError) as error:
                raise RuntimeError(
                    "cavitation active-set linear system is singular or unusable"
                ) from error
        if state.shape != (2 * count,) or not np.all(np.isfinite(state)):
            raise RuntimeError("cavitation active-set solve returned a non-finite state")

        pressure_interior = state[:count]
        saturation_interior = state[count:]
        invalid_pressure = full_active & (pressure_interior < -pressure_tolerance)
        invalid_saturation = cavitated & (
            saturation_interior > 1.0 + saturation_tolerance
        )
        if np.any(cavitated & (saturation_interior < -saturation_tolerance)):
            raise RuntimeError(
                "cavitated solution has negative fractional film content; "
                "the fully supplied inlet assumption is insufficient"
            )
        if np.any(invalid_pressure) or np.any(invalid_saturation):
            full_active[invalid_pressure] = False
            full_active[invalid_saturation] = True
            continue

        pressure = np.zeros(h.shape, dtype=np.float64)
        saturation = np.ones(h.shape, dtype=np.float64)
        pressure[1:-1, 1:-1] = pressure_interior.reshape(interior_ny, interior_nx)
        saturation[1:-1, 1:-1] = saturation_interior.reshape(interior_ny, interior_nx)
        saturation[1:-1, 0] = inlet_fractional_film_content
        saturation[1:-1, -1] = saturation[1:-1, -2]
        saturation[0, :] = saturation[1, :]
        saturation[-1, :] = saturation[-2, :]
        residual, fluxes = _mass_balance(h, pressure, saturation, dx, dy, reynolds_parameter)
        complementarity = pressure[1:-1, 1:-1] * (
            1.0 - saturation[1:-1, 1:-1]
        )
        return CavitatingReynoldsSolution(
            pressure=pressure,
            fractional_film_content=saturation,
            active_set_iterations=active_iteration,
            maximum_mass_balance_residual=float(np.max(np.abs(residual))),
            rms_mass_balance_residual=float(np.sqrt(np.mean(residual**2))),
            maximum_complementarity_violation=float(np.max(np.abs(complementarity))),
            west_boundary_flux=fluxes[0],
            east_boundary_flux=fluxes[1],
            south_boundary_flux=fluxes[2],
            north_boundary_flux=fluxes[3],
            inlet_fractional_film_content=inlet_fractional_film_content,
        )

    raise RuntimeError(
        "cavitation active set failed to converge within "
        f"{max_active_set_iterations} iterations"
    )


def _solve_active_system(system, rhs, equilibrate):
    """Solve A z=b using R A C u=R b, then recover z=C u in float64.

    R normalizes absolute row maxima; C then normalizes column maxima of
    R A. Both diagonals are strictly positive and finite. No perturbation,
    regularization, clipping or silent unscaled fallback is applied.
    """
    if not equilibrate:
        return np.asarray(spsolve(system, rhs), dtype=np.float64)
    row_maximum = abs(system).max(axis=1).toarray().ravel()
    if np.any(row_maximum <= 0) or not np.all(np.isfinite(row_maximum)):
        raise RuntimeError("active-set matrix has a zero or nonfinite row")
    row_scale = 1.0 / row_maximum
    scaled = diags(row_scale) @ system
    column_maximum = abs(scaled).max(axis=0).toarray().ravel()
    if np.any(column_maximum <= 0) or not np.all(np.isfinite(column_maximum)):
        raise RuntimeError("active-set matrix has a zero or nonfinite column")
    column_scale = 1.0 / column_maximum
    if not np.all(np.isfinite(row_scale)) or not np.all(np.isfinite(column_scale)):
        raise RuntimeError("active-set equilibration scales are nonfinite")
    state = spsolve(scaled @ diags(column_scale), row_scale * rhs)
    return np.asarray(column_scale * state, dtype=np.float64)


def _mass_balance(
    film: NDArray[np.float64],
    pressure: NDArray[np.float64],
    saturation: NDArray[np.float64],
    dx: float,
    dy: float,
    reynolds_parameter: float,
) -> tuple[NDArray[np.float64], tuple[float, float, float, float]]:
    """Compute conservative cell balances and outward boundary fluxes."""
    h3 = film**3
    h_center = film[1:-1, 1:-1]
    theta_center = saturation[1:-1, 1:-1]
    pressure_center = pressure[1:-1, 1:-1]
    h_east = 0.5 * (h_center + film[1:-1, 2:])
    h_west = 0.5 * (h_center + film[1:-1, :-2])
    theta_west = saturation[1:-1, :-2]
    q_east = (
        -0.5 * (h3[1:-1, 1:-1] + h3[1:-1, 2:])
        * (pressure[1:-1, 2:] - pressure_center)
        / dx
        + reynolds_parameter * h_east * theta_center
    )
    q_west = (
        -0.5 * (h3[1:-1, 1:-1] + h3[1:-1, :-2])
        * (pressure_center - pressure[1:-1, :-2])
        / dx
        + reynolds_parameter * h_west * theta_west
    )
    q_north = (
        -0.5 * (h3[1:-1, 1:-1] + h3[2:, 1:-1])
        * (pressure[2:, 1:-1] - pressure_center)
        / dy
    )
    q_south = (
        -0.5 * (h3[1:-1, 1:-1] + h3[:-2, 1:-1])
        * (pressure_center - pressure[:-2, 1:-1])
        / dy
    )
    residual = (q_east - q_west) / dx + (q_north - q_south) / dy
    fluxes = (
        -float(np.sum(q_west[:, 0])) * dy,
        float(np.sum(q_east[:, -1])) * dy,
        -float(np.sum(q_south[0, :])) * dx,
        float(np.sum(q_north[-1, :])) * dx,
    )
    return residual, fluxes
