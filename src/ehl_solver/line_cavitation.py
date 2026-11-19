"""Prescribed-film, stationary one-dimensional line-contact JFO flow."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, diags, hstack, vstack
from scipy.sparse.linalg import MatrixRankWarning

from ehl_solver.cavitation import _solve_active_system


@dataclass(frozen=True)
class CavitatingLineReynoldsSolution:
    """Dimensionless pressure and fractional film content on an x line."""

    pressure: NDArray[np.float64]
    fractional_film_content: NDArray[np.float64]
    active_set_iterations: int
    maximum_mass_balance_residual: float
    rms_mass_balance_residual: float
    maximum_complementarity_violation: float
    west_boundary_flux: float
    east_boundary_flux: float
    inlet_fractional_film_content: float

    @property
    def net_boundary_flux(self) -> float:
        """Return the sum of outward dimensionless boundary fluxes."""
        return self.west_boundary_flux + self.east_boundary_flux


def solve_cavitating_line_reynolds(
    film: ArrayLike,
    dx: float,
    reynolds_parameter: float,
    *,
    pressure_tolerance: float = 1e-10,
    saturation_tolerance: float = 1e-10,
    max_active_set_iterations: int = 100,
    inlet_fractional_film_content: float = 1.0,
    equilibrate_linear_system: bool = True,
) -> CavitatingLineReynoldsSolution:
    """Solve stationary 1-D JFO flow for a prescribed dimensionless film.

    ``film`` is ``H=h/h_ref`` on uniform dimensionless ``X=x/L`` nodes, and
    ``dx`` is ``dX``. Pressure is ``P=p/p_ref`` and the returned arrays have
    shape ``(nx,)``. The dimensionless flux per unit axial width is
    ``q=-H**3*dP/dX + Lambda*H*theta``, with ``Lambda`` equal to
    ``12*eta0*Ue*L/(h_ref**2*p_ref)``. Pressure is zero at both x boundaries;
    the west boundary has prescribed inlet saturation and the east is an
    outflow. Saturation is first-order upwinded for positive entrainment.

    The complementarity conditions are ``P>=0``, ``0<=theta<=1`` and
    ``P*(1-theta)=0``. The sparse active-set solve never clips pressure or
    saturation. This is a prescribed-film flow benchmark; it does not solve
    elastic film coupling or load closure.
    """
    h = np.asarray(film, dtype=np.float64)
    if h.ndim != 1 or h.size < 3:
        raise ValueError("film must be a 1-D array with at least 3 x nodes")
    if not np.all(np.isfinite(h)) or np.any(h <= 0.0):
        raise ValueError("film must contain only finite, strictly positive values")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be finite and positive")
    if not math.isfinite(reynolds_parameter) or reynolds_parameter <= 0.0:
        raise ValueError("reynolds_parameter must be finite and positive")
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
    if not isinstance(equilibrate_linear_system, bool):
        raise ValueError("equilibrate_linear_system must be a boolean")

    interior_count = h.size - 2
    h_cubed = h**3
    rows: list[int] = []
    columns: list[int] = []
    diffusion_values: list[float] = []
    transport_rows: list[int] = []
    transport_columns: list[int] = []
    transport_values: list[float] = []
    known_inlet_flux = np.zeros(interior_count, dtype=np.float64)
    lam = reynolds_parameter

    for i in range(1, h.size - 1):
        row = i - 1
        h_east = 0.5 * (h[i] + h[i + 1])
        h_west = 0.5 * (h[i - 1] + h[i])
        diffusion_east = 0.5 * (h_cubed[i] + h_cubed[i + 1]) / dx**2
        diffusion_west = 0.5 * (h_cubed[i - 1] + h_cubed[i]) / dx**2
        rows.append(row)
        columns.append(row)
        diffusion_values.append(diffusion_east + diffusion_west)
        transport_rows.append(row)
        transport_columns.append(row)
        transport_values.append(lam * h_east / dx)

        if i < h.size - 2:
            rows.append(row)
            columns.append(row + 1)
            diffusion_values.append(-diffusion_east)
        if i > 1:
            rows.append(row)
            columns.append(row - 1)
            diffusion_values.append(-diffusion_west)
            transport_rows.append(row)
            transport_columns.append(row - 1)
            transport_values.append(-lam * h_west / dx)
        else:
            known_inlet_flux[row] = (
                -lam * h_west * inlet_fractional_film_content / dx
            )

    shape = (interior_count, interior_count)
    diffusion_matrix = coo_matrix(
        (diffusion_values, (rows, columns)), shape=shape, dtype=np.float64
    ).tocsr()

    transport_matrix = coo_matrix(
        (transport_values, (transport_rows, transport_columns)),
        shape=shape,
        dtype=np.float64,
    ).tocsr()
    mass_matrix = hstack((diffusion_matrix, transport_matrix), format="csr")
    full_film = np.ones(interior_count, dtype=bool)
    seen_active_sets: set[bytes] = set()

    for active_iteration in range(1, max_active_set_iterations + 1):
        active_key = np.packbits(full_film).tobytes()
        if active_key in seen_active_sets:
            raise RuntimeError("line cavitation active set cycled without convergence")
        seen_active_sets.add(active_key)

        cavitated = ~full_film
        constraints = hstack(
            (
                diags(cavitated.astype(np.float64), format="csr"),
                diags(full_film.astype(np.float64), format="csr"),
            ),
            format="csr",
        )
        system = vstack((mass_matrix, constraints), format="csr")
        constraint_rhs = np.where(full_film, 1.0, 0.0)
        rhs = np.concatenate((-known_inlet_flux, constraint_rhs))
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                state = _solve_active_system(system, rhs, equilibrate_linear_system)
            except (MatrixRankWarning, RuntimeError) as error:
                raise RuntimeError(
                    "line cavitation active-set linear system is singular or unusable"
                ) from error
        if state.shape != (2 * interior_count,) or not np.all(np.isfinite(state)):
            raise RuntimeError("line cavitation solve returned a non-finite state")

        pressure_interior = state[:interior_count]
        saturation_interior = state[interior_count:]
        invalid_pressure = full_film & (pressure_interior < -pressure_tolerance)
        invalid_saturation = cavitated & (
            saturation_interior > 1.0 + saturation_tolerance
        )
        if np.any(cavitated & (saturation_interior < -saturation_tolerance)):
            raise RuntimeError(
                "line cavitated solution has negative fractional film content; "
                "the inlet supply assumption is insufficient"
            )
        if np.any(invalid_pressure) or np.any(invalid_saturation):
            full_film[invalid_pressure] = False
            full_film[invalid_saturation] = True
            continue

        pressure = np.zeros(h.size, dtype=np.float64)
        saturation = np.ones(h.size, dtype=np.float64)
        pressure[1:-1] = pressure_interior
        saturation[0] = inlet_fractional_film_content
        saturation[1:-1] = saturation_interior
        saturation[-1] = saturation[-2]
        residual, boundary_fluxes = _line_mass_balance(
            h, pressure, saturation, dx, lam
        )
        complementarity = pressure[1:-1] * (1.0 - saturation[1:-1])
        return CavitatingLineReynoldsSolution(
            pressure=pressure,
            fractional_film_content=saturation,
            active_set_iterations=active_iteration,
            maximum_mass_balance_residual=float(np.max(np.abs(residual))),
            rms_mass_balance_residual=float(np.sqrt(np.mean(residual**2))),
            maximum_complementarity_violation=float(np.max(np.abs(complementarity))),
            west_boundary_flux=boundary_fluxes[0],
            east_boundary_flux=boundary_fluxes[1],
            inlet_fractional_film_content=inlet_fractional_film_content,
        )

    raise RuntimeError(
        "line cavitation active set failed to converge within "
        f"{max_active_set_iterations} iterations"
    )


def _line_mass_balance(
    film: NDArray[np.float64],
    pressure: NDArray[np.float64],
    saturation: NDArray[np.float64],
    dx: float,
    reynolds_parameter: float,
) -> tuple[NDArray[np.float64], tuple[float, float]]:
    """Recompute original dimensionless face fluxes and cell residuals."""
    h_face = 0.5 * (film[:-1] + film[1:])
    h_cubed_face = 0.5 * (film[:-1] ** 3 + film[1:] ** 3)
    face_flux = (
        -h_cubed_face * np.diff(pressure) / dx
        + reynolds_parameter * h_face * saturation[:-1]
    )
    residual = np.diff(face_flux) / dx
    return residual, (-float(face_flux[0]), float(face_flux[-1]))
