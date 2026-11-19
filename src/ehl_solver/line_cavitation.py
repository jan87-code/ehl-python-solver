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
from ehl_solver.constitutive import IsothermalFluid


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


@dataclass(frozen=True)
class VariablePropertyLineReynoldsSolution:
    """SI pressure/content root and audit for variable-property line flow."""

    pressure_pa: NDArray[np.float64]
    fractional_film_content: NDArray[np.float64]
    audit: dict[str, float]
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class LineMassFluxComponents:
    """Dimensional mass-flux components on line-contact faces."""

    face_x_m: NDArray[np.float64]
    density_kg_m3: NDArray[np.float64]
    viscosity_pa_s: NDArray[np.float64]
    pressure_driven_kg_m_s: NDArray[np.float64]
    entrainment_kg_m_s: NDArray[np.float64]
    total_kg_m_s: NDArray[np.float64]


def compute_line_mass_flux_components(
    x_m: ArrayLike,
    film_m: ArrayLike,
    pressure_pa: ArrayLike,
    fractional_film_content: ArrayLike,
    fluid: IsothermalFluid,
    entrainment_speed_m_s: float,
) -> LineMassFluxComponents:
    """Decompose the original SI JFO mass flux for line-contact diagnostics.

    The face fluxes are per unit axial width [kg/(m s)]. The pressure-driven
    component uses the same face mobility as the independent SI audit; the
    entrainment component uses the upwind ``theta_west`` content.
    """
    x = np.asarray(x_m, dtype=np.float64)
    h = np.asarray(film_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    theta = np.asarray(fractional_film_content, dtype=np.float64)
    pressure_scale = (
        max(float(np.max(np.abs(pressure))), 1.0)
        if pressure.size and np.all(np.isfinite(pressure))
        else 1.0
    )
    pressure_roundoff_tolerance = (
        64.0 * np.finfo(np.float64).eps * pressure_scale
    )
    if (
        x.ndim != 1
        or x.size < 3
        or not all(np.all(np.isfinite(field)) for field in (x, h, pressure, theta))
        or any(field.shape != x.shape for field in (h, pressure, theta))
        or np.any(np.diff(x) <= 0.0)
        or np.any(h <= 0.0)
        or np.any(pressure < -pressure_roundoff_tolerance)
        or np.any((theta < 0.0) | (theta > 1.0))
    ):
        raise ValueError("line mass-flux diagnostics require finite admissible SI fields")
    if (
        not math.isfinite(entrainment_speed_m_s)
        or entrainment_speed_m_s <= 0.0
    ):
        raise ValueError("entrainment_speed_m_s must be positive finite")

    density, viscosity, _, _ = fluid.properties(pressure)
    mobility = density * h**3 / (12.0 * viscosity)
    capacity = density * h
    face_mobility = 0.5 * (mobility[:-1] + mobility[1:])
    face_capacity = 0.5 * (capacity[:-1] + capacity[1:])
    pressure_driven = -face_mobility * np.diff(pressure) / np.diff(x)
    entrainment = entrainment_speed_m_s * face_capacity * theta[:-1]
    return LineMassFluxComponents(
        0.5 * (x[:-1] + x[1:]),
        density,
        viscosity,
        pressure_driven,
        entrainment,
        pressure_driven + entrainment,
    )


class VariablePropertyLineReynoldsError(RuntimeError):
    """Retain the latest fixed-film variable-property iterate on failure."""

    def __init__(self, message, history, last_solution):
        super().__init__(message)
        self.history = tuple(history)
        self.last_solution = last_solution


def audit_variable_property_line_flux(
    x_m: ArrayLike,
    film_m: ArrayLike,
    pressure_pa: ArrayLike,
    fractional_film_content: ArrayLike,
    fluid: IsothermalFluid,
    entrainment_speed_m_s: float,
    *,
    length_scale_m: float,
    film_scale_m: float,
    pressure_scale_pa: float,
    strength: float = 1.0,
) -> dict[str, float]:
    """Independently recompute SI mass flux and dimensionless balance metrics."""
    x = np.asarray(x_m, dtype=np.float64)
    h = np.asarray(film_m, dtype=np.float64)
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    theta = np.asarray(fractional_film_content, dtype=np.float64)
    if (
        x.ndim != 1
        or x.size < 3
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0.0)
        or any(array.shape != x.shape for array in (h, pressure, theta))
        or not all(np.all(np.isfinite(array)) for array in (h, pressure, theta))
        or np.any(h <= 0.0)
    ):
        raise ValueError("line flux audit requires matching finite SI vectors and positive film")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (length_scale_m, film_scale_m, pressure_scale_pa)
    ):
        raise ValueError("line flux audit scales must be positive finite SI values")
    if not math.isfinite(entrainment_speed_m_s) or entrainment_speed_m_s <= 0.0:
        raise ValueError("entrainment_speed_m_s must be positive finite")

    rho, eta, _, _ = fluid.properties(pressure, strength=strength)
    mobility = rho * h**3 / (12.0 * eta)
    capacity = rho * h
    face_mobility = 0.5 * (mobility[:-1] + mobility[1:])
    face_capacity = 0.5 * (capacity[:-1] + capacity[1:])
    mass_flux = (
        -face_mobility * np.diff(pressure) / np.diff(x)
        + entrainment_speed_m_s * face_capacity * theta[:-1]
    )
    x_spacing = np.diff(x)
    cell_width = 0.5 * (x_spacing[:-1] + x_spacing[1:])
    residual = np.diff(mass_flux) / cell_width
    mass_flux_scale = (
        fluid.density_kg_m3
        * film_scale_m**3
        * pressure_scale_pa
        / (12.0 * fluid.viscosity_pa_s * length_scale_m)
    )
    dimensionless_maximum = (
        float(np.max(np.abs(residual)))
        * length_scale_m
        / mass_flux_scale
    )
    local_scale = np.maximum(
        np.abs(mass_flux[1:]) + np.abs(mass_flux[:-1]),
        mass_flux_scale * 1e-14,
    )
    relative_local = float(np.max(np.abs(np.diff(mass_flux)) / local_scale))
    global_scale = max(
        abs(float(mass_flux[0])) + abs(float(mass_flux[-1])),
        mass_flux_scale * 1e-14,
    )
    relative_global = abs(float(mass_flux[-1] - mass_flux[0])) / global_scale
    p_dimensionless = pressure / pressure_scale_pa
    return {
        "mass_maximum": dimensionless_maximum,
        "maximum_si_mass_divergence_kg_m2_s": float(np.max(np.abs(residual))),
        "relative_local_balance": relative_local,
        "relative_global_balance": float(relative_global),
        "net_boundary_mass_flow_kg_m_s": float(mass_flux[-1] - mass_flux[0]),
        "maximum_complementarity": float(
            np.max(np.abs(p_dimensionless * (1.0 - theta)))
        ),
        "minimum_pressure_pa": float(np.min(pressure)),
        "minimum_fractional_film_content": float(np.min(theta)),
        "maximum_fractional_film_content": float(np.max(theta)),
        "boundary_pressure_maximum_pa": float(
            max(abs(float(pressure[0])), abs(float(pressure[-1])))
        ),
    }


def solve_variable_property_line_reynolds(
    x_m: ArrayLike,
    film_m: ArrayLike,
    fluid: IsothermalFluid,
    entrainment_speed_m_s: float,
    *,
    length_scale_m: float,
    film_scale_m: float,
    pressure_scale_pa: float,
    inlet_fractional_film_content: float = 1.0,
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    pressure_tolerance: float = 1e-9,
    mass_tolerance: float = 1e-8,
    max_iterations_per_strength: int = 100,
    max_active_set_iterations: int = 100,
    initial_pressure_pa: ArrayLike | None = None,
) -> VariablePropertyLineReynoldsSolution:
    """Solve prescribed-film 1-D JFO flow with the shared SI fluid law.

    The conservative dimensionless mass flux is
    ``q_hat = -avg(R*H**3/V)*dP/dX + Lambda*avg(R*H)*theta_west``,
    where ``R=rho/rho0``, ``V=eta/eta0``,
    ``Lambda=12*eta0*Ue*L/(h_ref**2*p_ref)`` and
    ``X=x/L, H=h/h_ref, P=p/p_ref``. Face coefficients are frozen in a
    sparse active-set solve and updated by a bounded Picard iteration. Final
    acceptance uses the original, unrelaxed SI mass flux and JFO constraints.
    """
    x = np.asarray(x_m, dtype=np.float64)
    h = np.asarray(film_m, dtype=np.float64)
    if (
        x.ndim != 1
        or x.size < 5
        or h.shape != x.shape
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(h))
        or np.any(np.diff(x) <= 0.0)
        or np.any(h <= 0.0)
    ):
        raise ValueError("x_m and film_m must be matching finite ascending/positive 1-D SI data")
    dx = np.diff(x)
    dx_tolerance = 32 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(dx))), float(np.max(np.abs(x)))
    )
    if not np.allclose(dx, dx[0], rtol=1e-10, atol=dx_tolerance):
        raise ValueError("x_m must be uniformly spaced")
    if not isinstance(fluid, IsothermalFluid):
        raise ValueError("fluid must be an IsothermalFluid instance")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (
            entrainment_speed_m_s,
            length_scale_m,
            film_scale_m,
            pressure_scale_pa,
            pressure_tolerance,
            mass_tolerance,
        )
    ):
        raise ValueError("speed, SI scales and nonlinear tolerances must be positive finite")
    if (
        not math.isfinite(inlet_fractional_film_content)
        or not 0.0 <= inlet_fractional_film_content <= 1.0
    ):
        raise ValueError("inlet_fractional_film_content must lie in [0,1]")
    if (
        not strengths
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in strengths)
        or any(right <= left for left, right in zip(strengths, strengths[1:], strict=False))
    ):
        raise ValueError("strengths must be strictly increasing values in [0,1]")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (max_iterations_per_strength, max_active_set_iterations)
    ):
        raise ValueError("iteration limits must be positive integers")

    pressure = np.zeros_like(h)
    if initial_pressure_pa is not None:
        pressure = np.array(initial_pressure_pa, dtype=np.float64, copy=True)
        if (
            pressure.shape != h.shape
            or not np.all(np.isfinite(pressure))
            or np.any(pressure < 0.0)
        ):
            raise ValueError("initial_pressure_pa must be a finite nonnegative matching vector")
        pressure[[0, -1]] = 0.0
    dimensionless_film = h / film_scale_m
    X = x / length_scale_m
    dX = float(X[1] - X[0])
    lam = (
        12.0
        * fluid.viscosity_pa_s
        * entrainment_speed_m_s
        * length_scale_m
        / (film_scale_m**2 * pressure_scale_pa)
    )
    interior_count = x.size - 2
    full_film = np.ones(interior_count, dtype=bool)
    history: list[dict[str, float]] = []
    last_solution: VariablePropertyLineReynoldsSolution | None = None

    for strength in strengths:
        relative_density, relative_viscosity, _, _ = fluid.ratios(
            pressure, strength=strength
        )
        for iteration in range(1, max_iterations_per_strength + 1):
            mobility = (
                relative_density * dimensionless_film**3 / relative_viscosity
            )
            capacity = relative_density * dimensionless_film
            rows: list[int] = []
            columns: list[int] = []
            values: list[float] = []
            transport_rows: list[int] = []
            transport_columns: list[int] = []
            transport_values: list[float] = []
            known_inlet_flux = np.zeros(interior_count, dtype=np.float64)

            for i in range(1, x.size - 1):
                row = i - 1
                mobility_east = 0.5 * (mobility[i] + mobility[i + 1])
                mobility_west = 0.5 * (mobility[i - 1] + mobility[i])
                diffusion_east = mobility_east / dX**2
                diffusion_west = mobility_west / dX**2
                rows.append(row)
                columns.append(row)
                values.append(diffusion_east + diffusion_west)
                if i < x.size - 2:
                    rows.append(row)
                    columns.append(row + 1)
                    values.append(-diffusion_east)
                if i > 1:
                    rows.append(row)
                    columns.append(row - 1)
                    values.append(-diffusion_west)

                capacity_east = 0.5 * (capacity[i] + capacity[i + 1])
                transport_rows.append(row)
                transport_columns.append(row)
                transport_values.append(lam * capacity_east / dX)
                capacity_west = 0.5 * (capacity[i - 1] + capacity[i])
                if i > 1:
                    transport_rows.append(row)
                    transport_columns.append(row - 1)
                    transport_values.append(-lam * capacity_west / dX)
                else:
                    known_inlet_flux[row] = (
                        -lam * capacity_west * inlet_fractional_film_content / dX
                    )

            shape = (interior_count, interior_count)
            diffusion_matrix = coo_matrix(
                (values, (rows, columns)), shape=shape, dtype=np.float64
            ).tocsr()
            transport_matrix = coo_matrix(
                (transport_values, (transport_rows, transport_columns)),
                shape=shape,
                dtype=np.float64,
            ).tocsr()
            mass_matrix = hstack(
                (diffusion_matrix, transport_matrix), format="csr"
            )
            seen_active_sets: set[bytes] = set()
            active_iteration = 0

            try:
                for _ in range(max_active_set_iterations):
                    active_iteration += 1
                    active_key = np.packbits(full_film).tobytes()
                    if active_key in seen_active_sets:
                        raise RuntimeError("variable-property line JFO active set cycled")
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
                    rhs = np.concatenate(
                        (-known_inlet_flux, full_film.astype(np.float64))
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", MatrixRankWarning)
                        state = _solve_active_system(system, rhs, True)
                    if state.shape != (2 * interior_count,) or not np.all(np.isfinite(state)):
                        raise RuntimeError("variable-property line JFO returned a nonfinite state")
                    pressure_interior = state[:interior_count]
                    theta_interior = state[interior_count:]
                    invalid_pressure = full_film & (pressure_interior < -1e-10)
                    invalid_content = cavitated & (theta_interior > 1.0 + 1e-10)
                    if np.any(cavitated & (theta_interior < -1e-10)):
                        raise RuntimeError("variable-property line JFO produced negative content")
                    if np.any(invalid_pressure) or np.any(invalid_content):
                        full_film[invalid_pressure] = False
                        full_film[invalid_content] = True
                        continue
                    break
                else:
                    raise RuntimeError("variable-property line JFO active-set budget exhausted")

                candidate_pressure = np.zeros_like(h)
                candidate_theta = np.ones_like(h)
                candidate_pressure[1:-1] = pressure_interior * pressure_scale_pa
                candidate_theta[0] = inlet_fractional_film_content
                candidate_theta[1:-1] = theta_interior
                candidate_theta[-1] = candidate_theta[-2]
                audit = audit_variable_property_line_flux(
                    x,
                    h,
                    candidate_pressure,
                    candidate_theta,
                    fluid,
                    entrainment_speed_m_s,
                    length_scale_m=length_scale_m,
                    film_scale_m=film_scale_m,
                    pressure_scale_pa=pressure_scale_pa,
                    strength=strength,
                )
                relative_update = float(
                    np.linalg.norm(candidate_pressure - pressure)
                    / max(
                        np.linalg.norm(candidate_pressure),
                        pressure_scale_pa * 1e-12,
                    )
                )
                entry = {
                    "strength": float(strength),
                    "iteration": float(iteration),
                    "pressure_update_relative": relative_update,
                    "mass_maximum": audit["mass_maximum"],
                    "relative_local_balance": audit["relative_local_balance"],
                    "relative_global_balance": audit["relative_global_balance"],
                    "maximum_complementarity": audit["maximum_complementarity"],
                    "active_set_iterations": float(active_iteration),
                }
                history.append(entry)
                last_solution = VariablePropertyLineReynoldsSolution(
                    candidate_pressure,
                    candidate_theta,
                    audit,
                    tuple(history),
                )
                admissible = (
                    audit["mass_maximum"] <= mass_tolerance
                    and audit["relative_local_balance"] <= 1e-10
                    and audit["relative_global_balance"] <= 1e-10
                    and audit["maximum_complementarity"] <= 1e-10
                    and audit["minimum_pressure_pa"] >= -pressure_scale_pa * 1e-10
                    and audit["minimum_fractional_film_content"] >= -1e-10
                    and audit["maximum_fractional_film_content"] <= 1.0 + 1e-10
                    and audit["boundary_pressure_maximum_pa"] == 0.0
                )
                if relative_update <= pressure_tolerance and admissible:
                    pressure = candidate_pressure
                    break

                true_density, true_viscosity, _, _ = fluid.ratios(
                    candidate_pressure, strength=strength
                )
                relative_density = 0.5 * (relative_density + true_density)
                relative_viscosity = np.exp(
                    0.5
                    * (
                        np.log(relative_viscosity)
                        + np.log(true_viscosity)
                    )
                )
                pressure = candidate_pressure
            except (RuntimeError, ValueError, FloatingPointError) as error:
                raise VariablePropertyLineReynoldsError(
                    str(error), history, last_solution
                ) from error
        else:
            raise VariablePropertyLineReynoldsError(
                f"line property stage {strength} exhausted after "
                f"{max_iterations_per_strength} iterations",
                history,
                last_solution,
            )

    if last_solution is None:
        raise RuntimeError("variable-property line JFO produced no solution")
    return last_solution
