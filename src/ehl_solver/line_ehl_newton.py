"""Hybrid matrix-free Newton solve for stationary line-contact JFO EHL."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, vstack

from ehl_solver.backend import Backend, make_backend
from ehl_solver.line_cavitation import audit_variable_property_line_flux
from ehl_solver.line_contact import (
    PlaneStrainLineElasticityOperator,
)
from ehl_solver.line_ehl import LineContactEHLCase, _line_load_n_m


@dataclass(frozen=True)
class LineJFONewtonIteration:
    iteration: int
    residual_l2: float
    mass_residual_linf: float
    load_residual_relative: float
    gmres_true_relative_residual: float | None
    gmres_true_residual_history: tuple[float, ...]
    gmres_estimated_preconditioned_history: tuple[float, ...]
    gmres_reported_converged: bool
    gmres_iterations: int
    gmres_restart_count: int
    krylov_recovery_attempts: int
    preconditioner_fallback: bool
    preconditioner_schur_indicator: float | None
    superlu_setup_seconds: float
    superlu_solve_seconds: float
    line_search_step: float
    line_search_rejections: int
    pressure_maximum_pa: float
    pressure_maximum_location_m: float
    central_film_m: float
    minimum_film_m: float
    maximum_viscosity_pa_s: float
    maximum_density_kg_m3: float


@dataclass(frozen=True)
class LineJFONewtonSolution:
    state: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    fractional_film_content: NDArray[np.float64]
    film_thickness_m: NDArray[np.float64]
    elastic_displacement_m: NDArray[np.float64]
    central_film_offset_m: float
    target_load_n_m: float
    calculated_line_load_n_m: float
    relative_load_error: float
    audit: dict[str, float]
    history: tuple[LineJFONewtonIteration, ...]
    backend: str


class LineJFONewtonError(RuntimeError):
    """Retain the last line-contact state and hybrid linear diagnostics."""

    def __init__(self, message, state, history, diagnostics=None):
        super().__init__(message)
        self.state = np.asarray(state, dtype=np.float64).copy()
        self.history = tuple(history)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)


class LineJFOKrylovError(ArithmeticError):
    """Preserve the true-residual and SuperLU diagnostics for a failed solve."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _finite_history(values):
    return tuple(
        float(value) if math.isfinite(float(value)) else None
        for value in values
    )


def _gmres_diagnostics(result, *, failure_stage, preconditioner=None):
    diagnostics = {
        "failure_stage": failure_stage,
        "gmres_reason": result.reason,
        "gmres_iterations": result.iterations,
        "gmres_restart_count": result.restart_count,
        "gmres_reported_converged": result.converged,
        "gmres_reported_true_residual_history": _finite_history(
            result.true_residual_history
        ),
        "gmres_estimated_preconditioned_history": _finite_history(
            result.estimated_preconditioned_history
        ),
    }
    if preconditioner is not None:
        diagnostics.update(
            preconditioner="CPUBlockSchurPreconditioner (SciPy SuperLU)",
            preconditioner_fallback=preconditioner.used_bordered_fallback,
            preconditioner_schur_indicator=preconditioner.schur_indicator,
            superlu_setup_seconds=preconditioner.superlu_setup_seconds,
            superlu_solve_seconds=preconditioner.superlu_solve_seconds,
        )
    return diagnostics


def _failure_stage_from_error(error):
    message = str(error).lower()
    for stage in (
        "line-search",
        "pressure",
        "viscosity",
        "density",
        "elasticity",
        "film",
        "residual",
        "jvp",
    ):
        if stage in message:
            return stage
    return None


class LineEHLNewtonSystem:
    """1-D signed pressure/content/load residual with a matrix-free elastic JVP.

    The state is interior ``P=p/p_ref``, ``D=1-theta``, and ``h0/h_ref``.
    Face fluxes use the same averaged mobility/capacity and positive-x upwind
    content as the line JFO reference solver. The physical film is
    ``h=h0+h_geom+gamma*w``.
    """

    def __init__(
        self,
        case: LineContactEHLCase,
        target_load_n_m: float,
        *,
        backend: Backend | None = None,
        compliance_factor: float = 1.0,
        film_floor_m: float = 0.0,
        constitutive_strength: float = 1.0,
        memory_limit_bytes: int = 2**28,
    ):
        if not isinstance(case, LineContactEHLCase):
            raise ValueError("case must be a LineContactEHLCase")
        if (
            not math.isfinite(target_load_n_m)
            or target_load_n_m <= 0.0
            or not math.isfinite(compliance_factor)
            or not 0.0 <= compliance_factor <= 1.0
            or not math.isfinite(film_floor_m)
            or film_floor_m < 0.0
            or not math.isfinite(constitutive_strength)
            or not 0.0 <= constitutive_strength <= 1.0
        ):
            raise ValueError("invalid target load, compliance, film floor, or property strength")
        self.case = case
        self.target_load_n_m = float(target_load_n_m)
        self.compliance_factor = float(compliance_factor)
        self.film_floor_m = float(film_floor_m)
        self.constitutive_strength = float(constitutive_strength)
        self.backend = make_backend("torch", "cuda") if backend is None else backend
        if self.backend.name != "torch":
            raise ValueError("line JFO Newton requires an explicit PyTorch backend")
        self.xp = self.backend.xp
        self.nx = case.x_m.size
        self.n = self.nx - 2
        self.size = 2 * self.n + 1
        self.x_m = self.xp.as_tensor(
            np.array(case.x_m, copy=True),
            dtype=self.xp.float64,
            device=self.backend.device,
        )
        self.dx_dimensionless = case.dx_m / case.length_scale_m
        self.geometry_m = self.xp.as_tensor(
            np.array(case.geometry_m, copy=True),
            dtype=self.xp.float64,
            device=self.backend.device,
        )
        self.elasticity = PlaneStrainLineElasticityOperator(
            case.x_m,
            case.e_star_pa,
            case.elastic_reference_length_m,
            self.backend,
            memory_limit_bytes=memory_limit_bytes,
        )
        self.reynolds_parameter = (
            12.0
            * case.fluid.viscosity_pa_s
            * case.entrainment_speed_m_s
            * case.length_scale_m
            / (case.film_scale_m**2 * case.pressure_scale_pa)
        )
        self._center_right = int(np.searchsorted(case.x_m, 0.0, side="left"))
        self._center_left = max(0, self._center_right - 1)
        if self._center_right >= self.nx:
            self._center_right = self.nx - 1
        width = case.x_m[self._center_right] - case.x_m[self._center_left]
        self._center_right_weight = (
            0.0
            if width == 0.0
            else (0.0 - case.x_m[self._center_left]) / width
        )

    def vector(self, state: ArrayLike):
        value = self.xp.as_tensor(
            state, dtype=self.xp.float64, device=self.backend.device
        )
        if value.shape != (self.size,):
            raise ValueError(f"state must be a finite float64 vector of length {self.size}")
        self._require_finite("Newton state", value)
        return value

    def _require_finite(self, name, value):
        torch = self.xp
        nonfinite = ~torch.isfinite(value)
        if bool(nonfinite.any()):
            location = torch.nonzero(nonfinite, as_tuple=False)[0]
            index = tuple(int(item) for item in location.detach().cpu().tolist())
            sample = value[index] if index else value
            sample_value = float(sample.detach().cpu())
            raise FloatingPointError(
                f"non-finite {name} at index {index}: {sample_value!r}"
            )

    def _unpack(self, state):
        torch = self.xp
        pressure_dimensionless = torch.zeros(
            self.nx, dtype=torch.float64, device=self.backend.device
        )
        pressure_dimensionless[1:-1] = state[: self.n]
        deficit = torch.zeros_like(pressure_dimensionless)
        deficit[1:-1] = state[self.n : 2 * self.n]
        theta = torch.ones_like(pressure_dimensionless)
        theta[0] = self.case.inlet_fractional_film_content
        theta[1:-1] = 1.0 - deficit[1:-1]
        theta[-1] = theta[-2]
        pressure_pa = pressure_dimensionless * self.case.pressure_scale_pa
        self._require_finite("pressure [Pa]", pressure_pa)
        displacement_m = (
            self.elasticity(pressure_pa) * self.compliance_factor
        )
        self._require_finite("elasticity displacement [m]", displacement_m)
        offset_m = state[-1] * self.case.film_scale_m
        film_m = offset_m + self.geometry_m + displacement_m
        self._require_finite("film thickness [m]", film_m)
        return (
            pressure_dimensionless,
            deficit,
            theta,
            pressure_pa,
            film_m,
            displacement_m,
            offset_m,
        )

    def _properties(self, pressure_pa):
        torch = self.xp
        fluid = self.case.fluid
        pressure_ratio = pressure_pa / fluid.roelands_pressure_pa
        self._require_finite("viscosity pressure ratio", pressure_ratio)
        if bool(torch.any(pressure_ratio <= -0.5)):
            raise ValueError("line pressure is outside the Roelands validity domain")
        denominator = 1.0 + fluid.density_b_per_pa * pressure_pa
        self._require_finite("density denominator", denominator)
        if bool((denominator <= 0.0).any()):
            index = int(torch.nonzero(denominator <= 0.0, as_tuple=False)[0])
            raise ValueError(f"line density pole reached at node {index}")
        density_ratio = 1.0 + (
            self.constitutive_strength
            * fluid.density_a_per_pa
            * pressure_pa
            / denominator
        )
        self._require_finite("density ratio", density_ratio)
        if bool((density_ratio <= 0.0).any()):
            index = int(torch.nonzero(density_ratio <= 0.0, as_tuple=False)[0])
            raise ValueError(f"nonpositive density ratio at node {index}")
        log_eta_scale = math.log(
            fluid.viscosity_pa_s / fluid.viscosity_reference_pa_s
        )
        pressure_log = torch.log1p(pressure_ratio)
        self._require_finite("viscosity pressure logarithm", pressure_log)
        roelands_exponent = fluid.viscosity_index * pressure_log
        self._require_finite("Roelands viscosity exponent", roelands_exponent)
        maximum_exponent = float(torch.max(roelands_exponent))
        effective_log_eta_scale = (
            self.constitutive_strength * log_eta_scale
        )
        overflow_exponent = (
            math.log1p(
                math.log(np.finfo(np.float64).max)
                / effective_log_eta_scale
            )
            if effective_log_eta_scale > 0.0
            else math.inf
        )
        if maximum_exponent >= overflow_exponent:
            maximum_index = int(torch.argmax(roelands_exponent))
            maximum_pressure = float(pressure_pa[maximum_index])
            raise FloatingPointError(
                "Roelands viscosity overflow risk at node "
                f"{maximum_index}: p={maximum_pressure:.9e} Pa, "
                f"z*log(1+p/p_ref)_max={maximum_exponent:.9e}"
            )
        log_viscosity_ratio = (
            effective_log_eta_scale * torch.expm1(roelands_exponent)
        )
        self._require_finite("log viscosity ratio", log_viscosity_ratio)
        maximum_log_viscosity = float(torch.max(log_viscosity_ratio))
        if maximum_log_viscosity >= math.log(np.finfo(np.float64).max):
            maximum_index = int(torch.argmax(log_viscosity_ratio))
            maximum_pressure = float(pressure_pa[maximum_index])
            raise FloatingPointError(
                "Roelands viscosity overflow risk at node "
                f"{maximum_index}: p={maximum_pressure:.9e} Pa, "
                f"log(eta/eta0)_max={maximum_log_viscosity:.9e}"
            )
        viscosity_ratio = torch.exp(log_viscosity_ratio)
        self._require_finite("viscosity ratio", viscosity_ratio)
        if bool((viscosity_ratio <= 0.0).any()):
            index = int(
                torch.nonzero(viscosity_ratio <= 0.0, as_tuple=False)[0, 0]
            )
            raise FloatingPointError(
                f"nonpositive viscosity ratio at node {index}"
            )
        return density_ratio, viscosity_ratio, log_viscosity_ratio

    def _dimensionless_flux(self, pressure_dimensionless, theta, film_m):
        torch = self.xp
        density_ratio, viscosity_ratio, _ = self._properties(
            pressure_dimensionless * self.case.pressure_scale_pa
        )
        film = film_m / self.case.film_scale_m
        self._require_finite("dimensionless film", film)
        mobility = density_ratio * film**3 / viscosity_ratio
        capacity = density_ratio * film
        self._require_finite("Reynolds mobility", mobility)
        self._require_finite("entrainment capacity", capacity)
        face_mobility = 0.5 * (mobility[:-1] + mobility[1:])
        face_capacity = 0.5 * (capacity[:-1] + capacity[1:])
        pressure_flux = (
            -face_mobility
            * torch.diff(pressure_dimensionless)
            / self.dx_dimensionless
        )
        self._require_finite("pressure-driven flux", pressure_flux)
        entrainment_flux = (
            self.reynolds_parameter * face_capacity * theta[:-1]
        )
        self._require_finite("entrainment flux", entrainment_flux)
        flux = pressure_flux + entrainment_flux
        self._require_finite("total mass flux", flux)
        return flux

    def residual(self, state):
        torch = self.xp
        state = self.vector(state)
        (
            pressure_dimensionless,
            _,
            theta,
            _,
            film_m,
            _,
            _,
        ) = self._unpack(state)
        if (
            not bool(torch.isfinite(film_m).all())
            or bool(torch.any(film_m <= self.film_floor_m))
        ):
            raise ValueError("line Newton film is not strictly above its floor")
        flux = self._dimensionless_flux(
            pressure_dimensionless, theta, film_m
        )
        mass = torch.diff(flux) / self.dx_dimensionless
        complementarity = torch.minimum(
            state[: self.n], state[self.n : 2 * self.n]
        )
        self._require_finite("mass residual", mass)
        self._require_finite("JFO complementarity residual", complementarity)
        calculated_load = (
            self.case.pressure_scale_pa
            * self.case.dx_m
            * torch.sum(pressure_dimensionless[1:-1])
        )
        self._require_finite("calculated line load [N/m]", calculated_load)
        load = calculated_load / self.target_load_n_m - 1.0
        self._require_finite("load residual", load.reshape(1))
        result = torch.cat((mass, complementarity, load.reshape(1)))
        self._require_finite("line JFO residual", result)
        return result

    def jacobian_vector(self, state, direction):
        torch = self.xp
        state = self.vector(state)
        direction = self.vector(direction)
        (
            pressure_dimensionless,
            deficit,
            theta,
            pressure_pa,
            film_m,
            _,
            _,
        ) = self._unpack(state)
        pressure_direction = torch.zeros_like(pressure_dimensionless)
        pressure_direction[1:-1] = direction[: self.n]
        pressure_direction_pa = (
            pressure_direction * self.case.pressure_scale_pa
        )
        self._require_finite("pressure JVP [Pa]", pressure_direction_pa)
        deficit_direction = torch.zeros_like(deficit)
        deficit_direction[1:-1] = direction[self.n : 2 * self.n]
        theta_direction = -deficit_direction
        theta_direction[-1] = theta_direction[-2]
        elastic_direction = self.elasticity(pressure_direction_pa)
        self._require_finite("elasticity JVP [m]", elastic_direction)
        film_direction = (
            self.compliance_factor * elastic_direction
            / self.case.film_scale_m
            + direction[-1]
        )
        self._require_finite("film-thickness JVP", film_direction)

        density_ratio, viscosity_ratio, _ = self._properties(pressure_pa)
        fluid = self.case.fluid
        pressure_log = torch.log1p(
            pressure_pa / fluid.roelands_pressure_pa
        )
        density_direction = (
            self.constitutive_strength
            * fluid.density_a_per_pa
            * self.case.pressure_scale_pa
            / (1.0 + fluid.density_b_per_pa * pressure_pa) ** 2
            * pressure_direction
        )
        self._require_finite("density JVP", density_direction)
        log_eta_scale = math.log(
            fluid.viscosity_pa_s / fluid.viscosity_reference_pa_s
        )
        log_viscosity_direction = (
            self.constitutive_strength
            * log_eta_scale
            * fluid.viscosity_index
            / fluid.roelands_pressure_pa
            * torch.exp(
                (fluid.viscosity_index - 1.0) * pressure_log
            )
            * self.case.pressure_scale_pa
            * pressure_direction
        )
        self._require_finite("log-viscosity JVP", log_viscosity_direction)
        viscosity_direction = viscosity_ratio * log_viscosity_direction
        self._require_finite("viscosity JVP", viscosity_direction)

        film = film_m / self.case.film_scale_m
        dimensionless_film_direction = film_direction
        mobility = density_ratio * film**3 / viscosity_ratio
        mobility_direction = (
            density_direction * film**3 / viscosity_ratio
            + 3.0
            * density_ratio
            * film**2
            * dimensionless_film_direction
            / viscosity_ratio
            - density_ratio
            * film**3
            * viscosity_direction
            / viscosity_ratio**2
        )
        capacity = density_ratio * film
        capacity_direction = (
            density_direction * film
            + density_ratio * dimensionless_film_direction
        )
        face_mobility = 0.5 * (mobility[:-1] + mobility[1:])
        face_mobility_direction = 0.5 * (
            mobility_direction[:-1] + mobility_direction[1:]
        )
        face_capacity = 0.5 * (capacity[:-1] + capacity[1:])
        face_capacity_direction = 0.5 * (
            capacity_direction[:-1] + capacity_direction[1:]
        )
        pressure_difference = torch.diff(pressure_dimensionless)
        pressure_flux_direction = -(
            face_mobility_direction * pressure_difference
            + face_mobility * torch.diff(pressure_direction)
        ) / self.dx_dimensionless
        entrainment_flux_direction = self.reynolds_parameter * (
            face_capacity_direction * theta[:-1]
            + face_capacity * theta_direction[:-1]
        )
        mass_direction = torch.diff(
            pressure_flux_direction + entrainment_flux_direction
        ) / self.dx_dimensionless
        self._require_finite("mass-residual JVP", mass_direction)

        p_state = state[: self.n]
        d_state = state[self.n : 2 * self.n]
        p_direction = direction[: self.n]
        d_direction = direction[self.n : 2 * self.n]
        complementarity_direction = torch.where(
            p_state < d_state,
            p_direction,
            torch.where(
                d_state < p_state,
                d_direction,
                torch.minimum(p_direction, d_direction),
            ),
        )
        self._require_finite(
            "complementarity-residual JVP", complementarity_direction
        )
        load_direction = (
            self.case.pressure_scale_pa
            * self.case.dx_m
            * torch.sum(pressure_direction[1:-1])
            / self.target_load_n_m
        )
        self._require_finite("load-residual JVP", load_direction.reshape(1))
        result = torch.cat(
            (mass_direction, complementarity_direction, load_direction.reshape(1))
        )
        self._require_finite("line JFO Jacobian-vector product", result)
        return result

    def frozen_matrix(self, state):
        """Frozen-mobility sparse active-set matrix for CPU SuperLU."""
        state = self.vector(state)
        (
            pressure,
            deficit,
            theta,
            _,
            film_m,
            _,
            _,
        ) = self._unpack(state)
        density_ratio, viscosity_ratio, _ = self._properties(
            pressure * self.case.pressure_scale_pa
        )
        p = pressure.detach().cpu().numpy()
        d = deficit.detach().cpu().numpy()
        t = theta.detach().cpu().numpy()
        h = (film_m / self.case.film_scale_m).detach().cpu().numpy()
        rho = density_ratio.detach().cpu().numpy()
        eta = viscosity_ratio.detach().cpu().numpy()
        mobility = rho * h**3 / eta
        capacity = rho * h
        mobility_h0 = 3.0 * rho * h**2 / eta
        capacity_h0 = rho
        dx = self.dx_dimensionless
        n = self.n
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []

        for node in range(1, self.nx - 1):
            row = node - 1
            east_mobility = 0.5 * (mobility[node] + mobility[node + 1])
            west_mobility = 0.5 * (mobility[node - 1] + mobility[node])
            rows.append(row)
            columns.append(row)
            values.append((east_mobility + west_mobility) / dx**2)
            if node > 1:
                rows.append(row)
                columns.append(row - 1)
                values.append(-west_mobility / dx**2)
            if node < self.nx - 2:
                rows.append(row)
                columns.append(row + 1)
                values.append(-east_mobility / dx**2)

            east_capacity = 0.5 * (capacity[node] + capacity[node + 1])
            west_capacity = 0.5 * (capacity[node - 1] + capacity[node])
            rows.append(row)
            columns.append(n + row)
            values.append(-self.reynolds_parameter * east_capacity / dx)
            if node > 1:
                rows.append(row)
                columns.append(n + row - 1)
                values.append(self.reynolds_parameter * west_capacity / dx)

        mass_flux_h0 = (
            -0.5
            * (mobility_h0[:-1] + mobility_h0[1:])
            * np.diff(p)
            / dx
            + self.reynolds_parameter
            * 0.5
            * (capacity_h0[:-1] + capacity_h0[1:])
            * t[:-1]
        )
        mass_h0 = np.diff(mass_flux_h0) / dx
        rows.extend(range(n))
        columns.extend([2 * n] * n)
        values.extend(mass_h0.tolist())

        mass_block = coo_matrix(
            (values, (rows, columns)),
            shape=(n, 2 * n + 1),
            dtype=np.float64,
        ).tocsr()
        active_pressure = p[1:-1] < d[1:-1]
        complement_rows = np.arange(n)
        complement_columns = np.where(
            active_pressure, np.arange(n), n + np.arange(n)
        )
        complementarity = coo_matrix(
            (
                np.ones(n, dtype=np.float64),
                (complement_rows, complement_columns),
            ),
            shape=(n, 2 * n + 1),
        ).tocsr()
        load_row = coo_matrix(
            (
                np.full(
                    n,
                    self.case.pressure_scale_pa
                    * self.case.dx_m
                    / self.target_load_n_m,
                    dtype=np.float64,
                ),
                (np.zeros(n, dtype=np.int64), np.arange(n)),
            ),
            shape=(1, 2 * n + 1),
        ).tocsr()
        return vstack((mass_block, complementarity, load_row), format="csc")

    def feasibility_error(self, state, *, tolerance=1e-10):
        torch = self.xp
        try:
            value = self.vector(state)
        except (FloatingPointError, ValueError) as error:
            return str(error)
        if bool(torch.any(value[: self.n] < -tolerance)):
            index = int(torch.nonzero(value[: self.n] < -tolerance, as_tuple=False)[0, 0])
            return f"negative pressure state at interior index {index}"
        if bool(torch.any(value[self.n : 2 * self.n] < -tolerance)):
            index = int(
                torch.nonzero(
                    value[self.n : 2 * self.n] < -tolerance, as_tuple=False
                )[0, 0]
            )
            return f"negative film-content deficit at interior index {index}"
        if bool(torch.any(value[self.n : 2 * self.n] > 1.0 + tolerance)):
            index = int(
                torch.nonzero(
                    value[self.n : 2 * self.n] > 1.0 + tolerance,
                    as_tuple=False,
                )[0, 0]
            )
            return f"film-content deficit exceeds one at interior index {index}"
        try:
            _, _, _, _, film_m, _, _ = self._unpack(value)
        except (FloatingPointError, ValueError) as error:
            return str(error)
        if bool(torch.any(film_m <= self.film_floor_m)):
            index = int(
                torch.nonzero(film_m <= self.film_floor_m, as_tuple=False)[0, 0]
            )
            return (
                f"film thickness at node {index} is not strictly above "
                f"the floor {self.film_floor_m:.9e} m"
            )
        return None

    def feasible(self, state, *, tolerance=1e-10):
        return self.feasibility_error(state, tolerance=tolerance) is None

    def metrics(self, state, residual):
        torch = self.xp
        (
            pressure_dimensionless,
            _,
            _,
            pressure_pa,
            film_m,
            _,
            _,
        ) = self._unpack(state)
        density_ratio, viscosity_ratio, _ = self._properties(pressure_pa)
        maximum_index = int(torch.argmax(pressure_pa))
        central_film = (
            (1.0 - self._center_right_weight)
            * film_m[self._center_left]
            + self._center_right_weight * film_m[self._center_right]
        )
        maximum_viscosity = (
            self.case.fluid.viscosity_pa_s * torch.max(viscosity_ratio)
        )
        maximum_density = self.case.fluid.density_kg_m3 * torch.max(density_ratio)
        mass_residual = residual[: self.n]
        calculated_load = (
            self.case.pressure_scale_pa
            * self.case.dx_m
            * torch.sum(pressure_dimensionless[1:-1])
        )
        return {
            "residual_l2": float(torch.linalg.vector_norm(residual)),
            "mass_residual_linf": float(torch.max(torch.abs(mass_residual))),
            "load_residual_relative": float(
                calculated_load / self.target_load_n_m - 1.0
            ),
            "pressure_maximum_pa": float(pressure_pa[maximum_index]),
            "pressure_maximum_location_m": float(self.case.x_m[maximum_index]),
            "central_film_m": float(central_film),
            "minimum_film_m": float(torch.min(film_m)),
            "maximum_viscosity_pa_s": float(maximum_viscosity),
            "maximum_density_kg_m3": float(maximum_density),
        }

    def export_solution(self, state, history):
        value = self.vector(state)
        (
            _,
            _,
            theta,
            pressure_pa,
            film_m,
            displacement_m,
            offset_m,
        ) = self._unpack(value)
        pressure = pressure_pa.detach().cpu().numpy().copy()
        content = theta.detach().cpu().numpy().copy()
        film = film_m.detach().cpu().numpy().copy()
        displacement = displacement_m.detach().cpu().numpy().copy()
        offset = float(offset_m)
        audit = audit_variable_property_line_flux(
            self.case.x_m,
            film,
            pressure,
            content,
            self.case.fluid,
            self.case.entrainment_speed_m_s,
            length_scale_m=self.case.length_scale_m,
            film_scale_m=self.case.film_scale_m,
            pressure_scale_pa=self.case.pressure_scale_pa,
            strength=self.constitutive_strength,
        )
        calculated_load = _line_load_n_m(pressure, self.case.dx_m)
        relative_load_error = abs(
            calculated_load / self.target_load_n_m - 1.0
        )
        direct_operator = PlaneStrainLineElasticityOperator(
            self.case.x_m,
            self.case.e_star_pa,
            self.case.elastic_reference_length_m,
            make_backend("numpy"),
        )
        direct = np.convolve(
            pressure, direct_operator.influence_kernel, mode="full"
        )[self.nx - 1 : 2 * self.nx - 1]
        reconstructed = (
            offset + self.case.geometry_m + self.compliance_factor * direct
        )
        audit["direct_elasticity_relative_error"] = float(
            np.max(np.abs(direct - displacement))
            / max(
                np.max(np.abs(direct)),
                np.finfo(np.float64).tiny,
            )
        )
        audit["direct_film_reconstruction_maximum_m"] = float(
            np.max(np.abs(reconstructed - film))
        )
        accepted = (
            audit["mass_maximum"] <= 1e-8
            and audit["relative_local_balance"] <= 1e-10
            and audit["relative_global_balance"] <= 1e-10
            and audit["maximum_complementarity"] <= 1e-10
            and audit["minimum_pressure_pa"]
            >= -self.case.pressure_scale_pa * 1e-10
            and audit["minimum_fractional_film_content"] >= -1e-10
            and audit["maximum_fractional_film_content"] <= 1.0 + 1e-10
            and audit["boundary_pressure_maximum_pa"] == 0.0
            and np.all(film > self.film_floor_m)
            and relative_load_error <= 1e-8
            and audit["direct_elasticity_relative_error"] <= 1e-12
            and audit["direct_film_reconstruction_maximum_m"]
            <= 1e-12 * self.case.film_scale_m
        )
        if not accepted:
            raise ValueError(f"line Newton candidate failed independent audit: {audit}")
        return LineJFONewtonSolution(
            state=value.detach().cpu().numpy().copy(),
            pressure_pa=pressure,
            fractional_film_content=content,
            film_thickness_m=film,
            elastic_displacement_m=displacement,
            central_film_offset_m=offset,
            target_load_n_m=self.target_load_n_m,
            calculated_line_load_n_m=calculated_load,
            relative_load_error=relative_load_error,
            audit=audit,
            history=tuple(history),
            backend=f"{self.backend.platform}/{self.backend.device}",
        )


def _fraction_to_boundary(system, state, direction, *, fraction=0.99):
    torch = system.xp
    n = system.n
    value = 1.0
    lower_fields = state[: 2 * n]
    lower_direction = direction[: 2 * n]
    margin = lower_fields + 1e-10
    system._require_finite("line-search feasibility margin", margin)
    falling = lower_direction < 0.0
    if bool(falling.any()):
        value = min(
            value,
            fraction
            * float(
                torch.min(
                    margin[falling] / -lower_direction[falling]
                )
            ),
        )
    deficit = state[n : 2 * n]
    deficit_direction = direction[n : 2 * n]
    rising = deficit_direction > 0.0
    if bool(rising.any()):
        value = min(
            value,
            fraction
            * float(
                torch.min(
                    (1.0 + 1e-10 - deficit[rising])
                    / deficit_direction[rising]
                )
            ),
        )
    _, _, _, _, film, _, _ = system._unpack(state)
    pressure_direction = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64, device=state.device),
            direction[: system.n],
            torch.zeros(1, dtype=torch.float64, device=state.device),
        )
    )
    pressure_direction_pa = (
        pressure_direction * system.case.pressure_scale_pa
    )
    system._require_finite(
        "line-search pressure direction [Pa]", pressure_direction_pa
    )
    elastic_direction = system.elasticity(pressure_direction_pa)
    system._require_finite(
        "line-search elasticity direction [m]", elastic_direction
    )
    film_direction = (
        system.compliance_factor * elastic_direction
        + direction[-1] * system.case.film_scale_m
    )
    system._require_finite("line-search film direction [m]", film_direction)
    closing = film_direction < 0.0
    if bool(closing.any()):
        value = min(
            value,
            fraction
            * float(
                torch.min(
                    (film[closing] - system.film_floor_m)
                    / -film_direction[closing]
                )
            ),
        )
    if not math.isfinite(value):
        raise FloatingPointError("non-finite line-search step bound")
    return max(0.0, min(1.0, value))


def _true_residual_tolerance(residual_norm, krylov_rtol):
    return max(5.0 * krylov_rtol, min(1.0e-6, 0.1 * residual_norm))


def _solve_correction(
    system,
    state,
    residual,
    *,
    krylov_rtol,
    true_residual_tolerance,
    gmres_restart,
    maximum_gmres_iterations,
    memory_limit_bytes,
    progress_callback,
):
    torch = system.xp
    from ehl_solver.device_krylov import device_gmres
    from ehl_solver.hybrid_jfo import CPUBlockSchurPreconditioner

    matrix = system.frozen_matrix(state)
    preconditioner = CPUBlockSchurPreconditioner(
        matrix,
        system.n,
        state.device,
        progress_callback=progress_callback,
    )
    result = device_gmres(
        lambda vector: system.jacobian_vector(state, vector),
        -residual,
        preconditioner=preconditioner,
        rtol=krylov_rtol,
        restart=gmres_restart,
        max_iterations=maximum_gmres_iterations,
        memory_limit_bytes=memory_limit_bytes,
        progress_callback=progress_callback,
    )
    direction = result.solution
    if not bool(torch.isfinite(direction).all()):
        location = torch.nonzero(~torch.isfinite(direction), as_tuple=False)[0]
        index = tuple(int(item) for item in location.detach().cpu().tolist())
        raise LineJFOKrylovError(
            f"non-finite GMRES correction at state index {index}",
            _gmres_diagnostics(
                result,
                failure_stage="GMRES correction",
                preconditioner=preconditioner,
            ),
        )
    try:
        true_residual = (
            system.jacobian_vector(state, direction) + residual
        )
    except FloatingPointError as error:
        raise LineJFOKrylovError(
            f"independent true-residual JVP failed: {error}",
            _gmres_diagnostics(
                result,
                failure_stage="independent JVP",
                preconditioner=preconditioner,
            ),
        ) from error
    true_relative = float(
        torch.linalg.vector_norm(true_residual)
        / torch.linalg.vector_norm(residual).clamp_min(
            torch.finfo(torch.float64).tiny
        )
    )
    if (
        not math.isfinite(true_relative)
        or true_relative > true_residual_tolerance
    ):
        raise LineJFOKrylovError(
            "line JFO GMRES true-residual gate failed: "
            f"{true_relative:.3e} ({result.reason})",
            {
                **_gmres_diagnostics(
                    result,
                    failure_stage="independent_true_residual",
                    preconditioner=preconditioner,
                ),
                "failure_stage": "independent_true_residual",
                "gmres_true_relative_residual": (
                    true_relative if math.isfinite(true_relative) else None
                ),
                "independently_recomputed_true_relative_residual": (
                    true_relative if math.isfinite(true_relative) else None
                ),
                "gmres_rtol": krylov_rtol,
                "true_residual_tolerance": true_residual_tolerance,
                "gmres_reported_converged": result.converged,
                "preconditioner": (
                    "CPUBlockSchurPreconditioner (SciPy SuperLU)"
                ),
                "preconditioner_fallback": preconditioner.used_bordered_fallback,
                "preconditioner_schur_indicator": preconditioner.schur_indicator,
                "superlu_setup_seconds": preconditioner.superlu_setup_seconds,
                "superlu_solve_seconds": preconditioner.superlu_solve_seconds,
            },
        )
    return direction, result, true_relative, preconditioner


def solve_line_ehl_newton(
    system: LineEHLNewtonSystem,
    initial_state: ArrayLike,
    *,
    maximum_newton_iterations: int = 30,
    gmres_restart: int = 60,
    maximum_gmres_iterations: int = 600,
    krylov_rtol: float = 1e-8,
    maximum_krylov_recoveries: int = 2,
    armijo_constant: float = 1e-4,
    maximum_backtracks: int = 24,
    memory_limit_bytes: int = 2**30,
    progress_callback=None,
) -> LineJFONewtonSolution:
    """Solve one target line load with feasible, directional-Armijo Newton."""
    torch = system.xp
    if (
        not isinstance(maximum_newton_iterations, int)
        or isinstance(maximum_newton_iterations, bool)
        or maximum_newton_iterations < 1
        or not isinstance(gmres_restart, int)
        or isinstance(gmres_restart, bool)
        or not 1 <= gmres_restart <= 240
        or not isinstance(maximum_gmres_iterations, int)
        or isinstance(maximum_gmres_iterations, bool)
        or maximum_gmres_iterations < 1
        or not math.isfinite(krylov_rtol)
        or not 0.0 < krylov_rtol < 1.0
        or not isinstance(maximum_krylov_recoveries, int)
        or isinstance(maximum_krylov_recoveries, bool)
        or maximum_krylov_recoveries < 0
        or not math.isfinite(armijo_constant)
        or not 0.0 < armijo_constant < 1.0
        or not isinstance(maximum_backtracks, int)
        or isinstance(maximum_backtracks, bool)
        or maximum_backtracks < 0
        or not isinstance(memory_limit_bytes, int)
        or isinstance(memory_limit_bytes, bool)
        or memory_limit_bytes < 1
        or (
            progress_callback is not None
            and not callable(progress_callback)
        )
    ):
        raise ValueError("invalid line Newton/Krylov/line-search controls")
    state = system.vector(initial_state).clone()
    initial_failure = system.feasibility_error(state)
    if initial_failure is not None:
        raise ValueError(
            f"initial line Newton state is infeasible: {initial_failure}"
        )
    history: list[LineJFONewtonIteration] = []
    latest_diagnostics: dict[str, object] = {}

    def report(stage, *, iteration=None, emit=False, **metrics):
        if progress_callback is not None:
            progress_callback(
                stage=stage,
                iteration=iteration,
                emit=emit,
                target_load_n_m=system.target_load_n_m,
                **metrics,
            )

    for iteration in range(maximum_newton_iterations + 1):
        try:
            residual = system.residual(state)
            metrics = system.metrics(state, residual)
        except (FloatingPointError, ValueError) as error:
            latest_diagnostics["failure_stage"] = "residual"
            latest_diagnostics["failure_detail"] = str(error)
            raise LineJFONewtonError(
                str(error), state.detach().cpu().numpy(), history, latest_diagnostics
            ) from error
        latest_diagnostics.update(metrics)
        latest_diagnostics["newton_iteration"] = iteration
        report(
            "newton_residual",
            iteration=iteration,
            emit=True,
            **metrics,
        )
        if (
            metrics["mass_residual_linf"] <= 1e-8
            and abs(metrics["load_residual_relative"]) <= 1e-8
            and float(torch.max(torch.abs(residual[system.n : 2 * system.n])))
            <= 1e-10
        ):
            try:
                return system.export_solution(state, history)
            except ValueError as error:
                latest_diagnostics["independent_audit_failure"] = str(error)
        if iteration == maximum_newton_iterations:
            raise LineJFONewtonError(
                "line Newton iteration limit exhausted",
                state.detach().cpu().numpy(),
                history,
                {**latest_diagnostics, **metrics},
            )

        recovery_failures = []
        correction = None
        true_residual_tolerance = _true_residual_tolerance(
            metrics["residual_l2"], krylov_rtol
        )
        for recovery in range(maximum_krylov_recoveries + 1):
            try:
                direction, gmres, true_relative, preconditioner = _solve_correction(
                    system,
                    state,
                    residual,
                    krylov_rtol=krylov_rtol * 0.1**recovery,
                    true_residual_tolerance=true_residual_tolerance,
                    gmres_restart=gmres_restart,
                    maximum_gmres_iterations=(
                        maximum_gmres_iterations * 2**recovery
                    ),
                    memory_limit_bytes=memory_limit_bytes,
                    progress_callback=(
                        None
                        if progress_callback is None
                        else lambda _iteration=iteration, **event: report(
                            event.pop("stage"),
                            iteration=event.pop("iteration", _iteration),
                            **event,
                        )
                    ),
                )
                action = system.jacobian_vector(state, direction)
                merit_slope = float(torch.dot(residual, action))
                if math.isfinite(merit_slope) and merit_slope < 0.0:
                    correction = (
                        direction,
                        gmres,
                        true_relative,
                        preconditioner,
                        merit_slope,
                        recovery,
                    )
                    break
                recovery_error = LineJFOKrylovError(
                    "line Newton correction is not a descent direction",
                    {
                        "merit_directional_derivative": merit_slope,
                        "gmres_true_relative_residual": true_relative,
                        "gmres_iterations": gmres.iterations,
                        "gmres_restart_count": gmres.restart_count,
                        "preconditioner_fallback": (
                            preconditioner.used_bordered_fallback
                        ),
                        "preconditioner_schur_indicator": (
                            preconditioner.schur_indicator
                        ),
                    },
                )
            except LineJFOKrylovError as error:
                recovery_error = error
            except (FloatingPointError, ValueError, RuntimeError) as error:
                latest_diagnostics["linear_solve_failure"] = str(error)
                latest_diagnostics["failure_stage"] = (
                    _failure_stage_from_error(error)
                    or "GMRES or CPU preconditioner"
                )
                latest_diagnostics["failure_detail"] = str(error)
                raise LineJFONewtonError(
                    str(error),
                    state.detach().cpu().numpy(),
                    history,
                    latest_diagnostics,
                ) from error
            recovery_failures.append(recovery_error.diagnostics)
            if recovery == maximum_krylov_recoveries:
                latest_diagnostics.update(recovery_error.diagnostics)
                latest_diagnostics["krylov_recovery_failures"] = recovery_failures
                raise LineJFONewtonError(
                    str(recovery_error),
                    state.detach().cpu().numpy(),
                    history,
                    latest_diagnostics,
                ) from recovery_error
            report(
                "gmres_recovery",
                iteration=iteration,
                emit=True,
                recovery_attempt=recovery + 1,
                next_maximum_gmres_iterations=(
                    maximum_gmres_iterations * 2 ** (recovery + 1)
                ),
                recovery_reason=str(recovery_error),
                continuation_decision="retry GMRES with tighter tolerance and larger cap",
                **metrics,
            )
        if correction is None:
            raise AssertionError("bounded Krylov recovery did not produce a correction")
        (
            direction,
            gmres,
            true_relative,
            preconditioner,
            merit_slope,
            recovery_count,
        ) = correction
        merit = 0.5 * float(torch.dot(residual, residual))
        if not math.isfinite(merit):
            latest_diagnostics.update(
                failure_stage="line_search_merit",
                failure_detail="non-finite current Newton merit",
            )
            raise LineJFONewtonError(
                "line search received a non-finite Newton merit",
                state.detach().cpu().numpy(),
                history,
                latest_diagnostics,
            )
        try:
            feasible_step = _fraction_to_boundary(system, state, direction)
        except (FloatingPointError, ValueError) as error:
            latest_diagnostics.update(
                failure_stage="line_search_step_bound",
                failure_detail=str(error),
            )
            raise LineJFONewtonError(
                f"line-search step-bound evaluation failed: {error}",
                state.detach().cpu().numpy(),
                history,
                latest_diagnostics,
            ) from error
        step = feasible_step
        rejected = 0
        accepted_state = None
        last_reason = "no feasible Armijo step"
        for backtrack in range(maximum_backtracks + 1):
            if step <= 0.0:
                break
            trial = state + step * direction
            feasibility_error = system.feasibility_error(trial)
            if feasibility_error is None:
                try:
                    trial_residual = system.residual(trial)
                    trial_merit = 0.5 * float(
                        torch.dot(trial_residual, trial_residual)
                    )
                except (FloatingPointError, ValueError) as error:
                    trial_merit = math.inf
                    last_reason = (
                        f"line-search trial residual evaluation failed: {error}"
                    )
                else:
                    if not math.isfinite(trial_merit):
                        last_reason = "line-search trial merit is non-finite"
                    elif (
                        trial_merit
                        <= merit + armijo_constant * step * merit_slope
                        and trial_merit < merit
                    ):
                        accepted_state = trial
                        last_reason = "directional Armijo accepted"
                        break
                    else:
                        last_reason = "directional Armijo rejected"
            else:
                last_reason = (
                    f"line-search feasibility rejected: {feasibility_error}"
                )
            rejected += 1
            report(
                "line_search_rejected",
                iteration=iteration,
                emit=True,
                trial=backtrack + 1,
                step_length=step,
                reason=last_reason,
                continuation_decision="backtrack",
                **metrics,
            )
            step *= 0.5
        if accepted_state is None:
            latest_diagnostics.update(
                gmres_true_relative_residual=true_relative,
                merit_directional_derivative=merit_slope,
                feasible_step_limit=feasible_step,
                line_search_rejections=rejected,
                line_search_last_reason=last_reason,
                failure_stage="line_search",
                failure_detail=last_reason,
                preconditioner_fallback=preconditioner.used_bordered_fallback,
            )
            raise LineJFONewtonError(
                "line Newton directional Armijo search exhausted",
                state.detach().cpu().numpy(),
                history,
                latest_diagnostics,
            )
        state = accepted_state
        history.append(
            LineJFONewtonIteration(
                iteration=iteration,
                residual_l2=metrics["residual_l2"],
                mass_residual_linf=metrics["mass_residual_linf"],
                load_residual_relative=metrics["load_residual_relative"],
                gmres_true_relative_residual=true_relative,
                gmres_true_residual_history=tuple(
                    float(value) for value in gmres.true_residual_history
                ),
                gmres_estimated_preconditioned_history=tuple(
                    float(value)
                    for value in gmres.estimated_preconditioned_history
                ),
                gmres_reported_converged=gmres.converged,
                gmres_iterations=gmres.iterations,
                gmres_restart_count=gmres.restart_count,
                krylov_recovery_attempts=recovery_count,
                preconditioner_fallback=preconditioner.used_bordered_fallback,
                preconditioner_schur_indicator=preconditioner.schur_indicator,
                superlu_setup_seconds=preconditioner.superlu_setup_seconds,
                superlu_solve_seconds=preconditioner.superlu_solve_seconds,
                line_search_step=step,
                line_search_rejections=rejected,
                pressure_maximum_pa=metrics["pressure_maximum_pa"],
                pressure_maximum_location_m=metrics[
                    "pressure_maximum_location_m"
                ],
                central_film_m=metrics["central_film_m"],
                minimum_film_m=metrics["minimum_film_m"],
                maximum_viscosity_pa_s=metrics[
                    "maximum_viscosity_pa_s"
                ],
                maximum_density_kg_m3=metrics["maximum_density_kg_m3"],
            )
        )
        report(
            "newton_step_accepted",
            iteration=iteration,
            emit=True,
            continuation_decision="accept",
            gmres_true_relative_residual=true_relative,
            gmres_reported_converged=gmres.converged,
            gmres_iterations=gmres.iterations,
            gmres_restart_count=gmres.restart_count,
            krylov_recovery_attempts=recovery_count,
            preconditioner_fallback=preconditioner.used_bordered_fallback,
            preconditioner_schur_indicator=preconditioner.schur_indicator,
            line_search_step=step,
            line_search_rejections=rejected,
            **metrics,
        )

    raise AssertionError("unreachable line Newton loop exit")
