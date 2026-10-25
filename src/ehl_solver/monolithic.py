"""Matrix-free bordered Newton solve for signed full-film load control."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, spilu

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.pseudo_arclength import (
    PseudoArclengthResult,
    pseudo_arclength_continue,
)
from ehl_solver.reynolds import assemble_reynolds_system, reynolds_residual

logger = logging.getLogger(__name__)
MAX_INEXACT_KRYLOV_RTOL = 1e-2


@dataclass(frozen=True)
class KrylovSolveDiagnostic:
    """Per-solve GMRES stopping data and preconditioned residual history."""

    label: str
    preconditioner: str
    requested_rtol: float
    effective_rtol: float
    iterations: int
    info: int
    true_relative_residual: float
    preconditioned_residual_history: tuple[float, ...]
    termination_reason: str


@dataclass(frozen=True)
class PreconditionerBenchmark:
    """One timed bordered-correction attempt with a configured preconditioner."""

    preconditioner: str
    preconditioner_used: str
    converged: bool
    elapsed_seconds: float
    krylov_iterations: int
    maximum_true_relative_residual: float
    bordered_residual_relative: float
    termination_reason: str


@dataclass(frozen=True)
class FrozenFilmPreconditioner:
    """Scaled sparse Reynolds block and its reusable approximate inverse."""

    matrix: csr_matrix
    operator: LinearOperator
    strategy: str


@dataclass(frozen=True)
class BorderedCorrection:
    """Newton correction and diagnostics for one bordered linear solve."""

    pressure_correction: NDArray[np.float64]
    h0_correction_m: float
    pressure_krylov_iterations: int
    pressure_krylov_residual_relative: float
    schur_denominator: float
    schur_condition_indicator: float
    bordered_residual_relative: float
    used_fallback: bool
    preconditioner_used: str
    krylov_diagnostics: tuple[KrylovSolveDiagnostic, ...]


@dataclass(frozen=True)
class MonolithicIteration:
    """Accepted nonlinear iteration and dual-line-search diagnostics."""

    iteration: int
    reynolds_residual_scaled: float
    load_residual_n: float
    load_residual_scaled: float
    pressure_correction_relative: float
    h0_correction_relative: float
    minimum_film_thickness_m: float
    minimum_pressure_pa: float
    maximum_pressure_pa: float
    feasible_step_limit: float
    accepted_step_length: float
    merit_directional_derivative: float
    rejected_line_search_trials: int
    krylov_iterations: int
    krylov_residual_relative: float
    schur_denominator: float
    schur_condition_indicator: float
    bordered_residual_relative: float
    forcing_tolerance: float
    krylov_diagnostics: tuple[KrylovSolveDiagnostic, ...]
    linear_refinement_attempts: int = 0
    line_search_termination_reason: str = "armijo_satisfied"


@dataclass(frozen=True)
class MonolithicLoadSolution:
    """Converged signed-full-film pressure, film and central offset."""

    pressure_dimensionless: NDArray[np.float64]
    pressure_pa: NDArray[np.float64]
    film_thickness_m: NDArray[np.float64]
    elastic_displacement_m: NDArray[np.float64]
    central_film_offset_m: float
    target_load_n: float
    calculated_load_n: float
    load_residual_n: float
    history: tuple[MonolithicIteration, ...]
    operator_evaluations: int
    initial_h0_adjustment_m: float = 0.0


@dataclass(frozen=True)
class FeasibleInitialState:
    """Admissible pressure/offset initial state and explicit offset change."""

    pressure_dimensionless: NDArray[np.float64]
    central_film_offset_m: float
    original_minimum_film_thickness_m: float
    adjusted_minimum_film_thickness_m: float
    h0_adjustment_m: float


@dataclass(frozen=True)
class LoadContinuationStep:
    """Accepted or rejected adaptive load-continuation attempt."""

    attempt: int
    target_load_n: float
    step_load_n: float
    accepted: bool
    newton_iterations: int
    relative_load_error: float
    central_film_offset_m: float
    minimum_film_thickness_m: float
    reason: str
    failure_diagnostics: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class MonolithicLoadContinuation:
    """Final solution and complete adaptive load-stepping history."""

    solution: MonolithicLoadSolution
    steps: tuple[LoadContinuationStep, ...]


class MonolithicNewtonError(RuntimeError):
    """Nonlinear failure retaining completed Newton iteration diagnostics."""

    def __init__(
        self,
        message: str,
        history: tuple[MonolithicIteration, ...],
        diagnostics: dict[str, object] | None = None,
    ):
        super().__init__(message)
        self.history = history
        self.diagnostics = dict(diagnostics or {})


class BorderedLinearSolveError(RuntimeError):
    """Failed bordered Krylov correction retaining every completed GMRES record."""

    def __init__(
        self,
        message: str,
        krylov_diagnostics: tuple[KrylovSolveDiagnostic, ...],
    ):
        super().__init__(message)
        self.krylov_diagnostics = krylov_diagnostics


class BorderedReynoldsElasticityProblem:
    """Residual and matrix-free derivatives for unknown interior P and h0.

    Dimensionless pressure is the primary field; ``h0`` and geometric inputs
    are dimensional SI. Pressure boundaries remain fixed to zero. Pressure
    deformation is computed by the existing cell-integrated, zero-padded
    DC-FFT kernel with the documented positive gap-increasing sign.
    ``compliance_factor`` is a dimensionless continuation multiplier on the
    displacement; it does not alter the physical modulus or kernel.
    """

    def __init__(
        self,
        x_m: ArrayLike,
        y_m: ArrayLike,
        radius_x_m: float,
        radius_y_m: float,
        e_star_pa: float,
        length_scale_m: float,
        film_scale_m: float,
        pressure_scale_pa: float,
        reynolds_parameter: float,
        target_load_n: float,
        *,
        compliance_factor: float = 1.0,
        reynolds_residual_scale: float = 1.0,
        load_residual_scale_n: float | None = None,
        film_floor_m: float = 1e-12,
        krylov_rtol: float = 1e-8,
        krylov_max_iterations: int | None = None,
        gmres_restart: int = 80,
        preconditioner: str = "ilu",
        ilu_drop_tolerance: float = 1e-3,
        ilu_fill_factor: float = 10.0,
    ) -> None:
        self.x = np.asarray(x_m, dtype=np.float64)
        self.y = np.asarray(y_m, dtype=np.float64)
        if (
            self.x.ndim != 1
            or self.y.ndim != 1
            or min(self.x.size, self.y.size) < 3
            or not np.all(np.isfinite(self.x))
            or not np.all(np.isfinite(self.y))
            or np.any(np.diff(self.x) <= 0.0)
            or np.any(np.diff(self.y) <= 0.0)
        ):
            raise ValueError("coordinates must be finite ascending 1-D arrays")
        self.dx_m = float(self.x[1] - self.x[0])
        self.dy_m = float(self.y[1] - self.y[0])
        x_atol = 32.0 * np.finfo(np.float64).eps * max(
            abs(self.dx_m), float(np.max(np.abs(self.x))), np.finfo(float).tiny
        )
        y_atol = 32.0 * np.finfo(np.float64).eps * max(
            abs(self.dy_m), float(np.max(np.abs(self.y))), np.finfo(float).tiny
        )
        if not np.allclose(np.diff(self.x), self.dx_m, rtol=1e-10, atol=x_atol) or not np.allclose(
            np.diff(self.y), self.dy_m, rtol=1e-10, atol=y_atol
        ):
            raise ValueError("coordinates must be uniformly spaced")
        positive = (
            radius_x_m,
            radius_y_m,
            e_star_pa,
            length_scale_m,
            film_scale_m,
            pressure_scale_pa,
            reynolds_residual_scale,
            film_floor_m,
            krylov_rtol,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError(
                "radii, scales, floor and solver tolerances must be positive"
            )
        if not math.isfinite(reynolds_parameter) or reynolds_parameter < 0.0:
            raise ValueError("reynolds_parameter must be finite and nonnegative")
        if not math.isfinite(target_load_n):
            raise ValueError("target_load_n must be finite")
        if not math.isfinite(compliance_factor) or not 0.0 <= compliance_factor <= 1.0:
            raise ValueError("compliance_factor must be finite and in [0, 1]")
        if krylov_rtol > 1.0:
            raise ValueError("krylov_rtol must not exceed one")
        if krylov_max_iterations is not None and (
            not isinstance(krylov_max_iterations, int)
            or isinstance(krylov_max_iterations, bool)
            or krylov_max_iterations < 1
        ):
            raise ValueError("krylov_max_iterations must be positive or None")
        if (
            not isinstance(gmres_restart, int)
            or isinstance(gmres_restart, bool)
            or gmres_restart < 1
        ):
            raise ValueError("gmres_restart must be a positive integer")
        if preconditioner not in ("ilu", "jacobi", "none"):
            raise ValueError("preconditioner must be 'ilu', 'jacobi', or 'none'")
        if (
            not math.isfinite(ilu_drop_tolerance)
            or ilu_drop_tolerance < 0.0
            or not math.isfinite(ilu_fill_factor)
            or ilu_fill_factor < 1.0
        ):
            raise ValueError("ILU drop tolerance must be nonnegative and fill factor >= 1")
        self.radius_x_m = radius_x_m
        self.radius_y_m = radius_y_m
        self.e_star_pa = e_star_pa
        self.length_scale_m = length_scale_m
        self.film_scale_m = film_scale_m
        self.pressure_scale_pa = pressure_scale_pa
        self.reynolds_parameter = reynolds_parameter
        self.target_load_n = target_load_n
        self.compliance_factor = compliance_factor
        self.reynolds_residual_scale = reynolds_residual_scale
        self.load_residual_scale_n = (
            max(abs(target_load_n), pressure_scale_pa * length_scale_m**2 * 1e-6)
            if load_residual_scale_n is None
            else load_residual_scale_n
        )
        if (
            not math.isfinite(self.load_residual_scale_n)
            or self.load_residual_scale_n <= 0.0
        ):
            raise ValueError("load_residual_scale_n must be finite and positive")
        self.film_floor_m = film_floor_m
        self.krylov_rtol = krylov_rtol
        self.krylov_max_iterations = krylov_max_iterations
        self.gmres_restart = gmres_restart
        self.preconditioner = preconditioner
        self.ilu_drop_tolerance = ilu_drop_tolerance
        self.ilu_fill_factor = ilu_fill_factor
        self.dx = self.dx_m / length_scale_m
        self.dy = self.dy_m / length_scale_m
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="xy")
        self.rigid_geometry_m = (
            self.xx**2 / (2.0 * radius_x_m) + self.yy**2 / (2.0 * radius_y_m)
        )
        self.backend = make_backend("numpy")
        self.elastic_kernel = cell_integrated_kernel(
            (self.y.size, self.x.size),
            self.dx_m,
            self.dy_m,
            e_star_pa,
            self.backend,
        )
        self.interior_shape = (self.y.size - 2, self.x.size - 2)
        self.pressure_size = int(np.prod(self.interior_shape))
        self.operator_evaluations = 0

    def unpack_pressure(self, pressure_interior: ArrayLike) -> NDArray[np.float64]:
        pressure = np.zeros((self.y.size, self.x.size), dtype=np.float64)
        pressure[1:-1, 1:-1] = np.asarray(pressure_interior, dtype=np.float64).reshape(
            self.interior_shape
        )
        return pressure

    def state_fields(
        self, pressure_interior: ArrayLike, h0_dimensionless: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        pressure = self.unpack_pressure(pressure_interior)
        h0_m = h0_dimensionless * self.film_scale_m
        displacement = np.asarray(
            fft_convolve_same(
                pressure * self.pressure_scale_pa, self.elastic_kernel, self.backend
            ),
            dtype=np.float64,
        )
        film = h0_m + self.rigid_geometry_m + self.compliance_factor * displacement
        return pressure, film, displacement

    def residual(self, state: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(state, dtype=np.float64)
        if values.shape != (self.pressure_size + 1,) or not np.all(np.isfinite(values)):
            raise ValueError("state must be a finite pressure-interior vector plus h0/h_ref")
        pressure, film_m, _ = self.state_fields(values[:-1], float(values[-1]))
        if not np.all(np.isfinite(film_m)):
            raise ValueError("state film thickness contains non-finite values")
        if np.any(film_m <= self.film_floor_m):
            raise ValueError("state film thickness is at or below film_floor_m")
        reynolds = reynolds_residual(
            pressure,
            film_m / self.film_scale_m,
            self.dx,
            self.dy,
            self.reynolds_parameter,
        )
        load = _integrated_load_n(self.x, self.y, pressure * self.pressure_scale_pa)
        if not np.all(np.isfinite(reynolds)) or not math.isfinite(load):
            raise ValueError("state residual contains non-finite Reynolds or load values")
        self.operator_evaluations += 1
        return np.concatenate(
            (
                (reynolds / self.reynolds_residual_scale).ravel(),
                np.array([(load - self.target_load_n) / self.load_residual_scale_n]),
            )
        )

    def compliance_derivative(self, state: ArrayLike) -> NDArray[np.float64]:
        """Differentiate the scaled residual in compliance at fixed pressure/h0.

        This includes mobility and entrainment changes through dH/dgamma =
        displacement/h_ref. The fixed-state load derivative is exactly zero.
        No parameter perturbation or mutation is required, including at 0 or 1.
        """
        return self.jacobian_vector(
            state, np.zeros(self.pressure_size + 1), compliance_direction=1.0
        )

    def jacobian_vector(
        self, state: ArrayLike, direction: ArrayLike, *, compliance_direction: float = 0.0
    ) -> NDArray[np.float64]:
        """Apply the exact discrete state/optional compliance directional derivative."""
        z = np.asarray(state, dtype=np.float64)
        v = np.asarray(direction, dtype=np.float64)
        if z.shape != (self.pressure_size + 1,) or v.shape != z.shape:
            raise ValueError("state and direction must have pressure_size + 1 entries")
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(v)):
            raise ValueError("state and direction must contain only finite values")
        if not math.isfinite(compliance_direction):
            raise ValueError("compliance_direction must be finite")
        if not np.any(v) and compliance_direction == 0.0:
            return np.zeros_like(v)
        pressure, film_m, displacement = self.state_fields(z[:-1], float(z[-1]))
        if not np.all(np.isfinite(film_m)) or np.any(film_m <= self.film_floor_m):
            raise ValueError("Jacobian state must have film thickness above the floor")

        pressure_direction = self.unpack_pressure(v[:-1])
        displacement_direction = np.asarray(
            fft_convolve_same(
                pressure_direction * self.pressure_scale_pa,
                self.elastic_kernel,
                self.backend,
            ),
            dtype=np.float64,
        )
        film_direction = (
            v[-1] * self.film_scale_m
            + self.compliance_factor * displacement_direction
            + compliance_direction * displacement
        ) / self.film_scale_m

        h = film_m / self.film_scale_m
        h3 = h**3
        dh3 = 3.0 * h**2 * film_direction
        p_center = pressure[1:-1, 1:-1]
        dp_center = pressure_direction[1:-1, 1:-1]
        h3_center = h3[1:-1, 1:-1]
        dh3_center = dh3[1:-1, 1:-1]

        def directional_face_flux(
            p_neighbor: NDArray[np.float64],
            p_center_field: NDArray[np.float64],
            dp_neighbor: NDArray[np.float64],
            dp_center_field: NDArray[np.float64],
            h3_neighbor: NDArray[np.float64],
            h3_center_field: NDArray[np.float64],
            dh3_neighbor: NDArray[np.float64],
            dh3_center_field: NDArray[np.float64],
            spacing: float,
        ) -> NDArray[np.float64]:
            conductance = 0.5 * (h3_center_field + h3_neighbor)
            conductance_direction = 0.5 * (dh3_center_field + dh3_neighbor)
            pressure_difference = p_neighbor - p_center_field
            pressure_difference_direction = dp_neighbor - dp_center_field
            return (
                conductance_direction * pressure_difference
                + conductance * pressure_difference_direction
            ) / spacing**2

        east = directional_face_flux(
            pressure[1:-1, 2:],
            p_center,
            pressure_direction[1:-1, 2:],
            dp_center,
            h3[1:-1, 2:],
            h3_center,
            dh3[1:-1, 2:],
            dh3_center,
            self.dx,
        )
        west = directional_face_flux(
            pressure[1:-1, :-2],
            p_center,
            pressure_direction[1:-1, :-2],
            dp_center,
            h3[1:-1, :-2],
            h3_center,
            dh3[1:-1, :-2],
            dh3_center,
            self.dx,
        )
        north = directional_face_flux(
            pressure[2:, 1:-1],
            p_center,
            pressure_direction[2:, 1:-1],
            dp_center,
            h3[2:, 1:-1],
            h3_center,
            dh3[2:, 1:-1],
            dh3_center,
            self.dy,
        )
        south = directional_face_flux(
            pressure[:-2, 1:-1],
            p_center,
            pressure_direction[:-2, 1:-1],
            dp_center,
            h3[:-2, 1:-1],
            h3_center,
            dh3[:-2, 1:-1],
            dh3_center,
            self.dy,
        )
        reynolds_direction = east + west + north + south
        reynolds_direction -= self.reynolds_parameter * (
            film_direction[1:-1, 2:] - film_direction[1:-1, :-2]
        ) / (2.0 * self.dx)
        load_direction = float(np.dot(self.load_gradient(), v[:-1]))
        return np.concatenate(
            (
                (reynolds_direction / self.reynolds_residual_scale).ravel(),
                np.array([load_direction]),
            )
        )

    def load_gradient(self) -> NDArray[np.float64]:
        """Exact load-row derivative with respect to interior dimensionless P."""
        weights_x = np.ones(self.x.size, dtype=np.float64)
        weights_y = np.ones(self.y.size, dtype=np.float64)
        weights_x[[0, -1]] = 0.5
        weights_y[[0, -1]] = 0.5
        weights = np.outer(weights_y, weights_x)[1:-1, 1:-1]
        return (
            self.pressure_scale_pa
            * self.dx_m
            * self.dy_m
            * weights.ravel()
            / self.load_residual_scale_n
        )


def _integrated_load_n(
    x_m: NDArray[np.float64],
    y_m: NDArray[np.float64],
    pressure_pa: NDArray[np.float64],
) -> float:
    return float(trapezoid(trapezoid(pressure_pa, x_m, axis=1), y_m))


def build_frozen_film_preconditioner(
    problem: BorderedReynoldsElasticityProblem,
    state: ArrayLike,
    *,
    strategy: str | None = None,
) -> FrozenFilmPreconditioner:
    """Build the scaled pressure preconditioner at a feasible current state.

    The matrix is the negative conservative Reynolds diffusion operator
    divided by the configured residual scale, matching the sign and scaling
    of the pressure Jacobian. The returned factorization/operator is intended
    to be reused for both Schur solves and the full bordered fallback within
    one Newton correction. ILU failures fall back explicitly to Jacobi and
    then identity if the diagonal itself is invalid.
    """
    requested = problem.preconditioner if strategy is None else strategy
    if requested not in ("ilu", "jacobi", "none"):
        raise ValueError("strategy must be 'ilu', 'jacobi', or 'none'")
    z = np.asarray(state, dtype=np.float64)
    if z.shape != (problem.pressure_size + 1,) or not np.all(np.isfinite(z)):
        raise ValueError("state must be a finite pressure-interior vector plus h0/h_ref")
    _, film_m, _ = problem.state_fields(z[:-1], float(z[-1]))
    if not np.all(np.isfinite(film_m)) or np.any(film_m <= problem.film_floor_m):
        raise ValueError("preconditioner state film thickness must exceed the floor")
    reynolds_matrix, _ = assemble_reynolds_system(
        film_m / problem.film_scale_m,
        problem.dx,
        problem.dy,
        problem.reynolds_parameter,
    )
    matrix = (-reynolds_matrix / problem.reynolds_residual_scale).tocsr()
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    used = requested
    try:
        if requested == "ilu":
            factorization = spilu(
                matrix.tocsc(),
                drop_tol=problem.ilu_drop_tolerance,
                fill_factor=problem.ilu_fill_factor,
            )
            operator = LinearOperator(
                matrix.shape, matvec=factorization.solve, dtype=np.float64
            )
        elif requested == "jacobi":
            if np.any(~np.isfinite(diagonal)) or np.any(diagonal == 0.0):
                raise ArithmeticError("Jacobi preconditioner has an invalid diagonal")
            inverse_diagonal = 1.0 / diagonal
            operator = LinearOperator(
                matrix.shape,
                matvec=lambda vector: inverse_diagonal * vector,
                dtype=np.float64,
            )
        else:
            operator = LinearOperator(
                matrix.shape,
                matvec=lambda vector: np.asarray(vector, dtype=np.float64),
                dtype=np.float64,
            )
    except (ArithmeticError, RuntimeError, ValueError) as error:
        logger.warning(
            "%s Reynolds preconditioner failed (%s); falling back to Jacobi",
            requested,
            error,
        )
        used = "jacobi"
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal == 0.0):
            logger.warning("Jacobi fallback has invalid diagonal; using identity")
            used = "none"
            operator = LinearOperator(
                matrix.shape,
                matvec=lambda vector: np.asarray(vector, dtype=np.float64),
                dtype=np.float64,
            )
        else:
            inverse_diagonal = 1.0 / diagonal
            operator = LinearOperator(
                matrix.shape,
                matvec=lambda vector: inverse_diagonal * vector,
                dtype=np.float64,
            )
    return FrozenFilmPreconditioner(matrix, operator, used)


def make_feasible_initial_state(
    problem: BorderedReynoldsElasticityProblem,
    initial_pressure_dimensionless: ArrayLike,
    initial_h0_m: float,
    *,
    film_margin_fraction: float = 1e-3,
) -> FeasibleInitialState:
    """Raise ``h0`` explicitly when needed to make a finite initial film feasible.

    Pressure is retained exactly, including signed values. This is an
    initialization aid, not a pressure or film clamp: it changes only the
    scalar starting offset and reports the required change. The nonlinear
    equations still determine the converged offset. Non-finite elastic fields
    and invalid pressure boundaries are rejected.
    """
    pressure = np.asarray(initial_pressure_dimensionless, dtype=np.float64)
    if pressure.shape != (problem.y.size, problem.x.size):
        raise ValueError("initial pressure must have shape (ny, nx)")
    if (
        not np.all(np.isfinite(pressure))
        or np.any(pressure[0, :] != 0.0)
        or np.any(pressure[-1, :] != 0.0)
        or np.any(pressure[:, 0] != 0.0)
        or np.any(pressure[:, -1] != 0.0)
    ):
        raise ValueError("initial pressure must be finite and satisfy zero boundaries")
    if not math.isfinite(initial_h0_m) or initial_h0_m <= 0.0:
        raise ValueError("initial_h0_m must be finite and positive")
    if (
        not math.isfinite(film_margin_fraction)
        or film_margin_fraction <= 0.0
    ):
        raise ValueError("film_margin_fraction must be finite and positive")
    state = np.concatenate(
        (
            pressure[1:-1, 1:-1].ravel(),
            [initial_h0_m / problem.film_scale_m],
        )
    )
    _, film, _ = problem.state_fields(state[:-1], float(state[-1]))
    if not np.all(np.isfinite(film)):
        raise ValueError("initial film contains non-finite values")
    original_minimum = float(np.min(film))
    target_minimum = problem.film_floor_m + film_margin_fraction * problem.film_scale_m
    h0_adjustment = max(0.0, target_minimum - original_minimum)
    adjusted_h0 = initial_h0_m + h0_adjustment
    adjusted_film = film + h0_adjustment
    if np.min(adjusted_film) <= problem.film_floor_m:
        raise RuntimeError("could not construct a strictly feasible initial film")
    return FeasibleInitialState(
        pressure_dimensionless=pressure.copy(),
        central_film_offset_m=adjusted_h0,
        original_minimum_film_thickness_m=original_minimum,
        adjusted_minimum_film_thickness_m=float(np.min(adjusted_film)),
        h0_adjustment_m=h0_adjustment,
    )


def solve_bordered_correction(
    problem: BorderedReynoldsElasticityProblem,
    state: ArrayLike,
    *,
    schur_tolerance: float = 1e-10,
    bordered_tolerance: float | None = None,
    krylov_rtol: float | None = None,
    krylov_max_iterations: int | None = None,
    preconditioner_strategy: str | None = None,
) -> BorderedCorrection:
    """Solve a matrix-free Newton system through a Schur complement.

    The pressure block is solved by GMRES; its frozen-film Reynolds matrix
    provides an ILU preconditioner. If the Schur denominator is unsafe or the
    reconstructed bordered correction is inaccurate, the method falls back
    to GMRES on the complete bordered Jacobian with a block preconditioner.
    """
    z = np.asarray(state, dtype=np.float64)
    requested_rtol = problem.krylov_rtol if krylov_rtol is None else krylov_rtol
    maxiter = (
        problem.krylov_max_iterations
        if krylov_max_iterations is None
        else krylov_max_iterations
    )
    if (
        not math.isfinite(requested_rtol)
        or requested_rtol <= 0.0
        or requested_rtol > 1.0
        or not math.isfinite(schur_tolerance)
        or schur_tolerance <= 0.0
        or (
            bordered_tolerance is not None
            and (
                not math.isfinite(bordered_tolerance)
                or bordered_tolerance <= 0.0
            )
        )
        or (
            maxiter is not None
            and (
                not isinstance(maxiter, int)
                or isinstance(maxiter, bool)
                or maxiter < 1
            )
        )
    ):
        raise ValueError("Krylov and bordered tolerances must be finite and positive")
    rtol = min(requested_rtol, MAX_INEXACT_KRYLOV_RTOL)
    linear_residual_tolerance = min(
        MAX_INEXACT_KRYLOV_RTOL,
        max(1e-7, 2.0 * rtol)
        if bordered_tolerance is None
        else bordered_tolerance,
    )
    residual = problem.residual(z)
    pressure_count = problem.pressure_size
    h_direction = np.zeros_like(z)
    h_direction[-1] = 1.0
    b = problem.jacobian_vector(z, h_direction)[:-1]
    load_gradient = problem.load_gradient()
    restart = min(problem.gmres_restart, pressure_count + 1)

    requested_preconditioner = (
        problem.preconditioner
        if preconditioner_strategy is None
        else preconditioner_strategy
    )
    if requested_preconditioner not in ("ilu", "jacobi", "none"):
        raise ValueError("preconditioner_strategy must be 'ilu', 'jacobi', or 'none'")
    frozen_preconditioner = build_frozen_film_preconditioner(
        problem, z, strategy=requested_preconditioner
    )
    preconditioner = frozen_preconditioner.operator
    preconditioner_used = frozen_preconditioner.strategy

    def pressure_matvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        direction = np.zeros_like(z)
        direction[:-1] = vector
        return problem.jacobian_vector(z, direction)[:-1]

    pressure_operator = LinearOperator(
        (pressure_count, pressure_count), matvec=pressure_matvec, dtype=np.float64
    )
    iterations = 0
    krylov_diagnostics: list[KrylovSolveDiagnostic] = []
    current_preconditioned_residuals: list[float] = []

    def callback(residual_norm: float) -> None:
        nonlocal iterations
        iterations += 1
        current_preconditioned_residuals.append(float(residual_norm))

    def solve_pressure(
        rhs: NDArray[np.float64], label: str
    ) -> tuple[NDArray[np.float64], float]:
        nonlocal iterations
        iterations = 0
        current_preconditioned_residuals.clear()
        solution, info = gmres(
            pressure_operator,
            rhs,
            M=preconditioner,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            restart=restart,
            callback=callback,
            callback_type="pr_norm",
        )
        solution = np.asarray(solution, dtype=np.float64)
        relative_residual = float(
            np.linalg.norm(pressure_operator @ solution - rhs)
            / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
        )
        termination = (
            "converged"
            if info == 0 and math.isfinite(relative_residual)
            else f"GMRES info={info}"
        )
        krylov_diagnostics.append(
            KrylovSolveDiagnostic(
                label=label,
                preconditioner=preconditioner_used,
                requested_rtol=requested_rtol,
                effective_rtol=rtol,
                iterations=iterations,
                info=int(info),
                true_relative_residual=relative_residual,
                preconditioned_residual_history=tuple(
                    current_preconditioned_residuals
                ),
                termination_reason=termination,
            )
        )
        if (
            info != 0
            or not np.all(np.isfinite(solution))
            or not math.isfinite(relative_residual)
            or relative_residual
            > min(MAX_INEXACT_KRYLOV_RTOL, max(1.25 * rtol, 1e-12))
        ):
            raise RuntimeError(
                f"{label} GMRES failed or was inaccurate: {termination}; "
                f"true relative residual={relative_residual:.3e}, requested rtol={rtol:.3e}"
            )
        return solution, relative_residual

    try:
        x, residual_x = solve_pressure(-residual[:-1], "pressure_residual")
        count_x = iterations
        y, residual_y = solve_pressure(b, "h0_response")
        count_y = iterations
        pressure_krylov_residual = max(residual_x, residual_y)
        schur = -float(np.dot(load_gradient, y))
        schur_scale = max(
            np.linalg.norm(load_gradient) * np.linalg.norm(y),
            np.finfo(float).tiny,
        )
        indicator = abs(schur) / schur_scale
        if not math.isfinite(schur) or indicator <= schur_tolerance:
            raise ArithmeticError("Schur denominator is singular or ill-conditioned")
        delta_h0 = (-residual[-1] - float(np.dot(load_gradient, x))) / schur
        delta_p = x - y * delta_h0
        correction = np.concatenate((delta_p, [delta_h0]))
        full_jacobian_action = np.concatenate(
            (
                pressure_matvec(delta_p) + b * delta_h0,
                [float(np.dot(load_gradient, delta_p))],
            )
        )
        relative = float(
            np.linalg.norm(full_jacobian_action + residual)
            / max(np.linalg.norm(residual), np.finfo(float).tiny)
        )
        if (
            not math.isfinite(relative)
            or relative > linear_residual_tolerance
        ):
            raise ArithmeticError("Schur correction failed the bordered residual check")
        return BorderedCorrection(
            pressure_correction=delta_p.reshape(problem.interior_shape),
            h0_correction_m=delta_h0 * problem.film_scale_m,
            pressure_krylov_iterations=count_x + count_y,
            pressure_krylov_residual_relative=pressure_krylov_residual,
            schur_denominator=schur,
            schur_condition_indicator=indicator,
            bordered_residual_relative=relative,
            used_fallback=False,
            preconditioner_used=preconditioner_used,
            krylov_diagnostics=tuple(krylov_diagnostics),
        )
    except (ArithmeticError, RuntimeError) as schur_error:
        logger.info(
            "Schur solve unavailable (%s); using full bordered GMRES fallback",
            schur_error,
        )
        def full_matvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            return problem.jacobian_vector(z, vector)

        full_operator = LinearOperator(
            (pressure_count + 1, pressure_count + 1),
            matvec=full_matvec,
            dtype=np.float64,
        )
        block_inverse = LinearOperator(
            (pressure_count + 1, pressure_count + 1),
            matvec=lambda rhs: np.concatenate(
                (frozen_preconditioner.operator @ rhs[:-1], [rhs[-1]])
            ),
            dtype=np.float64,
        )
        iterations = 0
        current_preconditioned_residuals.clear()
        correction, info = gmres(
            full_operator,
            -residual,
            M=block_inverse,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            restart=restart,
            callback=callback,
            callback_type="pr_norm",
        )
        correction = np.asarray(correction, dtype=np.float64)
        linear_residual = full_operator @ correction + residual
        relative = float(
            np.linalg.norm(linear_residual)
            / max(np.linalg.norm(residual), np.finfo(float).tiny)
        )
        termination = (
            "converged"
            if info == 0 and math.isfinite(relative)
            else f"GMRES info={info}"
        )
        krylov_diagnostics.append(
            KrylovSolveDiagnostic(
                label="full_bordered_fallback",
                preconditioner=preconditioner_used,
                requested_rtol=requested_rtol,
                effective_rtol=rtol,
                iterations=iterations,
                info=int(info),
                true_relative_residual=relative,
                preconditioned_residual_history=tuple(
                    current_preconditioned_residuals
                ),
                termination_reason=termination,
            )
        )
        if (
            info != 0
            or not np.all(np.isfinite(correction))
            or not math.isfinite(relative)
            or relative > linear_residual_tolerance
        ):
            raise BorderedLinearSolveError(
                f"full bordered GMRES fallback failed or was inaccurate: "
                f"{termination}; true relative residual={relative:.3e}; "
                f"Schur failure: {schur_error}",
                tuple(krylov_diagnostics),
            ) from None
        return BorderedCorrection(
            pressure_correction=correction[:-1].reshape(problem.interior_shape),
            h0_correction_m=float(correction[-1]) * problem.film_scale_m,
            pressure_krylov_iterations=iterations,
            pressure_krylov_residual_relative=relative,
            schur_denominator=math.nan,
            schur_condition_indicator=0.0,
            bordered_residual_relative=relative,
            used_fallback=True,
            preconditioner_used=preconditioner_used,
            krylov_diagnostics=tuple(krylov_diagnostics),
        )


def compare_bordered_preconditioners(
    problem: BorderedReynoldsElasticityProblem,
    state: ArrayLike,
    *,
    strategies: tuple[str, ...] = ("ilu", "jacobi", "none"),
    krylov_rtol: float | None = None,
) -> tuple[PreconditionerBenchmark, ...]:
    """Time the same bordered Newton correction with selected preconditioners.

    Each strategy builds its sparse preconditioner once for the frozen-film
    pressure block and reuses it for both Schur solves and any bordered
    fallback. The nonlinear state and configured problem preconditioner are
    not modified. Timings include assembly and factorization.
    """
    if not strategies:
        raise ValueError("strategies must contain at least one preconditioner")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates")
    invalid = set(strategies) - {"ilu", "jacobi", "none"}
    if invalid:
        raise ValueError(f"unsupported preconditioner strategies: {sorted(invalid)}")
    results: list[PreconditionerBenchmark] = []
    for strategy in strategies:
        started = time.perf_counter()
        try:
            correction = solve_bordered_correction(
                problem,
                state,
                krylov_rtol=krylov_rtol,
                preconditioner_strategy=strategy,
            )
        except BorderedLinearSolveError as error:
            elapsed = time.perf_counter() - started
            diagnostics = error.krylov_diagnostics
            results.append(
                PreconditionerBenchmark(
                    preconditioner=strategy,
                    preconditioner_used=(
                        diagnostics[-1].preconditioner if diagnostics else strategy
                    ),
                    converged=False,
                    elapsed_seconds=elapsed,
                    krylov_iterations=sum(item.iterations for item in diagnostics),
                    maximum_true_relative_residual=max(
                        (item.true_relative_residual for item in diagnostics),
                        default=math.inf,
                    ),
                    bordered_residual_relative=math.inf,
                    termination_reason=str(error),
                )
            )
            continue
        elapsed = time.perf_counter() - started
        results.append(
            PreconditionerBenchmark(
                preconditioner=strategy,
                preconditioner_used=correction.preconditioner_used,
                converged=True,
                elapsed_seconds=elapsed,
                krylov_iterations=sum(
                    item.iterations for item in correction.krylov_diagnostics
                ),
                maximum_true_relative_residual=max(
                    (
                        item.true_relative_residual
                        for item in correction.krylov_diagnostics
                    ),
                    default=0.0,
                ),
                bordered_residual_relative=correction.bordered_residual_relative,
                termination_reason="converged",
            )
        )
    return tuple(results)


def _merit_directional_derivative(
    problem: BorderedReynoldsElasticityProblem,
    state: NDArray[np.float64],
    residual: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> float:
    """Differentiate the scaled least-squares merit along a Newton direction."""
    residual_direction = problem.jacobian_vector(state, direction)
    pressure_count = problem.pressure_size
    return float(
        np.dot(residual[:-1], residual_direction[:-1]) / pressure_count
        + residual[-1] * residual_direction[-1]
    )


@dataclass(frozen=True)
class _ArmijoLineSearchResult:
    accepted_state: NDArray[np.float64] | None
    feasible_step_limit: float
    accepted_step_length: float
    rejected_trials: int
    last_trial_step_length: float
    last_trial_merit: float
    termination_reason: str


def _armijo_line_search(
    problem: BorderedReynoldsElasticityProblem,
    state: NDArray[np.float64],
    direction: NDArray[np.float64],
    current_merit: float,
    merit_slope: float,
    *,
    fraction_to_boundary: float,
    armijo_constant: float,
    maximum_backtracks: int,
) -> _ArmijoLineSearchResult:
    """Find a strictly feasible common step satisfying the scaled Armijo test."""
    _, film_change, _ = problem.state_fields(
        state[:-1] + direction[:-1], float(state[-1] + direction[-1])
    )
    _, film_current, _ = problem.state_fields(state[:-1], float(state[-1]))
    delta_film = film_change - film_current
    negative = delta_film < 0.0
    feasible_limit = 1.0
    if np.any(negative):
        feasible_limit = min(
            1.0,
            fraction_to_boundary
            * float(
                np.min(
                    (film_current[negative] - problem.film_floor_m)
                    / -delta_film[negative]
                )
            ),
        )
    if not math.isfinite(feasible_limit) or feasible_limit <= 0.0:
        return _ArmijoLineSearchResult(
            None, feasible_limit, 0.0, 0, 0.0, math.inf,
            "no_positive_feasible_step",
        )

    step_length = feasible_limit
    rejected = 0
    last_trial_merit = math.inf
    last_trial_step_length = step_length
    for _ in range(maximum_backtracks + 1):
        last_trial_step_length = step_length
        trial_state = state + step_length * direction
        try:
            trial_residual = problem.residual(trial_state)
            trial_reynolds = float(
                np.linalg.norm(trial_residual[:-1]) / math.sqrt(problem.pressure_size)
            )
            trial_load = float(trial_residual[-1])
            trial_merit = 0.5 * (trial_reynolds**2 + trial_load**2)
        except ValueError:
            trial_merit = math.inf
        last_trial_merit = trial_merit
        if (
            math.isfinite(trial_merit)
            and trial_merit
            <= current_merit + armijo_constant * step_length * merit_slope
        ):
            return _ArmijoLineSearchResult(
                trial_state,
                feasible_limit,
                step_length,
                rejected,
                last_trial_step_length,
                last_trial_merit,
                "armijo_satisfied",
            )
        rejected += 1
        step_length *= 0.5
    return _ArmijoLineSearchResult(
        None,
        feasible_limit,
        0.0,
        rejected,
        last_trial_step_length,
        last_trial_merit,
        "backtracking_exhausted",
    )


def solve_monolithic_full_film_load(
    problem: BorderedReynoldsElasticityProblem,
    initial_pressure_dimensionless: ArrayLike,
    initial_h0_m: float,
    *,
    reynolds_tolerance: float = 1e-8,
    load_tolerance: float = 1e-6,
    max_iterations: int = 30,
    fraction_to_boundary: float = 0.99,
    armijo_constant: float = 1e-4,
    maximum_backtracks: int = 20,
    log_every: int = 1,
    initial_film_strategy: str = "strict",
) -> MonolithicLoadSolution:
    """Solve simultaneous pressure and h0 equations with dual line searches.

    First, a fraction-to-boundary feasibility search computes the maximum
    common step preserving film thickness above ``problem.film_floor_m``.
    Then Armijo backtracking on the scaled combined residual chooses the
    accepted step. Pressure and h0 always use this same step length.

    The pressure array has shape ``(ny, nx)`` and is dimensionless; all
    coordinates, ``h0``, film thickness, and elastic displacement are SI.
    ``reynolds_tolerance`` bounds the scaled Reynolds RMS,
    ``||R_p||_2/sqrt(N)``, and ``load_tolerance`` bounds the absolute scaled
    load residual. The film is reconstructed exactly from rigid geometry and
    elastic displacement at each residual evaluation. Signed negative
    pressure is retained; this function is a mathematical full-film solver,
    not a cavitation or physically validated EHL model.
    """
    if initial_film_strategy not in ("strict", "raise_h0"):
        raise ValueError("initial_film_strategy must be 'strict' or 'raise_h0'")
    if not 0.0 < fraction_to_boundary < 1.0:
        raise ValueError("fraction_to_boundary must be in (0, 1)")
    if not math.isfinite(armijo_constant) or not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must be in (0, 1)")
    if not all(
        math.isfinite(tolerance) and tolerance > 0.0
        for tolerance in (reynolds_tolerance, load_tolerance)
    ):
        raise ValueError("nonlinear tolerances must be finite and positive")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
        or not isinstance(maximum_backtracks, int)
        or isinstance(maximum_backtracks, bool)
        or maximum_backtracks < 0
        or not isinstance(log_every, int)
        or isinstance(log_every, bool)
        or log_every < 1
    ):
        raise ValueError(
            "iteration limits and log_every must be positive/nonnegative integers"
        )
    pressure = np.asarray(initial_pressure_dimensionless, dtype=np.float64)
    if pressure.shape != (problem.y.size, problem.x.size):
        raise ValueError("initial pressure must have shape (ny, nx)")
    if (
        not np.all(np.isfinite(pressure))
        or np.any(pressure[0, :] != 0.0)
        or np.any(pressure[-1, :] != 0.0)
        or np.any(pressure[:, 0] != 0.0)
        or np.any(pressure[:, -1] != 0.0)
    ):
        raise ValueError("initial pressure must be finite and satisfy zero boundaries")
    if not math.isfinite(initial_h0_m) or initial_h0_m <= 0.0:
        raise ValueError("initial_h0_m must be finite and positive")
    h0_adjustment_m = 0.0
    if initial_film_strategy == "raise_h0":
        feasible = make_feasible_initial_state(
            problem,
            pressure,
            initial_h0_m,
        )
        pressure = feasible.pressure_dimensionless
        initial_h0_m = feasible.central_film_offset_m
        h0_adjustment_m = feasible.h0_adjustment_m
        if h0_adjustment_m > 0.0:
            logger.info(
                "raised initial h0 by %.6e m to establish film positivity; "
                "original minimum film was %.6e m",
                h0_adjustment_m,
                feasible.original_minimum_film_thickness_m,
            )
    state = np.concatenate(
        (pressure[1:-1, 1:-1].ravel(), [initial_h0_m / problem.film_scale_m])
    )
    history: list[MonolithicIteration] = []
    previous_nonlinear_norm: float | None = None
    for iteration in range(1, max_iterations + 1):
        residual = problem.residual(state)
        reynolds_norm = float(np.linalg.norm(residual[:-1]) / math.sqrt(problem.pressure_size))
        load_scaled = float(residual[-1])
        load_residual_n = load_scaled * problem.load_residual_scale_n
        pressure, film, displacement = problem.state_fields(state[:-1], float(state[-1]))
        load_norm = abs(load_scaled)
        merit = 0.5 * (reynolds_norm**2 + load_norm**2)
        nonlinear_norm = math.sqrt(2.0 * merit)
        if previous_nonlinear_norm is None:
            forcing_tolerance = MAX_INEXACT_KRYLOV_RTOL
        else:
            residual_ratio = nonlinear_norm / max(
                previous_nonlinear_norm, np.finfo(np.float64).tiny
            )
            forcing_tolerance = min(
                MAX_INEXACT_KRYLOV_RTOL,
                max(1e-4, 0.9 * residual_ratio**1.5),
            )
        rigid_film = film - problem.compliance_factor * displacement
        peak_elastic_displacement = float(
            np.max(np.abs(problem.compliance_factor * displacement))
        )
        failure_diagnostics: dict[str, object] = {
            "iteration": iteration,
            "reynolds_residual_scaled": reynolds_norm,
            "load_residual_n": load_residual_n,
            "load_residual_scaled": load_scaled,
            "minimum_film_thickness_m": float(np.min(film)),
            "minimum_pressure_pa": float(
                np.min(pressure * problem.pressure_scale_pa)
            ),
            "maximum_pressure_pa": float(np.max(pressure * problem.pressure_scale_pa)),
            "maximum_elastic_displacement_m": peak_elastic_displacement,
            "minimum_rigid_film_thickness_m": float(np.min(rigid_film)),
            "elastic_deformation_to_rigid_gap_ratio": (
                peak_elastic_displacement
                / max(float(np.min(rigid_film)), np.finfo(float).tiny)
            ),
            "operator_evaluations": problem.operator_evaluations,
        }
        if reynolds_norm <= reynolds_tolerance and load_norm <= load_tolerance:
            if np.min(film) <= problem.film_floor_m:
                raise MonolithicNewtonError("converged state violates film floor", tuple(history))
            independently_evaluated_load = _integrated_load_n(
                problem.x, problem.y, pressure * problem.pressure_scale_pa
            )
            logger.info(
                "monolithic Newton converged at iteration %d: scaled Reynolds RMS=%.3e, "
                "load residual=%.3e N, h_min=%.6e m",
                iteration - 1,
                reynolds_norm,
                independently_evaluated_load - problem.target_load_n,
                float(np.min(film)),
            )
            return MonolithicLoadSolution(
                pressure_dimensionless=pressure,
                pressure_pa=pressure * problem.pressure_scale_pa,
                film_thickness_m=film,
                elastic_displacement_m=displacement,
                central_film_offset_m=float(state[-1]) * problem.film_scale_m,
                target_load_n=problem.target_load_n,
                calculated_load_n=independently_evaluated_load,
                load_residual_n=independently_evaluated_load - problem.target_load_n,
                history=tuple(history),
                operator_evaluations=problem.operator_evaluations,
                initial_h0_adjustment_m=h0_adjustment_m,
            )

        linear_diagnostics: list[KrylovSolveDiagnostic] = []
        linear_refinement_attempts = 0
        total_rejected_trials = 0
        while True:
            try:
                correction = solve_bordered_correction(
                    problem,
                    state,
                    krylov_rtol=forcing_tolerance,
                )
            except (ArithmeticError, RuntimeError) as error:
                if isinstance(error, BorderedLinearSolveError):
                    linear_diagnostics.extend(error.krylov_diagnostics)
                failure_diagnostics["krylov_diagnostics"] = tuple(linear_diagnostics)
                failure_diagnostics["linear_solver_error"] = str(error)
                raise MonolithicNewtonError(
                    f"bordered Newton correction failed: {error}",
                    tuple(history),
                    failure_diagnostics,
                ) from error
            linear_diagnostics.extend(correction.krylov_diagnostics)
            direction = np.concatenate(
                (
                    correction.pressure_correction.ravel(),
                    [correction.h0_correction_m / problem.film_scale_m],
                )
            )
            merit_slope = _merit_directional_derivative(
                problem, state, residual, direction
            )
            for recovery in range(2):
                if math.isfinite(merit_slope) and merit_slope < 0.0:
                    break
                try:
                    recovery_tolerance = max(
                        10.0 * np.finfo(np.float64).eps,
                        min(
                            forcing_tolerance * 0.1 ** (recovery + 1),
                            problem.krylov_rtol * 0.1**recovery,
                        ),
                    )
                    correction = solve_bordered_correction(
                        problem,
                        state,
                        krylov_rtol=recovery_tolerance,
                    )
                except (ArithmeticError, RuntimeError) as error:
                    if isinstance(error, BorderedLinearSolveError):
                        linear_diagnostics.extend(error.krylov_diagnostics)
                    failure_diagnostics["krylov_diagnostics"] = tuple(
                        linear_diagnostics
                    )
                    failure_diagnostics["linear_solver_error"] = str(error)
                    raise MonolithicNewtonError(
                        f"tighter Krylov recovery failed: {error}",
                        tuple(history),
                        failure_diagnostics,
                    ) from error
                linear_diagnostics.extend(correction.krylov_diagnostics)
                direction = np.concatenate(
                    (
                        correction.pressure_correction.ravel(),
                        [correction.h0_correction_m / problem.film_scale_m],
                    )
                )
                merit_slope = _merit_directional_derivative(
                    problem, state, residual, direction
                )
            if not math.isfinite(merit_slope) or merit_slope >= 0.0:
                failure_diagnostics.update(
                    {
                        "merit_directional_derivative": merit_slope,
                        "current_merit": merit,
                        "krylov_diagnostics": tuple(linear_diagnostics),
                    }
                )
                raise MonolithicNewtonError(
                    "Newton correction remained non-descent after tighter Krylov "
                    "recovery",
                    tuple(history),
                    failure_diagnostics,
                )

            line_search = _armijo_line_search(
                problem,
                state,
                direction,
                merit,
                merit_slope,
                fraction_to_boundary=fraction_to_boundary,
                armijo_constant=armijo_constant,
                maximum_backtracks=maximum_backtracks,
            )
            total_rejected_trials += line_search.rejected_trials
            if line_search.accepted_state is not None:
                state = line_search.accepted_state
                break
            achieved_linear_error = max(
                correction.bordered_residual_relative,
                max(
                    (
                        diagnostic.true_relative_residual
                        for diagnostic in correction.krylov_diagnostics
                    ),
                    default=math.inf,
                ),
            )
            refinement_accuracy_limit = max(
                1e-7, 2.0 * problem.krylov_rtol
            )
            if (
                linear_refinement_attempts < 2
                and achieved_linear_error > refinement_accuracy_limit
            ):
                linear_refinement_attempts += 1
                forcing_tolerance = max(
                    10.0 * np.finfo(np.float64).eps,
                    min(forcing_tolerance * 0.1, problem.krylov_rtol),
                )
                continue

            failure_diagnostics.update(
                {
                    "feasible_step_limit": line_search.feasible_step_limit,
                    "last_trial_step_length": line_search.last_trial_step_length,
                    "rejected_line_search_trials": total_rejected_trials,
                    "line_search_termination_reason": line_search.termination_reason,
                    "linear_refinement_attempts": linear_refinement_attempts,
                    "merit_directional_derivative": merit_slope,
                    "current_merit": merit,
                    "last_trial_merit": line_search.last_trial_merit,
                    "achieved_linear_error": achieved_linear_error,
                    "refinement_accuracy_limit": refinement_accuracy_limit,
                    "armijo_merit_limit": (
                        merit
                        + armijo_constant
                        * line_search.last_trial_step_length
                        * merit_slope
                    ),
                    "krylov_diagnostics": tuple(linear_diagnostics),
                    "schur_denominator": correction.schur_denominator,
                    "schur_condition_indicator": correction.schur_condition_indicator,
                    "bordered_residual_relative": correction.bordered_residual_relative,
                }
            )
            raise MonolithicNewtonError(
                "dual line search could not find an Armijo-decreasing feasible step",
                tuple(history),
                failure_diagnostics,
            )
        feasible_limit = line_search.feasible_step_limit
        step_length = line_search.accepted_step_length
        rejected = total_rejected_trials
        accepted_pressure, accepted_film, _ = problem.state_fields(
            state[:-1], float(state[-1])
        )
        history.append(
            MonolithicIteration(
                iteration=iteration,
                reynolds_residual_scaled=reynolds_norm,
                load_residual_n=load_residual_n,
                load_residual_scaled=load_scaled,
                pressure_correction_relative=float(
                    np.linalg.norm(step_length * correction.pressure_correction)
                    / max(
                        np.linalg.norm(state[:-1]),
                        np.finfo(np.float64).tiny,
                    )
                ),
                h0_correction_relative=abs(
                    step_length * correction.h0_correction_m
                    / max(abs(float(state[-1]) * problem.film_scale_m), problem.film_floor_m)
                ),
                minimum_film_thickness_m=float(np.min(accepted_film)),
                minimum_pressure_pa=float(np.min(accepted_pressure) * problem.pressure_scale_pa),
                maximum_pressure_pa=float(np.max(accepted_pressure) * problem.pressure_scale_pa),
                feasible_step_limit=feasible_limit,
                accepted_step_length=step_length,
                merit_directional_derivative=merit_slope,
                rejected_line_search_trials=rejected,
                krylov_iterations=sum(record.iterations for record in linear_diagnostics),
                krylov_residual_relative=max(
                    record.true_relative_residual for record in linear_diagnostics
                ),
                schur_denominator=correction.schur_denominator,
                schur_condition_indicator=correction.schur_condition_indicator,
                bordered_residual_relative=correction.bordered_residual_relative,
                forcing_tolerance=forcing_tolerance,
                linear_refinement_attempts=linear_refinement_attempts,
                line_search_termination_reason=line_search.termination_reason,
                krylov_diagnostics=tuple(linear_diagnostics),
            )
        )
        previous_nonlinear_norm = nonlinear_norm
        if iteration == 1 or iteration % log_every == 0:
            logger.info(
                "monolithic Newton iteration=%d: scaled Reynolds RMS=%.3e, "
                "load residual=%.3e N, alpha_feasible=%.3e, alpha=%.3e, "
                "rejected=%d, GMRES=%d (relative residual %.3e), "
                "Schur indicator=%.3e, h_min=%.6e m",
                iteration,
                reynolds_norm,
                load_residual_n,
                feasible_limit,
                step_length,
                rejected,
                sum(record.iterations for record in linear_diagnostics),
                max(
                    record.true_relative_residual
                    for record in linear_diagnostics
                ),
                correction.schur_condition_indicator,
                float(np.min(accepted_film)),
            )
    final_residual = problem.residual(state)
    final_pressure, final_film, _ = problem.state_fields(
        state[:-1], float(state[-1])
    )
    raise MonolithicNewtonError(
        f"monolithic Newton did not converge in {max_iterations} iterations",
        tuple(history),
        {
            "iteration": max_iterations,
            "reynolds_residual_scaled": float(
                np.linalg.norm(final_residual[:-1])
                / math.sqrt(problem.pressure_size)
            ),
            "load_residual_scaled": float(final_residual[-1]),
            "load_residual_n": float(final_residual[-1])
            * problem.load_residual_scale_n,
            "minimum_film_thickness_m": float(np.min(final_film)),
            "minimum_pressure_pa": float(
                np.min(final_pressure) * problem.pressure_scale_pa
            ),
            "maximum_pressure_pa": float(
                np.max(final_pressure) * problem.pressure_scale_pa
            ),
            "operator_evaluations": problem.operator_evaluations,
        },
    )


def continue_monolithic_load(
    problem: BorderedReynoldsElasticityProblem,
    initial_solution: MonolithicLoadSolution,
    target_load_n: float,
    *,
    initial_step_fraction: float = 0.25,
    minimum_step_fraction: float = 1e-3,
    maximum_step_fraction: float = 0.5,
    growth_factor: float = 1.5,
    maximum_retries_per_step: int = 8,
    newton_options: dict[str, float | int] | None = None,
) -> MonolithicLoadContinuation:
    """Adaptively continue the signed target load from a converged solution.

    The supplied ``initial_solution`` is the required verified starting
    state. Failed trials are recorded, and the last accepted pressure and h0
    are retained for retry; unconverged states are never used as warm starts.
    Step sizes are fractions of the total requested load change. This remains
    mathematical signed-full-film continuation and does not imply physical
    admissibility when pressure is negative.
    """
    if not math.isfinite(target_load_n):
        raise ValueError("target_load_n must be finite")
    if not (
        0.0 < minimum_step_fraction <= initial_step_fraction <= maximum_step_fraction
        <= 1.0
    ):
        raise ValueError("load step fractions must satisfy 0 < min <= initial <= max <= 1")
    if not math.isfinite(growth_factor) or growth_factor <= 1.0:
        raise ValueError("growth_factor must be finite and greater than one")
    if (
        not isinstance(maximum_retries_per_step, int)
        or isinstance(maximum_retries_per_step, bool)
        or maximum_retries_per_step < 0
    ):
        raise ValueError("maximum_retries_per_step must be nonnegative")
    initial_load = initial_solution.target_load_n
    options = dict(newton_options or {})
    if not math.isclose(
        problem.target_load_n,
        initial_load,
        rel_tol=1e-12,
        abs_tol=8.0 * np.finfo(np.float64).eps * problem.load_residual_scale_n,
    ):
        raise ValueError(
            "problem target load must match the supplied converged initial solution"
        )
    initial_pressure = np.asarray(
        initial_solution.pressure_dimensionless, dtype=np.float64
    )
    if initial_pressure.shape != (problem.y.size, problem.x.size):
        raise ValueError("initial solution pressure shape does not match the problem grid")
    initial_state = np.concatenate(
        (
            initial_pressure[1:-1, 1:-1].ravel(),
            [
                initial_solution.central_film_offset_m
                / problem.film_scale_m
            ],
        )
    )
    initial_residual = problem.residual(initial_state)
    _, reconstructed_film, _ = problem.state_fields(
        initial_state[:-1], float(initial_state[-1])
    )
    supplied_film = np.asarray(initial_solution.film_thickness_m, dtype=np.float64)
    if supplied_film.shape != reconstructed_film.shape or not np.allclose(
        supplied_film,
        reconstructed_film,
        rtol=1e-10,
        atol=32.0 * np.finfo(np.float64).eps * problem.film_scale_m,
    ):
        raise ValueError(
            "initial solution film is inconsistent with the continuation problem"
        )
    initial_reynolds_tolerance = float(options.get("reynolds_tolerance", 1e-8))
    initial_load_tolerance = float(options.get("load_tolerance", 1e-6))
    if not all(
        math.isfinite(tolerance) and tolerance > 0.0
        for tolerance in (initial_reynolds_tolerance, initial_load_tolerance)
    ):
        raise ValueError("continuation tolerances must be finite and positive")
    if (
        np.linalg.norm(initial_residual[:-1]) / math.sqrt(problem.pressure_size)
        > initial_reynolds_tolerance
        or abs(initial_residual[-1]) > initial_load_tolerance
    ):
        raise ValueError("initial solution is not converged for the continuation problem")
    total_change = target_load_n - initial_load
    if total_change == 0.0:
        return MonolithicLoadContinuation(initial_solution, ())

    accepted_solution = initial_solution
    accepted_load = initial_load
    fraction_step = initial_step_fraction
    attempt = 0
    retries = 0
    records: list[LoadContinuationStep] = []
    load_resolution = 8.0 * np.finfo(np.float64).eps * max(
        abs(target_load_n), abs(initial_load), np.finfo(np.float64).tiny
    )
    if abs(total_change) <= load_resolution:
        updated_load_residual = initial_solution.calculated_load_n - target_load_n
        if abs(updated_load_residual) > initial_load_tolerance * problem.load_residual_scale_n:
            raise ValueError(
                "requested load change is below floating-point continuation "
                "resolution but exceeds the configured load tolerance"
            )
        final_solution = replace(
            initial_solution,
            target_load_n=target_load_n,
            load_residual_n=updated_load_residual,
        )
        return MonolithicLoadContinuation(final_solution, ())
    while abs(target_load_n - accepted_load) > load_resolution:
        proposed_fraction = min(
            fraction_step,
            abs(target_load_n - accepted_load) / abs(total_change),
        )
        proposed_load = accepted_load + math.copysign(
            proposed_fraction * abs(total_change), total_change
        )
        problem.target_load_n = proposed_load
        attempt += 1
        try:
            candidate = solve_monolithic_full_film_load(
                problem,
                accepted_solution.pressure_dimensionless,
                accepted_solution.central_film_offset_m,
                **options,
            )
        except (MonolithicNewtonError, RuntimeError, ValueError) as error:
            failure_history = (
                error.history if isinstance(error, MonolithicNewtonError) else ()
            )
            failure_diagnostics = (
                error.diagnostics if isinstance(error, MonolithicNewtonError) else {}
            )
            logger.warning(
                "load continuation rejected attempt %d at %.9e N: %s",
                attempt,
                proposed_load,
                error,
            )
            records.append(
                LoadContinuationStep(
                    attempt=attempt,
                    target_load_n=proposed_load,
                    step_load_n=proposed_load - accepted_load,
                    accepted=False,
                    newton_iterations=len(failure_history),
                    relative_load_error=math.inf,
                    central_film_offset_m=accepted_solution.central_film_offset_m,
                    minimum_film_thickness_m=float(
                        np.min(accepted_solution.film_thickness_m)
                    ),
                    reason=str(error),
                    failure_diagnostics=tuple(failure_diagnostics.items()),
                )
            )
            retries += 1
            fraction_step *= 0.5
            if retries > maximum_retries_per_step or fraction_step < minimum_step_fraction:
                problem.target_load_n = accepted_load
                raise MonolithicNewtonError(
                    "load continuation cannot reach the target; last accepted "
                    f"load={accepted_load:.9e} N, requested={target_load_n:.9e} N, "
                    f"minimum fractional step={minimum_step_fraction:.3e}",
                    failure_history,
                    dict(failure_diagnostics),
                ) from error
            continue

        records.append(
            LoadContinuationStep(
                attempt=attempt,
                target_load_n=proposed_load,
                step_load_n=proposed_load - accepted_load,
                accepted=True,
                newton_iterations=len(candidate.history),
                relative_load_error=abs(candidate.load_residual_n)
                / max(abs(proposed_load), problem.load_residual_scale_n),
                central_film_offset_m=candidate.central_film_offset_m,
                minimum_film_thickness_m=float(np.min(candidate.film_thickness_m)),
                reason="converged",
            )
        )
        accepted_solution = candidate
        accepted_load = proposed_load
        logger.info(
            "load continuation accepted attempt %d at %.9e N: h0=%.6e m, "
            "Newton corrections=%d",
            attempt,
            proposed_load,
            candidate.central_film_offset_m,
            len(candidate.history),
        )
        retries = 0
        fraction_step = min(
            maximum_step_fraction, fraction_step * growth_factor
        )

    problem.target_load_n = target_load_n
    return MonolithicLoadContinuation(accepted_solution, tuple(records))


@dataclass(frozen=True)
class ComplianceContinuationStep:
    """A fixed-load compliance trial, including rejected-state diagnostics."""

    compliance_factor: float
    accepted: bool
    newton_iterations: int
    reason: str
    failure_diagnostics: tuple[tuple[str, object], ...] = ()
    krylov_iterations: int = 0
    rejected_line_search_trials: int = 0
    maximum_true_linear_residual: float = 0.0
    maximum_bordered_correction_residual: float = 0.0
    solution: MonolithicLoadSolution | None = None


@dataclass(frozen=True)
class MonolithicComplianceContinuation:
    solution: MonolithicLoadSolution
    steps: tuple[ComplianceContinuationStep, ...]


@dataclass(frozen=True)
class MonolithicPseudoArclengthContinuation:
    solution: MonolithicLoadSolution
    branch: PseudoArclengthResult


@dataclass(frozen=True)
class JacobianConditioningDiagnostic:
    compliance_factor: float
    largest_singular_value: float
    smallest_singular_value: float
    condition_number: float


def estimate_bordered_jacobian_conditioning(
    problem: BorderedReynoldsElasticityProblem,
    state: ArrayLike,
    *,
    maximum_state_size: int = 256,
) -> JacobianConditioningDiagnostic:
    """Explicitly assemble the small scaled Jacobian and estimate its spectrum."""
    values = np.asarray(state, dtype=np.float64)
    if (
        not isinstance(maximum_state_size, int)
        or isinstance(maximum_state_size, bool)
        or maximum_state_size < 1
        or values.shape != (problem.pressure_size + 1,)
        or values.size > maximum_state_size
    ):
        raise ValueError("state dimension is invalid or exceeds the explicit spectrum limit")
    if not np.all(np.isfinite(values)):
        raise ValueError("state must be finite")
    identity = np.eye(values.size, dtype=np.float64)
    matrix = np.column_stack(
        [
            problem.jacobian_vector(values, identity[:, index])
            for index in range(values.size)
        ]
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    condition = (
        math.inf
        if smallest == 0.0
        else largest / smallest
    )
    return JacobianConditioningDiagnostic(
        problem.compliance_factor,
        largest,
        smallest,
        condition,
    )


def continue_monolithic_compliance(
    problem: BorderedReynoldsElasticityProblem,
    initial_solution: MonolithicLoadSolution,
    target_compliance_factor: float = 1.0,
    *,
    initial_step: float = 0.1,
    minimum_step: float = 1e-4,
    maximum_step: float = 0.25,
    maximum_retries_per_step: int = 8,
    newton_options: dict[str, float | int] | None = None,
) -> MonolithicComplianceContinuation:
    """Continue compliance at fixed signed load using converged warm starts.

    Steps are absolute dimensionless compliance increments. Each trial uses
    the unchanged dual Newton line search. Infeasible warm starts are rejected,
    never shifted or clipped. On failure the problem is restored to its last
    accepted compliance and the exception retains all trial records and that
    converged solution. This is branch following, not evidence of existence
    of a solution at the requested compliance.
    """
    if not math.isfinite(target_compliance_factor) or not 0 <= target_compliance_factor <= 1:
        raise ValueError("target compliance must be finite and in [0, 1]")
    if not (0 < minimum_step <= initial_step <= maximum_step) or not math.isfinite(maximum_step):
        raise ValueError("compliance steps must satisfy 0 < min <= initial <= max < infinity")
    if (not isinstance(maximum_retries_per_step, int)
            or isinstance(maximum_retries_per_step, bool) or maximum_retries_per_step < 0):
        raise ValueError("maximum_retries_per_step must be nonnegative")
    options = dict(newton_options or {})
    if options.get("initial_film_strategy", "strict") != "strict":
        raise ValueError("compliance continuation requires strict film feasibility")
    # The zero-load-change path independently verifies residuals and film
    # consistency using exactly the same validation as load continuation.
    continue_monolithic_load(
        problem, initial_solution, problem.target_load_n, newton_options=options
    )
    accepted_factor = problem.compliance_factor
    accepted_solution = initial_solution
    step = initial_step
    retries = 0
    records: list[ComplianceContinuationStep] = []
    while accepted_factor != target_compliance_factor:
        increment = min(step, abs(target_compliance_factor - accepted_factor))
        proposed = accepted_factor + math.copysign(
            increment, target_compliance_factor - accepted_factor
        )
        if increment == abs(target_compliance_factor - accepted_factor):
            proposed = target_compliance_factor
        if proposed == accepted_factor:
            raise ValueError("compliance increment is below floating-point resolution")
        problem.compliance_factor = proposed
        try:
            candidate = solve_monolithic_full_film_load(
                problem, accepted_solution.pressure_dimensionless,
                accepted_solution.central_film_offset_m, **options,
            )
            # Recompute convergence and film consistency before accepting.
            continue_monolithic_load(
                problem, candidate, problem.target_load_n, newton_options=options
            )
        except (RuntimeError, ValueError) as error:
            problem.compliance_factor = accepted_factor
            diagnostics = error.diagnostics if isinstance(error, MonolithicNewtonError) else {}
            history = error.history if isinstance(error, MonolithicNewtonError) else ()
            failed_krylov = tuple(diagnostics.get("krylov_diagnostics", ()))
            failed_line_search_rejections = int(
                diagnostics.get("rejected_line_search_trials", 0)
            )
            failed_bordered_residual = float(
                diagnostics.get("bordered_residual_relative", 0.0)
            )
            records.append(ComplianceContinuationStep(
                proposed,
                False,
                len(history),
                str(error),
                tuple(diagnostics.items()),
                sum(entry.krylov_iterations for entry in history)
                + sum(record.iterations for record in failed_krylov),
                sum(entry.rejected_line_search_trials for entry in history)
                + failed_line_search_rejections,
                max(
                    (
                        *(
                        diagnostic.true_relative_residual
                        for entry in history
                        for diagnostic in entry.krylov_diagnostics
                        ),
                        *(
                            diagnostic.true_relative_residual
                            for diagnostic in failed_krylov
                        ),
                    ),
                    default=0.0,
                ),
                max(
                    (
                        failed_bordered_residual,
                        *(entry.bordered_residual_relative for entry in history),
                    ),
                    default=0.0,
                ),
            ))
            retries += 1
            step *= 0.5
            if retries > maximum_retries_per_step or step < minimum_step:
                raise MonolithicNewtonError(
                    "compliance continuation cannot reach target; "
                    f"last accepted factor={accepted_factor:.9e}", history,
                    {**diagnostics, "compliance_steps": tuple(records),
                     "last_accepted_solution": accepted_solution,
                     "last_accepted_compliance_factor": accepted_factor},
                ) from error
            continue
        records.append(ComplianceContinuationStep(
            proposed,
            True,
            len(candidate.history),
            "converged",
            krylov_iterations=sum(
                entry.krylov_iterations for entry in candidate.history
            ),
            rejected_line_search_trials=sum(
                entry.rejected_line_search_trials for entry in candidate.history
            ),
            maximum_true_linear_residual=max(
                (
                    diagnostic.true_relative_residual
                    for entry in candidate.history
                    for diagnostic in entry.krylov_diagnostics
                ),
                default=0.0,
            ),
            maximum_bordered_correction_residual=max(
                (
                    entry.bordered_residual_relative
                    for entry in candidate.history
                ),
                default=0.0,
            ),
            solution=candidate,
        ))
        accepted_factor = proposed
        accepted_solution = candidate
        retries = 0
        step = min(maximum_step, 1.5 * step)
    return MonolithicComplianceContinuation(accepted_solution, tuple(records))


def continue_monolithic_compliance_pseudo_arclength(
    problem: BorderedReynoldsElasticityProblem,
    initial_solution: MonolithicLoadSolution,
    *,
    target_compliance_factor: float = 1.0,
    initial_step: float = 0.02,
    minimum_step: float = 1e-4,
    maximum_step: float = 0.05,
    maximum_steps: int = 24,
    maximum_retries_per_step: int = 6,
    residual_tolerance: float = 1e-8,
    load_tolerance: float = 1e-6,
    maximum_state_size: int = 256,
    compliance_reference_modulus_pa: float | None = None,
) -> MonolithicPseudoArclengthContinuation:
    """Experimental dense predictor-corrector continuation in compliance.

    The full residual Jacobian is assembled from analytic matrix-free JVPs
    and solved directly. This intentionally limits the method to small
    systems; it is a branch diagnostic, not a large-grid solver. Pressure
    unknowns are weighted by their RMS norm, while ``h0/h_ref`` and
    compliance each have unit weight by default. Strict film positivity is checked at
    every predictor and corrector trial. The compliance derivative is analytic,
    including both pressure-driven mobility and entrainment film terms.
    Optional compliance_reference_modulus_pa [Pa] weights parameter motion
    by (E_ref/E_star)**2, i.e. arclength in c=gamma*E_ref/E_star. This
    matches steps for equivalent gamma/E_star problems at fixed geometry,
    scales and load. Returned parameters/tangents still use actual gamma;
    the default metric retains unit gamma weight.
    """
    parameter_weight = 1.0
    if compliance_reference_modulus_pa is not None:
        if (
            not math.isfinite(compliance_reference_modulus_pa)
            or compliance_reference_modulus_pa <= 0.0
        ):
            raise ValueError("compliance_reference_modulus_pa must be finite and positive [Pa]")
        modulus_ratio = compliance_reference_modulus_pa / problem.e_star_pa
        parameter_weight = modulus_ratio * modulus_ratio
        if not math.isfinite(parameter_weight) or parameter_weight <= 0.0:
            raise ValueError("normalized compliance metric is outside the float64 range")
    if (
        not isinstance(maximum_state_size, int)
        or isinstance(maximum_state_size, bool)
        or maximum_state_size < 2
        or problem.pressure_size + 1 > maximum_state_size
    ):
        raise ValueError(
            "pseudo-arclength continuation is limited to "
            f"{maximum_state_size} pressure/film unknowns"
        )
    if not math.isfinite(target_compliance_factor) or not (
        0.0 <= target_compliance_factor <= 1.0
    ):
        raise ValueError("target_compliance_factor must be finite and in [0, 1]")
    if not math.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    if not math.isfinite(load_tolerance) or load_tolerance <= 0.0:
        raise ValueError("load_tolerance must be finite and positive")
    if not math.isclose(
        initial_solution.target_load_n,
        problem.target_load_n,
        rel_tol=1e-12,
        abs_tol=8.0 * np.finfo(np.float64).eps * problem.load_residual_scale_n,
    ):
        raise ValueError("problem load must match the initial converged solution")

    state = np.concatenate(
        (
            initial_solution.pressure_dimensionless[1:-1, 1:-1].ravel(),
            [initial_solution.central_film_offset_m / problem.film_scale_m],
        )
    )
    initial_residual = problem.residual(state)
    if (
        np.linalg.norm(initial_residual[:-1]) / math.sqrt(problem.pressure_size)
        > residual_tolerance
        or abs(initial_residual[-1]) > load_tolerance
    ):
        raise ValueError("initial solution does not satisfy continuation tolerances")
    _, reconstructed_film, _ = problem.state_fields(state[:-1], float(state[-1]))
    if not np.allclose(
        initial_solution.film_thickness_m,
        reconstructed_film,
        rtol=1e-10,
        atol=32.0 * np.finfo(np.float64).eps * problem.film_scale_m,
    ):
        raise ValueError("initial solution film is inconsistent with the problem")

    def residual_at(
        values: NDArray[np.float64], compliance: float
    ) -> NDArray[np.float64]:
        if not 0.0 <= compliance <= 1.0:
            raise ValueError("compliance factor must remain in [0, 1]")
        problem.compliance_factor = compliance
        return problem.residual(values)

    def jacobian_at(
        values: NDArray[np.float64], compliance: float
    ) -> NDArray[np.float64]:
        problem.compliance_factor = compliance
        identity = np.eye(values.size, dtype=np.float64)
        return np.column_stack(
            [
                problem.jacobian_vector(values, identity[:, index])
                for index in range(values.size)
            ]
        )

    def compliance_derivative(
        values: NDArray[np.float64], compliance: float
    ) -> NDArray[np.float64]:
        problem.compliance_factor = compliance
        return problem.compliance_derivative(values)

    def feasible(values: NDArray[np.float64], compliance: float) -> bool:
        if not 0.0 <= compliance <= 1.0:
            return False
        problem.compliance_factor = compliance
        try:
            _, film_m, _ = problem.state_fields(values[:-1], float(values[-1]))
        except ValueError:
            return False
        return bool(
            np.all(np.isfinite(film_m))
            and np.all(film_m > problem.film_floor_m)
        )

    weights = np.ones(state.size, dtype=np.float64)
    weights[:-1] = 1.0 / problem.pressure_size
    branch = pseudo_arclength_continue(
        residual_at,
        jacobian_at,
        compliance_derivative,
        state,
        problem.compliance_factor,
        state_weights=weights,
        parameter_weight=parameter_weight,
        feasible=feasible,
        target_parameter=target_compliance_factor,
        initial_step=initial_step,
        minimum_step=minimum_step,
        maximum_step=maximum_step,
        maximum_steps=maximum_steps,
        maximum_retries_per_step=maximum_retries_per_step,
        residual_tolerance=min(residual_tolerance, load_tolerance),
        arclength_tolerance=min(residual_tolerance, load_tolerance),
    )
    problem.compliance_factor = branch.parameter
    pressure, film, displacement = problem.state_fields(
        branch.state[:-1], float(branch.state[-1])
    )
    pressure_pa = pressure * problem.pressure_scale_pa
    calculated_load = _integrated_load_n(problem.x, problem.y, pressure_pa)
    solution = MonolithicLoadSolution(
        pressure_dimensionless=pressure,
        pressure_pa=pressure_pa,
        film_thickness_m=film,
        elastic_displacement_m=displacement,
        central_film_offset_m=float(branch.state[-1]) * problem.film_scale_m,
        target_load_n=problem.target_load_n,
        calculated_load_n=calculated_load,
        load_residual_n=calculated_load - problem.target_load_n,
        history=(),
        operator_evaluations=problem.operator_evaluations,
        initial_h0_adjustment_m=initial_solution.initial_h0_adjustment_m,
    )
    return MonolithicPseudoArclengthContinuation(solution, branch)
