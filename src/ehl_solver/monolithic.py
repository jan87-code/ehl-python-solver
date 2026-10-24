"""Matrix-free bordered Newton solve for signed full-film load control."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.sparse.linalg import LinearOperator, gmres, spilu

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.elasticity import cell_integrated_kernel
from ehl_solver.reynolds import assemble_reynolds_system, reynolds_residual

logger = logging.getLogger(__name__)


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
    failure_diagnostics: tuple[tuple[str, float | int | str], ...] = ()


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
        diagnostics: dict[str, float | int | str] | None = None,
    ):
        super().__init__(message)
        self.history = history
        self.diagnostics = dict(diagnostics or {})


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
        finite_difference_step: float = 2e-7,
        krylov_rtol: float = 1e-8,
        krylov_max_iterations: int | None = None,
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
            finite_difference_step,
            krylov_rtol,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("radii, scales, floor and solver tolerances must be positive")
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
        self.finite_difference_step = finite_difference_step
        self.krylov_rtol = krylov_rtol
        self.krylov_max_iterations = krylov_max_iterations
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

    def jacobian_vector(self, state: ArrayLike, direction: ArrayLike) -> NDArray[np.float64]:
        """Centered matrix-free Jv including p->w->h->Reynolds and h0 paths."""
        z = np.asarray(state, dtype=np.float64)
        v = np.asarray(direction, dtype=np.float64)
        if z.shape != (self.pressure_size + 1,) or v.shape != z.shape:
            raise ValueError("state and direction must have pressure_size + 1 entries")
        norm_v = float(np.linalg.norm(v))
        if norm_v == 0.0:
            return np.zeros_like(v)
        step = self.finite_difference_step * (1.0 + np.linalg.norm(z)) / norm_v
        for _ in range(12):
            try:
                forward = self.residual(z + step * v)
                backward = self.residual(z - step * v)
                return (forward - backward) / (2.0 * step)
            except ValueError:
                try:
                    forward = self.residual(z + step * v)
                    base = self.residual(z)
                    return (forward - base) / step
                except ValueError:
                    try:
                        backward = self.residual(z - step * v)
                        base = self.residual(z)
                        return (base - backward) / step
                    except ValueError:
                        step *= 0.5
        raise RuntimeError("could not form a feasible Jacobian-vector product")

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


def solve_bordered_correction(
    problem: BorderedReynoldsElasticityProblem,
    state: ArrayLike,
    *,
    schur_tolerance: float = 1e-10,
    bordered_tolerance: float = 1e-7,
    krylov_rtol: float | None = None,
    krylov_max_iterations: int | None = None,
) -> BorderedCorrection:
    """Solve a matrix-free Newton system through a Schur complement.

    The pressure block is solved by GMRES; its frozen-film Reynolds matrix
    provides an ILU preconditioner. If the Schur denominator is unsafe or the
    reconstructed bordered correction is inaccurate, the method falls back
    to GMRES on the complete bordered Jacobian with a block preconditioner.
    """
    z = np.asarray(state, dtype=np.float64)
    residual = problem.residual(z)
    pressure_count = problem.pressure_size
    h_direction = np.zeros_like(z)
    h_direction[-1] = 1.0
    b = problem.jacobian_vector(z, h_direction)[:-1]
    load_gradient = problem.load_gradient()
    rtol = problem.krylov_rtol if krylov_rtol is None else krylov_rtol
    maxiter = (
        problem.krylov_max_iterations
        if krylov_max_iterations is None
        else krylov_max_iterations
    )
    restart = min(80, pressure_count + 1)

    pressure, film, _ = problem.state_fields(z[:-1], float(z[-1]))
    frozen_matrix, _ = assemble_reynolds_system(
        film / problem.film_scale_m,
        problem.dx,
        problem.dy,
        problem.reynolds_parameter,
    )
    ilu = spilu((-frozen_matrix / problem.reynolds_residual_scale).tocsc())
    preconditioner = LinearOperator(
        frozen_matrix.shape, matvec=ilu.solve, dtype=np.float64
    )

    def pressure_matvec(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        direction = np.zeros_like(z)
        direction[:-1] = vector
        return problem.jacobian_vector(z, direction)[:-1]

    pressure_operator = LinearOperator(
        (pressure_count, pressure_count), matvec=pressure_matvec, dtype=np.float64
    )
    iterations = 0

    def callback(_: NDArray[np.float64]) -> None:
        nonlocal iterations
        iterations += 1

    def solve_pressure(rhs: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        nonlocal iterations
        iterations = 0
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
        if info != 0 or not np.all(np.isfinite(solution)):
            raise RuntimeError(f"pressure-block GMRES failed with info={info}")
        solution = np.asarray(solution, dtype=np.float64)
        relative_residual = float(
            np.linalg.norm(pressure_operator @ solution - rhs)
            / max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
        )
        if not math.isfinite(relative_residual) or relative_residual > 10.0 * rtol:
            raise RuntimeError(
                "pressure-block GMRES returned an inaccurate solve: "
                f"relative residual={relative_residual:.3e}"
            )
        return solution, relative_residual

    try:
        x, residual_x = solve_pressure(-residual[:-1])
        count_x = iterations
        y, residual_y = solve_pressure(b)
        count_y = iterations
        pressure_krylov_residual = max(residual_x, residual_y)
        schur = -float(np.dot(load_gradient, y))
        scale = max(
            abs(float(np.dot(load_gradient, y))),
            np.linalg.norm(load_gradient) * np.linalg.norm(y) * np.finfo(float).eps,
            np.finfo(float).tiny,
        )
        indicator = abs(schur) / scale
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
        if not math.isfinite(relative) or relative > bordered_tolerance:
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
        )
    except (ArithmeticError, RuntimeError):
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
                (preconditioner @ rhs[:-1], [rhs[-1]])
            ),
            dtype=np.float64,
        )
        iterations = 0
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
        if info != 0 or not np.all(np.isfinite(correction)):
            raise RuntimeError(
                f"full bordered GMRES fallback failed with info={info}"
            ) from None
        linear_residual = full_operator @ correction + residual
        relative = float(
            np.linalg.norm(linear_residual)
            / max(np.linalg.norm(residual), np.finfo(float).tiny)
        )
        if relative > bordered_tolerance:
            raise RuntimeError(
                "full bordered GMRES fallback correction is inaccurate: "
                f"relative residual={relative:.3e}"
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
        )


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
    state = np.concatenate(
        (pressure[1:-1, 1:-1].ravel(), [initial_h0_m / problem.film_scale_m])
    )
    history: list[MonolithicIteration] = []
    for iteration in range(1, max_iterations + 1):
        residual = problem.residual(state)
        reynolds_norm = float(np.linalg.norm(residual[:-1]) / math.sqrt(problem.pressure_size))
        load_scaled = float(residual[-1])
        load_residual_n = load_scaled * problem.load_residual_scale_n
        pressure, film, displacement = problem.state_fields(state[:-1], float(state[-1]))
        load_norm = abs(load_scaled)
        merit = 0.5 * (reynolds_norm**2 + load_norm**2)
        failure_diagnostics: dict[str, float | int | str] = {
            "iteration": iteration,
            "reynolds_residual_scaled": reynolds_norm,
            "load_residual_n": load_residual_n,
            "load_residual_scaled": load_scaled,
            "minimum_film_thickness_m": float(np.min(film)),
            "maximum_pressure_pa": float(np.max(pressure * problem.pressure_scale_pa)),
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
            )

        try:
            correction = solve_bordered_correction(problem, state)
        except (ArithmeticError, RuntimeError) as error:
            raise MonolithicNewtonError(
                f"bordered Newton correction failed: {error}",
                tuple(history),
                failure_diagnostics,
            ) from error
        for recovery in range(3):
            direction = np.concatenate(
                (
                    correction.pressure_correction.ravel(),
                    [correction.h0_correction_m / problem.film_scale_m],
                )
            )
            merit_slope = _merit_directional_derivative(
                problem, state, residual, direction
            )
            if math.isfinite(merit_slope) and merit_slope < 0.0:
                break
            if recovery == 2:
                raise MonolithicNewtonError(
                    "Newton correction remained non-descent after tighter Krylov "
                    "recovery",
                    tuple(history),
                    {**failure_diagnostics, "merit_directional_derivative": merit_slope},
                )
            tightened_rtol = max(problem.krylov_rtol * 0.1 ** (recovery + 1), 1e-12)
            try:
                correction = solve_bordered_correction(
                    problem,
                    state,
                    krylov_rtol=tightened_rtol,
                )
            except (ArithmeticError, RuntimeError) as error:
                raise MonolithicNewtonError(
                    f"tighter Krylov recovery failed: {error}",
                    tuple(history),
                    failure_diagnostics,
                ) from error
        delta_pressure = np.zeros_like(pressure)
        delta_pressure[1:-1, 1:-1] = correction.pressure_correction
        delta_h0 = correction.h0_correction_m
        _, film_change, _ = problem.state_fields(
            state[:-1] + delta_pressure[1:-1, 1:-1].ravel(),
            float(state[-1]) + delta_h0 / problem.film_scale_m,
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
            raise MonolithicNewtonError(
                "Newton correction has no positive feasible step",
                tuple(history),
                failure_diagnostics,
            )

        step_length = feasible_limit
        accepted = False
        rejected = 0
        current_merit = merit
        for _ in range(maximum_backtracks + 1):
            trial_state = state + step_length * direction
            try:
                trial_residual = problem.residual(trial_state)
                trial_rp = float(
                    np.linalg.norm(trial_residual[:-1])
                    / math.sqrt(problem.pressure_size)
                )
                trial_rq = float(trial_residual[-1])
                trial_merit = 0.5 * (trial_rp**2 + trial_rq**2)
            except ValueError:
                trial_merit = math.inf
            if (
                math.isfinite(trial_merit)
                and trial_merit <= (1.0 - armijo_constant * step_length) * current_merit
            ):
                state = trial_state
                accepted = True
                break
            rejected += 1
            step_length *= 0.5
        if not accepted:
            raise MonolithicNewtonError(
                "dual line search could not find an Armijo-decreasing feasible step",
                tuple(history),
                {
                    **failure_diagnostics,
                    "feasible_step_limit": feasible_limit,
                    "last_trial_step_length": step_length,
                    "rejected_line_search_trials": rejected,
                },
            )
        _, accepted_film, _ = problem.state_fields(state[:-1], float(state[-1]))
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
                    step_length * delta_h0
                    / max(abs(float(state[-1]) * problem.film_scale_m), problem.film_floor_m)
                ),
                minimum_film_thickness_m=float(np.min(accepted_film)),
                maximum_pressure_pa=float(
                    max(float(np.max(state[:-1])), 0.0) * problem.pressure_scale_pa
                ),
                feasible_step_limit=feasible_limit,
                accepted_step_length=step_length,
                merit_directional_derivative=merit_slope,
                rejected_line_search_trials=rejected,
                krylov_iterations=correction.pressure_krylov_iterations,
                krylov_residual_relative=correction.pressure_krylov_residual_relative,
                schur_denominator=correction.schur_denominator,
                schur_condition_indicator=correction.schur_condition_indicator,
                bordered_residual_relative=correction.bordered_residual_relative,
            )
        )
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
                correction.pressure_krylov_iterations,
                correction.pressure_krylov_residual_relative,
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
