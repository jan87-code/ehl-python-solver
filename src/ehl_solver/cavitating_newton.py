"""Experimental monolithic CPU Newton solve of JFO pressure/content and film."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, gmres, splu

from ehl_solver.backend import make_backend
from ehl_solver.cavitating_coupled import (
    CavitatingElasticityProblem,
    CavitatingElasticitySolution,
    solve_cavitating_elasticity,
)
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.reynolds import assemble_reynolds_system


class CavitatingNewtonSystem:
    """Internal scaled residual/JVP for P, deficit D=1-theta, optional h0/h_ref.

    Public problem/solution fields are SI. Interior P is dimensionless p/p_ref;
    both P and D are dimensionless (ny-2,nx-2) fields flattened in C order.
    Mass rows are div(q)/mass_scale, complementarity rows min(P,D), and the
    optional force row is (integral(p)-target)/target. A positive SI target
    makes h0 an unknown; otherwise the problem's prescribed offset is used.
    """

    def __init__(
        self,
        problem: CavitatingElasticityProblem,
        *,
        compliance_factor=1.0,
        target_load_n=None,
        mass_scale=100.0,
    ):
        if not math.isfinite(compliance_factor) or not 0 <= compliance_factor <= 1:
            raise ValueError("compliance_factor must lie in [0,1]")
        if not math.isfinite(mass_scale) or mass_scale <= 0:
            raise ValueError("mass_scale must be positive")
        if target_load_n is not None and (not math.isfinite(target_load_n) or target_load_n <= 0):
            raise ValueError("target_load_n must be positive [N] or None")
        self.problem = problem
        self.gamma = compliance_factor
        self.target_load_n = target_load_n
        self.mass_scale = mass_scale
        self.n = (problem.shape[0] - 2) * (problem.shape[1] - 2)
        self.size = 2 * self.n + int(target_load_n is not None)

    def pack(self, solution):
        parts = [
            solution.pressure_pa[1:-1, 1:-1].ravel() / self.problem.pressure_scale_pa,
            1 - solution.fractional_film_content[1:-1, 1:-1].ravel(),
        ]
        if self.target_load_n is not None:
            parts.append([solution.central_film_offset_m / self.problem.film_scale_m])
        return np.concatenate(parts)

    def unpack(self, state):
        z = np.asarray(state, dtype=np.float64)
        if z.shape != (self.size,) or not np.all(np.isfinite(z)):
            raise ValueError("state must contain finite interior P,D and optional h0/h_ref")
        p = np.zeros(self.problem.shape)
        theta = np.ones(self.problem.shape)
        p[1:-1, 1:-1] = z[: self.n].reshape(p[1:-1, 1:-1].shape)
        theta[1:-1, 1:-1] = 1 - z[self.n : 2 * self.n].reshape(theta[1:-1, 1:-1].shape)
        theta[1:-1, 0] = self.problem.inlet_fractional_film_content
        theta[1:-1, -1] = theta[1:-1, -2]
        theta[0] = theta[1]
        theta[-1] = theta[-2]
        offset = (
            z[-1] * self.problem.film_scale_m
            if self.target_load_n is not None
            else self.problem.central_film_offset_m
        )
        film, w = self.problem.fields(
            p * self.problem.pressure_scale_pa,
            compliance_factor=self.gamma,
            central_film_offset_m=offset,
        )
        return p, theta, film, w, offset

    def residual(self, state):
        p, theta, film, _, _ = self.unpack(state)
        mass = self.problem.mass_residual(p * self.problem.pressure_scale_pa, theta, film)
        result = [
            mass.ravel() / self.mass_scale,
            np.minimum(state[: self.n], state[self.n : 2 * self.n]),
        ]
        if self.target_load_n is not None:
            result.append(
                [
                    (self.problem.load_n(p * self.problem.pressure_scale_pa) - self.target_load_n)
                    / self.target_load_n
                ]
            )
        return np.concatenate(result)

    def jacobian_vector(self, state, direction):
        v = np.asarray(direction, dtype=np.float64)
        if v.shape != (self.size,) or not np.all(np.isfinite(v)):
            raise ValueError("direction has invalid shape or nonfinite values")
        p, theta, film, _, _ = self.unpack(state)
        problem = self.problem
        dp = np.zeros(problem.shape)
        dt = np.zeros(problem.shape)
        dp[1:-1, 1:-1] = v[: self.n].reshape(dp[1:-1, 1:-1].shape)
        dt[1:-1, 1:-1] = -v[self.n : 2 * self.n].reshape(dt[1:-1, 1:-1].shape)
        dw = fft_convolve_same(
            dp * problem.pressure_scale_pa, problem.elastic_kernel, make_backend("numpy")
        )
        dh = self.gamma * dw / problem.film_scale_m
        if self.target_load_n is not None:
            dh = dh + v[-1]
        h = film / problem.film_scale_m
        h3, dh3 = h**3, 3 * h**2 * dh
        dqx = (
            -0.5 * (dh3[:, 1:] + dh3[:, :-1]) * np.diff(p, axis=1) / problem.dx
            - 0.5 * (h3[:, 1:] + h3[:, :-1]) * np.diff(dp, axis=1) / problem.dx
            + problem.reynolds_parameter
            * (
                0.5 * (dh[:, 1:] + dh[:, :-1]) * theta[:, :-1]
                + 0.5 * (h[:, 1:] + h[:, :-1]) * dt[:, :-1]
            )
        )
        dqy = (
            -0.5 * (dh3[1:] + dh3[:-1]) * np.diff(p, axis=0) / problem.dy
            - 0.5 * (h3[1:] + h3[:-1]) * np.diff(dp, axis=0) / problem.dy
        )
        mass = np.diff(dqx[1:-1], axis=1) / problem.dx + np.diff(dqy[:, 1:-1], axis=0) / problem.dy
        full = state[: self.n] >= state[self.n : 2 * self.n]
        result = [
            mass.ravel() / self.mass_scale,
            np.where(full, v[self.n : 2 * self.n], v[: self.n]),
        ]
        if self.target_load_n is not None:
            result.append([problem.load_n(dp * problem.pressure_scale_pa) / self.target_load_n])
        return np.concatenate(result)

    def frozen_matrix(self, state):
        """Sparse exact active-set Jacobian at frozen film; elastic columns omitted."""
        _, _, film, _, _ = self.unpack(state)
        problem = self.problem
        h = film / problem.film_scale_m
        a, _ = assemble_reynolds_system(h, problem.dx, problem.dy, problem.reynolds_parameter)
        east = (0.5 * (h[1:-1, 1:-1] + h[1:-1, 2:])).ravel()
        west = (0.5 * (h[1:-1, 1:-1] + h[1:-1, :-2])).ravel()
        west[np.arange(self.n) % (problem.shape[1] - 2) == 0] = 0
        d = diags(
            [
                -problem.reynolds_parameter * east / problem.dx,
                problem.reynolds_parameter * west[1:] / problem.dx,
            ],
            [0, -1],
            shape=(self.n, self.n),
        )
        full = state[: self.n] >= state[self.n : 2 * self.n]
        block = bmat(
            [
                [a / self.mass_scale, d / self.mass_scale],
                [diags((~full).astype(float)), diags(full.astype(float))],
            ],
            format="csc",
        )
        if self.target_load_n is None:
            return block
        direction = np.zeros(self.size)
        direction[-1] = 1
        column = self.jacobian_vector(state, direction)[: 2 * self.n]
        load_row = np.r_[
            np.full(
                self.n, problem.pressure_scale_pa * problem.dx_m * problem.dy_m / self.target_load_n
            ),
            np.zeros(self.n),
        ]
        return bmat(
            [
                [block, csr_matrix(column[:, None])],
                [csr_matrix(load_row[None]), csr_matrix((1, 1))],
            ],
            format="csc",
        )

    def solution(self, state):
        p, theta, film, w, offset = self.unpack(state)
        return CavitatingElasticitySolution(
            p * self.problem.pressure_scale_pa,
            theta,
            film,
            w,
            offset,
            self.gamma,
            self.problem.load_n(p * self.problem.pressure_scale_pa),
            (),
        )

    def merit_slope(self, state, residual, direction):
        action = self.jacobian_vector(state, direction)
        tied = state[: self.n] == state[self.n : 2 * self.n]
        action[self.n : 2 * self.n][tied] = np.minimum(
            direction[: self.n][tied], direction[self.n : 2 * self.n][tied]
        )
        return float(np.dot(residual, action) / residual.size)


@dataclass(frozen=True)
class CavitatingCorrection:
    direction: np.ndarray
    true_linear_residual: float
    linear_iterations: int
    used_bordered_fallback: bool
    schur_indicator: float | None
    linear_true_residuals: tuple[float, ...]
    krylov_histories: tuple[tuple[float, ...], ...] = ()
    field_setup_failures: tuple[dict, ...] = ()


class CavitatingLinearError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def solve_cavitating_correction(
    system,
    state,
    *,
    krylov_rtol=1e-9,
    maximum_krylov_cycles=20,
    gmres_restart=80,
    schur_tolerance=1e-10,
    progress_callback=None,
):
    """Verified pressure/content block solves and optional h0 Schur correction.

    Unsafe/failed Schur reconstruction recovers through full bordered GMRES.
    Each GMRES result and the complete reconstructed correction are checked
    using the analytic matrix-free Jacobian, independently of preconditioning.
    """
    if (
        not math.isfinite(krylov_rtol)
        or not 0 < krylov_rtol < 1
        or not math.isfinite(schur_tolerance)
        or schur_tolerance <= 0
    ):
        raise ValueError("Krylov/Schur tolerances must be positive; krylov_rtol < 1")
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1
        for n in (maximum_krylov_cycles, gmres_restart)
    ):
        raise ValueError("Krylov limits must be positive integers")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    jvp_seconds = 0.0
    preconditioner_seconds = 0.0
    iterations = 0

    def report(stage, **metrics):
        if progress_callback is not None:
            progress_callback(
                stage=stage,
                cumulative_jvp_seconds=jvp_seconds,
                cumulative_preconditioner_seconds=preconditioner_seconds,
                cumulative_superlu_seconds=preconditioner_seconds,
                **metrics,
            )

    def action(v):
        nonlocal jvp_seconds
        if progress_callback is None:
            return system.jacobian_vector(state, v)
        report("jvp")
        started = time.perf_counter()
        result = system.jacobian_vector(state, v)
        jvp_seconds += time.perf_counter() - started
        report("jvp_complete", emit=False)
        return result

    report("linear_residual")
    residual = system.residual(state)
    operator = LinearOperator(
        (system.size, system.size),
        matvec=action,
        dtype=np.float64,
    )
    report("matrix_assembly")
    matrix = system.frozen_matrix(state)
    true_residuals, counts, traces = [], [], []
    field_setup_failures = []
    field_factorization = getattr(system, "field_factorization", "full")
    last_true_relative_residual = None
    last_true_rhs_kind = None

    def linear_solve(op, rhs, lu, rhs_kind):
        nonlocal preconditioner_seconds, iterations
        nonlocal last_true_relative_residual, last_true_rhs_kind
        count = []
        restart_residuals = []

        def precondition(v):
            nonlocal preconditioner_seconds
            if progress_callback is None:
                return lu.solve(v)
            report("superlu_solve")
            started = time.perf_counter()
            result = lu.solve(v)
            preconditioner_seconds += time.perf_counter() - started
            report("superlu_solve_complete", emit=False)
            return result

        def on_iteration(value):
            nonlocal iterations
            count.append(float(value))
            iterations += 1
            if progress_callback is not None:
                restart_count = len(count) // gmres_restart
                if len(count) % gmres_restart == 0:
                    restart_residuals.append(float(value))
                stagnant = 0
                for previous, current in zip(
                    restart_residuals, restart_residuals[1:], strict=False
                ):
                    stagnant = stagnant + 1 if current >= 0.99 * previous else 0
                report(
                    "gmres_restart" if len(count) % gmres_restart == 0 else "gmres",
                    gmres_iterations=iterations,
                    gmres_solve_iteration=len(count),
                    gmres_rhs_kind=rhs_kind,
                    gmres_restart_count=restart_count,
                    gmres_stagnation_restarts=stagnant,
                    gmres_residual_estimate_kind=(
                        "left_preconditioned_norm_over_unpreconditioned_rhs_norm"
                    ),
                    gmres_residual_estimate=(
                        float(value) if math.isfinite(value) else None
                    ),
                    gmres_true_relative_residual=last_true_relative_residual,
                    gmres_true_residual_rhs_kind=last_true_rhs_kind,
                    gmres_true_residual_status=(
                        "stale"
                        if last_true_relative_residual is not None
                        else "not_computed"
                    ),
                )

        preconditioner = LinearOperator(op.shape, matvec=precondition, dtype=np.float64)
        report(
            "gmres_start",
            emit=True,
            gmres_rhs_kind=rhs_kind,
            gmres_solve_iteration=0,
            gmres_restart_count=0,
            gmres_stagnation_restarts=0,
            gmres_residual_estimate=None,
            gmres_residual_estimate_kind=(
                "left_preconditioned_norm_over_unpreconditioned_rhs_norm"
            ),
            gmres_true_relative_residual=last_true_relative_residual,
            gmres_true_residual_rhs_kind=last_true_rhs_kind,
            gmres_true_residual_status=(
                "stale" if last_true_relative_residual is not None else "not_computed"
            ),
        )
        answer, info = gmres(
            op,
            rhs,
            M=preconditioner,
            rtol=krylov_rtol,
            atol=0,
            maxiter=maximum_krylov_cycles,
            restart=gmres_restart,
            callback=on_iteration,
            callback_type="pr_norm",
        )
        report("gmres_end", emit=False)
        norm = float(np.linalg.norm(rhs))
        error = float(np.linalg.norm(op @ answer - rhs))
        measured = error / norm if norm > 0 else error
        counts.append(len(count))
        traces.append(tuple(count))
        true_residuals.append(measured)
        last_true_relative_residual = measured if math.isfinite(measured) else None
        last_true_rhs_kind = rhs_kind
        report(
            "gmres_true_residual",
            gmres_rhs_kind=rhs_kind,
            gmres_true_relative_residual=measured if math.isfinite(measured) else None,
            gmres_true_residual_rhs_kind=rhs_kind,
            gmres_true_residual_status="current",
            emit=True,
        )
        return answer, info == 0 and math.isfinite(measured) and measured <= 5 * krylov_rtol

    def error(message):
        raise CavitatingLinearError(
            message,
            {
                "true_linear_residual": true_residuals[-1] if true_residuals else math.inf,
                "linear_true_residuals": tuple(true_residuals),
                "linear_iterations": sum(counts),
                "krylov_histories": tuple(traces),
                "field_factorization": field_factorization,
                "field_setup_failures": tuple(field_setup_failures),
            },
        )

    if not np.any(residual):
        return CavitatingCorrection(np.zeros_like(residual), 0.0, 0, False, None, ())
    report("superlu_factorization", factorization="field", emit=True)
    factor_started = time.perf_counter()
    try:
        lu = (system.reduced_field_factor(state, matrix) if field_factorization == "reduced"
              else splu(matrix[: 2 * system.n, : 2 * system.n]))
    except RuntimeError as exc:
        field_setup_failures.append({"factorization": field_factorization, "reason": str(exc)})
        lu = None
    finally:
        preconditioner_seconds += time.perf_counter() - factor_started
    report("superlu_factorization_complete", factorization="field", emit=True)
    border_size = system.size - 2 * system.n
    if system.target_load_n is None or border_size == 0:
        if lu is None:
            error("frozen-film active-set preconditioner failed")
        correction, good = linear_solve(operator, -residual, lu, "field_residual")
        if not good:
            error("true Krylov residual check failed")
        return CavitatingCorrection(
            correction,
            true_residuals[-1],
            sum(counts),
            False,
            None,
            tuple(true_residuals),
            tuple(traces),
            tuple(field_setup_failures),
        )
    indicator = None
    if lu is not None:
        block = LinearOperator(
            (2 * system.n, 2 * system.n),
            dtype=np.float64,
            matvec=lambda v: (
                operator
                @ np.r_[v, np.zeros(border_size, dtype=np.float64)]
            )[: 2 * system.n],
        )
        pressure, good_pressure = linear_solve(
            block, -residual[: 2 * system.n], lu, "field_residual"
        )
        if good_pressure:
            responses = []
            good_responses = True
            for index in range(border_size):
                basis = np.zeros(system.size)
                basis[2 * system.n + index] = 1.0
                column = (operator @ basis)[: 2 * system.n]
                label = (
                    "offset_response"
                    if border_size == 1
                    else f"border_response_{index}"
                )
                response, good_response = linear_solve(
                    block, column, lu, label
                )
                responses.append(response)
                good_responses = good_responses and good_response
            if good_responses:
                field_responses = np.column_stack(responses)
                border_rows = matrix[
                    2 * system.n :, : 2 * system.n
                ].toarray()
                border_block = matrix[
                    2 * system.n :, 2 * system.n :
                ].toarray()
                schur = border_block - border_rows @ field_responses
                if border_size == 1:
                    denominator = float(schur[0, 0])
                    scale = float(
                        np.linalg.norm(border_rows[0])
                        * np.linalg.norm(field_responses[:, 0])
                    )
                    indicator = abs(denominator) / max(
                        scale, np.finfo(float).tiny
                    )
                else:
                    row_scale = np.linalg.norm(border_rows, axis=1)
                    column_scale = np.linalg.norm(field_responses, axis=0)
                    scaled_schur = schur / np.maximum(
                        row_scale[:, None] * column_scale[None, :],
                        np.finfo(float).tiny,
                    )
                    singular_values = np.linalg.svd(
                        scaled_schur, compute_uv=False
                    )
                    indicator = float(
                        singular_values[-1]
                        / max(singular_values[0], np.finfo(float).tiny)
                    )
                if math.isfinite(indicator) and indicator > schur_tolerance:
                    scalar_correction = np.linalg.solve(
                        schur,
                        -residual[2 * system.n :]
                        - border_rows @ pressure,
                    )
                    correction = np.r_[
                        pressure - field_responses @ scalar_correction,
                        scalar_correction,
                    ]
                    report("correction_verification")
                    measured = float(
                        np.linalg.norm(operator @ correction + residual)
                        / np.linalg.norm(residual)
                    )
                    if math.isfinite(measured) and measured <= 5 * krylov_rtol:
                        return CavitatingCorrection(
                            correction,
                            measured,
                            sum(counts),
                            False,
                            indicator,
                            tuple(true_residuals),
                            tuple(traces),
                            tuple(field_setup_failures),
                        )
    report("superlu_factorization", factorization="bordered", emit=True)
    factor_started = time.perf_counter()
    try:
        full_lu = splu(matrix)
    except RuntimeError:
        error("full bordered active-set preconditioner failed")
    finally:
        preconditioner_seconds += time.perf_counter() - factor_started
    report("superlu_factorization_complete", factorization="bordered", emit=True)
    correction, good = linear_solve(operator, -residual, full_lu, "bordered_residual")
    if not good:
        error("true bordered Krylov residual check failed")
    return CavitatingCorrection(
        correction,
        true_residuals[-1],
        sum(counts),
        True,
        indicator,
        tuple(true_residuals),
        tuple(traces),
        tuple(field_setup_failures),
    )


@dataclass(frozen=True)
class CavitatingNewtonIteration:
    """Start-of-correction residuals, merits, and explicitly named post-step fields."""

    iteration: int
    mass_maximum: float
    complementarity: float
    load_residual_n: float
    step_length: float
    rejected_trials: int
    linear_iterations: int
    true_linear_residual: float
    merit_before: float
    merit_after: float
    merit_slope: float
    minimum_film_m: float
    full_film_cells: int
    used_bordered_fallback: bool
    schur_indicator: float | None
    mass_maximum_after: float
    central_film_offset_m_after: float
    pressure_minimum_pa_after: float
    saturation_minimum_after: float
    saturation_maximum_after: float
    linear_true_residuals: tuple[float, ...]
    krylov_histories: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CavitatingNewtonResult:
    solution: CavitatingElasticitySolution
    history: tuple[CavitatingNewtonIteration, ...]
    target_load_n: float | None
    load_residual_n: float


class CavitatingNewtonError(RuntimeError):
    def __init__(self, message, history, last_solution, diagnostics=None):
        super().__init__(message)
        self.history = tuple(history)
        self.last_solution = last_solution
        self.diagnostics = diagnostics or {}


def solve_cavitating_newton(
    problem,
    *,
    initial_solution=None,
    compliance_factor=1.0,
    target_load_n=None,
    maximum_iterations=30,
    maximum_backtracks=20,
    krylov_rtol=1e-9,
    maximum_krylov_cycles=20,
    gmres_restart=80,
    mass_tolerance=1e-8,
    admissibility_tolerance=1e-10,
    load_tolerance=1e-8,
):
    """Solve pressure/content, elasticity and optional positive load monolithically.

    Analytic matrix-free JVP; frozen-film sparse LU preconditioner; verified
    true GMRES residual; feasible fraction-to-boundary + directional Armijo.
    Positive target_load_n [N] adds a simultaneous h0/h_ref unknown and force
    row. This is numerical verification, not a validated physical EHL solver.
    """
    if any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1
        for n in (maximum_iterations, maximum_backtracks, maximum_krylov_cycles, gmres_restart)
    ):
        raise ValueError("Newton/Krylov/backtrack limits must be positive integers")
    if (
        not all(
            math.isfinite(t) and t > 0
            for t in (krylov_rtol, mass_tolerance, admissibility_tolerance, load_tolerance)
        )
        or krylov_rtol >= 1
    ):
        raise ValueError("solver tolerances must be positive, with krylov_rtol < 1")
    system = CavitatingNewtonSystem(
        problem, compliance_factor=compliance_factor, target_load_n=target_load_n
    )
    if initial_solution is None:
        initial_solution = solve_cavitating_elasticity(problem, continuation_factors=(0.0,))
    z = system.pack(initial_solution)
    history = []
    bound = admissibility_tolerance
    lower = np.r_[np.full(system.n, -bound), np.full(system.n, -bound)]
    upper = np.r_[np.full(system.n, np.inf), np.full(system.n, 1 + bound)]

    def fail(message, **diagnostics):
        raise CavitatingNewtonError(message, history, system.solution(z), diagnostics)

    for iteration in range(maximum_iterations + 1):
        try:
            solution = system.solution(z)
            metrics = problem.metrics(
                solution.pressure_pa, solution.fractional_film_content, solution.film_thickness_m
            )
            residual = system.residual(z)
        except ValueError as error:
            raise CavitatingNewtonError(str(error), history, None) from error
        force_error = solution.load_n - target_load_n if target_load_n is not None else 0.0
        if max(metrics.pressure_violation, metrics.saturation_violation) > bound * (1 + 1e-5):
            fail("Newton initial/accepted state is inadmissible")
        if (
            metrics.mass_maximum <= mass_tolerance
            and metrics.maximum_relative_local_balance <= mass_tolerance
            and max(
                metrics.maximum_complementarity, np.max(np.abs(residual[system.n : 2 * system.n]))
            )
            <= bound
            and (target_load_n is None or abs(force_error / target_load_n) <= load_tolerance)
        ):
            return CavitatingNewtonResult(solution, tuple(history), target_load_n, force_error)
        if iteration == maximum_iterations:
            fail("Newton iteration limit exhausted", mass_maximum=metrics.mass_maximum)
        try:
            linear = solve_cavitating_correction(
                system,
                z,
                krylov_rtol=krylov_rtol,
                maximum_krylov_cycles=maximum_krylov_cycles,
                gmres_restart=gmres_restart,
            )
        except CavitatingLinearError as error:
            fail(str(error), **error.diagnostics)
        correction = linear.direction
        slope = system.merit_slope(z, residual, correction)
        merit = float(0.5 * np.mean(residual**2))
        if not math.isfinite(slope) or slope >= 0:
            fail("Newton correction is not a descent direction", slope=slope)
        step = 1.0
        state_part, direction_part = z[: 2 * system.n], correction[: 2 * system.n]
        for values, change, bounds in (
            (state_part, direction_part, lower),
            (-state_part, -direction_part, -upper),
        ):
            decreasing = change < 0
            if np.any(decreasing):
                step = min(
                    step,
                    0.99
                    * float(
                        np.min((values[decreasing] - bounds[decreasing]) / -change[decreasing])
                    ),
                )
        accepted = False
        for _rejected in range(maximum_backtracks + 1):
            candidate = z + step * correction
            try:
                candidate_fields = system.unpack(candidate)
                feasible = np.all(candidate[: 2 * system.n] >= lower) and np.all(
                    candidate[: 2 * system.n] <= upper
                )
                trial_residual = system.residual(candidate) if feasible else None
                trial_merit = float(0.5 * np.mean(trial_residual**2)) if feasible else math.inf
            except ValueError:
                trial_merit = math.inf
            if (
                step > 0
                and math.isfinite(trial_merit)
                and trial_merit <= merit + 1e-4 * step * slope
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            fail("feasibility/Armijo search exhausted", step=step, merit=merit)
        candidate_metrics = problem.metrics(
            candidate_fields[0] * problem.pressure_scale_pa,
            candidate_fields[1],
            candidate_fields[2],
        )
        history.append(
            CavitatingNewtonIteration(
                iteration + 1,
                metrics.mass_maximum,
                metrics.maximum_complementarity,
                force_error,
                step,
                _rejected,
                linear.linear_iterations,
                linear.true_linear_residual,
                merit,
                trial_merit,
                slope,
                float(candidate_fields[2].min()),
                int(np.count_nonzero(z[: system.n] >= z[system.n : 2 * system.n])),
                linear.used_bordered_fallback,
                linear.schur_indicator,
                candidate_metrics.mass_maximum,
                float(candidate_fields[4]),
                float(candidate_fields[0].min()) * problem.pressure_scale_pa,
                float(candidate_fields[1].min()),
                float(candidate_fields[1].max()),
                linear.linear_true_residuals,
                linear.krylov_histories,
            )
        )
        z = candidate
    raise AssertionError("unreachable Newton termination")
