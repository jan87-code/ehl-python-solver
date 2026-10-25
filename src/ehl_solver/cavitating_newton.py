"""Experimental monolithic CPU Newton solve of JFO pressure/content and film."""

from __future__ import annotations

import math
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
    residual = system.residual(state)
    operator = LinearOperator(
        (system.size, system.size),
        matvec=lambda v: system.jacobian_vector(state, v),
        dtype=np.float64,
    )
    matrix = system.frozen_matrix(state)
    true_residuals, counts, traces = [], [], []

    def linear_solve(op, rhs, lu):
        count = []
        preconditioner = LinearOperator(op.shape, matvec=lu.solve, dtype=np.float64)
        answer, info = gmres(
            op,
            rhs,
            M=preconditioner,
            rtol=krylov_rtol,
            atol=0,
            maxiter=maximum_krylov_cycles,
            restart=gmres_restart,
            callback=lambda value: count.append(float(value)),
            callback_type="pr_norm",
        )
        norm = float(np.linalg.norm(rhs))
        error = float(np.linalg.norm(op @ answer - rhs))
        measured = error / norm if norm > 0 else error
        counts.append(len(count))
        traces.append(tuple(count))
        true_residuals.append(measured)
        return answer, info == 0 and math.isfinite(measured) and measured <= 5 * krylov_rtol

    def error(message):
        raise CavitatingLinearError(
            message,
            {
                "true_linear_residual": true_residuals[-1] if true_residuals else math.inf,
                "linear_true_residuals": tuple(true_residuals),
                "linear_iterations": sum(counts),
                "krylov_histories": tuple(traces),
            },
        )

    if not np.any(residual):
        return CavitatingCorrection(np.zeros_like(residual), 0.0, 0, False, None, ())
    try:
        lu = splu(matrix[: 2 * system.n, : 2 * system.n])
    except RuntimeError:
        lu = None
    if system.target_load_n is None:
        if lu is None:
            error("frozen-film active-set preconditioner failed")
        correction, good = linear_solve(operator, -residual, lu)
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
        )
    indicator = None
    if lu is not None:
        block = LinearOperator(
            (2 * system.n, 2 * system.n),
            dtype=np.float64,
            matvec=lambda v: (operator @ np.r_[v, 0.0])[: 2 * system.n],
        )
        pressure, good_pressure = linear_solve(block, -residual[: 2 * system.n], lu)
        if good_pressure:
            h0_direction = np.zeros(system.size)
            h0_direction[-1] = 1
            column = (operator @ h0_direction)[: 2 * system.n]
            response, good_response = linear_solve(block, column, lu)
            if good_response:
                load_row = matrix[-1, : 2 * system.n].toarray().ravel()
                denominator = -float(load_row @ response)
                scale = float(np.linalg.norm(load_row) * np.linalg.norm(response))
                indicator = abs(denominator) / max(scale, np.finfo(float).tiny)
                if math.isfinite(indicator) and indicator > schur_tolerance:
                    dh0 = (-residual[-1] - load_row @ pressure) / denominator
                    correction = np.r_[pressure - response * dh0, dh0]
                    measured = float(
                        np.linalg.norm(operator @ correction + residual) / np.linalg.norm(residual)
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
                        )
    try:
        full_lu = splu(matrix)
    except RuntimeError:
        error("full bordered active-set preconditioner failed")
    correction, good = linear_solve(operator, -residual, full_lu)
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
