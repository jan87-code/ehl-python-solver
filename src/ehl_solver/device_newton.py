"""Experimental resident signed-full-film Newton/load solve; CPU reference preserved.

This is a mathematical model with signed pressure, not GPU JFO or physical EHL
validation. Python controls scalar decisions; all iterative state fields and
corrections stay on the explicitly selected torch device.
"""

import math
from dataclasses import dataclass, replace

from ehl_solver.device_bordered import DeviceBorderedError, solve_device_bordered
from ehl_solver.monolithic import (
    KrylovSolveDiagnostic,
    LoadContinuationStep,
    _ArmijoLineSearchResult,
)


@dataclass(frozen=True)
class DeviceNewtonIteration:
    reynolds_residual_scaled: float
    load_residual_scaled: float
    minimum_film_thickness_m: float
    feasible_step_limit: float
    accepted_step_length: float
    merit_directional_derivative: float
    rejected_line_search_trials: int
    forcing_tolerance: float
    bordered_residual_relative: float
    krylov_diagnostics: tuple
    linear_refinement_attempts: int
    used_fallback: bool


@dataclass(frozen=True)
class DeviceNewtonSolution:
    state: object
    pressure_dimensionless: object
    film_thickness_m: object
    elastic_displacement_m: object
    central_film_offset_m: float
    calculated_load_n: float
    history: tuple
    target_load_n: float


class DeviceNewtonError(RuntimeError):
    def __init__(self, message, state, history, diagnostics):
        super().__init__(message)
        self.state = state
        self.history = tuple(history)
        self.diagnostics = diagnostics


def device_line_search(
    evaluator, state, direction, current_merit, merit_slope, *,
    fraction_to_boundary=.99, armijo_constant=1e-4, maximum_backtracks=20,
):
    """Same fraction-to-film-boundary and directional Armijo rules as CPU.

    Film depends affinely on pressure/h0 in this model, so its exact directional
    change is computed directly, without requiring an infeasible unit trial.
    """
    import torch

    _, film, _ = evaluator.state_fields(state)
    delta_film = (
        direction[-1]*evaluator.film_scale_m
        + evaluator.compliance_factor * evaluator.elasticity(
            evaluator._pressure(direction[:-1]) * evaluator.pressure_scale_pa
        )
    )
    negative = delta_film < 0
    feasible = 1.
    if bool(negative.any()):
        feasible = min(1., fraction_to_boundary * float(
            ((film[negative] - evaluator.film_floor_m) / -delta_film[negative]).min()
        ))
    if not math.isfinite(feasible) or feasible <= 0:
        return _ArmijoLineSearchResult(None, feasible, 0., 0, 0., math.inf,
                                       "no_positive_feasible_step")
    alpha, rejected, last_merit = feasible, 0, math.inf
    for _ in range(maximum_backtracks + 1):
        trial = state + alpha*direction
        try:
            residual = evaluator.residual(trial)
            last_merit = .5 * float(
                torch.dot(residual[:-1], residual[:-1])/evaluator.pressure_size + residual[-1]**2
            )
        except ValueError:
            last_merit = math.inf
        if (math.isfinite(last_merit) and
                last_merit <= current_merit + armijo_constant*alpha*merit_slope):
            return _ArmijoLineSearchResult(trial, feasible, alpha, rejected, alpha,
                                           last_merit, "armijo_satisfied")
        rejected += 1
        last_alpha = alpha
        alpha *= .5
    return _ArmijoLineSearchResult(None, feasible, 0., rejected, last_alpha,
                                   last_merit, "backtracking_exhausted")


def solve_device_full_film_load(
    evaluator, initial_state, *, reynolds_tolerance=1e-8, load_tolerance=1e-6,
    max_iterations=30, fraction_to_boundary=.99, armijo_constant=1e-4,
    maximum_backtracks=20, krylov_rtol=1e-8, restart=40, max_krylov_iterations=400,
    preconditioner="laplacian", memory_limit_bytes=2**30,
):
    """Resident counterpart of CPU dual-line-search/inexact Newton formulation.

    initial_state is dimensionless interior P followed by h0/h_ref, already on
    the chosen device. Strict feasibility is required; there is no implicit
    initial h0 raise, clipping, regularization or CPU fallback.
    """
    import torch

    if evaluator.backend.name != "torch":
        raise ValueError("resident Newton requires an explicit torch backend")
    if not 0 < fraction_to_boundary < 1 or not 0 < armijo_constant < 1:
        raise ValueError("fraction_to_boundary and armijo_constant must be in (0,1)")
    if any(not math.isfinite(t) or t <= 0
           for t in (reynolds_tolerance, load_tolerance, krylov_rtol)) or krylov_rtol > 1:
        raise ValueError("invalid nonlinear or Krylov tolerances")
    if (not isinstance(max_iterations, int) or isinstance(max_iterations, bool)
            or max_iterations < 1 or not isinstance(maximum_backtracks, int)
            or isinstance(maximum_backtracks, bool) or maximum_backtracks < 0):
        raise ValueError("invalid Newton/backtracking iteration limits")
    state = evaluator._vector(initial_state).clone()
    history, previous_norm = [], None
    for _ in range(max_iterations):
        residual = evaluator.residual(state)
        reynolds_norm = float(torch.linalg.vector_norm(residual[:-1])) / math.sqrt(
            evaluator.pressure_size
        )
        load = float(residual[-1])
        pressure, film, w = evaluator.state_fields(state)
        merit = .5*(reynolds_norm**2 + load**2)
        norm = math.sqrt(2*merit)
        if reynolds_norm <= reynolds_tolerance and abs(load) <= load_tolerance:
            # Recompute force by nested trapezoids rather than the residual's
            # load-gradient dot product. This exports only a scalar.
            calculated = float(torch.trapezoid(torch.trapezoid(
                pressure*evaluator.pressure_scale_pa, dx=evaluator.dx_m, dim=1
            ), dx=evaluator.dy_m))
            if abs(calculated - evaluator.target_load_n) > (
                load_tolerance*evaluator.load_residual_scale_n
            ):
                raise DeviceNewtonError("independent force check failed", state, history, {})
            return DeviceNewtonSolution(
                state, pressure, film, w, float(state[-1])*evaluator.film_scale_m,
                calculated, tuple(history), evaluator.target_load_n,
            )
        forcing = .01 if previous_norm is None else min(
            .01, max(1e-4, .9*(norm/max(previous_norm, torch.finfo(state.dtype).tiny))**1.5)
        )
        records, refinements, rejected = [], 0, 0

        def correction(tolerance, state=state, records=records):
            try:
                result = solve_device_bordered(
                    evaluator, state, rtol=tolerance, restart=restart,
                    max_iterations=max_krylov_iterations, preconditioner=preconditioner,
                    memory_limit_bytes=memory_limit_bytes,
                )
            except DeviceBorderedError as error:
                append_records(error.diagnostics, tolerance)
                raise DeviceNewtonError(str(error), state, history,
                                         {"krylov_diagnostics": tuple(records)}) from error
            append_records(result.diagnostics, tolerance)
            return result

        def append_records(diagnostics, tolerance, records=records):
            for label, r in diagnostics:
                records.append(KrylovSolveDiagnostic(
                    label, preconditioner, tolerance, min(tolerance, .01), r.iterations,
                    0 if r.converged else 1, r.true_relative_residual,
                    r.estimated_preconditioned_history, r.reason,
                ))

        def slope(direction, state=state, residual=residual):
            dr = evaluator.jacobian_vector(state, direction)
            return float(torch.dot(residual[:-1], dr[:-1])/evaluator.pressure_size
                         + residual[-1]*dr[-1])

        while True:
            current = correction(forcing)
            merit_slope = slope(current.direction)
            for recovery in range(2):
                if math.isfinite(merit_slope) and merit_slope < 0:
                    break
                tolerance = max(10*torch.finfo(state.dtype).eps,
                                min(forcing*.1**(recovery+1), krylov_rtol*.1**recovery))
                current = correction(tolerance)
                merit_slope = slope(current.direction)
            if not math.isfinite(merit_slope) or merit_slope >= 0:
                raise DeviceNewtonError("non-descent after tighter Krylov recovery", state,
                                         history, {"krylov_diagnostics": tuple(records)})
            search = device_line_search(
                evaluator, state, current.direction, merit, merit_slope,
                fraction_to_boundary=fraction_to_boundary, armijo_constant=armijo_constant,
                maximum_backtracks=maximum_backtracks,
            )
            rejected += search.rejected_trials
            if search.accepted_state is not None:
                state = search.accepted_state
                break
            achieved = max(current.bordered_residual_relative,
                           max(r.true_relative_residual for _, r in current.diagnostics))
            if refinements < 2 and achieved > max(1e-7, 2*krylov_rtol):
                refinements += 1
                forcing = max(10*torch.finfo(state.dtype).eps, min(forcing*.1, krylov_rtol))
                continue
            raise DeviceNewtonError("dual line search failed", state, history, {
                "line_search": search, "krylov_diagnostics": tuple(records),
                "merit_slope": merit_slope, "rejected_trials": rejected,
            })
        _, accepted_film, _ = evaluator.state_fields(state)
        history.append(DeviceNewtonIteration(
            reynolds_norm, load, float(accepted_film.min()), search.feasible_step_limit,
            search.accepted_step_length, merit_slope, rejected, forcing,
            current.bordered_residual_relative, tuple(records), refinements, current.used_fallback,
        ))
        previous_norm = norm
    final_residual = evaluator.residual(state)
    raise DeviceNewtonError("Newton iteration limit", state, history, {
        "reynolds_residual_scaled": float(torch.linalg.vector_norm(final_residual[:-1]))
        / math.sqrt(evaluator.pressure_size),
        "load_residual_scaled": float(final_residual[-1]),
    })


@dataclass(frozen=True)
class DeviceLoadContinuation:
    solution: DeviceNewtonSolution
    steps: tuple


class DeviceContinuationError(RuntimeError):
    def __init__(self, solution, steps):
        super().__init__("device load continuation exhausted retries; last accepted state retained")
        self.solution, self.steps = solution, tuple(steps)


def continue_device_load(
    evaluator, initial_solution, target_load_n, *, initial_step_fraction=.25,
    minimum_step_fraction=1e-3, maximum_step_fraction=.5, growth_factor=1.5,
    maximum_retries_per_step=8, newton_options=None,
):
    """Resident adaptive signed-load stepping with CPU reference step semantics.

    Only verified accepted states are warm starts. Nonlinear failures halve
    steps and preserve diagnostics; unexpected device/runtime failures propagate.
    evaluator.target_load_n tracks the last accepted load on exhaustion.
    """
    import torch

    if not math.isfinite(target_load_n) or not (
        0 < minimum_step_fraction <= initial_step_fraction <= maximum_step_fraction <= 1
    ) or not math.isfinite(growth_factor) or growth_factor <= 1:
        raise ValueError("invalid continuation target/step fractions/growth")
    if (not isinstance(maximum_retries_per_step, int)
            or isinstance(maximum_retries_per_step, bool) or maximum_retries_per_step < 0):
        raise ValueError("invalid retry count")
    options = dict(newton_options or {})
    rt, lt = options.get("reynolds_tolerance", 1e-8), options.get("load_tolerance", 1e-6)
    if any(not math.isfinite(t) or t <= 0 for t in (rt, lt)):
        raise ValueError("invalid continuation tolerances")
    initial_load = initial_solution.target_load_n
    eps, tiny = torch.finfo(initial_solution.state.dtype).eps, torch.finfo(torch.float64).tiny
    if not math.isclose(evaluator.target_load_n, initial_load, rel_tol=1e-12,
                        abs_tol=8*eps*evaluator.load_residual_scale_n):
        raise ValueError("initial target differs from evaluator")
    residual = evaluator.residual(initial_solution.state)
    _, film, _ = evaluator.state_fields(initial_solution.state)
    if (not torch.allclose(initial_solution.film_thickness_m, film, rtol=1e-10,
                           atol=32*eps*evaluator.film_scale_m)
            or float(torch.linalg.vector_norm(residual[:-1])) / math.sqrt(
                evaluator.pressure_size) > rt
            or abs(float(residual[-1])) > lt):
        raise ValueError("initial solution is not consistent/converged")
    total = target_load_n - initial_load
    resolution = 8*eps*max(abs(target_load_n), abs(initial_load), tiny)
    if abs(total) <= resolution:
        if abs(initial_solution.calculated_load_n-target_load_n) > (
            lt*evaluator.load_residual_scale_n
        ):
            raise ValueError("unresolvable load change exceeds load tolerance")
        evaluator.target_load_n = target_load_n
        return DeviceLoadContinuation(replace(initial_solution, target_load_n=target_load_n), ())
    accepted, accepted_load = initial_solution, initial_load
    step, retries, records = initial_step_fraction, 0, []
    while abs(target_load_n - accepted_load) > resolution:
        fraction = min(step, abs(target_load_n-accepted_load)/abs(total))
        proposed = accepted_load + math.copysign(fraction*abs(total), total)
        evaluator.target_load_n = proposed
        try:
            candidate = solve_device_full_film_load(evaluator, accepted.state, **options)
            check = evaluator.residual(candidate.state)
            _, candidate_film, _ = evaluator.state_fields(candidate.state)
            if (candidate.target_load_n != proposed
                    or not torch.allclose(candidate_film, candidate.film_thickness_m,
                                          rtol=1e-10, atol=32*eps*evaluator.film_scale_m)
                    or float(torch.linalg.vector_norm(check[:-1])) / math.sqrt(
                        evaluator.pressure_size) > rt
                    or abs(float(check[-1])) > lt):
                raise DeviceNewtonError("continuation candidate failed independent check",
                                         candidate.state, candidate.history, {})
        except (DeviceNewtonError, ValueError) as error:
            details = dict(error.diagnostics) if isinstance(error, DeviceNewtonError) else {}
            if isinstance(error, DeviceNewtonError):
                details.update(failed_state=error.state, newton_history=error.history)
            records.append(LoadContinuationStep(
                len(records)+1, proposed, proposed-accepted_load, False,
                len(error.history) if isinstance(error, DeviceNewtonError) else 0, math.inf,
                accepted.central_film_offset_m, float(accepted.film_thickness_m.min()), str(error),
                tuple(details.items()),
            ))
            retries += 1
            step *= .5
            if retries > maximum_retries_per_step or step < minimum_step_fraction:
                evaluator.target_load_n = accepted_load
                raise DeviceContinuationError(accepted, records) from error
            continue
        except BaseException:
            evaluator.target_load_n = accepted_load
            raise
        records.append(LoadContinuationStep(
            len(records)+1, proposed, proposed-accepted_load, True, len(candidate.history),
            abs(candidate.calculated_load_n-proposed) / max(
                abs(proposed), evaluator.load_residual_scale_n),
            candidate.central_film_offset_m, float(candidate.film_thickness_m.min()), "converged",
        ))
        accepted, accepted_load, retries = candidate, proposed, 0
        step = min(maximum_step_fraction, step*growth_factor)
    return DeviceLoadContinuation(accepted, tuple(records))
