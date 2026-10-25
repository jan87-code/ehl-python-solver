"""Small-system predictor-corrector pseudo-arclength continuation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ResidualFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]
JacobianFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]
ParameterDerivativeFunction = Callable[
    [NDArray[np.float64], float], NDArray[np.float64]
]
FeasibilityFunction = Callable[[NDArray[np.float64], float], bool]


@dataclass(frozen=True)
class PseudoArclengthStep:
    attempt: int
    accepted: bool
    parameter: float
    arclength_step: float
    newton_iterations: int
    rejected_line_search_trials: int
    residual_rms: float
    arclength_residual: float
    tangent_parameter_component: float
    reason: str
    state: tuple[float, ...]
    smallest_jacobian_singular_value: float = math.nan
    jacobian_condition_number: float = math.nan
    feasible_step_limits: tuple[float, ...] = ()
    accepted_step_lengths: tuple[float, ...] = ()


@dataclass(frozen=True)
class PseudoArclengthResult:
    state: NDArray[np.float64]
    parameter: float
    tangent: NDArray[np.float64]
    steps: tuple[PseudoArclengthStep, ...]
    turning_point_attempts: tuple[int, ...]
    termination_reason: str


def _weighted_norm(vector: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.dot(weights, vector**2)))


def _tangent(
    jacobian: NDArray[np.float64],
    parameter_derivative: NDArray[np.float64],
    weights: NDArray[np.float64],
    previous: NDArray[np.float64] | None,
    parameter_direction: float,
) -> NDArray[np.float64]:
    augmented = np.column_stack((jacobian, parameter_derivative))
    _, singular_values, right = np.linalg.svd(augmented, full_matrices=True)
    if (
        singular_values.size != jacobian.shape[0]
        or not np.all(np.isfinite(singular_values))
        or singular_values[-1]
        <= np.finfo(np.float64).eps * max(augmented.shape) * singular_values[0]
    ):
        raise np.linalg.LinAlgError("continuation tangent system is rank deficient")
    vector = right[-1].copy()
    metric = np.concatenate((weights, [1.0]))
    norm = _weighted_norm(vector, metric)
    if not math.isfinite(norm) or norm == 0.0:
        raise np.linalg.LinAlgError("continuation tangent has invalid weighted norm")
    vector /= norm
    if previous is None:
        if vector[-1] * parameter_direction < 0.0:
            vector *= -1.0
    elif np.dot(metric * previous, vector) < 0.0:
        vector *= -1.0
    return vector


def pseudo_arclength_continue(
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    parameter_derivative: ParameterDerivativeFunction,
    initial_state: NDArray[np.float64],
    initial_parameter: float,
    *,
    state_weights: NDArray[np.float64] | None = None,
    feasible: FeasibilityFunction | None = None,
    parameter_direction: float = 1.0,
    target_parameter: float | None = None,
    initial_step: float = 0.02,
    minimum_step: float = 1e-4,
    maximum_step: float = 0.1,
    maximum_steps: int = 40,
    maximum_corrector_iterations: int = 12,
    maximum_backtracks: int = 16,
    maximum_retries_per_step: int = 6,
    residual_tolerance: float = 1e-8,
    arclength_tolerance: float = 1e-8,
    armijo_constant: float = 1e-4,
) -> PseudoArclengthResult:
    """Follow a solution branch using a scaled tangent predictor and corrector.

    The corrector solves the residual and tangent-plane constraint as one
    bordered Newton system. Dense direct solves and explicit Jacobians make
    this an experimental diagnostic for small systems, not a large-grid
    replacement for the matrix-free production Newton solver.
    """
    state = np.asarray(initial_state, dtype=np.float64).copy()
    parameter = float(initial_parameter)
    if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be a nonempty finite vector")
    if not math.isfinite(parameter):
        raise ValueError("initial_parameter must be finite")
    if not math.isfinite(parameter_direction) or parameter_direction == 0.0:
        raise ValueError("parameter_direction must be finite and nonzero")
    if target_parameter is not None and not math.isfinite(target_parameter):
        raise ValueError("target_parameter must be finite")
    if not (
        math.isfinite(minimum_step)
        and math.isfinite(initial_step)
        and math.isfinite(maximum_step)
        and 0.0 < minimum_step <= initial_step <= maximum_step
    ):
        raise ValueError("arclength steps must satisfy 0 < minimum <= initial <= maximum")
    if (
        not isinstance(maximum_steps, int)
        or isinstance(maximum_steps, bool)
        or maximum_steps < 1
        or not isinstance(maximum_corrector_iterations, int)
        or isinstance(maximum_corrector_iterations, bool)
        or maximum_corrector_iterations < 1
        or not isinstance(maximum_backtracks, int)
        or isinstance(maximum_backtracks, bool)
        or maximum_backtracks < 0
        or not isinstance(maximum_retries_per_step, int)
        or isinstance(maximum_retries_per_step, bool)
        or maximum_retries_per_step < 0
    ):
        raise ValueError("continuation iteration and retry limits are invalid")
    if (
        not math.isfinite(residual_tolerance)
        or residual_tolerance <= 0.0
        or not math.isfinite(arclength_tolerance)
        or arclength_tolerance <= 0.0
        or not math.isfinite(armijo_constant)
        or not 0.0 < armijo_constant < 1.0
    ):
        raise ValueError("continuation tolerances and Armijo constant are invalid")
    if feasible is not None and not feasible(state, parameter):
        raise ValueError("initial continuation state is infeasible")

    initial_residual = np.asarray(residual(state, parameter), dtype=np.float64)
    if (
        initial_residual.shape != state.shape
        or not np.all(np.isfinite(initial_residual))
        or np.linalg.norm(initial_residual) / math.sqrt(state.size) > residual_tolerance
    ):
        raise ValueError("initial continuation state is not a converged residual root")
    weights = (
        np.full(state.size, 1.0 / state.size, dtype=np.float64)
        if state_weights is None
        else np.asarray(state_weights, dtype=np.float64)
    )
    if (
        weights.shape != state.shape
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("state_weights must be finite, positive, and match state")
    current_jacobian = np.asarray(jacobian(state, parameter), dtype=np.float64)
    current_parameter_derivative = np.asarray(
        parameter_derivative(state, parameter), dtype=np.float64
    )
    if (
        current_jacobian.shape != (state.size, state.size)
        or current_parameter_derivative.shape != state.shape
        or not np.all(np.isfinite(current_jacobian))
        or not np.all(np.isfinite(current_parameter_derivative))
    ):
        raise ValueError("initial Jacobian and parameter derivative have invalid shapes")
    tangent = _tangent(
        current_jacobian,
        current_parameter_derivative,
        weights,
        None,
        parameter_direction,
    )
    step_size = initial_step
    records: list[PseudoArclengthStep] = []
    termination_reason = "maximum_steps"

    for attempt in range(1, maximum_steps + 1):
        accepted = False
        retry_count = 0
        while not accepted and retry_count <= maximum_retries_per_step:
            predictor = np.concatenate((state, [parameter])) + step_size * tangent
            predicted_state = predictor[:-1]
            predicted_parameter = float(predictor[-1])
            if feasible is not None and not feasible(predicted_state, predicted_parameter):
                records.append(
                    PseudoArclengthStep(
                        attempt, False, predicted_parameter, step_size, 0, 0,
                        math.inf, math.inf, float(tangent[-1]), "infeasible_predictor",
                        tuple(float(value) for value in predicted_state),
                    )
                )
                retry_count += 1
                step_size *= 0.5
                if step_size < minimum_step:
                    termination_reason = "minimum_arclength_step_after_infeasible_predictor"
                    break
                continue

            trial_state = predicted_state.copy()
            trial_parameter = predicted_parameter
            rejected_trials = 0
            corrector_converged = False
            reason = "maximum_corrector_iterations"
            residual_rms = math.inf
            arc_residual = math.inf
            newton_iterations = 0
            feasible_step_limits: list[float] = []
            accepted_step_lengths: list[float] = []
            metric = np.concatenate((weights, [1.0]))

            for corrector_iteration in range(1, maximum_corrector_iterations + 1):
                newton_iterations = corrector_iteration
                values = np.asarray(residual(trial_state, trial_parameter), dtype=np.float64)
                residual_rms = float(np.linalg.norm(values) / math.sqrt(state.size))
                displacement = np.concatenate((trial_state, [trial_parameter])) - predictor
                arc_residual = float(np.dot(metric * tangent, displacement))
                if (
                    residual_rms <= residual_tolerance
                    and abs(arc_residual) <= arclength_tolerance
                ):
                    corrector_converged = True
                    reason = "converged"
                    break

                trial_jacobian = np.asarray(
                    jacobian(trial_state, trial_parameter), dtype=np.float64
                )
                trial_parameter_derivative = np.asarray(
                    parameter_derivative(trial_state, trial_parameter),
                    dtype=np.float64,
                )
                bordered = np.empty((state.size + 1, state.size + 1), dtype=np.float64)
                bordered[:-1, :-1] = trial_jacobian
                bordered[:-1, -1] = trial_parameter_derivative
                bordered[-1, :] = metric * tangent
                right_hand_side = np.concatenate((-values, [-arc_residual]))
                try:
                    correction = np.linalg.solve(bordered, right_hand_side)
                except np.linalg.LinAlgError:
                    reason = "singular_bordered_corrector"
                    break

                residual_direction = bordered @ correction
                merit = 0.5 * (
                    np.dot(values, values) / state.size + arc_residual**2
                )
                merit_slope = float(
                    np.dot(values, residual_direction[:-1]) / state.size
                    + arc_residual * residual_direction[-1]
                )
                if not math.isfinite(merit_slope) or merit_slope >= 0.0:
                    reason = "non_descent_corrector"
                    break

                step_length = 1.0
                line_search_accepted = False
                largest_feasible_step = 0.0
                for _ in range(maximum_backtracks + 1):
                    candidate_state = trial_state + step_length * correction[:-1]
                    candidate_parameter = trial_parameter + step_length * correction[-1]
                    is_feasible = feasible is None or feasible(
                        candidate_state, candidate_parameter
                    )
                    if is_feasible:
                        largest_feasible_step = max(
                            largest_feasible_step, step_length
                        )
                        try:
                            candidate_residual = np.asarray(
                                residual(candidate_state, candidate_parameter),
                                dtype=np.float64,
                            )
                            candidate_displacement = (
                                np.concatenate((candidate_state, [candidate_parameter]))
                                - predictor
                            )
                            candidate_arc = float(
                                np.dot(metric * tangent, candidate_displacement)
                            )
                            candidate_merit = 0.5 * (
                                np.dot(candidate_residual, candidate_residual)
                                / state.size
                                + candidate_arc**2
                            )
                        except (FloatingPointError, ValueError):
                            candidate_merit = math.inf
                    else:
                        candidate_merit = math.inf
                    if (
                        math.isfinite(candidate_merit)
                        and candidate_merit
                        <= merit + armijo_constant * step_length * merit_slope
                    ):
                        trial_state = candidate_state
                        trial_parameter = float(candidate_parameter)
                        feasible_step_limits.append(largest_feasible_step)
                        accepted_step_lengths.append(step_length)
                        line_search_accepted = True
                        break
                    rejected_trials += 1
                    step_length *= 0.5
                if not line_search_accepted:
                    reason = "armijo_backtracking_exhausted"
                    break

            if corrector_converged:
                next_jacobian = np.asarray(
                    jacobian(trial_state, trial_parameter), dtype=np.float64
                )
                next_parameter_derivative = np.asarray(
                    parameter_derivative(trial_state, trial_parameter),
                    dtype=np.float64,
                )
                try:
                    next_tangent = _tangent(
                        next_jacobian,
                        next_parameter_derivative,
                        weights,
                        tangent,
                        parameter_direction,
                    )
                except np.linalg.LinAlgError:
                    reason = "rank_deficient_updated_tangent"
                    corrector_converged = False

            if corrector_converged:
                jacobian_singular_values = np.linalg.svd(
                    next_jacobian, compute_uv=False
                )
                smallest_jacobian_singular_value = float(
                    jacobian_singular_values[-1]
                )
                jacobian_condition_number = (
                    math.inf
                    if smallest_jacobian_singular_value == 0.0
                    else float(
                        jacobian_singular_values[0]
                        / smallest_jacobian_singular_value
                    )
                )
                records.append(
                    PseudoArclengthStep(
                        attempt,
                        True,
                        trial_parameter,
                        step_size,
                        newton_iterations,
                        rejected_trials,
                        residual_rms,
                        arc_residual,
                        float(next_tangent[-1]),
                        "converged",
                        tuple(float(value) for value in trial_state),
                        smallest_jacobian_singular_value,
                        jacobian_condition_number,
                        tuple(feasible_step_limits),
                        tuple(accepted_step_lengths),
                    )
                )
                state = trial_state
                parameter = trial_parameter
                tangent = next_tangent
                accepted = True
                if target_parameter is not None and (
                    parameter_direction * (parameter - target_parameter) >= 0.0
                ):
                    termination_reason = "target_parameter_reached"
                    break
                if newton_iterations <= 3 and rejected_trials == 0:
                    step_size = min(maximum_step, 1.25 * step_size)
                elif newton_iterations >= 7 or rejected_trials >= 4:
                    step_size = max(minimum_step, 0.7 * step_size)
            else:
                records.append(
                    PseudoArclengthStep(
                        attempt,
                        False,
                        trial_parameter,
                        step_size,
                        newton_iterations,
                        rejected_trials,
                        residual_rms,
                        arc_residual,
                        float(tangent[-1]),
                        reason,
                        tuple(float(value) for value in trial_state),
                        feasible_step_limits=tuple(feasible_step_limits),
                        accepted_step_lengths=tuple(accepted_step_lengths),
                    )
                )
                retry_count += 1
                step_size *= 0.5
                if step_size < minimum_step:
                    termination_reason = "minimum_arclength_step_after_corrector_failure"
                    break

        if termination_reason.startswith("minimum_arclength_step"):
            break
        if termination_reason == "target_parameter_reached":
            break
        if accepted:
            continue
        termination_reason = "maximum_retries_per_step"
        break
    else:
        termination_reason = "maximum_steps"

    accepted_steps = [step for step in records if step.accepted]
    turning_points = tuple(
        current.attempt
        for previous, current in zip(accepted_steps, accepted_steps[1:], strict=False)
        if previous.tangent_parameter_component
        * current.tangent_parameter_component
        < 0.0
    )
    return PseudoArclengthResult(
        state.copy(),
        parameter,
        tangent.copy(),
        tuple(records),
        turning_points,
        termination_reason,
    )
