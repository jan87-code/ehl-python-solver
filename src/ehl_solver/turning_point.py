"""Bounded small-system localization of a parameter turning point on a branch."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from ehl_solver.pseudo_arclength import (
    FeasibilityFunction,
    JacobianFunction,
    ParameterDerivativeFunction,
    PseudoArclengthResult,
    ResidualFunction,
    pseudo_arclength_continue,
)


@dataclass(frozen=True)
class TurningPointResult:
    """A retained bracket and all attempted refinements, including failures.

    State/parameter conventions are supplied by the residual callbacks, as
    for pseudo_arclength_continue. bracket_width uses the supplied weighted
    metric; normalized_parameter_tangent is sqrt(parameter_weight)*t_lambda.
    This locates a stationary parameter, without proving a simple fold or a
    physical instability. Endpoint tangents are oriented along the bracket.
    """

    left: PseudoArclengthResult
    right: PseudoArclengthResult
    best: PseudoArclengthResult
    probes: tuple[PseudoArclengthResult, ...]
    bracket_width: float
    bracket_widths: tuple[float, ...]
    normalized_parameter_tangent: float
    termination_reason: str


def localize_parameter_turn(
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    parameter_derivative: ParameterDerivativeFunction,
    left_state: NDArray[np.float64],
    left_parameter: float,
    right_state: NDArray[np.float64],
    right_parameter: float,
    *,
    state_weights: NDArray[np.float64] | None = None,
    parameter_weight: float = 1.0,
    feasible: FeasibilityFunction | None = None,
    bracket_tolerance: float = 1e-6,
    tangent_tolerance: float = 1e-6,
    residual_tolerance: float = 1e-10,
    maximum_refinements: int = 24,
    maximum_state_size: int = 512,
) -> TurningPointResult:
    """Shrink a local root bracket with opposing oriented parameter tangents.

    Each probe follows the left root for half the weighted chord length,
    using one existing tangent-plane predictor/corrector step with strict
    feasibility and Armijo globalization. Both endpoint distances must be
    below 0.9 of the old width; otherwise the original bracket is retained.
    Opposing tangent signs select the new bracket. Both bracket width and
    normalized tangent tolerances must pass before returning 'localized'.
    Failed/budget-exhausted results retain the last bracket and probe history.
    No callback state is mutated here; callers own any mutable problem state.
    """
    left_values = np.asarray(left_state, dtype=np.float64)
    right_values = np.asarray(right_state, dtype=np.float64)
    if (
        not isinstance(maximum_state_size, int) or isinstance(maximum_state_size, bool)
        or maximum_state_size < 1 or left_values.ndim != 1 or left_values.size == 0
        or left_values.size > maximum_state_size or right_values.shape != left_values.shape
    ):
        raise ValueError("turning-point state sizes are invalid or exceed the dense limit")
    if (
        not isinstance(maximum_refinements, int) or isinstance(maximum_refinements, bool)
        or maximum_refinements < 1
        or not all(math.isfinite(value) and value > 0 for value in
                   (bracket_tolerance, tangent_tolerance, residual_tolerance))
    ):
        raise ValueError("turning-point tolerances and refinement limit must be positive")
    weights = (
        np.full(left_values.size, 1.0 / left_values.size)
        if state_weights is None else np.asarray(state_weights, dtype=np.float64)
    )
    options = dict(
        state_weights=weights, parameter_weight=parameter_weight, feasible=feasible,
        residual_tolerance=residual_tolerance, arclength_tolerance=residual_tolerance,
    )
    # Returning at the initial target validates feasibility, root, derivatives
    # and metric without advancing either supplied endpoint.
    left = pseudo_arclength_continue(
        residual, jacobian, parameter_derivative, left_values, left_parameter,
        target_parameter=left_parameter, **options,
    )
    right = pseudo_arclength_continue(
        residual, jacobian, parameter_derivative, right_values, right_parameter,
        target_parameter=right_parameter, **options,
    )
    metric = np.r_[weights, parameter_weight]

    def vector(point):
        return np.r_[point.state, point.parameter]

    def distance(a, b):
        difference = vector(a) - vector(b)
        return float(np.sqrt(np.dot(metric, difference**2)))

    chord = vector(right) - vector(left)
    if not math.isfinite(distance(left, right)) or distance(left, right) == 0.0:
        raise ValueError("turning-point endpoints must define a finite nonzero bracket")
    if np.dot(metric * left.tangent, chord) < 0:
        left = replace(left, tangent=-left.tangent)
    if np.dot(metric * right.tangent, chord) < 0:
        right = replace(right, tangent=-right.tangent)
    if left.tangent[-1] * right.tangent[-1] >= 0:
        raise ValueError("turning-point endpoints must have opposing oriented parameter tangents")

    probes = []
    widths = [distance(left, right)]
    reason = "maximum_refinements"
    parameter_scale = math.sqrt(parameter_weight)
    for _ in range(maximum_refinements):
        best = min((left, right), key=lambda point: abs(point.tangent[-1]))
        width = distance(left, right)
        if (
            width <= bracket_tolerance
            and abs(parameter_scale * best.tangent[-1]) <= tangent_tolerance
        ):
            reason = "localized"
            break
        step = 0.5 * width
        probe = pseudo_arclength_continue(
            residual, jacobian, parameter_derivative, left.state, left.parameter,
            parameter_direction=math.copysign(1.0, left.tangent[-1]),
            initial_step=step, minimum_step=step, maximum_step=step,
            maximum_steps=1, maximum_retries_per_step=0, **options,
        )
        if np.dot(metric * left.tangent, probe.tangent) < 0:
            probe = replace(probe, tangent=-probe.tangent)
        probes.append(probe)
        if not probe.steps or not probe.steps[-1].accepted:
            reason = "corrector_failed"
            widths.append(width)
            break
        if max(distance(left, probe), distance(probe, right)) >= 0.9 * width:
            reason = "noncontracting_probe"
            widths.append(width)
            break
        if probe.tangent[-1] == 0.0:
            left = right = probe
        elif left.tangent[-1] * probe.tangent[-1] > 0:
            left = probe
        else:
            right = probe
        widths.append(distance(left, right))
    best = min((left, right), key=lambda point: abs(point.tangent[-1]))
    width = distance(left, right)
    normalized_tangent = float(parameter_scale * best.tangent[-1])
    if width <= bracket_tolerance and abs(normalized_tangent) <= tangent_tolerance:
        reason = "localized"
    return TurningPointResult(
        left, right, best, tuple(probes), width, tuple(widths), normalized_tangent, reason
    )


@dataclass(frozen=True)
class TurningPointConditions:
    """Numerical nondegeneracy evidence at an approximate stationary root.

    phi has unit weighted state norm; psi has unit Euclidean residual norm.
    The normalized parameter is c=sqrt(parameter_weight)*parameter.
    Coefficients a=psi.T F_c and b=psi.T F_zz[phi,phi] predict c''=-b/a
    at an exact simple turn. Approximate singular values and finite differences
    do not constitute a rigorous rank/nondegeneracy or continuum proof.
    """

    residual_rms: float
    singular_values: tuple[float, ...]
    smallest_to_next_singular_ratio: float
    finite_difference_jacobian_relative_error: float
    right_null_direction: tuple[float, ...]
    transversality_coefficient: float
    normalized_transversality: float
    curvature_steps: tuple[float, ...]
    curvature_coefficients: tuple[float, ...]
    normalized_curvature_projections: tuple[float, ...]
    predicted_parameter_curvatures: tuple[float, ...]
    relative_curvature_step_variation: float


def audit_turning_point_conditions(
    residual: ResidualFunction,
    jacobian: JacobianFunction,
    parameter_derivative: ParameterDerivativeFunction,
    state: NDArray[np.float64],
    parameter: float,
    *,
    state_weights: NDArray[np.float64] | None = None,
    parameter_weight: float = 1.0,
    feasible: FeasibilityFunction | None = None,
    curvature_steps: tuple[float, ...] = (1e-2, 3e-3, 1e-3),
    jacobian_difference_step: float = 1e-5,
    residual_tolerance: float = 1e-9,
    maximum_state_size: int = 512,
) -> TurningPointConditions:
    """Check a near-null direction, transversality and residual curvature.

    An independently differenced full residual Jacobian checks the supplied
    analytic matrix. Curvature comes from centered *residual* differences
    along the weighted-normalized right singular vector at fixed parameter.
    All perturbed states must be feasible; no step repair or clipping occurs.
    Residual/parameter conventions are the caller's explicit nondimensional
    ones, as in localize_parameter_turn. Scalar systems report a NaN
    smallest-to-next ratio because no second singular value exists.
    """
    values = np.asarray(state, dtype=np.float64)
    if (
        not isinstance(maximum_state_size, int) or isinstance(maximum_state_size, bool)
        or values.ndim != 1 or not 0 < values.size <= maximum_state_size
        or not np.all(np.isfinite(values)) or not math.isfinite(parameter)
    ):
        raise ValueError("condition-audit state is invalid or exceeds the dense limit")
    if (
        not math.isfinite(parameter_weight) or parameter_weight <= 0
        or not math.isfinite(jacobian_difference_step) or jacobian_difference_step <= 0
        or not math.isfinite(residual_tolerance) or residual_tolerance <= 0
        or not curvature_steps
        or not all(math.isfinite(step) and step > 0 for step in curvature_steps)
    ):
        raise ValueError("condition-audit weights, steps and tolerances must be positive")
    weights = (
        np.full(values.size, 1 / values.size) if state_weights is None
        else np.asarray(state_weights, dtype=np.float64)
    )
    if weights.shape != values.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("state_weights must be finite, positive and match the state")

    def evaluate(point):
        if feasible is not None and not feasible(point, parameter):
            raise ValueError("condition-audit perturbation is infeasible")
        result = np.asarray(residual(point, parameter), dtype=np.float64)
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise ValueError("condition-audit residual must be a finite vector matching state")
        return result

    root_residual = evaluate(values)
    rms = float(np.sqrt(np.mean(root_residual**2)))
    if rms > residual_tolerance:
        raise ValueError("condition-audit state is not a converged residual root")
    matrix = np.asarray(jacobian(values, parameter), dtype=np.float64)
    parameter_column = np.asarray(parameter_derivative(values, parameter), dtype=np.float64)
    if (
        matrix.shape != (values.size, values.size) or not np.all(np.isfinite(matrix))
        or parameter_column.shape != values.shape or not np.all(np.isfinite(parameter_column))
    ):
        raise ValueError("condition-audit derivatives have invalid shapes or values")
    left, singular_values, right = np.linalg.svd(matrix)
    phi = right[-1] / np.sqrt(np.dot(weights, right[-1]**2))
    normalized_column = parameter_column / math.sqrt(parameter_weight)
    psi = left[:, -1]
    if np.dot(psi, normalized_column) < 0:
        psi = -psi
    a = float(np.dot(psi, normalized_column))
    column_norm = float(np.linalg.norm(normalized_column))
    transversality = a / column_norm if column_norm > 0 else 0.0
    difference_matrix = np.empty_like(matrix)
    for index in range(values.size):
        step = jacobian_difference_step * max(1.0, abs(values[index]))
        direction = np.zeros(values.size)
        direction[index] = step
        difference_matrix[:, index] = (
            evaluate(values + direction) - evaluate(values - direction)
        ) / (2 * step)
    difference_norm = float(np.linalg.norm(difference_matrix))
    matrix_error = float(np.linalg.norm(matrix - difference_matrix))
    relative_error = (matrix_error / difference_norm if difference_norm > 0
                      else 0.0 if matrix_error == 0 else math.inf)
    coefficients, projections = [], []
    for step in curvature_steps:
        second = (evaluate(values + step * phi) - 2 * root_residual
                  + evaluate(values - step * phi)) / step**2
        b = float(np.dot(psi, second))
        norm = float(np.linalg.norm(second))
        coefficients.append(b)
        projections.append(abs(b) / norm if norm > 0 else 0.0)
    coefficient_scale = max(abs(b) for b in coefficients)
    # Compare to the smallest-step estimate irrespective of input ordering.
    smallest_step_b = coefficients[int(np.argmin(curvature_steps))]
    variation = (max(abs(b - smallest_step_b) for b in coefficients) / coefficient_scale
                 if coefficient_scale > 0 else 0.0)
    ratio = (float(singular_values[-1] / singular_values[-2]) if values.size > 1
             and singular_values[-2] > 0 else math.inf if values.size > 1 else math.nan)
    return TurningPointConditions(
        rms, tuple(float(value) for value in singular_values), ratio, relative_error,
        tuple(float(value) for value in phi), a, transversality,
        tuple(curvature_steps), tuple(coefficients), tuple(projections),
        tuple(-b / a if a != 0 else math.nan for b in coefficients), variation,
    )
