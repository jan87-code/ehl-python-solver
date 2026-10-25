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
