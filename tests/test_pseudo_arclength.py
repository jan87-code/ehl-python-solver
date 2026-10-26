import numpy as np
import pytest

from ehl_solver.pseudo_arclength import pseudo_arclength_continue


def test_pseudo_arclength_tracks_known_fold_in_scalar_nonlinear_system():
    def residual(state, parameter):
        return np.array([state[0] ** 2 + parameter - 1.0])

    def jacobian(state, _parameter):
        return np.array([[2.0 * state[0]]])

    def parameter_derivative(_state, _parameter):
        return np.array([1.0])

    result = pseudo_arclength_continue(
        residual,
        jacobian,
        parameter_derivative,
        np.array([-1.0]),
        0.0,
        target_parameter=1.01,
        initial_step=0.08,
        maximum_step=0.1,
        maximum_steps=32,
    )

    accepted = [step for step in result.steps if step.accepted]
    assert len(accepted) > 10
    assert result.turning_point_attempts
    assert any(step.parameter > 0.999 for step in accepted)
    assert accepted[-1].state[0] > 0.0
    assert accepted[-1].parameter < 0.999
    assert min(step.smallest_jacobian_singular_value for step in accepted) < 0.01
    assert all(step.residual_rms <= 1e-8 for step in accepted)
    assert all(abs(step.arclength_residual) <= 1e-8 for step in accepted)
    assert result.termination_reason == "maximum_steps"


def test_rejected_predictor_retains_attempted_state_for_diagnostics():
    result = pseudo_arclength_continue(
        lambda state, parameter: np.array([state[0] - parameter]),
        lambda _state, _parameter: np.array([[1.0]]),
        lambda _state, _parameter: np.array([-1.0]),
        np.array([0.0]),
        0.0,
        feasible=lambda _state, parameter: parameter <= 0.01,
        initial_step=0.02,
        maximum_step=0.02,
        maximum_retries_per_step=0,
    )

    rejected = result.steps[0]
    assert not rejected.accepted
    assert rejected.reason == "infeasible_predictor"
    np.testing.assert_allclose(
    rejected.parameter,
    rejected.state[0],
    rtol=1e-13,
    atol=1e-15,
    )   
    assert rejected.parameter > 0.01


@pytest.mark.parametrize("direction, initial, target", [(1.0, 0.0, 0.05), (-1.0, 0.1, 0.05)])
def test_target_corrector_solves_exact_endpoint_inside_feasibility_bounds(
    direction, initial, target
):
    result = pseudo_arclength_continue(
        lambda state, parameter: np.array([state[0] - parameter**2]),
        lambda _state, _parameter: np.array([[1.0]]),
        lambda _state, parameter: np.array([-2.0 * parameter]),
        np.array([initial**2]),
        initial,
        parameter_direction=direction,
        target_parameter=target,
        initial_step=0.1,
        maximum_step=0.1,
        feasible=lambda _state, parameter: (
            min(initial, target) <= parameter <= max(initial, target)
        ),
    )
    assert result.termination_reason == "target_parameter_reached"
    assert result.parameter == target
    np.testing.assert_allclose(result.state, [target**2], atol=1e-10)
    assert len(result.steps) == 1
    assert result.steps[-1].residual_rms <= 1e-8
    assert result.steps[-1].arclength_residual == 0.0


def test_initial_target_returns_verified_root_without_stepping():
    result = pseudo_arclength_continue(
        lambda state, parameter: state - parameter,
        lambda _state, _parameter: np.eye(1),
        lambda _state, _parameter: -np.ones(1),
        np.array([0.0]), 0.0, target_parameter=0.0,
    )
    assert result.parameter == 0.0
    assert result.steps == ()
    assert result.termination_reason == "target_parameter_reached"


def test_curved_branch_corrector_crossing_recorrects_at_target():
    # At x=0, lambda=x+x**2 has a unit slope. The tangent predicts
    # lambda=0.0707, while the unconstrained corrector crosses 0.072.
    result = pseudo_arclength_continue(
        lambda state, parameter: np.array([state[0] + state[0]**2 - parameter]),
        lambda state, _parameter: np.array([[1.0 + 2.0 * state[0]]]),
        lambda _state, _parameter: np.array([-1.0]),
        np.array([0.0]), 0.0, target_parameter=0.072,
        initial_step=0.1, maximum_step=0.1, maximum_steps=1,
    )
    assert result.termination_reason == "target_parameter_reached"
    assert result.parameter == 0.072
    np.testing.assert_allclose(
        result.state + result.state**2, [result.parameter], atol=1e-8
    )


@pytest.mark.parametrize("target", [1.01, 0.5])
def test_weighted_parameter_metric_preserves_branch_under_linear_reparameterization(target):
    def follow(parameter_scale):
        return pseudo_arclength_continue(
            lambda state, parameter: state**2 + parameter_scale * parameter - 1,
            lambda state, _parameter: np.array([[2 * state[0]]]),
            lambda _state, _parameter: np.array([parameter_scale]),
            np.array([-1.0]), 0.0, target_parameter=target / parameter_scale,
            parameter_weight=parameter_scale**2,
            initial_step=0.08, maximum_step=0.1, maximum_steps=32,
        )
    reference, scaled = follow(1.0), follow(10.0)
    assert reference.turning_point_attempts == scaled.turning_point_attempts
    assert reference.termination_reason == scaled.termination_reason
    np.testing.assert_allclose(
        [step.state for step in reference.steps], [step.state for step in scaled.steps], atol=1e-12
    )
    np.testing.assert_allclose(
        [step.parameter for step in reference.steps],
        [10 * step.parameter for step in scaled.steps], atol=1e-12,
    )
    assert np.dot(scaled.tangent[:-1], scaled.tangent[:-1]) + 100 * scaled.tangent[-1]**2 \
        == pytest.approx(1.0)


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_parameter_metric_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="parameter_weight"):
        pseudo_arclength_continue(
            lambda state, parameter: state - parameter,
            lambda _state, _parameter: np.eye(1),
            lambda _state, _parameter: -np.ones(1),
            np.array([0.0]), 0.0, parameter_weight=weight,
        )
