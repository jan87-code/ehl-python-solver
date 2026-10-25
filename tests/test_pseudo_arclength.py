import numpy as np

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
    assert rejected.parameter == rejected.state[0]
    assert rejected.parameter > 0.01
