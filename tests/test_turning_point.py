import numpy as np
import pytest

from ehl_solver.turning_point import localize_parameter_turn


def _quadratic_fold(scale=1.0, **options):
    return localize_parameter_turn(
        lambda state, parameter: state**2 + scale * parameter - 1,
        lambda state, _parameter: np.array([[2 * state[0]]]),
        lambda _state, _parameter: np.array([scale]),
        np.array([-0.1]), 0.99 / scale, np.array([0.1]), 0.99 / scale,
        parameter_weight=scale**2, **options,
    )


def test_localizes_known_fold_with_root_and_tangent_bracket_checks():
    result = _quadratic_fold()
    assert result.termination_reason == "localized"
    assert result.bracket_width <= 1e-6
    assert abs(result.normalized_parameter_tangent) <= 1e-6
    assert abs(result.best.state[0]) < 5e-7
    assert result.best.parameter == pytest.approx(1.0, abs=1e-10)
    assert result.left.tangent[-1] * result.right.tangent[-1] <= 0
    assert all(probe.steps[-1].residual_rms <= 1e-10 for probe in result.probes)
    np.testing.assert_allclose(result.best.state**2 + result.best.parameter, 1, atol=1e-10)


def test_localization_preserves_linear_parameter_scaling():
    reference, scaled = _quadratic_fold(), _quadratic_fold(scale=10.0)
    assert reference.termination_reason == scaled.termination_reason == "localized"
    np.testing.assert_allclose(reference.best.state, scaled.best.state, atol=1e-12)
    assert reference.best.parameter == pytest.approx(10 * scaled.best.parameter, abs=1e-12)
    assert reference.bracket_width == pytest.approx(scaled.bracket_width, abs=1e-12)


def test_localizes_cubic_turn_with_transverse_state_and_nonuniform_weights():
    def state(x):
        return np.array([x, 2 * x])

    result = localize_parameter_turn(
        lambda z, parameter: np.array([z[0]**3 - z[0] + parameter, z[1] - 2*z[0]]),
        lambda z, _parameter: np.array([[3*z[0]**2 - 1, 0], [-2, 1]]),
        lambda _z, _parameter: np.array([1.0, 0.0]),
        state(0.45), 0.45 - 0.45**3, state(0.65), 0.65 - 0.65**3,
        state_weights=np.array([0.5, 0.25]),
    )
    assert result.termination_reason == "localized"
    np.testing.assert_allclose(result.best.state, state(1 / np.sqrt(3)), atol=1e-6)
    assert result.best.parameter == pytest.approx(2 / (3*np.sqrt(3)), abs=1e-10)


def test_failed_corrector_retains_original_bracket_and_failed_probe():
    # Feasible endpoints, but an infeasible tangent predictor near the fold.
    result = _quadratic_fold(feasible=lambda _state, parameter: parameter <= 1.0)
    assert result.termination_reason == "corrector_failed"
    assert result.left.parameter == result.right.parameter == 0.99
    assert result.probes[-1].steps[-1].reason == "infeasible_predictor"
    assert not result.probes[-1].steps[-1].accepted


def test_refinement_budget_is_not_reported_as_localization():
    result = _quadratic_fold(maximum_refinements=1)
    assert result.termination_reason == "maximum_refinements"
    assert result.bracket_width > 1e-6
    assert result.probes


def test_same_sign_endpoint_tangents_are_rejected():
    with pytest.raises(ValueError, match="opposing"):
        localize_parameter_turn(
            lambda state, parameter: state - parameter,
            lambda _state, _parameter: np.eye(1),
            lambda _state, _parameter: -np.ones(1),
            np.array([0.0]), 0.0, np.array([0.1]), 0.1,
        )


def test_noncontracting_probe_keeps_the_previous_bracket(monkeypatch):
    from dataclasses import replace

    import ehl_solver.turning_point as module

    original = module.pseudo_arclength_continue

    def jump_to_distant_root(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("target_parameter") is None:
            return replace(result, state=np.array([10.0]), parameter=-99.0)
        return result

    monkeypatch.setattr(module, "pseudo_arclength_continue", jump_to_distant_root)
    result = _quadratic_fold()
    assert result.termination_reason == "noncontracting_probe"
    assert result.left.parameter == result.right.parameter == 0.99
    assert result.probes[-1].parameter == -99.0
    assert result.bracket_widths[-1] == result.bracket_widths[0]


@pytest.mark.parametrize("options", [
    {"maximum_refinements": 0}, {"maximum_state_size": 0},
    {"bracket_tolerance": float("nan")}, {"tangent_tolerance": 0.0},
])
def test_invalid_localization_controls_are_rejected(options):
    with pytest.raises(ValueError):
        _quadratic_fold(**options)
