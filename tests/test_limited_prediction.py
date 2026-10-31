"""Exact MC endpoint audits of secant continuation initialization."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ehl_solver.limited_continuation as module
from ehl_solver.limited_audit import audit_limited_state
from ehl_solver.limited_newton import LimitedNewtonError
from ehl_solver.limited_system import LimitedLoadProbe
from ehl_solver.prototype import PointContactCase


def sha(state):
    return hashlib.sha256(state.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def cold_path():
    case = PointContactCase(
        **json.loads((Path(__file__).parents[1] / "cases/circular_50n.json").read_text())
    )
    stages = []
    path = module.continue_limited_transport(
        case, stage_observer=lambda i, c, t, r: stages.append(r)
    )
    return path, stages


def test_cold_endpoint_passes_independent_exact_equations_and_direct_elasticity(cold_path):
    path, stages = cold_path
    root = path.result
    assert root.transport_fraction == 1.0 and root.history == ()
    audit = audit_limited_state(LimitedLoadProbe(root.case), root.state, direct=True)
    assert audit["numerically_accepted"] and audit["full_limited_equations"]
    assert audit["direct_elasticity_checked"]
    assert len(path.attempts) <= 24
    last = path.attempts[-1]
    assert last["audit_only"] and last["transport_fraction"] == 1.0
    assert last["initialization"]["newton_correction"] is False
    assert last["initialization"]["eligible_for_final_audit"]
    assert stages[-1].audit["numerically_accepted"]


def test_all_prediction_probes_replay_from_accepted_roots(cold_path):
    path, stages = cold_path
    roots = {sha(r.state): r for r in stages if not isinstance(r, LimitedNewtonError)}
    eligible = []
    rejected = []
    for attempt in path.attempts:
        if "endpoint_prediction_probe" not in attempt:
            continue
        info = attempt["endpoint_prediction_probe"]
        previous, last = roots[info["previous_state_sha256"]], roots[info["last_state_sha256"]]
        before = (previous.state.copy(), last.state.copy())
        candidate, replay = module._endpoint_prediction(previous, last)
        assert replay == info
        np.testing.assert_array_equal(previous.state, before[0])
        np.testing.assert_array_equal(last.state, before[1])
        if candidate is None:
            rejected.append(replay)
            assert not replay["eligible_for_final_audit"]
            continue
        eligible.append(candidate)
        direction = info["factor"] * (last.state - previous.state)
        np.testing.assert_array_equal(candidate, last.state + direction)
        probe = LimitedLoadProbe(last.case)
        r = probe.residual(last.state)
        slope = float(r @ probe.derivative(last.state, direction) / probe.size)
        assert slope == info["actual_directional_slope"] and slope < 0
        assert info["merit_after"] < info["merit_before"]
        assert info["merit_after"] <= info["merit_before"] + 1e-4 * slope
        assert probe.feasible(candidate)
        assert audit_limited_state(probe, candidate, direct=True)["numerically_accepted"]
    assert eligible and rejected
    np.testing.assert_array_equal(eligible[-1], path.result.state)


def test_prediction_sources_are_reaudited_and_must_be_ordered(cold_path):
    _, stages = cold_path
    previous, last = stages[0], stages[1]
    with pytest.raises(ValueError, match="Ordered"):
        module._endpoint_prediction(last, previous)
    with pytest.raises(ValueError, match="Ordered"):
        module._endpoint_prediction(previous, previous)
    with pytest.raises(ValueError, match="Ordered"):
        module._endpoint_prediction(previous, replace(last, case=replace(last.case, load_n=51)))
    damaged = replace(previous, state=previous.state * 1.01)
    with pytest.raises(ValueError, match="independent stage audit"):
        module._endpoint_prediction(damaged, last)


def test_final_direct_audit_can_reject_eligible_prediction(cold_path, monkeypatch):
    import ehl_solver.limited_newton as solver

    root = cold_path[0].result
    actual = solver.audit_limited_state

    def disagree(*args, **kwargs):
        audit = actual(*args, **kwargs)
        if kwargs.get("direct", False):
            audit["numerically_accepted"] = False
        return audit

    monkeypatch.setattr(solver, "audit_limited_state", disagree)
    outcome, info = module._attempt(root.case, root.state, 1.0, 0, None, audit_only=True)
    assert isinstance(outcome, LimitedNewtonError)
    assert "direct elastic/final SI audit disagrees" in str(outcome)
    assert info["status"] == "rejected_stage" and not info["audit"]["numerically_accepted"]
    assert outcome.history == ()
    np.testing.assert_array_equal(outcome.state, root.state)


def test_failed_predicted_endpoint_rolls_back_to_last_accepted_root(cold_path, monkeypatch):
    _, stages = cold_path
    first, second = stages[:2]
    seen = []

    def solve(case, state, **kwargs):
        fraction = kwargs["transport_fraction"]
        seen.append(
            (fraction, None if state is None else state.copy(), kwargs["maximum_iterations"])
        )
        if fraction == 0:
            return first
        if fraction == 0.5:
            return second
        raise LimitedNewtonError("injected endpoint/retry failure", state + 123, [], {})

    def predict(previous, last):
        return last.state + 0.123, {"method": "injected prediction"}

    monkeypatch.setattr(module, "solve_limited_load", solve)
    monkeypatch.setattr(module, "_endpoint_prediction", predict)
    with pytest.raises(module.LimitedPathError) as exc:
        module.continue_limited_transport(first.case, maximum_attempts=4)
    assert [row[0] for row in seen] == [0, 0.5, 1, 0.75]
    assert seen[2][2] == 0  # Final verification cannot run Newton from the predictor.
    np.testing.assert_array_equal(seen[3][1], second.state)
    assert exc.value.last_result.transport_fraction == 0.5
    assert exc.value.attempts[2]["initialization"]["method"] == "injected prediction"


def test_prediction_option_is_explicit_boolean():
    with pytest.raises(ValueError, match="boolean"):
        module.continue_limited_transport(PointContactCase(), endpoint_prediction=1)


def test_prediction_can_be_disabled_without_changing_retry_seeds(cold_path, monkeypatch):
    _, stages = cold_path
    first, second = stages[:2]

    def solve(case, state, **kwargs):
        fraction = kwargs['transport_fraction']
        if fraction == 0:
            return first
        if fraction == 0.5:
            return second
        np.testing.assert_array_equal(state, second.state)
        raise LimitedNewtonError('injected non-predicted stage failure', state, [], {})

    def forbidden(*args):
        raise AssertionError('Prediction was disabled')

    monkeypatch.setattr(module, 'solve_limited_load', solve)
    monkeypatch.setattr(module, '_endpoint_prediction', forbidden)
    with pytest.raises(module.LimitedPathError) as exc:
        module.continue_limited_transport(first.case, maximum_attempts=3, endpoint_prediction=False)
    assert [a['transport_fraction'] for a in exc.value.attempts] == [0, 0.5, 0.9]
    assert all('endpoint_prediction_probe' not in a for a in exc.value.attempts)
