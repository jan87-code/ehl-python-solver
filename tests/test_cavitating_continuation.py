from dataclasses import replace

import numpy as np
import pytest
from test_cavitating_coupled import make_problem

from ehl_solver.cavitating_continuation import (
    CavitatingContinuationError,
    continue_cavitating_load,
)
from ehl_solver.cavitating_newton import CavitatingNewtonError, solve_cavitating_newton
from ehl_solver.verification import audit_cavitating_solution


@pytest.fixture
def initial_root():
    problem = make_problem(1e7, (7, 9))
    result = solve_cavitating_newton(problem, target_load_n=0.002)
    return problem, result.solution


@pytest.mark.parametrize("target", [0.0015, 0.0025])
def test_adaptive_force_path_closes_target_and_matches_direct_solution(initial_root, target):
    problem, initial = initial_root
    result = continue_cavitating_load(problem, initial, target)
    assert result.trials[-1].accepted
    assert result.trials[-1].target_load_n == target
    direct = solve_cavitating_newton(problem, initial_solution=initial, target_load_n=target)
    np.testing.assert_allclose(
        result.solution.pressure_pa, direct.solution.pressure_pa, rtol=1e-6, atol=1e-4
    )
    for trial in result.trials:
        if trial.accepted:
            audit = audit_cavitating_solution(problem, trial.solution)
            assert abs(audit.load_n - trial.target_load_n) / trial.target_load_n <= 1e-8
            assert audit.mass_maximum < 1e-8
            assert audit.maximum_complementarity < 1e-10
    assert initial.load_n == pytest.approx(0.002, rel=1e-8)
    assert problem.central_film_offset_m == 20e-6


def test_rejected_trial_retries_from_last_root_with_reduced_step(initial_root, monkeypatch):
    import ehl_solver.cavitating_continuation as module

    problem, initial = initial_root
    original = module.solve_cavitating_newton
    calls = []

    def reject_once(problem, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise CavitatingNewtonError(
                "injected trial failure", (), initial, {"true_linear_residual": 0.1}
            )
        return original(problem, **kwargs)

    monkeypatch.setattr(module, "solve_cavitating_newton", reject_once)
    result = continue_cavitating_load(problem, initial, 0.0025)
    assert not result.trials[0].accepted
    assert result.trials[0].diagnostics["true_linear_residual"] == 0.1
    assert calls[0]["initial_solution"] is calls[1]["initial_solution"] is initial
    assert calls[1]["target_load_n"] < calls[0]["target_load_n"]
    assert result.trials[-1].accepted


def test_exhaustion_retains_last_verified_root_and_full_failures(initial_root, monkeypatch):
    import ehl_solver.cavitating_continuation as module

    problem, initial = initial_root

    def always_fail(*_args, **_kwargs):
        raise CavitatingNewtonError("bounded failure", (), initial)

    monkeypatch.setattr(module, "solve_cavitating_newton", always_fail)
    with pytest.raises(CavitatingContinuationError) as caught:
        continue_cavitating_load(problem, initial, 0.0025, maximum_retries_per_stage=2)
    assert caught.value.last_solution is initial
    assert len(caught.value.trials) == 2
    assert not any(trial.accepted for trial in caught.value.trials)


def test_independent_candidate_verification_retains_corrupted_return_and_refuses_it(
    initial_root, monkeypatch
):
    import ehl_solver.cavitating_continuation as module

    problem, initial = initial_root
    fake = replace(initial, film_thickness_m=initial.film_thickness_m + 1e-6)

    def false_success(*_args, **kwargs):
        result = solve_cavitating_newton(
            problem, initial_solution=initial, target_load_n=kwargs["target_load_n"]
        )
        return replace(result, solution=replace(fake, load_n=kwargs["target_load_n"]))

    monkeypatch.setattr(module, "solve_cavitating_newton", false_success)
    with pytest.raises(CavitatingContinuationError) as caught:
        continue_cavitating_load(problem, initial, 0.0025, maximum_retries_per_stage=1)
    assert caught.value.last_solution is initial
    assert caught.value.trials[0].solution is not None
    assert not caught.value.trials[0].accepted
    assert caught.value.trials[0].newton_history


def test_invalid_start_is_rejected_before_any_continuation(initial_root):
    problem, initial = initial_root
    with pytest.raises(ValueError, match="verification"):
        continue_cavitating_load(
            problem, replace(initial, film_thickness_m=initial.film_thickness_m + 1e-6), 0.0025
        )


def test_near_identical_target_returns_verified_root_without_roundoff_sized_steps(initial_root):
    problem, initial = initial_root
    target = np.nextafter(initial.load_n, np.inf)
    result = continue_cavitating_load(problem, initial, target)
    assert result.solution is initial
    assert result.trials == ()


def test_rejected_short_final_step_is_reduced_instead_of_repeating_target(
    initial_root, monkeypatch
):
    import ehl_solver.cavitating_continuation as module

    problem, initial = initial_root
    original = module.solve_cavitating_newton
    calls = []
    target = 0.0025
    rejected = False

    def reject_first_target(problem, **kwargs):
        nonlocal rejected
        calls.append(kwargs["target_load_n"])
        if kwargs["target_load_n"] == target and not rejected:
            rejected = True
            raise CavitatingNewtonError(
                "reject shortened final step", (), kwargs["initial_solution"]
            )
        return original(problem, **kwargs)

    monkeypatch.setattr(module, "solve_cavitating_newton", reject_first_target)
    result = continue_cavitating_load(
        problem, initial, target, initial_step_fraction=0.6, maximum_step_fraction=0.6
    )
    first_target = calls.index(target)
    assert calls[first_target + 1] < target
    assert result.trials[-1].accepted
