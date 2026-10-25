from dataclasses import replace

import numpy as np
import pytest
from test_cavitating_coupled import make_problem

from ehl_solver.cavitating_coupled import solve_cavitating_elasticity
from ehl_solver.cavitating_newton import (
    CavitatingNewtonError,
    CavitatingNewtonSystem,
    solve_cavitating_newton,
)
from ehl_solver.verification import audit_cavitating_solution


def mixed_state(system):
    p = np.full(system.n, 0.1)
    d = np.zeros(system.n)
    p[::2], d[::2] = 0, 0.2
    z = np.r_[p, d]
    return np.r_[z, 1.0] if system.target_load_n is not None else z


@pytest.mark.parametrize("gamma", [0.0, 1.0])
@pytest.mark.parametrize("load", [None, 0.002])
def test_complete_analytic_cavitation_jacobian_matches_residual_columns(gamma, load):
    system = CavitatingNewtonSystem(
        make_problem(1e6, (5, 7)), compliance_factor=gamma, target_load_n=load
    )
    z = mixed_state(system)
    identity = np.eye(system.size)
    matrix = np.column_stack([system.jacobian_vector(z, v) for v in identity])
    difference = np.column_stack(
        [(system.residual(z + 1e-5 * v) - system.residual(z - 1e-5 * v)) / (2e-5) for v in identity]
    )
    assert np.linalg.norm(matrix - difference) / np.linalg.norm(matrix) < 2e-8
    direction = np.random.default_rng(320).normal(size=system.size)
    direction /= np.linalg.norm(direction)
    action = system.jacobian_vector(z, direction)
    for step in (1e-3, 1e-4, 1e-5):
        finite = (system.residual(z + step * direction) - system.residual(z - step * direction)) / (
            2 * step
        )
        assert np.linalg.norm(action - finite) / np.linalg.norm(action) < 2e-6
    if gamma == 0:
        np.testing.assert_allclose(
            system.frozen_matrix(z).toarray(), matrix, rtol=1e-12, atol=1e-12
        )


def test_merit_direction_at_complementarity_ties_is_a_true_one_sided_derivative():
    system = CavitatingNewtonSystem(make_problem(1e7, (5, 7)))
    state = np.full(system.size, 0.1)
    residual = system.residual(state)
    direction = np.random.default_rng(583).normal(size=system.size)
    expected = system.merit_slope(state, residual, direction)
    step = 1e-7
    measured = (
        0.5 * np.mean(system.residual(state + step * direction) ** 2) - 0.5 * np.mean(residual**2)
    ) / step
    assert measured == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("modulus", [1e8, 1e7, 1e6])
def test_monolithic_cavitation_agrees_with_independent_fixed_point_reference(modulus):
    problem = make_problem(modulus)
    result = solve_cavitating_newton(problem)
    reference = solve_cavitating_elasticity(problem)
    np.testing.assert_allclose(
        result.solution.pressure_pa, reference.pressure_pa, rtol=2e-6, atol=1e-5
    )
    np.testing.assert_allclose(
        result.solution.film_thickness_m, reference.film_thickness_m, rtol=1e-9, atol=1e-15
    )
    np.testing.assert_allclose(
        result.solution.fractional_film_content, reference.fractional_film_content, atol=1e-8
    )
    audit = audit_cavitating_solution(problem, result.solution)
    assert audit.mass_maximum < 1e-8
    assert audit.maximum_complementarity < 1e-10
    assert audit.minimum_pressure_pa > -1e-10 * problem.pressure_scale_pa
    assert audit.minimum_saturation >= -1e-10
    assert audit.maximum_saturation <= 1 + 1e-10
    assert audit.load_n > 0
    for row in result.history:
        assert row.true_linear_residual <= 5e-9
        assert sum(len(trace) for trace in row.krylov_histories) == row.linear_iterations
        assert max(row.linear_true_residuals) <= 5e-9
        assert row.pressure_minimum_pa_after >= -1e-10 * problem.pressure_scale_pa
        assert row.saturation_minimum_after >= -1e-10
        assert row.saturation_maximum_after <= 1 + 1e-10
        assert row.merit_after <= row.merit_before + 1e-4 * row.step_length * row.merit_slope
        assert row.minimum_film_m > problem.film_floor_m
    if modulus == 1e6:
        assert any(row.step_length < 0.99 for row in result.history)


def test_iteration_exhaustion_keeps_last_admissible_candidate_and_history():
    problem = make_problem(1e6)
    with pytest.raises(CavitatingNewtonError, match="iteration limit") as caught:
        solve_cavitating_newton(problem, maximum_iterations=1)
    assert len(caught.value.history) == 1
    assert caught.value.last_solution is not None
    assert caught.value.diagnostics["mass_maximum"] > 1e-8


def test_true_linear_residual_rejects_false_gmres_success(monkeypatch):
    import ehl_solver.cavitating_newton as module

    def false_success(operator, rhs, **_kwargs):
        return np.zeros_like(rhs), 0

    monkeypatch.setattr(module, "gmres", false_success)
    with pytest.raises(CavitatingNewtonError, match="true Krylov") as caught:
        solve_cavitating_newton(make_problem())
    assert caught.value.diagnostics["true_linear_residual"] == pytest.approx(1.0)
    assert caught.value.last_solution is not None


def test_inadmissible_initial_pressure_is_not_clipped():
    problem = make_problem()
    initial = solve_cavitating_elasticity(problem, continuation_factors=(0.0,))
    bad = initial.pressure_pa.copy()
    bad[2, 2] = -1.0
    with pytest.raises(CavitatingNewtonError, match="inadmissible") as caught:
        solve_cavitating_newton(problem, initial_solution=replace(initial, pressure_pa=bad))
    assert caught.value.last_solution.pressure_pa[2, 2] == -1.0


@pytest.mark.parametrize("force_fallback", [False, True])
def test_schur_and_full_bordered_corrections_match_complete_dense_linearization(force_fallback):
    from ehl_solver.cavitating_newton import solve_cavitating_correction

    system = CavitatingNewtonSystem(make_problem(1e7, (5, 7)), target_load_n=0.002)
    z = mixed_state(system)
    correction = solve_cavitating_correction(
        system, z, schur_tolerance=2 if force_fallback else 1e-10
    )
    jacobian = np.column_stack([system.jacobian_vector(z, v) for v in np.eye(system.size)])
    residual = system.residual(z)
    exact = np.linalg.solve(jacobian, -residual)
    assert (
        np.linalg.norm(jacobian @ correction.direction + residual) / np.linalg.norm(residual) < 5e-9
    )
    np.testing.assert_allclose(correction.direction, exact, rtol=2e-7, atol=1e-8)
    assert correction.used_bordered_fallback == force_fallback


def test_failed_pressure_krylov_recovers_using_full_bordered_system(monkeypatch):
    import ehl_solver.cavitating_newton as module

    original = module.gmres
    calls = []

    def fail_first(operator, rhs, **kwargs):
        calls.append(operator.shape)
        return (np.zeros_like(rhs), 1) if len(calls) == 1 else original(operator, rhs, **kwargs)

    monkeypatch.setattr(module, "gmres", fail_first)
    system = CavitatingNewtonSystem(make_problem(1e7, (5, 7)), target_load_n=0.002)
    correction = module.solve_cavitating_correction(system, mixed_state(system))
    assert correction.used_bordered_fallback
    assert calls == [(2 * system.n, 2 * system.n), (system.size, system.size)]
    assert correction.true_linear_residual < 5e-9
    assert correction.linear_true_residuals[0] == 1.0


@pytest.mark.parametrize("modulus", [1e8, 1e7, 1e6])
def test_positive_load_and_offset_are_solved_simultaneously_and_recover_known_root(modulus):
    problem = make_problem(modulus)
    reference = solve_cavitating_elasticity(problem)
    displaced = replace(problem, central_film_offset_m=22e-6)
    result = solve_cavitating_newton(displaced, target_load_n=reference.load_n)
    audit = audit_cavitating_solution(displaced, result.solution)
    assert abs(audit.load_n - reference.load_n) / reference.load_n < 1e-8
    assert audit.mass_maximum < 1e-8
    assert audit.maximum_complementarity < 1e-10
    assert result.solution.central_film_offset_m == pytest.approx(20e-6, rel=2e-8)
    np.testing.assert_allclose(
        result.solution.pressure_pa, reference.pressure_pa, rtol=1e-6, atol=1e-4
    )
    assert abs(result.load_residual_n) / reference.load_n < 1e-8
    assert any(row.schur_indicator is not None for row in result.history)


def test_load_controller_rejects_nonpositive_target_instead_of_solving_signed_case():
    with pytest.raises(ValueError, match="target_load_n"):
        solve_cavitating_newton(make_problem(), target_load_n=-0.002)
