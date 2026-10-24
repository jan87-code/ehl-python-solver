import logging

import numpy as np
import pytest
from scipy.integrate import trapezoid

import ehl_solver.monolithic as monolithic
from ehl_solver.coupled import solve_coupled_reynolds_elasticity
from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    MonolithicNewtonError,
    continue_monolithic_load,
    solve_bordered_correction,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import dimensionless_reynolds_parameter, solve_reynolds_pressure

VISCOSITY_PA_S = 0.04
SURFACE_SPEED_M_S = 0.85
LENGTH_SCALE_M = 5e-4
FILM_SCALE_M = 20e-6
PRESSURE_SCALE_PA = (
    VISCOSITY_PA_S
    * SURFACE_SPEED_M_S
    * 0.5
    * LENGTH_SCALE_M
    / FILM_SCALE_M**2
)
REYNOLDS_PARAMETER = dimensionless_reynolds_parameter(
    VISCOSITY_PA_S,
    SURFACE_SPEED_M_S,
    0.0,
    LENGTH_SCALE_M,
    FILM_SCALE_M,
    PRESSURE_SCALE_PA,
)


def _grid():
    return (
        np.linspace(-0.5e-3, 0.75e-3, 17),
        np.linspace(-0.5e-3, 0.5e-3, 13),
    )


def _pressure_at_h0(x, y, h0, e_star=1e8):
    xx, yy = np.meshgrid(x, y, indexing="xy")
    film = (
        h0 + xx**2 / (2.0 * 0.01) + yy**2 / (2.0 * 0.02)
    ) / FILM_SCALE_M
    result = solve_reynolds_pressure(
        film,
        (x[1] - x[0]) / LENGTH_SCALE_M,
        (y[1] - y[0]) / LENGTH_SCALE_M,
        REYNOLDS_PARAMETER,
        rtol=1e-12,
        atol=1e-14,
    )
    return result.pressure


def _load(pressure, x, y):
    return float(
        trapezoid(
            trapezoid(pressure * PRESSURE_SCALE_PA, x, axis=1),
            y,
        )
    )


def _problem(
    target_load,
    e_star=1e8,
    *,
    compliance_factor=1.0,
    krylov_rtol=1e-8,
    film_floor_m=1e-12,
):
    x, y = _grid()
    return BorderedReynoldsElasticityProblem(
        x,
        y,
        0.01,
        0.02,
        e_star,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        target_load,
        compliance_factor=compliance_factor,
        reynolds_residual_scale=100.0,
        film_floor_m=film_floor_m,
        krylov_rtol=krylov_rtol,
        krylov_max_iterations=300,
    )


def test_matrix_free_bordered_jvp_matches_independent_finite_difference():
    x, y = _grid()
    pressure = _pressure_at_h0(x, y, 18e-6)
    target = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    problem = _problem(target)
    state = np.concatenate(
        (pressure[1:-1, 1:-1].ravel(), [18e-6 / FILM_SCALE_M])
    )
    rng = np.random.default_rng(9021)
    direction = rng.normal(size=state.size)
    direction /= np.linalg.norm(direction)

    observed = problem.jacobian_vector(state, direction)
    step = 1e-5
    expected = (problem.residual(state + step * direction) - problem.residual(
        state - step * direction
    )) / (2.0 * step)

    np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-8)


def test_schur_correction_satisfies_complete_bordered_linearization():
    x, y = _grid()
    pressure = _pressure_at_h0(x, y, 18e-6)
    target = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    problem = _problem(target)
    state = np.concatenate(
        (pressure[1:-1, 1:-1].ravel(), [18e-6 / FILM_SCALE_M])
    )
    residual = problem.residual(state)
    correction = solve_bordered_correction(problem, state)
    correction_vector = np.concatenate(
        (
            correction.pressure_correction.ravel(),
            [correction.h0_correction_m / FILM_SCALE_M],
        )
    )
    independently_checked = (
        problem.residual(state + 1e-5 * correction_vector)
        - problem.residual(state - 1e-5 * correction_vector)
    ) / (2e-5)

    assert not correction.used_fallback
    assert correction.schur_condition_indicator > 0.0
    assert correction.pressure_krylov_residual_relative < 1e-7
    assert correction.bordered_residual_relative < 1e-7
    np.testing.assert_allclose(
        independently_checked + residual,
        0.0,
        rtol=2e-4,
        atol=2e-7,
    )


def test_singular_schur_indicator_recovers_with_full_bordered_gmres():
    x, y = _grid()
    pressure = _pressure_at_h0(x, y, 18e-6)
    target = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    problem = _problem(target)
    state = np.concatenate(
        (pressure[1:-1, 1:-1].ravel(), [18e-6 / FILM_SCALE_M])
    )

    correction = solve_bordered_correction(
        problem, state, schur_tolerance=2.0
    )

    assert correction.used_fallback
    assert correction.pressure_krylov_residual_relative < 1e-7
    assert correction.bordered_residual_relative < 1e-7


def test_failed_pressure_krylov_recovers_through_full_bordered_solve(monkeypatch):
    x, y = _grid()
    pressure = _pressure_at_h0(x, y, 18e-6)
    target = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    problem = _problem(target)
    state = np.concatenate(
        (pressure[1:-1, 1:-1].ravel(), [18e-6 / FILM_SCALE_M])
    )
    original_gmres = monolithic.gmres
    calls = 0

    def fail_pressure_block_once(operator, rhs, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.zeros_like(rhs), 1
        return original_gmres(operator, rhs, **kwargs)

    monkeypatch.setattr(monolithic, "gmres", fail_pressure_block_once)
    correction = solve_bordered_correction(problem, state)

    assert correction.used_fallback
    assert correction.pressure_krylov_residual_relative < 1e-7
    assert correction.bordered_residual_relative < 1e-7
    assert calls >= 2


def test_zero_compliance_load_newton_matches_reynolds_reference():
    x, y = _grid()
    target_pressure = _pressure_at_h0(x, y, 20e-6)
    target = _load(target_pressure, x, y)
    initial_pressure = _pressure_at_h0(x, y, 18e-6)
    solution = solve_monolithic_full_film_load(
        _problem(target, compliance_factor=0.0),
        initial_pressure,
        18e-6,
        reynolds_tolerance=1e-10,
        load_tolerance=1e-10,
    )
    expected = _pressure_at_h0(x, y, solution.central_film_offset_m)

    assert solution.central_film_offset_m == pytest.approx(20e-6, abs=2e-10)
    assert abs(solution.load_residual_n) < abs(target) * 1e-8
    assert np.all(solution.film_thickness_m > 0.0)
    np.testing.assert_allclose(solution.pressure_dimensionless, expected, atol=2e-8)
    assert len(solution.history) >= 1
    residual = monolithic.reynolds_residual(
        solution.pressure_dimensionless,
        solution.film_thickness_m / FILM_SCALE_M,
        (x[1] - x[0]) / LENGTH_SCALE_M,
        (y[1] - y[0]) / LENGTH_SCALE_M,
        REYNOLDS_PARAMETER,
    )
    assert np.linalg.norm(residual) / np.sqrt(residual.size) / 100.0 <= 1e-10


def test_full_compliance_newton_matches_weak_coupling_reference():
    x, y = _grid()
    reference = solve_coupled_reynolds_elasticity(
        x,
        y,
        FILM_SCALE_M,
        0.01,
        0.02,
        1e8,
        LENGTH_SCALE_M,
        FILM_SCALE_M,
        PRESSURE_SCALE_PA,
        REYNOLDS_PARAMETER,
        log_every=1000,
    )
    target = float(
        trapezoid(trapezoid(reference.pressure_pa, x, axis=1), y)
    )
    initial = _pressure_at_h0(x, y, 18e-6)
    solution = solve_monolithic_full_film_load(
        _problem(target),
        initial,
        18e-6,
        reynolds_tolerance=1e-7,
        load_tolerance=1e-7,
    )

    assert solution.central_film_offset_m == pytest.approx(FILM_SCALE_M, abs=2e-9)
    assert abs(solution.load_residual_n) < abs(target) * 1e-7
    assert solution.pressure_pa.min() < 0.0
    assert np.max(np.abs(solution.elastic_displacement_m)) > 1e-9
    assert all(
        record.accepted_step_length <= record.feasible_step_limit
        for record in solution.history
    )
    assert np.max(np.abs(solution.film_thickness_m - (
        solution.central_film_offset_m
        + np.meshgrid(x, y, indexing="xy")[0] ** 2 / (2.0 * 0.01)
        + np.meshgrid(x, y, indexing="xy")[1] ** 2 / (2.0 * 0.02)
        + solution.elastic_displacement_m
    ))) < 1e-12
    np.testing.assert_allclose(
        solution.pressure_dimensionless,
        reference.pressure_dimensionless,
        rtol=2e-4,
        atol=2e-7,
    )


def test_moderate_compliance_closes_load_with_measurable_deformation(caplog):
    x, y = _grid()
    target = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    problem = _problem(target, e_star=1e7)

    with caplog.at_level(logging.INFO, logger="ehl_solver.monolithic"):
        solution = solve_monolithic_full_film_load(
            problem,
            _pressure_at_h0(x, y, 18e-6),
            18e-6,
            reynolds_tolerance=1e-7,
            load_tolerance=1e-7,
            log_every=2,
        )

    assert solution.central_film_offset_m * 1e6 == pytest.approx(18.7415, abs=0.02)
    assert np.min(solution.film_thickness_m) > problem.film_floor_m
    assert np.max(np.abs(solution.elastic_displacement_m)) > 500e-9
    assert abs(solution.load_residual_n) < 1e-12
    assert len(solution.history) == 3
    assert "monolithic Newton iteration=1" in caplog.text
    assert "monolithic Newton converged" in caplog.text


def test_extreme_compliance_rigid_initial_state_is_rejected_as_infeasible():
    x, y = _grid()
    target = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    problem = _problem(target, e_star=1e5)
    pressure = _pressure_at_h0(x, y, 18e-6)
    state = np.concatenate(
        (pressure[1:-1, 1:-1].ravel(), [18e-6 / FILM_SCALE_M])
    )

    with pytest.raises(ValueError, match="film thickness"):
        problem.residual(state)


def test_dual_line_search_rejects_infeasible_steps_and_keeps_diagnostics():
    x, y = _grid()
    target_pressure = _pressure_at_h0(x, y, 10e-6)
    target = _load(target_pressure, x, y)
    problem = _problem(target, film_floor_m=17.5e-6)
    with pytest.raises(MonolithicNewtonError) as failure:
        solve_monolithic_full_film_load(
            problem,
            _pressure_at_h0(x, y, 18e-6),
            18e-6,
            max_iterations=10,
            reynolds_tolerance=1e-9,
            load_tolerance=1e-8,
        )
    assert failure.value.history
    assert "reynolds_residual_scaled" in failure.value.diagnostics
    assert "minimum_film_thickness_m" in failure.value.diagnostics
    assert any(record.feasible_step_limit < 1.0 for record in failure.value.history)
    assert any(record.rejected_line_search_trials > 0 for record in failure.value.history)
    assert all(
        record.minimum_film_thickness_m > problem.film_floor_m
        for record in failure.value.history
    )


def test_newton_reports_nonconvergence_instead_of_accepting_small_steps():
    x, y = _grid()
    target = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    with pytest.raises(MonolithicNewtonError):
        solve_monolithic_full_film_load(
            _problem(target),
            _pressure_at_h0(x, y, 18e-6),
            18e-6,
            max_iterations=1,
        )


def test_adaptive_load_continuation_retries_from_last_converged_state(monkeypatch):
    x, y = _grid()
    first_pressure = _pressure_at_h0(x, y, 20e-6)
    first_load = _load(first_pressure, x, y)
    second_load = _load(_pressure_at_h0(x, y, 22e-6), x, y)
    problem = _problem(first_load, compliance_factor=0.0)
    initial = solve_monolithic_full_film_load(
        problem,
        _pressure_at_h0(x, y, 18e-6),
        18e-6,
        reynolds_tolerance=1e-10,
        load_tolerance=1e-10,
    )
    original_solve = monolithic.solve_monolithic_full_film_load
    failed_once = False

    def fail_first_attempt(*args, **kwargs):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise MonolithicNewtonError(
                "intentional rejected continuation trial", ()
            )
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(
        monolithic, "solve_monolithic_full_film_load", fail_first_attempt
    )
    continuation = continue_monolithic_load(
        problem,
        initial,
        second_load,
        initial_step_fraction=0.5,
        minimum_step_fraction=1e-3,
        maximum_step_fraction=0.5,
        newton_options={"reynolds_tolerance": 1e-10, "load_tolerance": 1e-10},
    )

    assert continuation.steps
    assert not continuation.steps[0].accepted
    assert continuation.steps[0].central_film_offset_m == pytest.approx(
        initial.central_film_offset_m
    )
    assert all(step.accepted for step in continuation.steps[1:])
    assert continuation.solution.target_load_n == pytest.approx(second_load)
    assert continuation.solution.central_film_offset_m == pytest.approx(22e-6, abs=2e-9)
    assert abs(continuation.solution.load_residual_n) < abs(second_load) * 1e-8


def test_failed_load_continuation_preserves_reference_and_failure_diagnostics(
    monkeypatch,
):
    x, y = _grid()
    first_load = _load(_pressure_at_h0(x, y, 20e-6), x, y)
    second_load = _load(_pressure_at_h0(x, y, 22e-6), x, y)
    problem = _problem(first_load, compliance_factor=0.0)
    initial = solve_monolithic_full_film_load(
        problem,
        _pressure_at_h0(x, y, 18e-6),
        18e-6,
        reynolds_tolerance=1e-10,
        load_tolerance=1e-10,
    )

    def always_fail(*args, **kwargs):
        raise MonolithicNewtonError(
            "controlled continuation failure",
            (),
            {"reynolds_residual_scaled": 2e-3},
        )

    monkeypatch.setattr(monolithic, "solve_monolithic_full_film_load", always_fail)
    with pytest.raises(MonolithicNewtonError) as failure:
        continue_monolithic_load(
            problem,
            initial,
            second_load,
            initial_step_fraction=0.5,
            minimum_step_fraction=1e-3,
            maximum_retries_per_step=0,
        )

    assert problem.target_load_n == pytest.approx(first_load)
    assert failure.value.diagnostics["reynolds_residual_scaled"] == pytest.approx(
        2e-3
    )
