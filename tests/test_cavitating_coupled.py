from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.cavitating_coupled import (
    CavitatingCouplingError,
    CavitatingElasticityProblem,
    solve_cavitating_elasticity,
)
from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.reynolds import reynolds_residual
from ehl_solver.verification import audit_cavitating_solution


def make_problem(modulus=1e7, shape=(9, 13), **kwargs):
    return CavitatingElasticityProblem(
        np.linspace(-0.5e-3, 0.75e-3, shape[1]),
        np.linspace(-0.5e-3, 0.5e-3, shape[0]),
        20e-6,
        0.01,
        0.02,
        modulus,
        5e-4,
        20e-6,
        21250,
        12,
        **kwargs,
    )


def test_zero_compliance_matches_fixed_film_jfo_reference():
    problem = make_problem()
    film, _ = problem.fields(np.zeros(problem.shape), compliance_factor=0)
    reference = solve_cavitating_reynolds(film / problem.film_scale_m, problem.dx, problem.dy, 12)
    actual = solve_cavitating_elasticity(problem, continuation_factors=(0.0,))
    np.testing.assert_array_equal(
        actual.pressure_pa, reference.pressure * problem.pressure_scale_pa
    )
    np.testing.assert_array_equal(actual.fractional_film_content, reference.fractional_film_content)
    np.testing.assert_array_equal(actual.film_thickness_m, film)
    assert actual.load_n > 0


def test_full_film_limit_matches_conservative_reynolds_stencil():
    problem = make_problem()
    rng = np.random.default_rng(740)
    p = rng.normal(size=problem.shape)
    film, _ = problem.fields(np.zeros(problem.shape), compliance_factor=0)
    observed = problem.mass_residual(p * problem.pressure_scale_pa, np.ones(problem.shape), film)
    expected = -reynolds_residual(p, film / problem.film_scale_m, problem.dx, problem.dy, 12)
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("modulus", [1e8, 1e7, 1e6])
def test_coupled_positive_load_mass_complementarity_and_direct_elasticity(modulus, monkeypatch):
    problem = make_problem(modulus)
    result = solve_cavitating_elasticity(problem)
    measured = problem.metrics(
        result.pressure_pa, result.fractional_film_content, result.film_thickness_m
    )
    assert result.compliance_factor == 1
    assert measured.mass_maximum <= 1e-8
    assert measured.maximum_complementarity < 1e-10
    assert result.pressure_pa.min() >= 0
    assert result.fractional_film_content.min() < 0.9
    assert result.fractional_film_content.max() <= 1
    assert result.film_thickness_m.min() > 20e-6
    assert result.load_n > 0

    def forbidden(*_args, **_kwargs):
        raise AssertionError("independent audit cannot use coupled reconstruction/metrics")

    monkeypatch.setattr(problem, "fields", forbidden)
    monkeypatch.setattr(problem, "metrics", forbidden)
    audit = audit_cavitating_solution(problem, result)
    assert audit.mass_rms == pytest.approx(measured.mass_rms, abs=1e-12)
    assert audit.mass_maximum < 1.001e-8
    assert abs(audit.net_boundary_flux) < 1e-8
    assert audit.net_boundary_flux == pytest.approx(audit.summed_volume_balance, abs=1e-12)
    assert audit.film_coupling_max_error_m < 1e-19
    assert audit.displacement_max_error_m < 1e-19
    assert audit.pressure_boundary_max_pa == 0
    assert audit.inlet_saturation_error == 0
    assert audit.load_metadata_error_n == 0


def test_relaxation_changes_work_not_the_converged_state():
    problem = make_problem()
    a = solve_cavitating_elasticity(problem, film_relaxation=0.5)
    b = solve_cavitating_elasticity(problem, film_relaxation=0.8)
    assert len(a.history) != len(b.history)
    np.testing.assert_allclose(a.pressure_pa, b.pressure_pa, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.film_thickness_m, b.film_thickness_m, rtol=1e-10)
    np.testing.assert_allclose(a.fractional_film_content, b.fractional_film_content, atol=1e-10)


def test_tiny_film_steps_do_not_hide_mass_residual_and_error_preserves_state():
    problem = make_problem(1e6)
    with pytest.raises(CavitatingCouplingError) as caught:
        solve_cavitating_elasticity(
            problem,
            continuation_factors=(0.0, 1.0),
            film_relaxation=1e-12,
            update_tolerance=0.01,
            maximum_iterations_per_stage=2,
        )
    failure = caught.value
    assert failure.history[-1].metrics.mass_rms > 1e-4
    assert failure.last_solution.compliance_factor == 1
    assert len(failure.last_solution.history) == len(failure.history)


def test_inner_active_set_failure_is_not_returned_as_a_solution():
    with pytest.raises(CavitatingCouplingError) as caught:
        solve_cavitating_elasticity(make_problem(), maximum_active_set_iterations=1)
    assert caught.value.last_solution is None
    assert caught.value.__cause__ is not None


def test_audit_detects_corrupted_film_saturation_and_load_metadata():
    problem = make_problem()
    result = solve_cavitating_elasticity(problem)
    corrupted = replace(
        result,
        film_thickness_m=result.film_thickness_m + 1e-6,
        fractional_film_content=result.fractional_film_content * 0.9,
        load_n=result.load_n + 1.0,
    )
    audit = audit_cavitating_solution(problem, corrupted)
    assert audit.film_coupling_max_error_m == pytest.approx(1e-6)
    assert audit.maximum_complementarity > 0.001
    assert audit.inlet_saturation_error == pytest.approx(0.1)
    assert audit.load_metadata_error_n == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "options",
    [
        {"film_relaxation": 0},
        {"maximum_iterations_per_stage": 0},
        {"mass_tolerance": float("nan")},
        {"continuation_factors": (0.5, 1.0)},
    ],
)
def test_invalid_iteration_options_are_rejected(options):
    with pytest.raises(ValueError):
        solve_cavitating_elasticity(make_problem(), **options)


def test_reduced_supply_preserves_mass_and_reduces_coupled_positive_force():
    flooded = make_problem()
    starved = make_problem(inlet_fractional_film_content=0.99)
    a, b = solve_cavitating_elasticity(flooded), solve_cavitating_elasticity(starved)
    assert 0 < b.load_n < a.load_n
    assert audit_cavitating_solution(starved, b).mass_maximum < 1.001e-8
    np.testing.assert_allclose(b.fractional_film_content[1:-1, 0], 0.99)


def test_nonpositive_coupled_film_is_rejected_without_clamping():
    problem = make_problem(1e6)
    with pytest.raises(ValueError, match="film"):
        problem.fields(-np.full(problem.shape, 1e9))


@pytest.mark.parametrize(
    "changes",
    [
        {"e_star_pa": 0.0},
        {"film_scale_m": -1.0},
        {"x_m": [0, 1e-4, 3e-4]},
        {"inlet_fractional_film_content": 1.1},
    ],
)
def test_geometry_and_physical_inputs_are_validated(changes):
    problem = make_problem()
    arguments = {
        name: getattr(problem, name)
        for name in (
            "x_m",
            "y_m",
            "central_film_offset_m",
            "radius_x_m",
            "radius_y_m",
            "e_star_pa",
            "length_scale_m",
            "film_scale_m",
            "pressure_scale_pa",
            "reynolds_parameter",
            "inlet_fractional_film_content",
        )
    }
    with pytest.raises(ValueError):
        CavitatingElasticityProblem(**(arguments | changes))
