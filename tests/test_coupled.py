import numpy as np
import pytest

from ehl_solver.coupled import solve_coupled_reynolds_elasticity
from ehl_solver.reynolds import (
    dimensionless_reynolds_parameter,
    parabolic_point_contact_film,
    reynolds_residual,
    solve_reynolds_pressure,
)

VISCOSITY_PA_S = 0.04
SURFACE_SPEED_1_M_S = 0.85
SURFACE_SPEED_2_M_S = 0.0
LENGTH_SCALE_M = 5e-4
FILM_SCALE_M = 20e-6
PRESSURE_SCALE_PA = (
    VISCOSITY_PA_S
    * (SURFACE_SPEED_1_M_S + SURFACE_SPEED_2_M_S)
    / 2
    * LENGTH_SCALE_M
    / FILM_SCALE_M**2
)
REYNOLDS_PARAMETER = dimensionless_reynolds_parameter(
    VISCOSITY_PA_S,
    SURFACE_SPEED_1_M_S,
    SURFACE_SPEED_2_M_S,
    LENGTH_SCALE_M,
    FILM_SCALE_M,
    PRESSURE_SCALE_PA,
)


def _solve(**kwargs):
    x = np.linspace(-0.5e-3, 0.75e-3, 17)
    y = np.linspace(-0.5e-3, 0.5e-3, 13)
    options = {
        "central_film_offset_m": FILM_SCALE_M,
        "radius_x_m": 0.01,
        "radius_y_m": 0.02,
        "e_star_pa": 1e8,
        "length_scale_m": LENGTH_SCALE_M,
        "film_scale_m": FILM_SCALE_M,
        "pressure_scale_pa": PRESSURE_SCALE_PA,
        "reynolds_parameter": REYNOLDS_PARAMETER,
        "pressure_relaxation": 0.6,
        "film_relaxation": 0.6,
        "pressure_tolerance": 1e-6,
        "film_tolerance": 1e-6,
        "reynolds_tolerance": 1e-8,
        "max_iterations_per_stage": 100,
        "inner_rtol": 1e-10,
        "inner_atol": 1e-12,
        "log_every": 1000,
    }
    options.update(kwargs)
    return x, y, solve_coupled_reynolds_elasticity(x, y, **options)


def test_zero_compliance_limit_matches_prescribed_film_solver():
    x, y, solution = _solve(
        continuation_factors=(0.0,),
        pressure_relaxation=1.0,
        film_relaxation=1.0,
        pressure_tolerance=1e-9,
        film_tolerance=1e-9,
        reynolds_tolerance=1e-10,
        inner_rtol=1e-12,
        inner_atol=1e-14,
    )
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    rigid_film = parabolic_point_contact_film(x_grid, y_grid, FILM_SCALE_M, 0.01, 0.02)
    expected = solve_reynolds_pressure(
        rigid_film / FILM_SCALE_M,
        (x[1] - x[0]) / LENGTH_SCALE_M,
        (y[1] - y[0]) / LENGTH_SCALE_M,
        REYNOLDS_PARAMETER,
        rtol=1e-12,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        solution.pressure_dimensionless, expected.pressure, rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(solution.film_thickness_m, rigid_film, rtol=1e-14)


def test_coupled_film_relation_residuals_and_continuation_converge():
    x, y, solution = _solve()
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    rigid_film = parabolic_point_contact_film(x_grid, y_grid, FILM_SCALE_M, 0.01, 0.02)
    expected_film = rigid_film + solution.elastic_displacement_m

    np.testing.assert_allclose(
        solution.film_thickness_m, expected_film, rtol=1e-9, atol=1e-14
    )
    assert np.all(np.isfinite(solution.film_thickness_m))
    assert np.all(solution.film_thickness_m > 0.0)
    assert solution.pressure_pa.min() < 0.0
    assert solution.pressure_pa.max() > 0.0

    pressure = solution.pressure_dimensionless
    film = solution.film_thickness_m / FILM_SCALE_M
    dx = (x[1] - x[0]) / LENGTH_SCALE_M
    dy = (y[1] - y[0]) / LENGTH_SCALE_M
    residual = reynolds_residual(pressure, film, dx, dy, REYNOLDS_PARAMETER)
    forcing = REYNOLDS_PARAMETER * (film[1:-1, 2:] - film[1:-1, :-2]) / (2 * dx)
    diffusion = residual + forcing
    normalized_residual = np.sqrt(np.mean(residual**2)) / (
        np.sqrt(np.mean(forcing**2)) + np.sqrt(np.mean(diffusion**2))
    )
    assert solution.history[-1].reynolds_residual_relative == pytest.approx(
        normalized_residual, rel=1e-12
    )

    for factor in (0.0, 0.25, 0.5, 0.75, 1.0):
        stage = [entry for entry in solution.history if entry.compliance_factor == factor]
        assert stage
        assert stage[-1].pressure_update_relative <= 1e-6
        assert stage[-1].film_update_relative <= 1e-6
        assert stage[-1].reynolds_residual_relative <= 1e-8
    assert np.max(np.abs(solution.elastic_displacement_m)) > 0.0


def test_relaxation_settings_change_iteration_count_not_converged_solution():
    _, _, fast = _solve(pressure_relaxation=1.0, film_relaxation=1.0)
    _, _, relaxed = _solve(pressure_relaxation=0.6, film_relaxation=0.6)

    assert len(fast.history) < len(relaxed.history)
    np.testing.assert_allclose(
        fast.pressure_dimensionless,
        relaxed.pressure_dimensionless,
        rtol=2e-5,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        fast.film_thickness_m, relaxed.film_thickness_m, rtol=1e-9, atol=1e-13
    )


def test_nonpositive_elastic_film_iterate_is_rejected_not_clipped():
    with pytest.raises(RuntimeError, match="nonpositive film thickness"):
        _solve(e_star_pa=1e4, max_iterations_per_stage=20)


def test_outer_iteration_limit_reports_failure_metrics():
    with pytest.raises(RuntimeError, match="failed at compliance_factor"):
        _solve(max_iterations_per_stage=1)


def test_snapshots_are_optional_and_capture_accepted_pressure_and_film():
    _, _, without_snapshots = _solve()
    assert without_snapshots.snapshots == ()

    _, _, with_snapshots = _solve(plot_every=10, snapshot_iterations=(1, 5))
    selected = {snapshot.global_iteration for snapshot in with_snapshots.snapshots}
    assert selected == {1, 5, 10, 20, 30, 40, 50, 60, 70, 80}
    for snapshot in with_snapshots.snapshots:
        accepted = with_snapshots.history[snapshot.global_iteration - 1]
        assert snapshot.global_iteration == accepted.global_iteration
        assert snapshot.compliance_factor == accepted.compliance_factor
        assert snapshot.pressure_pa.dtype == np.float64
        assert snapshot.film_thickness_m.dtype == np.float64
        assert np.all(snapshot.film_thickness_m > 0.0)
    with_snapshots.snapshots[0].pressure_pa[:] = 0.0
    assert np.any(with_snapshots.pressure_pa != 0.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"pressure_relaxation": 0.0},
        {"film_relaxation": 1.1},
        {"pressure_tolerance": 0.0},
        {"continuation_factors": (0.5, 1.0)},
        {"max_iterations_per_stage": 0},
    ],
)
def test_invalid_nonlinear_solver_options_are_rejected(changes):
    x = np.linspace(-0.5e-3, 0.5e-3, 5)
    y = np.linspace(-0.5e-3, 0.5e-3, 5)
    options = {
        "central_film_offset_m": FILM_SCALE_M,
        "radius_x_m": 0.01,
        "radius_y_m": 0.02,
        "e_star_pa": 1e8,
        "length_scale_m": LENGTH_SCALE_M,
        "film_scale_m": FILM_SCALE_M,
        "pressure_scale_pa": PRESSURE_SCALE_PA,
        "reynolds_parameter": REYNOLDS_PARAMETER,
    }
    options.update(changes)

    with pytest.raises(ValueError):
        solve_coupled_reynolds_elasticity(x, y, **options)
