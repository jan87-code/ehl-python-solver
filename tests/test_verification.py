from dataclasses import replace

import numpy as np
import pytest

from ehl_solver.monolithic import (
    BorderedReynoldsElasticityProblem,
    solve_monolithic_full_film_load,
)
from ehl_solver.reynolds import solve_reynolds_pressure
from ehl_solver.verification import audit_full_film_solution


@pytest.fixture
def small_full_film_case():
    x = np.linspace(-0.5e-3, 0.75e-3, 9)
    y = np.linspace(-0.5e-3, 0.5e-3, 7)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    film_scale = 20e-6
    film = (film_scale + xx**2 / 0.02 + yy**2 / 0.04) / film_scale
    pressure = solve_reynolds_pressure(
        film, (x[1] - x[0]) / 5e-4, (y[1] - y[0]) / 5e-4, 12.0,
        rtol=1e-12, atol=1e-14,
    ).pressure
    # Zero pressure boundaries make trapezoidal and rectangle loads identical.
    load = float(np.sum(pressure) * 21250.0 * (x[1] - x[0]) * (y[1] - y[0]))
    problem = BorderedReynoldsElasticityProblem(
        x, y, 0.01, 0.02, 1e7, 5e-4, film_scale, 21250.0, 12.0, load,
        reynolds_residual_scale=100.0,
    )
    solution = solve_monolithic_full_film_load(problem, pressure, film_scale)
    return problem, solution


def test_direct_audit_matches_nonlinear_residual_without_calling_it(
    small_full_film_case, monkeypatch
):
    problem, solution = small_full_film_case
    state = np.r_[solution.pressure_dimensionless[1:-1, 1:-1].ravel(),
                  solution.central_film_offset_m / problem.film_scale_m]
    residual = problem.residual(state)
    evaluations = problem.operator_evaluations

    def forbidden(*_args, **_kwargs):
        raise AssertionError("audit must not use the FFT state reconstruction or residual")

    monkeypatch.setattr(problem, "residual", forbidden)
    monkeypatch.setattr(problem, "state_fields", forbidden)
    audit = audit_full_film_solution(problem, solution)
    assert problem.operator_evaluations == evaluations
    assert audit.reynolds_rms_scaled == pytest.approx(
        np.sqrt(np.mean(residual[:-1]**2)), abs=1e-13
    )
    assert audit.load_residual_scaled == pytest.approx(residual[-1], abs=1e-12)
    assert audit.reynolds_rms_scaled < 1e-8
    assert audit.minimum_film_m > problem.film_floor_m
    assert audit.boundary_pressure_max_pa == 0.0
    assert audit.displacement_max_error_m < 1e-19
    assert audit.film_coupling_max_error_m < 1e-19
    assert audit.net_boundary_flux == pytest.approx(audit.summed_local_flux_balance, abs=1e-12)


def test_audit_detects_inconsistent_stored_film_displacement_and_scaling(small_full_film_case):
    problem, solution = small_full_film_case
    corrupted = replace(
        solution, film_thickness_m=solution.film_thickness_m + 1e-6,
        elastic_displacement_m=solution.elastic_displacement_m + 2e-6,
        pressure_dimensionless=solution.pressure_dimensionless + 0.1,
    )
    audit = audit_full_film_solution(problem, corrupted)
    assert audit.film_coupling_max_error_m == pytest.approx(1e-6)
    assert audit.displacement_max_error_m == pytest.approx(2e-6)
    assert audit.pressure_scaling_max_error_pa == pytest.approx(2125.0)
    assert audit.reynolds_rms_scaled > 1e-4


def test_audit_local_global_flux_identity_on_unconverged_state(small_full_film_case):
    problem, solution = small_full_film_case
    pressure = solution.pressure_pa.copy()
    pressure[2, 3] += 100.0
    audit = audit_full_film_solution(problem, replace(solution, pressure_pa=pressure))
    assert audit.maximum_local_flux_balance > 1e-3
    assert audit.film_coupling_max_error_m > 1e-12
    assert audit.load_residual_n != 0.0
    assert audit.net_boundary_flux == pytest.approx(audit.summed_local_flux_balance, abs=1e-12)


@pytest.mark.parametrize("change", ["nonfinite", "negative_film", "grid_limit"])
def test_audit_rejects_invalid_or_unbounded_inputs(small_full_film_case, change):
    problem, solution = small_full_film_case
    kwargs = {}
    if change == "nonfinite":
        solution = replace(solution, pressure_pa=np.full_like(solution.pressure_pa, np.nan))
    elif change == "negative_film":
        solution = replace(solution, film_thickness_m=-solution.film_thickness_m)
    else:
        kwargs["maximum_grid_points"] = 20
    with pytest.raises(ValueError):
        audit_full_film_solution(problem, solution, **kwargs)
