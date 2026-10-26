import numpy as np
import pytest
from scipy.integrate import trapezoid

from ehl_solver.backend import make_backend
from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.elasticity import elastic_deformation
from ehl_solver.exterior import exterior_source_budget, solve_exterior_response


def point_case(n=33):
    h0, radius, transverse = 1e-7, 0.01, 0.02
    length, ly = np.sqrt(2 * radius * h0), np.sqrt(2 * transverse * h0)
    x, y = np.linspace(-8, 8, n) * length, np.linspace(-8, 8, n) * ly
    h = h0 + x[None, :] ** 2 / (2 * radius) + y[:, None] ** 2 / (2 * transverse)
    pscale = 12 * 0.04 * 0.001 * length / h0**2
    return x, y, h, dict(length_scale_m=length, film_scale_m=h0, pressure_scale_pa=pscale)


@pytest.mark.parametrize("supply", [1.0, 0.9])
def test_exterior_response_matches_independent_uniform_jfo_including_supply(supply):
    x, y, h, kw = point_case()
    original = solve_cavitating_reynolds(
        h / kw["film_scale_m"],
        (x[1] - x[0]) / kw["length_scale_m"],
        (y[1] - y[0]) / kw["length_scale_m"],
        1,
        inlet_fractional_film_content=supply,
    )
    actual = solve_exterior_response(
        x,
        y,
        h,
        0.04,
        0.001,
        core_bounds_m=(x[12], x[20], y[12], y[20]),
        inlet_content=supply,
        **kw,
    )
    np.testing.assert_allclose(
        actual.pressure_pa / kw["pressure_scale_pa"], original.pressure, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.fractional_film_content, original.fractional_film_content, atol=1e-12
    )
    assert actual.audit["mass_maximum"] < 1e-8
    assert actual.audit["relative_local_balance"] < 1e-10
    assert actual.audit["relative_global_balance"] < 1e-10
    assert actual.audit["maximum_complementarity"] < 1e-10
    assert actual.history[-1]["exterior_dry_nodes"] > 0
    assert actual.history[-1]["relative_row_backward_error"] < 1e-10
    assert np.all(actual.fractional_film_content[1:-1, 0] == supply)


def test_moving_numerical_cut_preserves_full_load_and_exterior_elastic_sources():
    x, y, h, kw = point_case()
    results = [
        solve_exterior_response(
            x,
            y,
            h,
            0.04,
            0.001,
            core_bounds_m=(x[k], x[-k - 1], y[k], y[-k - 1]),
            **kw,
        )
        for k in (12, 10)
    ]
    np.testing.assert_allclose(results[0].pressure_pa, results[1].pressure_pa, atol=1e-8)
    targets = [[0, 0], [x[14], 0]]
    expected = elastic_deformation(
        results[0].pressure_pa, x[1] - x[0], y[1] - y[0], 1e9, make_backend("numpy")
    )
    budgets = [exterior_source_budget(x, y, r, 1e9, targets) for r in results]
    q = trapezoid(trapezoid(results[0].pressure_pa, x, axis=1), y)
    for b in budgets:
        assert b["total_load_n"] == pytest.approx(q, rel=1e-12)
        assert b["exterior_source_load_n"] > 0
        assert min(b["exterior_displacement_m"]) > 0
        np.testing.assert_allclose(
            b["total_displacement_m"], [expected[16, 16], expected[16, 14]], rtol=1e-12
        )
        assert b["maximum_displacement_partition_error_m"] < 1e-22
    assert budgets[0]["exterior_source_load_n"] > budgets[1]["exterior_source_load_n"]


def test_exterior_rejects_invalid_core_and_exhaustion_without_fallback():
    x, y, h, kw = point_case()
    with pytest.raises(ValueError, match="strict interior"):
        solve_exterior_response(
            x, y, h, 0.04, 0.001, core_bounds_m=(x[0], x[-1], y[0], y[-1]), **kw
        )
    with pytest.raises(RuntimeError, match="iteration limit"):
        solve_exterior_response(
            x,
            y,
            h,
            0.04,
            0.001,
            core_bounds_m=(x[12], x[20], y[12], y[20]),
            maximum_iterations=1,
            **kw,
        )
