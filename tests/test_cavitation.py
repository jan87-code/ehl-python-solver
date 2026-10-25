import numpy as np
import pytest
from scipy.integrate import trapezoid

from ehl_solver.cavitation import solve_cavitating_reynolds
from ehl_solver.reynolds import solve_reynolds_pressure


def _converging_diverging_film(nx=41, ny=17):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(-0.5, 0.5, ny)
    xx, _ = np.meshgrid(x, y, indexing="xy")
    return 1.4 - 0.4 * xx + 0.8 * xx**2


def test_full_film_active_set_reproduces_existing_reynolds_stencil():
    x = np.linspace(0.0, 1.0, 33)
    y = np.linspace(-0.5, 0.5, 15)
    xx, _ = np.meshgrid(x, y, indexing="xy")
    film = 1.4 - 0.4 * xx
    expected = solve_reynolds_pressure(film, x[1] - x[0], y[1] - y[0], 2.0)
    actual = solve_cavitating_reynolds(film, x[1] - x[0], y[1] - y[0], 2.0)

    np.testing.assert_allclose(actual.pressure, expected.pressure, atol=1e-11)
    np.testing.assert_allclose(actual.fractional_film_content, 1.0)
    assert actual.maximum_complementarity_violation == 0.0
    assert actual.maximum_mass_balance_residual < 1e-8


def test_jfo_solution_is_admissible_and_locally_mass_conservative():
    x = np.linspace(0.0, 1.0, 41)
    y = np.linspace(-0.5, 0.5, 17)
    film = _converging_diverging_film(x.size, y.size)
    full_film = solve_reynolds_pressure(
        film, x[1] - x[0], y[1] - y[0], 2.0
    )
    solution = solve_cavitating_reynolds(
        film, x[1] - x[0], y[1] - y[0], 2.0
    )

    assert full_film.pressure.min() < 0.0
    assert not np.allclose(solution.pressure, np.maximum(full_film.pressure, 0.0))
    assert np.min(solution.pressure) >= -1e-10
    assert np.min(solution.fractional_film_content) >= -1e-10
    assert np.max(solution.fractional_film_content) <= 1.0 + 1e-10
    assert np.max(solution.pressure) > 0.0
    assert np.any(solution.fractional_film_content[1:-1, 1:-1] < 1.0 - 1e-8)
    assert solution.maximum_complementarity_violation < 1e-8
    assert solution.maximum_mass_balance_residual < 1e-8
    assert abs(solution.net_boundary_flux) < 1e-8


def test_converging_diverging_benchmark_load_refines_and_boundaries_conserve():
    integrated_loads = []
    for nx, ny in ((21, 9), (41, 17), (81, 33)):
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(-0.5, 0.5, ny)
        solution = solve_cavitating_reynolds(
            _converging_diverging_film(nx, ny),
            x[1] - x[0],
            y[1] - y[0],
            2.0,
        )
        integrated_loads.append(
            trapezoid(trapezoid(solution.pressure, y, axis=0), x)
        )
        assert abs(solution.net_boundary_flux) < 1e-8
        assert solution.maximum_mass_balance_residual < 1e-8

    differences = np.abs(np.diff(integrated_loads))
    assert differences[1] < differences[0]
    assert integrated_loads[0] < integrated_loads[1] < integrated_loads[2]


def test_rupture_and_reformation_front_pressure_jumps_shrink_with_refinement():
    normalized_front_jumps = []
    reformation_locations = []
    for nx, ny in ((41, 21), (61, 31), (101, 51)):
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(-0.5, 0.5, ny)
        xx, _ = np.meshgrid(x, y, indexing="xy")
        film = 1.5 + 0.05 * np.cos(4.0 * np.pi * xx)
        solution = solve_cavitating_reynolds(
            film, x[1] - x[0], y[1] - y[0], 2.0
        )

        center = ny // 2
        pressure = solution.pressure[center]
        saturation = solution.fractional_film_content[center]
        cavitated = saturation < 1.0 - 1e-8
        fronts = np.flatnonzero(cavitated[1:] != cavitated[:-1])

        assert fronts.size == 3
        assert np.array_equal(cavitated[fronts], [False, True, False])
        for front in fronts:
            if cavitated[front]:
                assert pressure[front] == 0.0
                assert pressure[front + 1] >= 0.0
            else:
                assert pressure[front + 1] == 0.0
                assert pressure[front] >= 0.0
        assert np.all(
            solution.pressure[1:-1, 1:-1][
                solution.fractional_film_content[1:-1, 1:-1] < 1.0 - 1e-8
            ]
            == 0.0
        )
        front_side_pressure = np.array(
            [
                pressure[fronts[0]],
                pressure[fronts[1] + 1],
                pressure[fronts[2]],
            ]
        )
        normalized_front_jumps.append(
            float(np.max(front_side_pressure) / np.max(pressure))
        )
        reformation_locations.append(x[fronts[1] + 1])
        assert solution.maximum_mass_balance_residual < 1e-8
        assert solution.maximum_complementarity_violation < 1e-8

    assert normalized_front_jumps[-1] < 0.25 * normalized_front_jumps[0]
    assert abs(reformation_locations[-1] - reformation_locations[0]) < 0.03


def test_reduced_inlet_supply_reduces_pressure_and_preserves_mass_balance():
    x = np.linspace(0.0, 1.0, 81)
    y = np.linspace(-0.5, 0.5, 33)
    film = _converging_diverging_film(x.size, y.size)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    flooded = solve_cavitating_reynolds(film, dx, dy, 2.0)
    partially_starved = solve_cavitating_reynolds(
        film,
        dx,
        dy,
        2.0,
        inlet_fractional_film_content=0.99,
    )
    severely_starved = solve_cavitating_reynolds(
        film,
        dx,
        dy,
        2.0,
        inlet_fractional_film_content=0.9,
    )

    def pressure_integral(pressure):
        return float(trapezoid(trapezoid(pressure, y, axis=0), x))

    flooded_load = pressure_integral(flooded.pressure)
    partial_load = pressure_integral(partially_starved.pressure)

    assert partially_starved.inlet_fractional_film_content == 0.99
    assert partially_starved.pressure.max() > 0.0
    assert 0.0 < partial_load < flooded_load
    assert np.min(partially_starved.pressure) >= 0.0
    assert np.min(partially_starved.fractional_film_content) >= 0.0
    assert np.max(partially_starved.fractional_film_content) <= 1.0
    assert partially_starved.maximum_mass_balance_residual < 1e-8
    assert abs(partially_starved.net_boundary_flux) < 1e-8

    assert severely_starved.pressure.max() == 0.0
    assert pressure_integral(severely_starved.pressure) == 0.0
    np.testing.assert_allclose(
        severely_starved.fractional_film_content[1:-1, 0],
        0.9,
    )
    assert severely_starved.east_boundary_flux == pytest.approx(
        -severely_starved.west_boundary_flux, abs=1e-10
    )
    assert severely_starved.maximum_mass_balance_residual < 1e-8
    assert abs(severely_starved.net_boundary_flux) < 1e-8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dx", 0.0),
        ("dy", float("nan")),
        ("reynolds_parameter", 0.0),
        ("inlet_fractional_film_content", -0.01),
        ("inlet_fractional_film_content", 1.01),
        ("inlet_fractional_film_content", float("nan")),
    ],
)
def test_cavitation_rejects_invalid_numerical_inputs(field, value):
    film = np.ones((5, 5), dtype=np.float64)
    arguments = {
        "dx": 0.25,
        "dy": 0.25,
        "reynolds_parameter": 1.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        solve_cavitating_reynolds(film, **arguments)
