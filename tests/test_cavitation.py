import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix

from ehl_solver.cavitating_coupled import CavitatingElasticityProblem
from ehl_solver.cavitation import _solve_active_system, solve_cavitating_reynolds
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
        ("equilibrate_linear_system", "false"),
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


def test_equilibrated_linear_solution_recovers_original_coordinates_and_dense_root():
    # Unequal mass/constraint units and pressure/content magnitudes.
    matrix = np.array([[2e12, -3e11, 2e4], [0, 1, 0], [5e7, 1e3, 8]])
    rhs = np.array([5e4, 0.8, 1.7])
    expected = np.linalg.solve(matrix, rhs)
    actual = _solve_active_system(csr_matrix(matrix), rhs, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    scale = np.abs(matrix) @ np.abs(actual) + np.abs(rhs)
    assert np.max(np.abs(matrix @ actual - rhs) / scale) < 1e-14


@pytest.mark.parametrize("matrix", [np.zeros((2, 2)), [[1, 0], [2, 0]]])
def test_equilibration_rejects_unusable_rows_or_columns(matrix):
    with pytest.raises(RuntimeError, match="zero or nonfinite"):
        _solve_active_system(csr_matrix(matrix, dtype=float), np.ones(2), True)


def test_rigid_farfield_equilibration_preserves_root_and_original_mass_equation():
    h0, rx, ry, eta, speed = 1e-7, 0.01, 0.02, 0.04, 0.001
    length = np.sqrt(2 * rx * h0)
    pressure_scale = 12 * eta * speed * length / h0**2
    x = np.arange(-64, 25) * 0.125 * length
    y = np.arange(-64, 65) * 0.125 * np.sqrt(2) * length
    problem = CavitatingElasticityProblem(
        x, y, h0, rx, ry, 1e11, length, h0, pressure_scale, 1
    )
    film = h0 + problem.rigid_geometry_m
    scaled = solve_cavitating_reynolds(film / h0, problem.dx, problem.dy, 1)
    original = solve_cavitating_reynolds(
        film / h0, problem.dx, problem.dy, 1, equilibrate_linear_system=False
    )
    np.testing.assert_allclose(scaled.pressure, original.pressure, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        scaled.fractional_film_content, original.fractional_film_content, atol=1e-9
    )
    # The coupled implementation independently differences complete face arrays.
    independent = problem.mass_residual(
        scaled.pressure * pressure_scale, scaled.fractional_film_content, film
    )
    assert np.max(np.abs(independent)) < 2e-10
    assert scaled.maximum_mass_balance_residual < 2e-10
    assert scaled.maximum_complementarity_violation < 1e-12
    assert scaled.pressure.min() >= -1e-10
    assert scaled.fractional_film_content.min() >= 0
    assert scaled.fractional_film_content.max() <= 1 + 1e-10
    assert abs(scaled.net_boundary_flux) < 1e-8
