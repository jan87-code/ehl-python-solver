import numpy as np
import pytest
from scipy.integrate import trapezoid

from ehl_solver.line_cavitation import solve_cavitating_line_reynolds
from ehl_solver.line_contact import integrate_line_pressure


def _converging_diverging_film(size):
    x = np.linspace(0.0, 1.0, size)
    return x, 1.4 - 0.4 * x + 0.8 * x**2


def _independent_flux_residual(film, pressure, saturation, dx, reynolds_parameter):
    face_film = 0.5 * (film[:-1] + film[1:])
    face_film_cubed = 0.5 * (film[:-1] ** 3 + film[1:] ** 3)
    face_flux = (
        -face_film_cubed * np.diff(pressure) / dx
        + reynolds_parameter * face_film * saturation[:-1]
    )
    return np.diff(face_flux) / dx, face_flux


def test_full_film_line_solution_matches_independent_tridiagonal_system():
    x = np.linspace(0.0, 1.0, 41)
    film = 1.4 - 0.4 * x
    dx = x[1] - x[0]
    reynolds_parameter = 2.0
    interior = film[1:-1]
    interior_cubed = interior**3
    matrix = np.zeros((interior.size, interior.size))
    rhs = np.zeros(interior.size)

    for i in range(1, film.size - 1):
        row = i - 1
        h_east_cubed = 0.5 * (interior_cubed[row] + film[i + 1] ** 3)
        h_west_cubed = 0.5 * (film[i - 1] ** 3 + interior_cubed[row])
        matrix[row, row] = (h_east_cubed + h_west_cubed) / dx**2
        if row + 1 < interior.size:
            matrix[row, row + 1] = -h_east_cubed / dx**2
        if row > 0:
            matrix[row, row - 1] = -h_west_cubed / dx**2
        h_east = 0.5 * (film[i] + film[i + 1])
        h_west = 0.5 * (film[i - 1] + film[i])
        rhs[row] = -reynolds_parameter * (h_east - h_west) / dx

    solution = solve_cavitating_line_reynolds(
        film, dx, reynolds_parameter
    )

    np.testing.assert_allclose(solution.pressure[1:-1], np.linalg.solve(matrix, rhs))
    np.testing.assert_array_equal(solution.pressure[[0, -1]], 0.0)
    np.testing.assert_allclose(solution.fractional_film_content, 1.0)
    assert solution.active_set_iterations == 1
    assert solution.maximum_complementarity_violation == 0.0
    assert solution.maximum_mass_balance_residual < 1e-9
    assert abs(solution.net_boundary_flux) < 1e-10


def test_line_jfo_is_admissible_locally_conservative_and_load_auditable():
    x, film = _converging_diverging_film(41)
    dx = x[1] - x[0]
    solution = solve_cavitating_line_reynolds(film, dx, 2.0)
    residual, flux = _independent_flux_residual(
        film,
        solution.pressure,
        solution.fractional_film_content,
        dx,
        2.0,
    )

    assert solution.pressure.shape == film.shape
    assert solution.fractional_film_content.shape == film.shape
    np.testing.assert_array_equal(solution.pressure[[0, -1]], 0.0)
    assert solution.fractional_film_content[0] == 1.0
    assert 0.0 <= solution.pressure.min()
    assert 0.0 <= solution.fractional_film_content.min()
    assert solution.fractional_film_content.max() <= 1.0
    assert np.any(solution.fractional_film_content[1:-1] < 1.0 - 1e-8)
    assert np.all(
        solution.pressure[1:-1][solution.fractional_film_content[1:-1] < 1.0 - 1e-8]
        == 0.0
    )
    assert solution.maximum_complementarity_violation < 1e-10
    assert solution.maximum_mass_balance_residual < 1e-10
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)
    assert flux[0] == pytest.approx(-solution.west_boundary_flux, abs=1e-14)
    assert flux[-1] == pytest.approx(solution.east_boundary_flux, abs=1e-14)
    assert abs(solution.net_boundary_flux) < 1e-10

    length_scale_m = 0.01
    pressure_scale_pa = 1e6
    x_m = length_scale_m * x
    pressure_pa = pressure_scale_pa * solution.pressure
    line_load_n_m = integrate_line_pressure(pressure_pa, x_m).line_load_n_m
    independent_line_load_n_m = trapezoid(pressure_pa, x_m)
    assert line_load_n_m == pytest.approx(independent_line_load_n_m, rel=2e-14)
    assert line_load_n_m == pytest.approx(
        pressure_scale_pa * length_scale_m * trapezoid(solution.pressure, x),
        rel=2e-14,
    )


def test_line_jfo_pressure_integral_refines_on_converging_diverging_film():
    pressure_integrals = []
    for size in (21, 41, 81):
        x, film = _converging_diverging_film(size)
        solution = solve_cavitating_line_reynolds(film, x[1] - x[0], 2.0)
        pressure_integrals.append(trapezoid(solution.pressure, x))
        assert solution.maximum_mass_balance_residual < 1e-9
        assert abs(solution.net_boundary_flux) < 1e-9

    changes = np.abs(np.diff(pressure_integrals))
    assert pressure_integrals[0] < pressure_integrals[1] < pressure_integrals[2]
    assert changes[1] < changes[0]


def test_line_jfo_inlet_starvation_reduces_pressure_and_conserves_flux():
    x, film = _converging_diverging_film(41)
    dx = x[1] - x[0]
    flooded = solve_cavitating_line_reynolds(film, dx, 2.0)
    starved = solve_cavitating_line_reynolds(
        film, dx, 2.0, inlet_fractional_film_content=0.99
    )

    assert starved.inlet_fractional_film_content == 0.99
    assert starved.fractional_film_content[0] == 0.99
    assert trapezoid(starved.pressure, x) < trapezoid(flooded.pressure, x)
    assert starved.pressure.max() < flooded.pressure.max()
    assert starved.maximum_mass_balance_residual < 1e-10
    assert abs(starved.net_boundary_flux) < 1e-10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dx", 0.0),
        ("reynolds_parameter", 0.0),
        ("inlet_fractional_film_content", -0.01),
        ("inlet_fractional_film_content", 1.01),
        ("pressure_tolerance", float("nan")),
        ("saturation_tolerance", -1.0),
        ("max_active_set_iterations", 0),
        ("equilibrate_linear_system", "false"),
    ],
)
def test_line_jfo_rejects_invalid_inputs(field, value):
    arguments = {
        "dx": 0.25,
        "reynolds_parameter": 1.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        solve_cavitating_line_reynolds(np.ones(9), **arguments)
