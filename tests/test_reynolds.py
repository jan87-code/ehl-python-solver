import numpy as np
import pytest

from ehl_solver.reynolds import (
    assemble_reynolds_system,
    reynolds_residual,
    solve_reynolds_pressure,
)


def test_constant_pressure_and_film_have_zero_residual():
    pressure = np.full((5, 7), 0.4)
    film = np.full((5, 7), 1.2)

    residual = reynolds_residual(pressure, film, 0.25, 0.5, 3.0)

    assert residual.shape == (3, 5)
    np.testing.assert_array_equal(residual, 0.0)


def test_linear_film_wedge_uses_mean_surface_speed_parameter():
    ny, nx = 5, 7
    dx = 0.25
    x = np.arange(nx) * dx
    film = np.broadcast_to(1.0 + 0.2 * x, (ny, nx))
    pressure = np.zeros_like(film)
    reynolds_parameter = 3.0

    residual = reynolds_residual(pressure, film, dx, 0.5, reynolds_parameter)

    np.testing.assert_allclose(residual, -reynolds_parameter * 0.2, atol=1e-15)


def _manufactured_solution_error(n: int) -> float:
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx = dy = 1.0 / (n - 1)
    slope_x, slope_y = 0.2, 0.1
    reynolds_parameter = 0.7

    film = 1.0 + slope_x * xx + slope_y * yy
    pressure = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    pressure_x = np.pi * np.cos(np.pi * xx) * np.sin(np.pi * yy)
    pressure_y = np.pi * np.sin(np.pi * xx) * np.cos(np.pi * yy)
    laplacian_pressure = -2.0 * np.pi**2 * pressure
    exact_divergence = (
        film**3 * laplacian_pressure
        + 3.0 * film**2 * (slope_x * pressure_x + slope_y * pressure_y)
    )
    source = exact_divergence - reynolds_parameter * slope_x

    residual = reynolds_residual(
        pressure, film, dx, dy, reynolds_parameter, source=source
    )
    return float(np.sqrt(np.mean(residual**2)))


def test_manufactured_solution_residual_converges_at_second_order():
    errors = [_manufactured_solution_error(n) for n in (17, 33, 65)]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.08)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.08)


def _manufactured_pressure_case(n: int):
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx = dy = 1.0 / (n - 1)
    slope_x, slope_y = 0.2, 0.1
    reynolds_parameter = 0.7
    film = 1.0 + slope_x * xx + slope_y * yy
    exact_pressure = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    pressure_x = np.pi * np.cos(np.pi * xx) * np.sin(np.pi * yy)
    pressure_y = np.pi * np.sin(np.pi * xx) * np.cos(np.pi * yy)
    laplacian_pressure = -2.0 * np.pi**2 * exact_pressure
    exact_divergence = (
        film**3 * laplacian_pressure
        + 3.0 * film**2 * (slope_x * pressure_x + slope_y * pressure_y)
    )
    source = exact_divergence - reynolds_parameter * slope_x
    return film, exact_pressure, source, dx, dy, reynolds_parameter


def test_sparse_system_matches_conservative_residual():
    rng = np.random.default_rng(52)
    film = 1.0 + rng.uniform(0.0, 0.2, size=(6, 8))
    pressure = np.zeros_like(film)
    pressure[1:-1, 1:-1] = rng.standard_normal((4, 6))
    source = rng.standard_normal(film.shape)
    dx, dy, reynolds_parameter = 0.17, 0.23, 0.8

    matrix, rhs = assemble_reynolds_system(
        film, dx, dy, reynolds_parameter, source=source
    )
    algebraic_residual = matrix @ pressure[1:-1, 1:-1].ravel() - rhs
    conservative_residual = reynolds_residual(
        pressure, film, dx, dy, reynolds_parameter, source=source
    )

    assert matrix.dtype == np.float64
    assert matrix.shape == (24, 24)
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T, atol=1e-14)
    assert np.linalg.eigvalsh(matrix.toarray()).min() > 0.0
    np.testing.assert_allclose(
        algebraic_residual.reshape(conservative_residual.shape),
        -conservative_residual,
        rtol=1e-13,
        atol=1e-12,
    )


def _manufactured_pressure_error(n: int) -> float:
    film, exact_pressure, source, dx, dy, reynolds_parameter = (
        _manufactured_pressure_case(n)
    )
    solution = solve_reynolds_pressure(
        film, dx, dy, reynolds_parameter, source=source, rtol=1e-12, atol=1e-13
    )
    return float(np.sqrt(np.mean((solution.pressure - exact_pressure) ** 2)))


def test_manufactured_pressure_solution_converges_at_second_order():
    errors = [_manufactured_pressure_error(n) for n in (17, 33, 65)]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.1)


def test_manufactured_pressure_solution_has_zero_edges_and_small_diagnostics():
    film, exact_pressure, source, dx, dy, reynolds_parameter = (
        _manufactured_pressure_case(33)
    )
    solution = solve_reynolds_pressure(
        film, dx, dy, reynolds_parameter, source=source, rtol=1e-12, atol=1e-13
    )

    assert solution.pressure.dtype == np.float64
    np.testing.assert_array_equal(solution.pressure[0, :], 0.0)
    np.testing.assert_array_equal(solution.pressure[-1, :], 0.0)
    np.testing.assert_array_equal(solution.pressure[:, 0], 0.0)
    np.testing.assert_array_equal(solution.pressure[:, -1], 0.0)
    assert np.sqrt(np.mean((solution.pressure - exact_pressure) ** 2)) < 5e-4
    assert solution.algebraic_residual_relative_l2 < 1e-11
    assert solution.mass_balance_relative_error < 1e-11


def test_converging_wedge_pressure_is_positive_and_mass_conserving():
    ny, nx = 17, 41
    x = np.linspace(0.0, 1.0, nx)
    film = np.broadcast_to(1.4 - 0.4 * x, (ny, nx)).copy()
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)

    solution = solve_reynolds_pressure(
        film, dx, dy, reynolds_parameter=2.0, rtol=1e-12, atol=1e-13
    )

    assert solution.pressure.max() > 0.0
    assert np.all(solution.pressure[1:-1, 1:-1] > 0.0)
    np.testing.assert_allclose(
        solution.pressure, solution.pressure[::-1, :], rtol=1e-12, atol=1e-13
    )
    assert solution.algebraic_residual_relative_l2 < 1e-11
    assert solution.mass_balance_relative_error < 1e-11
    assert abs(solution.mass_balance_absolute_error) < 1e-10


def test_full_film_solver_preserves_negative_pressure_without_clipping():
    film = np.ones((7, 9))
    source = np.ones_like(film)

    solution = solve_reynolds_pressure(
        film, 0.1, 0.2, reynolds_parameter=0.0, source=source
    )

    assert np.all(solution.pressure[1:-1, 1:-1] < 0.0)


@pytest.mark.parametrize(
    "pressure,film,dx,dy,reynolds_parameter,source",
    [
        (np.ones(5), np.ones(5), 1.0, 1.0, 0.0, None),
        (np.ones((3, 3)), np.ones((4, 3)), 1.0, 1.0, 0.0, None),
        (np.ones((2, 3)), np.ones((2, 3)), 1.0, 1.0, 0.0, None),
        (np.ones((3, 3)), np.zeros((3, 3)), 1.0, 1.0, 0.0, None),
        (np.ones((3, 3)), np.ones((3, 3)), 0.0, 1.0, 0.0, None),
        (np.ones((3, 3)), np.ones((3, 3)), 1.0, 1.0, -1.0, None),
        (np.ones((3, 3)), np.ones((3, 3)), 1.0, 1.0, 0.0, np.ones((2, 3))),
        (np.ones((3, 3)), np.ones((3, 3)), 1.0, 1.0, 0.0, np.full((3, 3), np.nan)),
    ],
)
def test_invalid_reynolds_inputs_raise_value_error(
    pressure, film, dx, dy, reynolds_parameter, source
):
    with pytest.raises(ValueError):
        reynolds_residual(pressure, film, dx, dy, reynolds_parameter, source)


@pytest.mark.parametrize(
    "rtol,atol,max_iterations",
    [(-1.0, 0.0, None), (0.0, -1.0, None), (0.0, 0.0, None), (1e-8, 0.0, 0)],
)
def test_invalid_solver_options_raise_value_error(rtol, atol, max_iterations):
    with pytest.raises(ValueError):
        solve_reynolds_pressure(
            np.ones((3, 3)),
            1.0,
            1.0,
            0.0,
            rtol=rtol,
            atol=atol,
            max_iterations=max_iterations,
        )


def test_solver_reports_nonconvergence():
    x = np.linspace(0.0, 1.0, 20)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    film = 1.0 + 0.2 * xx + 0.1 * yy
    source = np.sin(np.pi * xx) * np.sin(np.pi * yy)

    with pytest.raises(RuntimeError, match="did not converge"):
        solve_reynolds_pressure(
            film, x[1] - x[0], x[1] - x[0], 0.5, source=source, max_iterations=1
        )
