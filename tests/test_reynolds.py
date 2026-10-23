import numpy as np
import pytest

from ehl_solver.reynolds import reynolds_residual


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
    ],
)
def test_invalid_reynolds_inputs_raise_value_error(
    pressure, film, dx, dy, reynolds_parameter, source
):
    with pytest.raises(ValueError):
        reynolds_residual(pressure, film, dx, dy, reynolds_parameter, source)
