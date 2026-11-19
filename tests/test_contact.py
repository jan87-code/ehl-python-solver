import numpy as np
import pytest

from ehl_solver.contact import integrate_contact_pressure


def test_uniform_pressure_integrates_force_and_zero_symmetric_moments():
    x = np.linspace(-2e-3, 2e-3, 5)
    y = np.linspace(-3e-3, 3e-3, 7)
    pressure = np.full((y.size, x.size), 2e6)
    dx, dy = x[1] - x[0], y[1] - y[0]

    loads = integrate_contact_pressure(pressure, x, y)

    assert loads.force_n == pytest.approx(pressure.sum() * dx * dy)
    assert loads.moment_x_n_m == pytest.approx(0.0, abs=1e-15)
    assert loads.moment_y_n_m == pytest.approx(0.0, abs=1e-15)


def test_contact_moments_use_right_handed_pressure_force_signs():
    x = np.arange(5, dtype=np.float64) * 1e-3
    y = np.arange(5, dtype=np.float64) * 2e-3
    pressure = np.zeros((5, 5))
    pressure[3, 4] = 1e6

    loads = integrate_contact_pressure(pressure, x, y)

    assert loads.force_n == pytest.approx(2.0)
    assert loads.moment_x_n_m == pytest.approx(-12e-3)
    assert loads.moment_y_n_m == pytest.approx(8e-3)


@pytest.mark.parametrize(
    'pressure,x,y',
    [
        (np.ones((3, 3)), [0.0, 1.0], [0.0, 1.0]),
        (np.ones((3, 3)), [0.0, 1.0, 2.1], [0.0, 1.0, 2.0]),
        (-np.ones((3, 3)), [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
    ],
)
def test_contact_integrator_rejects_invalid_pressure_or_grid(pressure, x, y):
    with pytest.raises(ValueError):
        integrate_contact_pressure(pressure, x, y)
