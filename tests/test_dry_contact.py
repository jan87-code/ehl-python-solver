import numpy as np
import pytest

from ehl_solver.backend import make_backend
from ehl_solver.dry_contact import DryContactConvergenceError, solve_dry_contact
from ehl_solver.geometry import CircularPointContactGeometry, EllipticalPointContactGeometry
from ehl_solver.hertz import circular_point_contact, elliptical_point_contact


@pytest.mark.parametrize('grid_size', [33, 65])
def test_dry_circular_contact_closes_load_and_converges_to_hertz(grid_size):
    load, radius, e_star = 5.0, 0.015, 113e9
    reference = circular_point_contact(load, radius, e_star)
    x = np.linspace(-3 * reference.radius_m, 3 * reference.radius_m, grid_size)
    result = solve_dry_contact(
        x, x, CircularPointContactGeometry(radius), load, e_star
    )
    center_pressure = float(result.pressure_pa[grid_size // 2, grid_size // 2])

    assert result.loads.force_n == pytest.approx(load, rel=2e-14)
    assert result.relative_kkt_residual <= 2e-6
    assert result.maximum_penetration_m < 2e-5 * reference.approach_m
    assert result.approach_m == pytest.approx(reference.approach_m, rel=3e-3)
    assert center_pressure == pytest.approx(reference.peak_pressure_pa, rel=3e-3)
    assert np.all(result.pressure_pa >= 0)


def test_dry_elliptical_contact_matches_hertz_axes_peak_and_approach():
    load, radius_x, radius_y, e_star = 5.0, 0.015, 0.06, 113e9
    reference = elliptical_point_contact(load, radius_x, radius_y, e_star)
    x = np.linspace(-3 * reference.semi_axis_x_m, 3 * reference.semi_axis_x_m, 65)
    y = np.linspace(-3 * reference.semi_axis_y_m, 3 * reference.semi_axis_y_m, 65)
    result = solve_dry_contact(
        x, y, EllipticalPointContactGeometry(radius_x, radius_y), load, e_star
    )
    center_pressure = float(result.pressure_pa[32, 32])

    assert result.loads.force_n == pytest.approx(load, rel=2e-14)
    assert result.relative_kkt_residual <= 2e-6
    assert result.approach_m == pytest.approx(reference.approach_m, rel=3e-3)
    assert center_pressure == pytest.approx(reference.peak_pressure_pa, rel=3e-3)
    active_y, active_x = np.nonzero(result.pressure_pa > 0)
    resolved_x = np.max(np.abs(x[active_x]))
    resolved_y = np.max(np.abs(y[active_y]))
    assert resolved_x == pytest.approx(reference.semi_axis_x_m, abs=x[1] - x[0])
    assert resolved_y == pytest.approx(reference.semi_axis_y_m, abs=y[1] - y[0])


def test_dry_contact_retains_last_iterate_when_iteration_budget_is_exhausted():
    reference = circular_point_contact(5.0, 0.015, 113e9)
    x = np.linspace(-3 * reference.radius_m, 3 * reference.radius_m, 33)

    with pytest.raises(DryContactConvergenceError) as error:
        solve_dry_contact(
            x,
            x,
            CircularPointContactGeometry(0.015),
            5.0,
            113e9,
            tolerance=1e-14,
            maximum_iterations=1,
        )

    assert error.value.last_result.iterations == 1
    assert error.value.last_result.loads.force_n == pytest.approx(5.0, rel=2e-14)


def test_dry_contact_torch_cpu_matches_numpy_reference():
    torch = pytest.importorskip('torch')
    load, radius, e_star = 5.0, 0.015, 113e9
    reference = circular_point_contact(load, radius, e_star)
    x = np.linspace(-3 * reference.radius_m, 3 * reference.radius_m, 33)
    geometry = CircularPointContactGeometry(radius)
    numpy_result = solve_dry_contact(x, x, geometry, load, e_star)
    torch_result = solve_dry_contact(
        x, x, geometry, load, e_star, backend=make_backend('torch', 'cpu')
    )

    assert torch_result.pressure_pa.dtype == torch.float64
    np.testing.assert_allclose(
        torch_result.pressure_pa.numpy(), numpy_result.pressure_pa, rtol=2e-5, atol=1e-3
    )


def test_dry_contact_torch_gpu_matches_numpy_when_available():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('No CUDA or ROCm GPU available')
    load, radius, e_star = 5.0, 0.015, 113e9
    reference = circular_point_contact(load, radius, e_star)
    x = np.linspace(-3 * reference.radius_m, 3 * reference.radius_m, 33)
    geometry = CircularPointContactGeometry(radius)
    numpy_result = solve_dry_contact(x, x, geometry, load, e_star)
    gpu_result = solve_dry_contact(
        x, x, geometry, load, e_star, backend=make_backend('torch', 'cuda')
    )

    assert gpu_result.pressure_pa.dtype == torch.float64
    np.testing.assert_allclose(
        gpu_result.pressure_pa.cpu().numpy(), numpy_result.pressure_pa, rtol=2e-5, atol=1e-3
    )
