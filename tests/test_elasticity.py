import math

import numpy as np
import pytest
from scipy.signal import convolve2d

from ehl_solver.backend import make_backend
from ehl_solver.elasticity import cell_integrated_kernel, elastic_deformation
from ehl_solver.hertz import circular_point_contact, pressure_radial


def test_kernel_is_symmetric_positive_and_integrates_singular_cell():
    dx, dy, e_star = 2.0e-6, 3.0e-6, 113.0e9
    kernel = cell_integrated_kernel((5, 7), dx, dy, e_star)
    center = (4, 6)
    expected = (
        2 * dx * math.asinh(dy / dx) + 2 * dy * math.asinh(dx / dy)
    ) / (math.pi * e_star)

    assert kernel.shape == (9, 13)
    assert kernel.dtype == np.float64
    assert np.all(kernel > 0)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], rtol=1e-14)
    assert kernel[center] == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("shape", [(3, 5), (6, 4)])
def test_elastic_deformation_matches_direct_linear_convolution(shape):
    rng = np.random.default_rng(31)
    pressure = rng.uniform(0, 2e8, size=shape)
    dx, dy, e_star = 1.7e-6, 2.3e-6, 113e9
    kernel = cell_integrated_kernel(shape, dx, dy, e_star)
    actual = elastic_deformation(pressure, dx, dy, e_star)
    full = convolve2d(pressure, kernel, mode="full")
    y0, x0 = kernel.shape[0] // 2, kernel.shape[1] // 2
    expected = full[y0 : y0 + shape[0], x0 : x0 + shape[1]]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)


def _hertz_displacement_error(n: int) -> float:
    load, radius, e_star = 5.0, 0.015, 113.0e9
    hertz = circular_point_contact(load, radius, e_star)
    half_width = 4 * hertz.radius_m
    coordinates = np.linspace(-half_width, half_width, n)
    spacing = coordinates[1] - coordinates[0]
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    pressure = np.array(
        [
            pressure_radial(math.hypot(float(xv), float(yv)), hertz)
            for xv, yv in zip(x.flat, y.flat, strict=True)
        ]
    ).reshape(x.shape)
    displacement = elastic_deformation(pressure, spacing, spacing, e_star)
    expected_center = hertz.approach_m
    return abs(float(displacement[n // 2, n // 2]) - expected_center) / expected_center


def test_hertz_center_displacement_converges_with_grid_refinement():
    errors = [_hertz_displacement_error(n) for n in (33, 65, 129)]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 0.01


def test_torch_cpu_matches_numpy_when_available():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(32)
    pressure = rng.uniform(0, 2e8, size=(5, 7))
    numpy_result = elastic_deformation(pressure, 2e-6, 3e-6, 113e9)
    torch_result = elastic_deformation(
        pressure, 2e-6, 3e-6, 113e9, make_backend("torch", "cpu")
    )

    assert torch_result.dtype == torch.float64
    np.testing.assert_allclose(torch_result.numpy(), numpy_result, rtol=1e-12, atol=1e-18)


def test_torch_gpu_matches_numpy_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA or ROCm GPU available")
    rng = np.random.default_rng(33)
    pressure = rng.uniform(0, 2e8, size=(5, 7))
    numpy_result = elastic_deformation(pressure, 2e-6, 3e-6, 113e9)
    gpu_result = elastic_deformation(
        pressure, 2e-6, 3e-6, 113e9, make_backend("torch", "cuda")
    )

    assert gpu_result.dtype == torch.float64
    np.testing.assert_allclose(gpu_result.cpu().numpy(), numpy_result, rtol=1e-11, atol=1e-18)


@pytest.mark.parametrize(
    "grid_shape,dx,dy,e_star",
    [((0, 3), 1.0, 1.0, 1.0), ((3, 3), 0.0, 1.0, 1.0), ((3, 3), 1.0, 1.0, 0.0)],
)
def test_invalid_kernel_parameters(grid_shape, dx, dy, e_star):
    with pytest.raises(ValueError):
        cell_integrated_kernel(grid_shape, dx, dy, e_star)


def test_deformation_requires_2d_pressure():
    with pytest.raises(ValueError, match="2-D"):
        elastic_deformation(np.ones(3), 1.0, 1.0, 1.0)
