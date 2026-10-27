import importlib

import numpy as np
import pytest
from scipy.signal import convolve2d

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same
from ehl_solver.elasticity import cell_integrated_kernel, elastic_deformation


@pytest.fixture(scope="module")
def hip_backend():
    try:
        native = importlib.import_module("ehl_solver._hip_native")
    except ImportError:
        pytest.skip("Native HIP extension is not built")
    try:
        if native.device_count() < 1:
            pytest.skip("No HIP GPU is available")
        return make_backend("hip")
    except RuntimeError as exc:
        pytest.skip(f"HIP runtime is unavailable: {exc}")


def _numpy_convolution(signal, kernel):
    full = convolve2d(signal, kernel, mode="full")
    y0, x0 = kernel.shape[0] // 2, kernel.shape[1] // 2
    return full[y0 : y0 + signal.shape[0], x0 : x0 + signal.shape[1]]


def _assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    difference = np.abs(actual - expected)
    significant = np.abs(expected) > np.max(np.abs(expected)) * 1e-10
    assert np.max(difference) <= 2e-12 + 2e-11 * np.max(np.abs(expected))
    assert np.max(difference[significant] / np.abs(expected[significant])) < 2e-11


def test_hip_impulse_response_and_kernel_centering(hip_backend):
    signal = np.zeros((7, 9), dtype=np.float64)
    signal[0, 0] = 1.0
    kernel = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    actual = fft_convolve_same(signal, kernel, hip_backend)
    _assert_close(actual, _numpy_convolution(signal, kernel))


@pytest.mark.parametrize("shape,kernel_shape", [((5, 8), (4, 3)), ((8, 5), (3, 4))])
def test_hip_asymmetric_rectangular_convolution(hip_backend, shape, kernel_shape):
    rng = np.random.default_rng(204)
    signal = rng.standard_normal(shape)
    kernel = rng.standard_normal(kernel_shape)
    _assert_close(
        fft_convolve_same(signal, kernel, hip_backend),
        _numpy_convolution(signal, kernel),
    )


def test_hip_boundary_behavior_has_no_wraparound(hip_backend):
    signal = np.zeros((9, 7), dtype=np.float64)
    signal[0, 0] = 1.0
    kernel = np.ones((5, 3), dtype=np.float64)
    actual = fft_convolve_same(signal, kernel, hip_backend)
    _assert_close(actual, _numpy_convolution(signal, kernel))
    np.testing.assert_allclose(actual[-2:, :], 0.0, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(actual[:, -1:], 0.0, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize("shape", [(13, 9), (16, 11)])
def test_hip_randomized_pressure_matches_numpy_displacement(hip_backend, shape):
    rng = np.random.default_rng(205)
    pressure_pa = rng.uniform(0.0, 2.0e8, size=shape)
    dx, dy, e_star_pa = 1.7e-6, 2.3e-6, 113.0e9
    numpy_result = elastic_deformation(pressure_pa, dx, dy, e_star_pa)
    hip_result = elastic_deformation(
        pressure_pa, dx, dy, e_star_pa, backend=hip_backend
    )
    assert hip_result.dtype == np.float64
    assert np.max(np.abs(hip_result - numpy_result)) < 1e-18
    relative = np.abs(hip_result - numpy_result) / np.maximum(
        np.abs(numpy_result), np.max(np.abs(numpy_result)) * 1e-14
    )
    assert np.max(relative) < 2e-10


def test_hip_matches_independent_direct_elastic_convolution(hip_backend):
    pressure_pa = np.zeros((7, 11), dtype=np.float64)
    pressure_pa[1, 8] = 1.4e8
    pressure_pa[5, 2] = 7.0e7
    kernel = cell_integrated_kernel(pressure_pa.shape, 2e-6, 3e-6, 113e9)
    expected = _numpy_convolution(pressure_pa, kernel)
    actual = elastic_deformation(pressure_pa, 2e-6, 3e-6, 113e9, hip_backend)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-18)
    assert np.all(actual >= 0.0)
