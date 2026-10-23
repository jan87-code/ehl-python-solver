import numpy as np
import pytest
from scipy.signal import convolve2d

from ehl_solver.backend import make_backend
from ehl_solver.convolution import fft_convolve_same


@pytest.mark.parametrize("shape,kshape", [((3, 5), (3, 3)), ((7, 9), (5, 3)), ((6, 4), (4, 2))])
def test_fft_vs_direct(shape, kshape):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape)
    k = rng.standard_normal(kshape)
    actual = fft_convolve_same(x, k, make_backend())
    full = convolve2d(x, k, mode="full")
    y0, x0 = kshape[0] // 2, kshape[1] // 2
    expected = full[y0 : y0 + shape[0], x0 : x0 + shape[1]]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_no_circular_wraparound():
    x = np.zeros((5, 5))
    x[0, 0] = 1
    k = np.ones((3, 3))
    out = fft_convolve_same(x, k, make_backend())
    np.testing.assert_allclose(out[4, :], 0, atol=1e-12)
    np.testing.assert_allclose(out[:, 4], 0, atol=1e-12)


def test_bad_dimensions():
    with pytest.raises(ValueError):
        fft_convolve_same(np.ones(5), np.ones((3, 3)), make_backend())
