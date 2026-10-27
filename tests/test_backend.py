import importlib.util

import numpy as np
import pytest

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.convolution import fft_convolve_same


def test_numpy_backend():
    b = make_backend()
    assert b.device == "cpu"
    assert as_float64([1, 2], b).dtype == np.float64


def test_unknown_backend():
    with pytest.raises(ValueError):
        make_backend("invalid")


def test_hip_backend_does_not_fall_back_when_extension_is_missing():
    if importlib.util.find_spec("ehl_solver._hip_native") is not None:
        pytest.skip("Native HIP extension is installed")
    with pytest.raises(RuntimeError, match="extension is not built"):
        make_backend("hip")


def test_torch_cpu_optional():
    pytest.importorskip("torch")
    b = make_backend("torch", "cpu")
    x = np.arange(35).reshape(5, 7)
    k = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
    result = fft_convolve_same(x, k, b).cpu().numpy()
    reference = fft_convolve_same(x, k, make_backend())
    np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)


def test_torch_gpu_optional():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA or ROCm GPU available")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((21, 17))
    k = rng.standard_normal((7, 5))
    gpu = fft_convolve_same(x, k, make_backend("torch", "cuda"))
    cpu = fft_convolve_same(x, k, make_backend())
    np.testing.assert_allclose(gpu.cpu().numpy(), cpu, rtol=1e-11, atol=1e-11)
