"""Device parity against independent spatial convolution and whole-face fluxes."""

import subprocess
import sys

import numpy as np
import pytest
from scipy.signal import convolve2d

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.device import DeviceFullFilmEvaluator
from ehl_solver.elasticity import ElasticityOperator, cell_integrated_kernel, elastic_deformation
from ehl_solver.hertz import circular_point_contact
from ehl_solver.monolithic import BorderedReynoldsElasticityProblem
from ehl_solver.reynolds import parabolic_point_contact_film, reynolds_residual


@pytest.fixture(params=["numpy", "cpu", "cuda"])
def backend(request):
    if request.param == "numpy":
        return make_backend()
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm device")
    return make_backend("torch", request.param)


def host(value, backend):
    return value if backend.name == "numpy" else value.detach().cpu().numpy()


@pytest.mark.parametrize("shape", [(3, 5), (6, 4), (1, 3), (33, 33)])
def test_cached_fft_direct_edges_signed_and_reuse(backend, shape):
    op = ElasticityOperator(shape, 1.7e-6, 2.3e-6, 113e9, backend)
    spectrum = op.kernel_spectrum
    kernel = cell_integrated_kernel(shape, 1.7e-6, 2.3e-6, 113e9)
    p = np.random.default_rng(52).normal(size=shape) * 1e8
    for field in (p, np.eye(*shape) * 1e8, np.zeros(shape)):
        full = convolve2d(field, kernel, mode="full")
        ny, nx = shape
        expected = full[ny - 1:2 * ny - 1, nx - 1:2 * nx - 1]
        actual = op(as_float64(field, backend))
        np.testing.assert_allclose(host(actual, backend), expected, rtol=3e-12, atol=2e-22)
        assert op.kernel_spectrum is spectrum
        if backend.name == "torch":
            assert str(actual.device).startswith(backend.device)
            assert actual.dtype == backend.xp.float64
    corner = np.zeros(shape)
    corner[0, -1] = 1e8
    np.testing.assert_allclose(
        host(op(as_float64(corner, backend)), backend),
        elastic_deformation(corner, 1.7e-6, 2.3e-6, 113e9),
        rtol=3e-12, atol=2e-22,
    )
    with pytest.raises(ValueError, match="shape"):
        op(np.ones((2, 2)))


def test_memory_and_parameter_guards():
    with pytest.raises(MemoryError):
        ElasticityOperator((513, 513), 1e-6, 1e-6, 1e11, memory_limit_bytes=1024)
    for shape in ((0, 3), (True, 3), (3,)):
        with pytest.raises(ValueError):
            ElasticityOperator(shape, 1e-6, 1e-6, 1e11)
    with pytest.raises(ValueError):
        ElasticityOperator((3, 3), 0, 1e-6, 1e11)


def test_cached_hertz_center_displacement_refines(backend):
    hertz = circular_point_contact(5., .015, 113e9)
    errors = []
    for n in (33, 65, 129):
        coordinates = np.linspace(-4 * hertz.radius_m, 4 * hertz.radius_m, n)
        x, y = np.meshgrid(coordinates, coordinates)
        inside = 1 - (x**2 + y**2) / hertz.radius_m**2
        # Analytical dry-contact pressure support, not solver-pressure clipping.
        pressure = hertz.peak_pressure_pa * np.sqrt(np.maximum(inside, 0))
        spacing = coordinates[1] - coordinates[0]
        op = ElasticityOperator((n, n), spacing, spacing, 113e9, backend)
        w = host(op(as_float64(pressure, backend)), backend)
        errors.append(abs(w[n//2, n//2] / hertz.approach_m - 1))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .01


def test_resident_calls_do_not_export_fields_and_fft_errors_propagate(backend, monkeypatch):
    op = ElasticityOperator((5, 7), 2e-6, 3e-6, 113e9, backend)
    p = as_float64(np.ones((5, 7)) * 1e8, backend)
    h = as_float64(np.ones((5, 7)), backend)

    def forbidden(*args, **kwargs):
        raise AssertionError("resident calculation attempted a host field export")

    if backend.name == "torch":
        monkeypatch.setattr(backend.xp.Tensor, "cpu", forbidden)
        monkeypatch.setattr(backend.xp.Tensor, "numpy", forbidden)
    for _ in range(3):
        assert op(p).shape == p.shape
        assert reynolds_residual(p, h, .1, .2, 12., backend=backend).shape == (3, 5)

    def failed_fft(*args, **kwargs):
        raise RuntimeError("injected FFT failure")

    monkeypatch.setattr(backend.xp.fft, "rfft2", failed_fft)
    with pytest.raises(RuntimeError, match="injected FFT failure"):
        op(p)


def test_reynolds_whole_face_divergence_and_geometry(backend):
    rng = np.random.default_rng(4)
    # Nonzero boundary pressure, anisotropic spacing, signed interior pressure.
    p = rng.normal(size=(7, 9))
    x, y = np.meshgrid(np.linspace(-1e-3, 1e-3, 9), np.linspace(-7e-4, 7e-4, 7))
    film = parabolic_point_contact_film(x, y, 20e-6, .01, .02)
    device_film = parabolic_point_contact_film(
        as_float64(x, backend), as_float64(y, backend), 20e-6, .01, .02,
        backend=backend,
    )
    np.testing.assert_allclose(host(device_film, backend), film, rtol=2e-15)
    h = film / 20e-6
    source = rng.normal(size=p.shape)
    dx, dy, lam = .17, .23, 12.
    # Independent flux storage on complete interfaces, with opposite sign to R.
    qx = -.5 * (h[:, 1:]**3 + h[:, :-1]**3) * np.diff(p, axis=1) / dx
    qx += lam * .5 * (h[:, 1:] + h[:, :-1])
    qy = -.5 * (h[1:, :]**3 + h[:-1, :]**3) * np.diff(p, axis=0) / dy
    expected = -(np.diff(qx, axis=1)[1:-1] / dx + np.diff(qy, axis=0)[:, 1:-1] / dy)
    expected -= source[1:-1, 1:-1]
    actual = reynolds_residual(
        as_float64(p, backend), device_film / 20e-6, dx, dy, lam,
        as_float64(source, backend), backend=backend,
    )
    np.testing.assert_allclose(host(actual, backend), expected, rtol=3e-13, atol=2e-11)
    np.testing.assert_allclose(
        host(actual, backend), reynolds_residual(p, h, dx, dy, lam, source),
        rtol=3e-13, atol=2e-11,
    )
    for invalid in (h * 0, h * np.nan, -h):
        with pytest.raises(ValueError, match="positive"):
            reynolds_residual(p, invalid, dx, dy, lam, backend=backend)


def test_cpu_import_and_execution_without_torch():
    # Fresh process catches accidental imports even when pytest has loaded torch.
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("torch deliberately unavailable")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import numpy as np
from ehl_solver.elasticity import ElasticityOperator
from ehl_solver.reynolds import reynolds_residual
assert ElasticityOperator((3, 3), 1e-6, 1e-6, 1e11)(np.ones((3, 3))).shape == (3, 3)
assert np.all(reynolds_residual(np.zeros((3, 3)), np.ones((3, 3)), 1., 1., 1.) == 0)
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_complete_full_film_residual_and_jvp(backend):
    problem = BorderedReynoldsElasticityProblem(
        np.linspace(-.5e-3, .75e-3, 9), np.linspace(-.5e-3, .5e-3, 7),
        .01, .02, 1e8, 5e-4, 20e-6, 21250., 12., -.001,
        compliance_factor=.6, reynolds_residual_scale=100.,
    )
    evaluator = DeviceFullFilmEvaluator(problem, backend)
    rng = np.random.default_rng(33)
    z = np.r_[rng.normal(size=problem.pressure_size) * .02, 1.]
    v = rng.normal(size=z.shape)
    v /= np.linalg.norm(v)
    zd, vd = as_float64(z, backend), as_float64(v, backend)
    for actual, expected in zip(
        evaluator.state_fields(zd), problem.state_fields(z[:-1], z[-1]), strict=True
    ):
        np.testing.assert_allclose(host(actual, backend), expected, rtol=2e-12, atol=1e-20)
    np.testing.assert_allclose(
        host(evaluator.residual(zd), backend), problem.residual(z), rtol=2e-12, atol=2e-13
    )
    for gamma_direction in (0., .3):
        actual = host(evaluator.jacobian_vector(
            zd, vd, compliance_direction=gamma_direction
        ), backend)
        expected = problem.jacobian_vector(z, v, compliance_direction=gamma_direction)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        step = 2e-6
        problem.compliance_factor = .6 + step * gamma_direction
        plus = problem.residual(z + step * v)
        problem.compliance_factor = .6 - step * gamma_direction
        minus = problem.residual(z - step * v)
        problem.compliance_factor = .6
        np.testing.assert_allclose(actual, (plus - minus) / (2 * step), rtol=2e-7, atol=1e-8)
    # Full Jacobian audit includes all pressure columns, h0 and force closure.
    eye = np.eye(z.size)
    observed = np.column_stack([
        host(evaluator.jacobian_vector(zd, as_float64(column, backend)), backend)
        for column in eye
    ])
    expected = np.column_stack([problem.jacobian_vector(z, column) for column in eye])
    np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=2e-12)
    assert np.any(observed[-1, :-1] != 0)
    assert observed[-1, -1] == 0
    for method in (evaluator.residual, lambda state: evaluator.jacobian_vector(state, vd)):
        with pytest.raises(ValueError, match="strictly"):
            method(as_float64(np.r_[np.zeros(problem.pressure_size), -10.], backend))
        with pytest.raises(ValueError, match="finite"):
            method(as_float64(z * np.nan, backend))
