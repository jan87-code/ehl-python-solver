"""Full donor/JFO device residual and bordered derivative verification."""

import numpy as np
import pytest

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.device_jfo import DeviceDonorJFO
from ehl_solver.limited_system import LimitedLoadLinearization, LimitedLoadProbe
from ehl_solver.prototype import PointContactCase


@pytest.fixture(params=["cpu", "cuda"])
def backend(request):
    torch = pytest.importorskip("torch")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is unavailable")
    return make_backend("torch", request.param)


def _case_state():
    case = PointContactCase(ny=9, nx=13)
    probe = LimitedLoadProbe(case, transport_fraction=0)
    rng = np.random.default_rng(42)
    z = np.r_[rng.uniform(0.1, 0.3, probe.n), rng.uniform(0.01, 0.05, probe.n), 0.5]
    return case, probe, z


def _directions(probe):
    rng = np.random.default_rng(43)
    random = rng.normal(size=probe.size) * 1e-3
    localized_pressure = np.zeros(probe.size)
    localized_pressure[probe.n // 2] = 1e-3
    localized_content = np.zeros(probe.size)
    localized_content[probe.n + 1] = 1e-3
    offset = np.zeros(probe.size)
    offset[-1] = 1e-3
    return random, localized_pressure, localized_content, offset


def test_complete_donor_residual_and_jvp_with_finite_differences(backend):
    case, cpu, state = _case_state()
    device = DeviceDonorJFO(case, backend)
    z = as_float64(state, backend)
    actual = device.residual(z).cpu().numpy()
    np.testing.assert_allclose(actual, cpu.residual(state), rtol=2e-12, atol=2e-12)
    guide = np.zeros(cpu.size)
    model = LimitedLoadLinearization(case, state, guide, transport_fraction=0)
    action = device.linearize(z, as_float64(guide, backend))
    for direction in _directions(cpu):
        cpu_jvp = model.jacobian_vector(state, direction)
        device_jvp = action(as_float64(direction, backend)).cpu().numpy()
        np.testing.assert_allclose(device_jvp, cpu_jvp, rtol=2e-11, atol=1e-12)
        step = 1e-3
        fd = (device.residual(z + step * as_float64(direction, backend))
              - device.residual(z - step * as_float64(direction, backend))) / (2 * step)
        error = np.linalg.norm(fd.cpu().numpy() - device_jvp) / np.linalg.norm(device_jvp)
        assert error < 2e-7


def test_ties_active_overrides_and_resident_operations(backend, monkeypatch):
    torch = pytest.importorskip("torch")
    case, cpu, state = _case_state()
    state[:cpu.n] = state[cpu.n:2 * cpu.n]
    guide = np.random.default_rng(44).normal(size=cpu.size)
    direction = np.random.default_rng(45).normal(size=cpu.size) * 1e-4
    device = DeviceDonorJFO(case, backend)
    z, g, v = (as_float64(a, backend) for a in (state, guide, direction))
    model = LimitedLoadLinearization(case, state, guide, transport_fraction=0)
    np.testing.assert_allclose(
        device.linearize(z, g)(v).cpu().numpy(),
        model.jacobian_vector(state, direction), rtol=2e-11, atol=1e-12,
    )
    np.testing.assert_allclose(
        device.directional_derivative(z, v).cpu().numpy(),
        cpu.derivative(state, direction), rtol=2e-11, atol=1e-12,
    )
    full = np.arange(cpu.n) % 2 == 0
    model = LimitedLoadLinearization(
        case, state, guide, active_full=full, transport_fraction=0
    )
    mask = torch.tensor(full, device=z.device)
    action = device.linearize(z, g, active_full=mask)
    np.testing.assert_allclose(
        action(v).cpu().numpy(), model.jacobian_vector(state, direction),
        rtol=2e-11, atol=1e-12,
    )
    with pytest.raises(ValueError, match="bordered vector"):
        device.residual(state)
    invalid = z.clone()
    invalid[-1] = -100
    with pytest.raises(ValueError, match="film"):
        device.residual(invalid)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected host field transfer")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    assert device.residual(z).device == z.device
    assert action(v).device == z.device
    assert device.directional_derivative(z, v).device == z.device
