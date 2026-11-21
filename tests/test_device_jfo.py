"""Full donor/JFO device residual and bordered derivative verification."""

import numpy as np
import pytest
from scipy.sparse import bmat, csc_matrix, diags

from ehl_solver.backend import as_float64, make_backend
from ehl_solver.cavitating_newton import solve_cavitating_correction
from ehl_solver.device_jfo import DeviceDonorJFO
from ehl_solver.device_jfo_krylov import solve_device_jfo_correction
from ehl_solver.device_jfo_preconditioner import ExperimentalDeviceDonorJFOPreconditioner
from ehl_solver.finite_roller_ehl import (
    FiniteRollerEHLCase,
    finite_roller_hertz_seed,
    verify_finite_roller_cuda_parity,
)
from ehl_solver.hybrid_jfo import (
    CPUBlockSchurPreconditioner,
    HybridJFOCorrectionSolver,
    _force_constrained_moment_sensitivity,
)
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


def test_force_constrained_moment_sensitivity_uses_bordered_schur_response():
    sensitivity, normalized, poor = _force_constrained_moment_sensitivity(
        np.array([[2.0, 0.5], [1.0, 3.0]]),
        moment_scale_n_m=0.05,
        angle_scale_rad=5e-4,
        tolerance=1e-10,
    )
    assert normalized == pytest.approx(2.75)
    assert sensitivity == pytest.approx(275.0)
    assert poor is False

    sensitivity, normalized, poor = _force_constrained_moment_sensitivity(
        np.array([[2.0, 0.5], [1.0, 0.25]]),
        moment_scale_n_m=0.05,
        angle_scale_rad=5e-4,
        tolerance=1e-10,
    )
    assert normalized == pytest.approx(0.0)
    assert sensitivity == pytest.approx(0.0)
    assert poor is True

    assert _force_constrained_moment_sensitivity(
        np.array([[0.0, 0.5], [1.0, 3.0]]),
        moment_scale_n_m=0.05,
        angle_scale_rad=5e-4,
        tolerance=1e-10,
    ) == (None, None, True)


@pytest.mark.parametrize("workspace_limit_bytes", [0, -1, True, 1.0])
def test_hybrid_workspace_limit_must_be_a_positive_integer(workspace_limit_bytes):
    with pytest.raises(ValueError, match="workspace_limit_bytes"):
        HybridJFOCorrectionSolver(
            PointContactCase(), workspace_limit_bytes=workspace_limit_bytes
        )


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


@pytest.mark.parametrize(
    "transport_fraction", [0.0, 0.35, 0.9991455078125, 1.0]
)
def test_blended_mc_device_residual_and_frozen_jvp_match_cpu(
    backend, transport_fraction, monkeypatch
):
    torch = pytest.importorskip("torch")
    case, donor, state = _case_state()
    rng = np.random.default_rng(20261118)
    guide = rng.normal(size=donor.size) * 1e-3
    direction = rng.normal(size=donor.size) * 1e-3
    active_full = state[:donor.n] >= state[donor.n:2 * donor.n]
    cpu = LimitedLoadLinearization(
        case,
        state,
        guide,
        active_full=active_full,
        transport_fraction=transport_fraction,
    )
    device = DeviceDonorJFO(
        case, backend, transport_fraction=transport_fraction
    )
    z, g, v = (
        as_float64(value, backend) for value in (state, guide, direction)
    )
    expected_residual = cpu.residual(state)
    expected_jvp = cpu.jacobian_vector(state, direction)
    limiter_branches = (
        None
        if cpu.limiter_branches is None
        else tuple(
            backend.xp.as_tensor(
                branch.copy(),
                dtype=backend.xp.int64,
                device=backend.device,
            )
            for branch in cpu.limiter_branches
        )
    )
    mask = backend.xp.as_tensor(
        active_full, dtype=backend.xp.bool, device=backend.device
    )
    actual_residual = device.linear_residual(z, mask)
    action = device.linearize(
        z, g, active_full=mask, limiter_branches=limiter_branches
    )
    actual_jvp = action(v)
    np.testing.assert_allclose(
        actual_residual.cpu().numpy(), expected_residual, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        actual_jvp.cpu().numpy(), expected_jvp, rtol=5e-11, atol=2e-12
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unexpected host field transfer")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    assert device.residual(z).device == z.device
    assert action(v).device == z.device


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


def test_local_device_preconditioner_matches_cpu_frozen_matrix(backend):
    case, cpu, state = _case_state()
    full = state[:cpu.n] >= state[cpu.n:2 * cpu.n]
    guide = np.zeros(cpu.size)
    model = LimitedLoadLinearization(
        case, state, guide, preconditioner="local", active_full=full,
        transport_fraction=0,
    )
    device = DeviceDonorJFO(case, backend)
    z = as_float64(state, backend)
    mask = backend.xp.as_tensor(full, dtype=backend.xp.bool, device=backend.device)
    inverse = ExperimentalDeviceDonorJFOPreconditioner(device, z, mask)
    direction = np.random.default_rng(2027).normal(size=2 * cpu.n) * 1e-4
    actual = inverse.matrix_vector(as_float64(direction, backend)).cpu().numpy()
    expected = model.frozen_matrix(state)[:2 * cpu.n, :2 * cpu.n] @ direction
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-9)

    rhs = as_float64(np.random.default_rng(2028).normal(size=cpu.size), backend)
    preconditioned = inverse(rhs)
    assert preconditioned.shape == rhs.shape
    assert preconditioned.dtype == backend.xp.float64
    assert preconditioned.device == rhs.device
    assert bool(backend.xp.isfinite(preconditioned).all())


def test_selected_newton_residual_matches_cpu_active_rows(backend):
    case, cpu, state = _case_state()
    full = state[:cpu.n] >= state[cpu.n:2 * cpu.n]
    device = DeviceDonorJFO(case, backend)
    z = as_float64(state, backend)
    mask = backend.xp.as_tensor(full, dtype=backend.xp.bool, device=backend.device)
    model = LimitedLoadLinearization(
        case, state, np.zeros(cpu.size), preconditioner="local",
        active_full=full, transport_fraction=0,
    )
    actual = device.linear_residual(z, mask).cpu().numpy()
    np.testing.assert_allclose(actual, model.residual(state), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        device.residual(z).cpu().numpy(), cpu.residual(state), rtol=2e-12, atol=2e-12
    )


def test_device_bordered_gmres_correction_matches_cpu_superlu(backend):
    case, cpu, state = _case_state()
    full = state[:cpu.n] >= state[cpu.n:2 * cpu.n]
    model = LimitedLoadLinearization(
        case, state, np.zeros(cpu.size), preconditioner="local",
        active_full=full, transport_fraction=0,
    )
    reference = solve_cavitating_correction(
        model, state, maximum_krylov_cycles=10, gmres_restart=60, krylov_rtol=1e-9
    )
    device = DeviceDonorJFO(case, backend)
    z = as_float64(state, backend)
    mask = backend.xp.as_tensor(full, dtype=backend.xp.bool, device=backend.device)
    actual = solve_device_jfo_correction(
        device, z, mask, rtol=1e-9, restart=60, max_iterations=600
    )
    correction = actual.direction.cpu().numpy()
    relative = np.linalg.norm(correction - reference.direction) / np.linalg.norm(
        reference.direction
    )
    assert actual.linear_residual_relative <= 5e-9
    assert relative < 5e-7


def test_cpu_schur_preconditioner_applies_full_frozen_border(backend):
    case, cpu, state = _case_state()
    full = state[:cpu.n] >= state[cpu.n:2 * cpu.n]
    model = LimitedLoadLinearization(
        case, state, np.zeros(cpu.size), preconditioner="local",
        active_full=full, transport_fraction=0,
    )
    matrix = model.frozen_matrix(state)
    inverse = CPUBlockSchurPreconditioner(matrix, cpu.n, backend.device)
    rhs = np.random.default_rng(2029).normal(size=cpu.size)
    actual = inverse(as_float64(rhs, backend)).cpu().numpy()
    np.testing.assert_allclose(matrix @ actual, rhs, rtol=2e-10, atol=2e-10)
    assert inverse.application_count == 1
    assert inverse.superlu_setup_seconds > 0
    assert inverse.superlu_solve_seconds > 0
    assert inverse.device_to_host_bytes == rhs.nbytes
    assert inverse.host_to_device_bytes == rhs.nbytes
    fallback = CPUBlockSchurPreconditioner(
        matrix, cpu.n, backend.device, schur_tolerance=1.0
    )
    fallback_actual = fallback(as_float64(rhs, backend)).cpu().numpy()
    assert fallback.used_bordered_fallback
    np.testing.assert_allclose(matrix @ fallback_actual, rhs, rtol=2e-10, atol=2e-10)
    with pytest.raises(ValueError, match="schur_tolerance"):
        CPUBlockSchurPreconditioner(
            matrix, cpu.n, backend.device, schur_tolerance=0.0
        )


def test_cpu_schur_preconditioner_solves_two_scalar_borders(backend):
    n = 3
    field = diags(np.arange(1.0, 2 * n + 1), format="csc")
    columns = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, -0.5],
            [-0.5, 1.0],
        ]
    )
    matrix = bmat(
        [
            [field, csc_matrix(columns)],
            [csc_matrix(columns.T), csc_matrix((2, 2))],
        ],
        format="csc",
    )
    rhs = np.random.default_rng(907).normal(size=2 * n + 2)
    inverse = CPUBlockSchurPreconditioner(matrix, n, backend.device)
    actual = inverse(as_float64(rhs, backend)).cpu().numpy()
    np.testing.assert_allclose(matrix @ actual, rhs, rtol=2e-12, atol=2e-12)
    assert inverse.border_size == 2
    assert not inverse.used_bordered_fallback

    fallback = CPUBlockSchurPreconditioner(
        matrix, n, backend.device, schur_tolerance=1.0
    )
    fallback_actual = fallback(as_float64(rhs, backend)).cpu().numpy()
    assert fallback.used_bordered_fallback
    np.testing.assert_allclose(
        matrix @ fallback_actual, rhs, rtol=2e-12, atol=2e-12
    )


def test_moment_controlled_device_jvp_and_hybrid_correction_match_cpu(backend):
    case = FiniteRollerEHLCase(nx=9, ny=13)
    base_probe = LimitedLoadProbe(case, transport_fraction=0.0)
    state = np.r_[finite_roller_hertz_seed(case), 0.0]
    guide = np.zeros_like(state)
    active_full = state[:base_probe.n] >= state[base_probe.n : 2 * base_probe.n]
    model = LimitedLoadLinearization(
        case,
        state,
        guide,
        preconditioner="local",
        active_full=active_full,
        transport_fraction=0.0,
        moment_target_n_m=0.0,
    )
    device = DeviceDonorJFO(
        case, backend, transport_fraction=0.0, moment_target_n_m=0.0
    )
    z = as_float64(state, backend)
    active = backend.xp.as_tensor(
        active_full, dtype=backend.xp.bool, device=backend.device
    )
    np.testing.assert_allclose(
        device.linear_residual(z, active).cpu().numpy(),
        model.residual(state),
        rtol=2e-12,
        atol=2e-12,
    )
    if backend.device == "cuda":
        parity = verify_finite_roller_cuda_parity(
            case, state, model.probe
        )
        assert parity["passed"]

    direction = np.random.default_rng(2029).normal(size=model.size) * 1e-3
    direction[-1] = 0.4
    actual_jvp = device.linearize(z, as_float64(guide, backend), active_full=active)(
        as_float64(direction, backend)
    ).cpu().numpy()
    expected_jvp = model.jacobian_vector(state, direction)
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=5e-11, atol=2e-12)

    reference = solve_cavitating_correction(
        model, state, maximum_krylov_cycles=10, gmres_restart=60
    )
    hybrid = HybridJFOCorrectionSolver(case, backend)
    actual = hybrid(model, state)
    np.testing.assert_allclose(
        model.jacobian_vector(state, actual.direction) + model.residual(state),
        0.0,
        rtol=0.0,
        atol=5e-9 * np.linalg.norm(model.residual(state)),
    )
    assert actual.true_linear_residual <= 5e-9
    sensitivity = actual.timings[
        "force_constrained_moment_sensitivity_n_m_per_rad"
    ]
    normalized_sensitivity = actual.timings[
        "normalized_force_constrained_moment_sensitivity"
    ]
    assert sensitivity is not None
    assert normalized_sensitivity == pytest.approx(
        sensitivity
        * model.probe.angle_scale_rad
        / model.probe.moment_scale_n_m
    )
    assert isinstance(actual.timings["poor_angular_sensitivity"], bool)
    assert np.linalg.norm(actual.direction - reference.direction) / np.linalg.norm(
        reference.direction
    ) < 5e-7


def test_hybrid_cuda_operator_cpu_schur_correction_matches_reference(backend):
    case, cpu, state = _case_state()
    full = state[:cpu.n] >= state[cpu.n:2 * cpu.n]
    model = LimitedLoadLinearization(
        case, state, np.zeros(cpu.size), preconditioner="local",
        active_full=full, transport_fraction=0.5,
    )
    reference = solve_cavitating_correction(
        model, state, maximum_krylov_cycles=10, gmres_restart=60, krylov_rtol=1e-9
    )
    hybrid = HybridJFOCorrectionSolver(case, backend)
    progress = []
    actual = hybrid(
        model, state, progress_callback=lambda **event: progress.append(event)
    )
    relative = np.linalg.norm(actual.direction - reference.direction) / np.linalg.norm(
        reference.direction
    )
    assert actual.true_linear_residual <= 5e-9
    assert actual.linear_iterations <= 600
    assert relative < 5e-7
    assert actual.timings["cpu_preconditioner_applications"] > 0
    assert actual.timings["gmres_iterations"] == actual.linear_iterations
    assert actual.timings["gmres_restart_count"] >= 0
    assert actual.timings["cpu_superlu_setup_seconds"] > 0
    assert actual.timings["cpu_superlu_solve_seconds"] > 0
    assert actual.timings["linear_rhs_h2d_bytes"] == model.size * 8
    assert actual.timings["cpu_residual_parity_h2d_bytes"] == model.size * 8
    assert actual.timings["cpu_residual_parity_linf"] <= 1e-10
    assert any(event["stage"] == "preconditioner_setup" for event in progress)
    assert any(event["stage"] == "superlu_factorization" for event in progress)
    assert any(event["stage"] == "superlu_solve" for event in progress)
    ready = next(event for event in progress if event["stage"] == "preconditioner_ready")
    assert ready["cumulative_preconditioner_seconds"] >= 0
    assert ready["cumulative_superlu_seconds"] > 0
    validation = next(
        event for event in progress if event["stage"] == "linear_validation"
    )
    assert validation["gmres_iterations"] == actual.linear_iterations
    assert "gmres_restart_count" in validation
    with pytest.raises(ValueError, match="tolerance or iteration budget"):
        hybrid(model, state, schur_tolerance=0.0)
